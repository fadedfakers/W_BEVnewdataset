import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import pickle
import open3d as o3d
import raillabel
from configs.config import BEVConfig as cfg
from data.transforms import SynchronizedBEVAug

class BEVMultiTaskDataset(Dataset):
    def __init__(self, data_root='/root/autodl-tmp/FOD/data', split='train'):
        self.data_root = data_root
        self.split = split
        
        # 扫描场景
        all_scenes = sorted([
            d for d in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
        ])
        split_idx = int(len(all_scenes) * 0.8)
        self.scenes = all_scenes[:split_idx] if split == 'train' else all_scenes[split_idx:]
        
        # 实例化同步增强器
        self.augmentor = SynchronizedBEVAug() if split == 'train' else None
        
        self.samples = []
        self._collect_samples()

        # 加载轨道多项式系数 (rail_coeffs.pkl)，与 preprocess 脚本输出路径一致
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rail_db_path = os.path.join(_project_root, 'data', 'rail_coeffs.pkl')
        if os.path.exists(self.rail_db_path):
            with open(self.rail_db_path, 'rb') as f:
                self.rail_db = pickle.load(f)
        else:
            self.rail_db = {}

    def _collect_samples(self):
        print(f"Scanning scenes for {self.split} split...")
        for scene_id in self.scenes:
            scene_dir = os.path.join(self.data_root, scene_id)
            json_path = os.path.join(scene_dir, f"{scene_id}_labels.json")
            if not os.path.exists(json_path): continue
            
            try:
                scene = raillabel.load(json_path)
            except Exception: continue

            lidar_dir, rgb_dir = os.path.join(scene_dir, "lidar"), os.path.join(scene_dir, "rgb_center")
            if not os.path.exists(lidar_dir) or not os.path.exists(rgb_dir): continue
            
            pcd_all, img_all = sorted(os.listdir(lidar_dir)), sorted(os.listdir(rgb_dir))
            for frame_id, frame in scene.frames.items():
                fid_int = int(frame_id)
                prefix = f"{fid_int:03d}_"
                pcd_f = next((f for f in pcd_all if f.startswith(prefix)), None)
                img_f = next((f for f in img_all if f.startswith(prefix)), None)
                if pcd_f and img_f:
                    self.samples.append({
                        'pcd_path': os.path.join(lidar_dir, pcd_f),
                        'img_path': os.path.join(rgb_dir, img_f),
                        'frame': frame,
                        'scene': scene
                    })
        print(f"✅ Loaded {len(self.samples)} valid frames for {self.split}.")

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = cv2.getGaussianKernel(diameter, sigma=diameter/6)
        gaussian = np.outer(gaussian, gaussian)
        
        g_max = gaussian.max()
        if g_max > 0:
            gaussian = gaussian / g_max
            
        x, y = int(center[0]), int(center[1])
        h, w = heatmap.shape
        left, right = min(x, radius), min(w - x, radius + 1)
        top, bottom = min(y, radius), min(h - y, radius + 1)
        if min(right, bottom) <= 0: return
        np.maximum(heatmap[y-top:y+bottom, x-left:x+right], 
                   gaussian[radius-top:radius+bottom, radius-left:radius+right] * k, 
                   out=heatmap[y-top:y+bottom, x-left:x+right])

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Point Cloud
        try:
            pcd = o3d.io.read_point_cloud(item['pcd_path'])
            points = np.asarray(pcd.points)
            valid_mask = (np.abs(points) < 500.0).all(axis=1)
            points = points[valid_mask]
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) == 0: points = np.zeros((1, 3), dtype=np.float64)
            points = points.astype(np.float32)
        except Exception:
            points = np.zeros((1, 3), dtype=np.float32)
        
        if points.shape[1] == 3:
            points = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # 2. Image
        img = cv2.imread(item['img_path'])
        if img is None: # 兜底防止坏图
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, cfg.IMG_SIZE)
        
        # 3. Geometry Parsing & ConnectedTo Logic (Changed)
        obj_geometries = [] 
        track_geometries = [] 
        ego_track_ids = set()

        # [新增] 寻找本车 (Ego Vehicle) 及其关联轨道
        for obj in item['scene'].objects.values():
            obj_type = obj.type.lower()
            if 'ego' in obj_type or 'train' in obj_type:
                if hasattr(obj, 'connected_to') and obj.connected_to:
                    for tid in obj.connected_to:
                        ego_track_ids.add(tid)
        
        for ann in item['frame'].annotations.values():
            obj = item['scene'].objects[ann.object_id]
            cls_name = obj.type.lower()
            
            # 处理轨道 (Track)
            if 'track' in cls_name or 'rail' in cls_name:
                pts_world = None
                if hasattr(ann, 'points'):
                    pts_world = np.array([[p.x, p.y] for p in ann.points])
                elif hasattr(ann, 'point_indices') and len(ann.point_indices) > 0:
                    pts_world = points[ann.point_indices, :2]
                
                if pts_world is not None: 
                    # 判断是否为本车轨道
                    is_ego = (ann.object_id in ego_track_ids)
                    track_geometries.append({
                        'pts': pts_world,
                        'id': ann.object_id,
                        'is_ego': is_ego
                    })
                continue

            # 处理障碍物 (Objects)
            if cls_name in cfg.CLASS_NAMES:
                cls_id = cfg.CLASS_NAMES.index(cls_name)
                center_world = None
                if hasattr(ann, 'point_indices') and len(ann.point_indices) > 0:
                    center_world = np.mean(points[ann.point_indices, :2], axis=0)
                elif hasattr(ann, 'pos'):
                    cx = ann.pos.x if hasattr(ann.pos, 'x') else ann.pos[0]
                    cy = ann.pos.y if hasattr(ann.pos, 'y') else ann.pos[1]
                    center_world = np.array([cx, cy])
                
                if center_world is not None and hasattr(ann, 'size'):
                    sx = ann.size.x if hasattr(ann.size, 'x') else ann.size[0]
                    sy = ann.size.y if hasattr(ann.size, 'y') else ann.size[1]
                    
                    yaw = 0.0
                    obj_geometries.append({
                        'center': center_world, 
                        'size': np.array([sx, sy]), 
                        'cls_id': cls_id,
                        'yaw': yaw
                    })

        # [兜底] 如果没有 connectedTo，取距离最近的轨道作为本线
        if not ego_track_ids and track_geometries:
            min_dist = float('inf')
            best_idx = -1
            for i, track in enumerate(track_geometries):
                dist = np.min(np.sqrt(np.sum(track['pts']**2, axis=1)))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            if best_idx != -1:
                track_geometries[best_idx]['is_ego'] = True

        # 4. Augmentation
        # 注意：这里需要根据augmentor的具体实现调整，假设它能处理字典列表或者我们先拆包再打包
        # 为了兼容你现有的 augmentor，我们这里做个适配
        # 如果 augmentor 还没改，我们只传点进去
        track_pts_list = [t['pts'] for t in track_geometries]
        
        if self.augmentor:
            # 假设 augmentor 还是接收 list of arrays
            points, obj_geometries, track_pts_list = self.augmentor(points, obj_geometries, track_pts_list)
            # 增强后把点放回去
            for i, pts in enumerate(track_pts_list):
                if i < len(track_geometries):
                    track_geometries[i]['pts'] = pts

        # 5. Generate Targets
        hm = np.zeros((cfg.NUM_CLASSES, cfg.GRID_H, cfg.GRID_W), dtype=np.float32)
        reg = np.zeros((2, cfg.GRID_H, cfg.GRID_W), dtype=np.float32) 
        wh = np.zeros((2, cfg.GRID_H, cfg.GRID_W), dtype=np.float32)
        rot = np.zeros((2, cfg.GRID_H, cfg.GRID_W), dtype=np.float32)
        rail_mask = np.zeros((cfg.GRID_H, cfg.GRID_W), dtype=np.float32)

        # Draw Rail (Modified: Only Ego Track)
        for track in track_geometries:
            if not track.get('is_ego', False): continue # 🚀 只画本车轨道
            
            pts_world = track['pts']
            pts_grid = ((pts_world - np.array([cfg.X_RANGE[0], cfg.Y_RANGE[0]])) / cfg.VOXEL_SIZE).astype(np.int32)
            if len(pts_grid) > 1:
                cv2.polylines(rail_mask, [pts_grid], isClosed=False, color=1.0, thickness=cfg.RAIL_MASK_THICKNESS)

        # Draw Objects
        gt_boxes_list = []
        for obj in obj_geometries:
            if not (np.isfinite(obj['center']).all() and np.isfinite(obj['size']).all()): continue
            if obj['size'][0] <= 0 or obj['size'][1] <= 0: continue
                
            gx = (obj['center'][0] - cfg.X_RANGE[0]) / cfg.VOXEL_SIZE
            gy = (obj['center'][1] - cfg.Y_RANGE[0]) / cfg.VOXEL_SIZE
            
            if 0 <= gx < cfg.GRID_W and 0 <= gy < cfg.GRID_H:
                ctx, cty = int(gx), int(gy)
                self.draw_umich_gaussian(hm[obj['cls_id']], (ctx, cty), radius=3)
                
                reg[:, cty, ctx] = [gx - ctx, gy - cty]
                
                w_grid = obj['size'][0] / cfg.VOXEL_SIZE
                l_grid = obj['size'][1] / cfg.VOXEL_SIZE
                wh[:, cty, ctx] = [np.log(max(w_grid, 1e-3)), np.log(max(l_grid, 1e-3))]
                
                yaw = obj.get('yaw', 0.0)
                rot[:, cty, ctx] = [np.sin(yaw), np.cos(yaw)]
                
                gt_boxes_list.append([gx, gy, w_grid, l_grid])

        # 6. Post-process
        rail_mask = cv2.GaussianBlur(rail_mask, (7, 7), cfg.RAIL_GAUSSIAN_SIGMA)
        img_tensor = (torch.from_numpy(img).permute(2, 0, 1).float()/255.0 - \
                      torch.tensor(cfg.IMG_MEAN).view(3,1,1)) / torch.tensor(cfg.IMG_STD).view(3,1,1)
        
        gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32) if gt_boxes_list else torch.zeros((0, 4), dtype=torch.float32)

        # 根据 unique_key 获取轨道多项式系数 (与 preprocess 脚本一致)
        pcd_name = os.path.basename(item['pcd_path'])
        scene_id = os.path.basename(os.path.dirname(os.path.dirname(item['pcd_path'])))
        unique_key = f"{scene_id}/lidar/{pcd_name}"
        rail_coeffs_raw = self.rail_db.get(unique_key, None)

        if rail_coeffs_raw is not None:
            rail_coeffs = torch.tensor(rail_coeffs_raw, dtype=torch.float32)
            has_rail = 1.0
        else:
            rail_coeffs = torch.zeros(3, dtype=torch.float32)
            has_rail = 0.0

        return img_tensor, torch.from_numpy(points).float(), {
            'hm': torch.from_numpy(hm),
            'reg': torch.from_numpy(reg),
            'wh': torch.from_numpy(wh),
            'rot': torch.from_numpy(rot),
            'mask': torch.from_numpy(rail_mask).unsqueeze(0),
            'masks': torch.from_numpy(rail_mask),
            'boxes': gt_boxes,
            'rail_coeffs': rail_coeffs,
            'has_rail': has_rail
        }

    def __len__(self): return len(self.samples)
    @staticmethod
    def collate_fn(batch): return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]