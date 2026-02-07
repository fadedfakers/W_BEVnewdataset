import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.box_ops import decode_boxes, bev_nms
from utils.intrusion_logic import IntrusionLogic

def visualize_2x2(image, lidar_bev, pred_mask, det_boxes, scores, save_path, raw_max, alarms):
    # 🔥 实例化逻辑类：使用更宽的轨道判定 (4.0m) 以捕获边缘目标
    logic = IntrusionLogic(roi_width_meters=4.0, voxel_size=cfg.VOXEL_SIZE)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    
    # 1. RGB
    mean = np.array(cfg.IMG_MEAN).reshape(1, 1, 3)
    std = np.array(cfg.IMG_STD).reshape(1, 1, 3)
    image_display = (image * std + mean) * 255.0
    image_display = np.clip(image_display, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(image_display)
    axes[0, 0].set_title(f"Input RGB (Max Signal: {raw_max:.5f})")
    axes[0, 0].axis('off')
    
    # 2. LiDAR
    if lidar_bev.ndim == 3:
        lidar_img = lidar_bev.max(axis=0)
    else:
        lidar_img = lidar_bev
    axes[0, 1].imshow(lidar_img, cmap='viridis', origin='upper')
    axes[0, 1].set_title("LiDAR BEV Features")
    axes[0, 1].axis('off')
    
    # 3. Rail Mask & Fit
    if raw_max > 1e-6:
        norm_mask = pred_mask / raw_max
    else:
        norm_mask = pred_mask

    rail_coeffs = logic.fit_rail_lines(norm_mask)
    
    fitting_canvas = (norm_mask * 255).astype(np.uint8)
    fitting_canvas = cv2.cvtColor(fitting_canvas, cv2.COLOR_GRAY2BGR)
    
    # 绘制拟合线
    if rail_coeffs is not None:
        H, W = pred_mask.shape 
        xs = np.arange(W)
        a, b, c = rail_coeffs
        ys = a * xs**2 + b * xs + c 
        valid_mask = (ys >= 0) & (ys < H)
        pts = np.stack([xs[valid_mask], ys[valid_mask]], axis=1).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(fitting_canvas, [pts], isClosed=False, color=(255, 0, 0), thickness=3)
        title_suffix = "(Fit Success)"
    else:
        title_suffix = "(Fit Failed)"

    axes[1, 0].imshow(fitting_canvas, origin='upper')
    axes[1, 0].set_title(f"Rail Segmentation {title_suffix}")
    axes[1, 0].axis('off')
    
    # 4. Safety Analysis
    result_bev = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # (A) 画轨道区域 (半透明效果模拟)
    if rail_coeffs is not None:
        # 使用逻辑类中的宽度配置
        width_px = int(logic.roi_width_px)
        pts_top = np.stack([xs[valid_mask], ys[valid_mask] - width_px//2], axis=1).astype(np.int32)
        pts_bot = np.stack([xs[valid_mask], ys[valid_mask] + width_px//2], axis=1).astype(np.int32)
        pts_all = np.concatenate([pts_top, pts_bot[::-1]])
        cv2.fillPoly(result_bev, [pts_all], color=(0, 100, 100)) # 暗青色底
        cv2.polylines(result_bev, [pts], isClosed=False, color=(0, 255, 255), thickness=2) # 黄色中心线

    # (C) 绘制报警框
    for alarm in alarms:
        box = alarm['box']
        score = alarm['score']
        level = alarm['level']
        msg = alarm['msg']
        
        x1, y1, x2, y2 = map(int, box)
        
        # 颜色：CRITICAL=红, WARNING=橙
        color = (255, 0, 0) if level == "CRITICAL" else (255, 165, 0)
        
        # 画粗框
        cv2.rectangle(result_bev, (x1, y1), (x2, y2), color, 3)
        
        # 画标签背景，确保字能看清
        label = f"{msg}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_bev, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(result_bev, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 统计信息
    n_crit = sum(1 for a in alarms if a['level'] == 'CRITICAL')
    n_warn = sum(1 for a in alarms if a['level'] == 'WARNING')
    
    # 顶部状态栏
    status_color = (255, 0, 0) if n_crit > 0 else ((255, 165, 0) if n_warn > 0 else (0, 255, 0))
    status_text = "STATUS: DANGER" if (n_crit + n_warn) > 0 else "STATUS: SAFE"
    
    cv2.rectangle(result_bev, (0, 0), (pred_mask.shape[1], 40), status_color, -1)
    cv2.putText(result_bev, f"{status_text} | CRITICAL: {n_crit} | WARNING: {n_warn}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    axes[1, 1].imshow(result_bev, origin='upper')
    axes[1, 1].set_title("Safety Analysis (High Sensitivity Mode)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Saved BEST sample to {save_path}")

def find_and_visualize_best(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading model: {checkpoint_path}")
    
    model = WBEVFusionNet(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    dataset = BEVMultiTaskDataset(data_root=data_root, split='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    
    # 🔥 激进版配置：为了捕获 0.47 分的检测框
    # 1. 放宽 ROI 到 4.0 米
    # 2. 手动调整阈值：高阈值 0.35, 低阈值 0.15
    logic = IntrusionLogic(roi_width_meters=4.0, voxel_size=cfg.VOXEL_SIZE)
    logic.CONF_HIGH = 0.35  # 只要 > 0.35 就算 CRITICAL
    logic.CONF_LOW = 0.15   # 只要 > 0.15 且在轨道内就算 WARNING
    
    print(f"🔍 Scanning frames (Sensitivity: High=0.35, Low=0.15)...")
    results = [] 
    
    with torch.no_grad():
        for i, (images, points, targets) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            points_list = [p.to(device) for p in points]
            outputs = model(images, points_list)
            
            # 1. 轨道拟合
            mask_prob = torch.sigmoid(outputs['mask_pred'][0, 0])
            max_val = mask_prob.max().item()
            mask_np = mask_prob.cpu().numpy()
            
            if max_val > 1e-6:
                norm_mask = mask_np / max_val
            else:
                norm_mask = mask_np
            rail_coeffs = logic.fit_rail_lines(norm_mask)
            
            # 2. 检测框
            fake_output = {
                'box_pred': outputs['box_pred'],
                'cls_pred': outputs['cls_pred'],
                'mask_pred': None 
            }
            det_boxes_batch = decode_boxes(fake_output, K=100, threshold=0.01)
            det_boxes = det_boxes_batch[0]
            keep = bev_nms(det_boxes, iou_threshold=0.1)
            det_boxes = det_boxes[keep]
            
            scores = det_boxes[:, 5].cpu().numpy()
            boxes_pixel = det_boxes.cpu().numpy()
            
            # 3. 运行逻辑 (使用激进阈值)
            boxes_list = []
            for box in boxes_pixel:
                x, y, w, l, rot = box[:5]
                boxes_list.append([x - w/2, y - l/2, x + w/2, y + l/2])
                
            alarms = logic.check_intrusion(boxes_list, scores, rail_coeffs, mask_np.shape)
            
            # 4. 危险评分
            danger_score = 0
            for a in alarms:
                if a['level'] == 'CRITICAL':
                    danger_score += 10.0 + a['score']
                elif a['level'] == 'WARNING':
                    danger_score += 5.0 + a['score']
            
            # 如果没有报警，按最高分排序，方便看看有没有差点报警的
            if len(alarms) == 0:
                danger_score = scores.max() if len(scores) > 0 else 0
            
            results.append({
                "id": i,
                "danger_score": danger_score,
                "alarms": alarms,
                "max_val": max_val,
                "boxes": boxes_pixel,
                "scores": scores,
                "mask_prob": mask_np,
                "img": images.cpu(),
                "pts": points_list[0].cpu()
            })

    results.sort(key=lambda x: x['danger_score'], reverse=True)
    top_N = 20
    top_results = results[:top_N]
    
    print(f"\n🏆 Top {top_N} Most Dangerous Frames:")
    
    for rank, res in enumerate(top_results):
        n_crit = sum(1 for a in res['alarms'] if a['level'] == 'CRITICAL')
        n_warn = sum(1 for a in res['alarms'] if a['level'] == 'WARNING')
        print(f"  #{rank+1}: Frame {res['id']} | DangerScore: {res['danger_score']:.2f} | Critical: {n_crit}, Warning: {n_warn}, MaxConf: {res['scores'].max() if len(res['scores'])>0 else 0:.2f}")
        
        img_tensor = res['img'].to(device)
        pts_tensor = res['pts'].to(device)
        
        with torch.no_grad():
            lidar_features = model.lidar_backbone([pts_tensor])
            lidar_bev_real = lidar_features[0].cpu().numpy()
        
        img_np = res['img'][0].permute(1, 2, 0).numpy()
        save_name = f"danger_sample_rank{rank+1}_frame{res['id']}.png"
        
        visualize_2x2(
            img_np, 
            lidar_bev_real, 
            res['mask_prob'], 
            res['boxes'], 
            res['scores'], 
            save_name, 
            res['max_val'],
            res['alarms']
        )

if __name__ == "__main__":
    DATA_ROOT = "/root/autodl-tmp/FOD/data"
    CKPT = "/root/autodl-tmp/FOD/W-BEVFusion/checkpoints/model_e95.pth"
    
    if os.path.exists(CKPT):
        find_and_visualize_best(CKPT, DATA_ROOT)
    else:
        print(f"❌ Checkpoint not found: {CKPT}")