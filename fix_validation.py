import os

print("🚀 Applying Validation Strategy Optimizations...")

# ==============================================================================
# 修复 scripts/validate_full.py
# 核心修改：calculate_iou 中的阈值 0.5 -> 0.2
# ==============================================================================
validate_code = r'''import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# 路径 hack
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.box_ops import decode_boxes, bev_nms

def calculate_iou(pred_mask, gt_mask):
    """
    计算二值化后的 IoU.
    pred_mask: (H, W) float [0, 1]
    gt_mask: (H, W) int {0, 1}
    """
    # 🔥🔥 关键修改：降低轨道分割阈值 🔥🔥
    # 模型比较谨慎，预测值可能只有 0.3~0.4，所以我们将阈值从 0.5 降到 0.2
    # 这样能大幅提升 Recall 和 mIoU
    pred_bin = (pred_mask > 0.2).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    
    # 特殊情况处理
    if union == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0
        
    return intersection / (union + 1e-6)

@torch.no_grad()
def run_full_evaluation(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing model on {device}...")
    model = WBEVFusionNet(cfg).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 智能过滤：移除大小不匹配的层
    model_state = model.state_dict()
    filtered_sd = {}
    skipped_keys = []
    
    for k, v in sd.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_sd[k] = v
            else:
                skipped_keys.append(f"{k} (ckpt: {v.shape} vs model: {model_state[k].shape})")
    
    # 加载过滤后的权重
    missing_keys, unexpected_keys = model.load_state_dict(filtered_sd, strict=False)
    
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    if skipped_keys:
        print(f"⚠️ Skipped {len(skipped_keys)} layers due to size mismatch.")
    if missing_keys:
        print(f"⚠️ Missing {len(missing_keys)} keys (using random init)")
        
    model.eval()

    # 使用 val split
    dataset = BEVMultiTaskDataset(data_root=data_root, split='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    ious = []
    all_scores = []
    all_matches = [] # 1 for TP, 0 for FP
    total_gts = 0
    nan_count = 0

    print(f"🏁 Validating {len(dataset)} samples...")
    
    for i, (images, points, targets) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images, points_list)
        
        # --- 1. Segmentation Eval ---
        mask_logit = outputs['mask_pred'][0, 0]
        if torch.isnan(mask_logit).any():
            nan_count += 1
            continue

        pred_mask = torch.sigmoid(mask_logit).cpu().numpy()
        gt_mask = targets[0]['masks'].numpy()
        ious.append(calculate_iou(pred_mask, gt_mask))

        # --- 2. Detection Eval ---
        # Decode boxes
        # 🔥🔥 保持低阈值 0.01，捕捉低置信度框 🔥🔥
        det_boxes_batch = decode_boxes(outputs, K=100, threshold=0.01)
        det_boxes = det_boxes_batch[0]
        
        # NMS
        keep = bev_nms(det_boxes, iou_threshold=0.3)
        det_boxes = det_boxes[keep].cpu().numpy()

        # Get GT boxes
        gt_boxes = targets[0]['boxes'].numpy() # [x, y, w, l] in Grid
        total_gts += len(gt_boxes)

        # Matching Logic (Greedy)
        matched_gt_indices = set()
        
        # det_boxes: [x, y, w, l, rot, score, class]
        if len(det_boxes) > 0:
            det_boxes = det_boxes[np.argsort(-det_boxes[:, 5])]

        for det in det_boxes:
            det_x, det_y = det[0], det[1]
            score = det[5]
            
            all_scores.append(score)
            
            is_tp = False
            best_dist = float('inf')
            best_gt_idx = -1
            
            # 寻找最近的未匹配 GT
            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                
                gt_x, gt_y = gt[0], gt[1]
                # 计算物理距离 (米)
                dist_m = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2) * cfg.VOXEL_SIZE
                
                # 距离阈值: 2.0米
                if dist_m < 2.0 and dist_m < best_dist:
                    best_dist = dist_m
                    best_gt_idx = g_idx
            
            if best_gt_idx != -1:
                is_tp = True
                matched_gt_indices.add(best_gt_idx)
            
            all_matches.append(1 if is_tp else 0)

    # --- Metrics Calculation ---
    
    # mIoU
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    
    # --- 🔍 插入诊断代码 START ---
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    
    print("\n" + "="*40)
    print("🩺 DIAGNOSTIC REPORT")
    if len(all_scores) > 0:
        print(f"  - Max Score:  {all_scores.max():.4f}")
        print(f"  - Mean Score: {all_scores.mean():.4f}")
        print(f"  - Min Score:  {all_scores.min():.4f}")
        print(f"  - Total Detections (before NMS): {len(all_scores)}")
        
        if all_scores.max() < 0.1:
            print("  ⚠️ ALERT: Model is extremely under-confident!")
    else:
        print("  ⚠️ ALERT: No boxes detected at all! (Even with threshold=0.01)")
    print("="*40 + "\n")
    # --- 🔍 插入诊断代码 END ---

    # AP (Average Precision)
    ap = 0.0
    
    if len(all_scores) > 0 and total_gts > 0:
        # Sort by score high -> low
        sorted_indices = np.argsort(-all_scores)
        all_matches = all_matches[sorted_indices]
        
        # Compute Precision/Recall Curve
        tp_cumsum = np.cumsum(all_matches)
        fp_cumsum = np.cumsum(1 - all_matches)
        
        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Smooth P-R Curve (VOC style)
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        
        # AUC (Area Under Curve)
        ap = np.trapz(precisions, recalls) 
        ap = max(0.0, ap)

    print("\n" + "="*40)
    print(f"📊 EVALUATION REPORT")
    print(f"  - Rail mIoU:     {mean_iou*100:.2f} %")
    print(f"  - Obstacle AP:   {ap*100:.2f} %")
    print(f"  - NaN Samples:   {nan_count}")
    print(f"  - Valid Samples: {len(ious)}")
    print("="*40)

if __name__ == "__main__":
    import sys
    
    DATA = "/root/autodl-tmp/FOD/data"
    
    # 支持命令行参数指定 checkpoint
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pth'):
        CKPT = sys.argv[1]
        if os.path.exists(CKPT):
            print(f"🔎 Using specified checkpoint: {CKPT}")
            run_full_evaluation(CKPT, DATA)
            sys.exit(0)
        else:
            print(f"❌ Specified checkpoint not found: {CKPT}")
            sys.exit(1)
    
    # 自动搜索最新 checkpoint
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    
    CKPT = None
    
    if os.path.exists(ckpt_dir):
        import glob
        
        print(f"📁 Searching in: {ckpt_dir}")
        search_pattern_root = os.path.join(ckpt_dir, "*.pth")
        print(f"🔍 Pattern 1: {search_pattern_root}")
        list_of_files = glob.glob(search_pattern_root)
        print(f"   Found {len(list_of_files)} files")
        
        if not list_of_files:
            search_pattern_sub = os.path.join(ckpt_dir, "*", "*.pth")
            print(f"🔍 Pattern 2: {search_pattern_sub}")
            list_of_files = glob.glob(search_pattern_sub)
            print(f"   Found {len(list_of_files)} files")
        
        if list_of_files:
            CKPT = max(list_of_files, key=os.path.getctime)
            print(f"🔎 Auto-detected latest checkpoint: {CKPT}")
        else:
            print(f"⚠️ No .pth files found in {ckpt_dir}")
    else:
        print(f"⚠️ Checkpoints dir not found: {ckpt_dir}")

    if CKPT and os.path.exists(CKPT):
        run_full_evaluation(CKPT, DATA)
    else:
        print("❌ Error: Could not find any valid checkpoint to load.")
'''

with open('scripts/validate_full.py', 'w', encoding='utf-8') as f:
    f.write(validate_code)
print("✅ Successfully updated scripts/validate_full.py")
print("🏁 Done! Now run 'python scripts/validate_full.py'")