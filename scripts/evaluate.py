import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 路径 hack
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.box_ops import decode_boxes, bev_nms
from utils.intrusion_logic import check_intrusion, fit_rail_lines

def visualize_2x2(image, lidar_bev, pred_mask, det_boxes, alerts, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9)) # 宽屏布局
    
    # --- 🔥 关键修复：动态拉伸 Mask 信号 🔥 ---
    # 打印原始最大值，确认模型是否“害羞”
    raw_max = pred_mask.max()
    print(f"🔎 [Debug] {save_path}: Raw Mask Max = {raw_max:.4f}")

    # 如果有任何信号（哪怕很微弱），就把它拉伸到 0~1
    if raw_max > 0.001: 
        # 归一化：让最亮的地方变成 1.0
        norm_mask = pred_mask / raw_max
        # 再次二值化：现在阈值 0.2 就很安全了
        mask_bin = (norm_mask > 0.2).astype(np.uint8)
        # 用拉伸后的 Mask 去做拟合
        rail_lines = fit_rail_lines(norm_mask, voxel_size=cfg.VOXEL_SIZE)
    else:
        mask_bin = np.zeros_like(pred_mask, dtype=np.uint8)
        rail_lines = []

    # 形态学处理（让显示更好看）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_processed = cv2.dilate(mask_bin, kernel, iterations=1)

    # (2) 准备 LiDAR 可视化图
    if lidar_bev.ndim == 3:
        lidar_img = lidar_bev.max(axis=0) 
    else:
        lidar_img = lidar_bev
    lidar_vis = (lidar_img - lidar_img.min()) / (lidar_img.max() - lidar_img.min() + 1e-6)
    lidar_vis = (lidar_vis * 255).astype(np.uint8)
    lidar_vis = cv2.cvtColor(lidar_vis, cv2.COLOR_GRAY2RGB)

    # (3) 准备 Mask 画布
    H, W = pred_mask.shape
    fitting_canvas = np.zeros((H, W), dtype=np.uint8)

    # --- 🎨 1. 左上：Input RGB Image ---
    mean = np.array(cfg.IMG_MEAN).reshape(1, 1, 3)
    std = np.array(cfg.IMG_STD).reshape(1, 1, 3)
    image_display = (image * std + mean) * 255.0
    image_display = np.clip(image_display, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(image_display)
    axes[0, 0].set_title(f"Input RGB (Mask Max: {raw_max:.3f})")
    axes[0, 0].axis('off')
    
    # --- 🎨 2. 右上：LiDAR BEV + 红色拟合线 ---
    if len(rail_lines) > 0:
        cv2.polylines(lidar_vis, rail_lines, isClosed=False, color=(255, 0, 0), thickness=2)
        lidar_title = "LiDAR BEV + Rail Curve (Red)"
    else:
        lidar_title = "LiDAR BEV (No Rail Detected)"
    axes[0, 1].imshow(lidar_vis, origin='upper')
    axes[0, 1].set_title(lidar_title)
    axes[0, 1].axis('off')
    
    # --- 🎨 3. 左下：Predicted Rail Mask ---
    title_suffix = ""
    if len(rail_lines) > 0:
        # 画亮白色粗线
        cv2.polylines(fitting_canvas, rail_lines, isClosed=False, color=255, thickness=5)
        title_suffix = "(Curve Fitting Success)"
    else:
        # 如果拟合失败，显示拉伸后的原始 Mask
        fitting_canvas = mask_processed * 255
        title_suffix = "(Fit Failed - Raw Mask)"

    axes[1, 0].imshow(fitting_canvas, cmap='gray', origin='upper')
    axes[1, 0].set_title(f"Predicted Rail {title_suffix}")
    axes[1, 0].axis('off')
    
    # --- 🎨 4. 右下：Safety Analysis ---
    result_bev = np.zeros((H, W, 3), dtype=np.uint8)
    
    # (A) 轨道 (青色)
    line_mask = np.zeros((H, W), dtype=np.uint8)
    if len(rail_lines) > 0:
        cv2.polylines(line_mask, rail_lines, isClosed=False, color=1, thickness=10)
    result_bev[line_mask > 0] = [0, 255, 255]
    
    # (B) 检测框
    if det_boxes is not None:
        for box in det_boxes:
            x, y, w, l, rot, score = box[:6]
            if score < 0.01: continue 
            cv2.circle(result_bev, (int(x), int(y)), 2, (0, 255, 0), -1)
            x1, y1 = int(x - w/2), int(y - l/2)
            x2, y2 = int(x + w/2), int(y + l/2)
            cv2.rectangle(result_bev, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(result_bev, f"{score:.2f}", (int(x)+2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 255, 200), 1)

    # (C) 报警
    if alerts:
        for alert in alerts:
            x1, y1, x2, y2 = alert['bbox_grid']
            score = alert['score']
            color = (255, 0, 0) if alert['alert'] == "RED" else (255, 255, 0)
            cv2.rectangle(result_bev, (x1, y1), (x2, y2), color, 3)
            z_info = alert.get('z_info', '') 
            cv2.putText(result_bev, f"{alert['alert']} {score:.2f} {z_info}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    axes[1, 1].imshow(result_bev, origin='upper')
    axes[1, 1].set_title(f"Safety Analysis")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Saved visualization to {save_path}")

def evaluate(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading model from {checkpoint_path}...")
    model = WBEVFusionNet(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    model.eval()
    
    try:
        dataset = BEVMultiTaskDataset(data_root=data_root, split='val')
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    
    print("📸 Starting visualization loop...")
    with torch.no_grad():
        for i, (images, points, targets) in enumerate(dataloader):
            if i >= 10: break 
            
            images = images.to(device)
            points_list = [p.to(device) for p in points]
            outputs = model(images, points_list)
            
            det_boxes_batch = decode_boxes(outputs, K=100, threshold=0.01) 
            det_boxes = det_boxes_batch[0] 
            keep = bev_nms(det_boxes, iou_threshold=0.1)
            det_boxes = det_boxes[keep]
            
            rail_mask_logit = outputs['mask_pred'][0, 0]
            rail_mask = (torch.sigmoid(rail_mask_logit) > 0.5).float()
            
            alerts = check_intrusion(det_boxes, rail_mask)
            
            img_np = images[0].permute(1, 2, 0).cpu().numpy()
            lidar_bev_map = model.lidar_backbone(points_list)[0].cpu().numpy()
            mask_np = rail_mask.cpu().numpy()
            det_boxes_np = det_boxes.cpu().numpy()
            
            save_path = f"vis_sample_{i:02d}.png"
            visualize_2x2(img_np, lidar_bev_map, mask_np, det_boxes_np, alerts, save_path)

if __name__ == "__main__":
    DATA_ROOT = cfg.DATA_ROOT
    CKPT_DIR = cfg.CKPT_DIR
    target_ckpt = "model_e80.pth"
    CHECKPOINT = os.path.join(CKPT_DIR, target_ckpt)
    
    if os.path.exists(CHECKPOINT):
        print(f"🎯 Targeted specific checkpoint: {CHECKPOINT}")
        evaluate(CHECKPOINT, DATA_ROOT)
    else:
        print(f"⚠️ Target '{target_ckpt}' not found!")