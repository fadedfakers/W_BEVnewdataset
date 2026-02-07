"""
evaluate_v2.py - Phase 2 升级版可视化
直接从模型输出提取 poly_pred (a, b, c)，将多项式曲线绘制在 BEV 图上。
"""
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.box_ops import decode_boxes, bev_nms
from utils.intrusion_logic import IntrusionLogic


def physical_coeffs_to_grid_points(poly_coeffs, x_range_meters=(0, 100), n_pts=100):
    """
    将物理坐标下的多项式系数 (a,b,c) 转换为 BEV Grid 上的点序列。
    与 IntrusionLogic 坐标系一致：x_phys=深度(m), y_phys=横向(m)
    Grid: col=深度(X), row=横向(Y), origin='upper' 时 row=0 在顶部
    """
    x_phys = np.linspace(x_range_meters[0], x_range_meters[1], n_pts)
    a, b, c = poly_coeffs
    y_phys = a * x_phys**2 + b * x_phys + c

    # 物理 -> Grid: col = x_phys/VOXEL (深度), row = (y_phys - Y_RANGE[0])/VOXEL (横向)
    grid_col = np.clip((x_phys / cfg.VOXEL_SIZE).astype(np.int32), 0, cfg.GRID_W - 1)
    grid_row = np.clip(((y_phys - cfg.Y_RANGE[0]) / cfg.VOXEL_SIZE).astype(np.int32), 0, cfg.GRID_H - 1)

    pts = np.column_stack([grid_col, grid_row])
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < cfg.GRID_W) & (pts[:, 1] >= 0) & (pts[:, 1] < cfg.GRID_H)
    return pts[valid]


def visualize_phase2(image, lidar_bev, pred_mask, poly_coeffs, det_boxes, alerts, save_path):
    """
    专门为 Phase 2 设计的 2x2 可视化
    poly_coeffs: [a, b, c] 模型回归出的系数（物理坐标）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    H, W = pred_mask.shape

    # --- 1. 左上：RGB 图像 ---
    mean = np.array(cfg.IMG_MEAN).reshape(1, 1, 3)
    std = np.array(cfg.IMG_STD).reshape(1, 1, 3)
    image_display = (image * std + mean) * 255.0
    image_display = np.clip(image_display, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(image_display)
    axes[0, 0].set_title("Front Camera")
    axes[0, 0].axis('off')

    # --- 2. 右上：LiDAR + 预测的多项式曲线 (黄色) ---
    if lidar_bev.ndim == 3:
        lidar_img = lidar_bev.max(axis=0)
    else:
        lidar_img = lidar_bev
    lidar_vis = (lidar_img - lidar_img.min()) / (lidar_img.max() - lidar_img.min() + 1e-6)
    lidar_vis = (lidar_vis * 255).astype(np.uint8)
    lidar_vis = cv2.cvtColor(lidar_vis, cv2.COLOR_GRAY2RGB)

    if poly_coeffs is not None and np.isfinite(poly_coeffs).all():
        pts = physical_coeffs_to_grid_points(poly_coeffs)
        if len(pts) > 1:
            cv2.polylines(lidar_vis, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
        lidar_title = "LiDAR + Predicted Poly-Curve (Yellow)"
    else:
        lidar_title = "LiDAR BEV (No Poly Pred)"
    axes[0, 1].imshow(lidar_vis, origin='upper')
    axes[0, 1].set_title(lidar_title)
    axes[0, 1].axis('off')

    # --- 3. 左下：Segmentation Mask ---
    axes[1, 0].imshow(pred_mask, cmap='jet', origin='upper')
    axes[1, 0].set_title(f"Segmentation Mask (Max Conf: {pred_mask.max():.2f})")
    axes[1, 0].axis('off')

    # --- 4. 右下：综合安全分析 ---
    safety_vis = lidar_vis.copy()
    # 绘制多项式曲线
    if poly_coeffs is not None and np.isfinite(poly_coeffs).all():
        pts = physical_coeffs_to_grid_points(poly_coeffs)
        if len(pts) > 1:
            cv2.polylines(safety_vis, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
    # 绘制检测框
    if det_boxes is not None:
        for box in det_boxes:
            x, y, w, l, rot, score = box[:6]
            if score < 0.01:
                continue
            cv2.circle(safety_vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            x1, y1 = int(x - w / 2), int(y - l / 2)
            x2, y2 = int(x + w / 2), int(y + l / 2)
            cv2.rectangle(safety_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(safety_vis, f"{score:.2f}", (int(x) + 2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 255, 200), 1)
    # 绘制报警 (IntrusionLogic 返回格式: box, score, level)
    if alerts:
        for alert in alerts:
            box = alert.get('box')
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
            else:
                continue
            score = alert.get('score', 0)
            level = alert.get('level', 'WARNING')
            color = (255, 0, 0) if level == "CRITICAL" else (255, 255, 0)
            cv2.rectangle(safety_vis, (x1, y1), (x2, y2), color, 3)
            cv2.putText(safety_vis, f"{level} {score:.2f}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    axes[1, 1].imshow(safety_vis, origin='upper')
    axes[1, 1].set_title("Safety Analysis")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Saved visualization to {save_path}")


def evaluate(checkpoint_path, data_root, num_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading model from {checkpoint_path}...")
    model = WBEVFusionNet(cfg).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
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
    logic = IntrusionLogic(voxel_size=cfg.VOXEL_SIZE, y_range_min=cfg.Y_RANGE[0])

    print("📸 Phase 2: Starting visualization loop (poly_pred on BEV)...")
    with torch.no_grad():
        for i, (images, points, targets) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            points_list = [p.to(device) for p in points]
            outputs = model(images, points_list)

            det_boxes_batch = decode_boxes(outputs, K=100, threshold=0.01)
            det_boxes = det_boxes_batch[0]
            keep = bev_nms(det_boxes, iou_threshold=0.1)
            det_boxes = det_boxes[keep]

            rail_mask_logit = outputs['mask_pred'][0, 0]
            rail_mask = torch.sigmoid(rail_mask_logit).float()
            mask_np = rail_mask.cpu().numpy()

            # 提取 poly_pred (a, b, c)
            poly_coeffs = None
            if 'poly_pred' in outputs:
                poly_coeffs = outputs['poly_pred'][0].cpu().numpy()

            # 侵入检测：统一使用 IntrusionLogic 的坐标转换，避免手写映射
            if poly_coeffs is not None and np.isfinite(poly_coeffs).all() and np.abs(poly_coeffs).sum() > 1e-6:
                rail_coeffs_grid = logic.convert_physical_to_grid_coeffs(poly_coeffs)
            else:
                rail_coeffs_grid = logic.fit_rail_lines((mask_np > 0.2).astype(np.float32))

            boxes_for_intrusion = []
            scores_for_intrusion = []
            det_boxes_np = det_boxes.cpu().numpy()
            for box in det_boxes_np:
                x, y, w, l = box[0], box[1], box[2], box[3]
                x1, y1 = x - w / 2, y - l / 2
                x2, y2 = x + w / 2, y + l / 2
                boxes_for_intrusion.append([x1, y1, x2, y2])
                scores_for_intrusion.append(float(box[5]))

            H, W = mask_np.shape[0], mask_np.shape[1]
            alerts = logic.check_intrusion(boxes_for_intrusion, scores_for_intrusion, rail_coeffs_grid, (H, W))

            img_np = images[0].permute(1, 2, 0).cpu().numpy()
            lidar_bev_map = model.lidar_backbone(points_list)[0].cpu().numpy()

            save_path = f"vis_phase2_sample_{i:02d}.png"
            visualize_phase2(img_np, lidar_bev_map, mask_np, poly_coeffs, det_boxes_np, alerts, save_path)


def _get_ckpt_dir():
    if os.path.exists(cfg.CKPT_DIR):
        return cfg.CKPT_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoints")


if __name__ == "__main__":
    DATA_ROOT = cfg.DATA_ROOT
    ckpt_dir = _get_ckpt_dir()
    target_ckpt = "model_e80.pth"
    CHECKPOINT = os.path.join(ckpt_dir, target_ckpt)

    if os.path.exists(CHECKPOINT):
        print(f"🎯 Targeted checkpoint: {CHECKPOINT}")
        evaluate(CHECKPOINT, DATA_ROOT)
    else:
        import glob
        list_of_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
        if list_of_files:
            CHECKPOINT = max(list_of_files, key=os.path.getctime)
            print(f"⚠️ Using latest: {CHECKPOINT}")
            evaluate(CHECKPOINT, DATA_ROOT)
        else:
            print(f"❌ No checkpoint found in {ckpt_dir}")
