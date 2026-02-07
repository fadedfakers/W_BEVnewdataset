"""
validate_full_v2.py - Phase 2 升级版量化
在原有 IoU、AP 基础上，新增 Poly Curve MAE：衡量模型回归曲线与真实铁轨中心的几何精度。
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

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
    pred_bin = (pred_mask > 0.2).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)

    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()

    if union == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0

    return intersection / (union + 1e-6)


def smart_load_state_dict(model, ckpt_path):
    """
    深度模糊匹配加载：解决 checkpoint 与当前模型 key 层级/命名差异。
    支持：精确匹配、层级修正（补/删 .0）、名称对齐（mask_head <-> seg_head）、形状唯一性强制匹配。
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    new_state_dict = {}
    matched_ckpt_keys = set()

    print("🔍 启动深度模糊匹配...")

    # 第一遍：精确匹配 + 层级/名称修正
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
            matched_ckpt_keys.add(k)
            continue

        possible_keys = [
            k.replace('.weight', '.0.weight').replace('.bias', '.0.bias'),
            k.replace('.0.', '.'),
            k.replace('mask_head', 'seg_head'),
            k.replace('seg_head', 'mask_head'),
        ]

        for pk in possible_keys:
            if pk in model_dict and model_dict[pk].shape == v.shape:
                new_state_dict[pk] = v
                matched_ckpt_keys.add(k)
                break

    # 第二遍：形状唯一性匹配（针对剩余缺失层）
    unmatched_model_keys = [k for k in model_dict.keys() if k not in new_state_dict]
    unmatched_ckpt_keys = [k for k in state_dict.keys() if k not in matched_ckpt_keys]

    if unmatched_model_keys:
        print(f"   尝试对剩余 {len(unmatched_model_keys)} 层进行形状匹配...")
        for mk in unmatched_model_keys:
            m_shape = model_dict[mk].shape
            candidates = [ck for ck in unmatched_ckpt_keys if state_dict[ck].shape == m_shape]
            if len(candidates) == 1:
                new_state_dict[mk] = state_dict[candidates[0]]
                unmatched_ckpt_keys.remove(candidates[0])
                print(f"   ✨ 形状强制匹配: {candidates[0]} -> {mk}")

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Smart load: 最终映射 {len(new_state_dict)} / {len(model_dict)} 层")
    if msg.missing_keys:
        print(f"   ⚠️ 仍缺失 {len(msg.missing_keys)} 层 (随机初始化)")
    return msg


def calculate_poly_error(pred_coeffs, gt_coeffs, x_range_meters=(0, 50)):
    """
    计算预测曲线与真实曲线之间的平均横向误差 (单位: 米)
    pred_coeffs, gt_coeffs: [a, b, c]，y = ax^2 + bx + c (物理坐标)
    """
    x = np.linspace(x_range_meters[0], x_range_meters[1], 100)

    y_pred = pred_coeffs[0] * x**2 + pred_coeffs[1] * x + pred_coeffs[2]
    y_gt = gt_coeffs[0] * x**2 + gt_coeffs[1] * x + gt_coeffs[2]

    mae = np.mean(np.abs(y_pred - y_gt))
    return mae


@torch.no_grad()
def run_full_evaluation(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing model on {device}...")
    model = WBEVFusionNet(cfg).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    smart_load_state_dict(model, checkpoint_path)

    model.eval()

    dataset = BEVMultiTaskDataset(data_root=data_root, split='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    ious = []
    all_scores = []
    all_matches = []
    total_gts = 0
    nan_count = 0
    poly_errors = []

    print(f"🏁 Validating {len(dataset)} samples (Phase 2: + Poly MAE)...")

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

        # --- 2. Poly Curve Error (Phase 2 新增) ---
        has_rail = targets[0].get('has_rail', 0)
        has_rail = has_rail.item() if hasattr(has_rail, 'item') else float(has_rail)
        if 'rail_coeffs' in targets[0] and has_rail > 0:
            if 'poly_pred' in outputs:
                gt_c = targets[0]['rail_coeffs'].numpy()
                pred_c = outputs['poly_pred'][0].detach().cpu().numpy()
                if np.isfinite(pred_c).all() and np.abs(pred_c).sum() > 1e-6:
                    poly_err = calculate_poly_error(pred_c, gt_c)
                    poly_errors.append(poly_err)

        # --- 3. Detection Eval ---
        det_boxes_batch = decode_boxes(outputs, K=100, threshold=0.01)
        det_boxes = det_boxes_batch[0]

        keep = bev_nms(det_boxes, iou_threshold=0.3)
        det_boxes = det_boxes[keep].cpu().numpy()

        gt_boxes = targets[0]['boxes'].numpy()
        total_gts += len(gt_boxes)

        matched_gt_indices = set()

        if len(det_boxes) > 0:
            det_boxes = det_boxes[np.argsort(-det_boxes[:, 5])]

        for det in det_boxes:
            det_x, det_y = det[0], det[1]
            score = det[5]

            all_scores.append(score)

            is_tp = False
            best_dist = float('inf')
            best_gt_idx = -1

            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue

                gt_x, gt_y = gt[0], gt[1]
                dist_m = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2) * cfg.VOXEL_SIZE

                if dist_m < 2.0 and dist_m < best_dist:
                    best_dist = dist_m
                    best_gt_idx = g_idx

            if best_gt_idx != -1:
                is_tp = True
                matched_gt_indices.add(best_gt_idx)

            all_matches.append(1 if is_tp else 0)

    # --- Metrics Calculation ---

    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0

    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)

    print("\n" + "=" * 40)
    print("🩺 DIAGNOSTIC REPORT")
    if len(all_scores) > 0:
        print(f"  - Max Score:  {all_scores.max():.4f}")
        print(f"  - Mean Score: {all_scores.mean():.4f}")
        print(f"  - Min Score:  {all_scores.min():.4f}")
        print(f"  - Total Detections: {len(all_scores)}")
        if all_scores.max() < 0.1:
            print("  ⚠️ ALERT: Model is extremely under-confident!")
    else:
        print("  ⚠️ ALERT: No boxes detected at all!")
    print("=" * 40 + "\n")

    ap = 0.0
    if len(all_scores) > 0 and total_gts > 0:
        sorted_indices = np.argsort(-all_scores)
        all_matches = all_matches[sorted_indices]

        tp_cumsum = np.cumsum(all_matches)
        fp_cumsum = np.cumsum(1 - all_matches)

        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        ap = np.trapz(precisions, recalls)
        ap = max(0.0, ap)

    mean_poly_mae = np.mean(poly_errors) if len(poly_errors) > 0 else float('nan')

    print("\n" + "=" * 40)
    print(f"📊 EVALUATION REPORT (Phase 2)")
    print(f"  - Rail mIoU:       {mean_iou*100:.2f} %")
    print(f"  - Obstacle AP:     {ap*100:.2f} %")
    print(f"  - Poly Curve MAE:  {mean_poly_mae:.3f} m  (n={len(poly_errors)} 有效样本)")
    print(f"  - NaN Samples:     {nan_count}")
    print(f"  - Valid Samples:   {len(ious)}")
    print("=" * 40)
    if len(poly_errors) > 0:
        print("💡 Phase 2 目标: Poly MAE < 0.2 m (20 cm) 表示几何精度良好")
    elif len(poly_errors) == 0:
        print("💡 Poly MAE 无数据: 验证集可能无 has_rail 样本，或模型 poly_pred 全零/无效")


def _get_ckpt_dir():
    if os.path.exists(cfg.CKPT_DIR):
        return cfg.CKPT_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoints")


if __name__ == "__main__":
    import glob

    DATA = cfg.DATA_ROOT
    ckpt_dir = _get_ckpt_dir()

    CKPT = None

    if len(sys.argv) > 1 and sys.argv[1].endswith('.pth'):
        CKPT = sys.argv[1]

    elif os.path.exists(ckpt_dir):
        potential = os.path.join(ckpt_dir, "model_e80.pth")
        if os.path.exists(potential):
            CKPT = potential
            print(f"🎯 Targeted checkpoint: {CKPT}")
        else:
            print(f"⚠️ model_e80.pth not found, falling back to latest...")
            list_of_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
            if list_of_files:
                CKPT = max(list_of_files, key=os.path.getctime)
                print(f"🔎 Auto-detected: {CKPT}")

    if CKPT and os.path.exists(CKPT):
        run_full_evaluation(CKPT, DATA)
    else:
        print(f"❌ Error: Checkpoint not found. Searched in {ckpt_dir}")
        print("💡 Hint: python validate_full_v2.py /path/to/model.pth")
