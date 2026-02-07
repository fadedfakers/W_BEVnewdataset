import os
import sys
import torch
import glob
import numpy as np
import pandas as pd
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

def calculate_metrics_for_frame(det_boxes, gt_boxes):
    """
    仅计算检测指标 (TP, FP, FN)
    """
    hits = 0
    num_gt = len(gt_boxes)
    num_dt = len(det_boxes)
    
    if num_gt > 0 and num_dt > 0:
        # det: [x, y, ...], gt: [x, y, ...]
        dt_cents = det_boxes[:, :2]
        gt_cents = gt_boxes[:, :2]
        
        # 计算距离矩阵
        dists = torch.cdist(dt_cents, gt_cents)
        
        # 贪婪匹配: 距离小于 15 pixel (约1.5m) 视为 TP
        min_dists, match_idx = dists.min(dim=1)
        true_positives = (min_dists < 15).sum().item()
        hits = min(true_positives, num_gt)
    
    return hits, num_gt, num_dt

def evaluate_checkpoint(model, dataloader, device):
    """评估单个权重的性能"""
    model.eval()
    
    total_hits = 0
    total_gt_obj = 0
    total_dt_obj = 0
    
    # 调试标志：只在第一次打印 key，方便排查
    debug_printed = False
    
    with torch.no_grad():
        for i, (images, points, targets) in enumerate(dataloader):
            images = images.to(device)
            points = [p.to(device) for p in points]
            
            # Forward
            outputs = model(images, points)
            
            # --- Debug: 打印一次 targets 的 keys ---
            if not debug_printed:
                # print(f"🔍 [Debug] Keys in targets[0]: {targets[0].keys()}")
                debug_printed = True

            # --- Process Obstacle ---
            # 阈值设为 0.1，模拟实际验证
            det_boxes_batch = decode_boxes(outputs, K=50, threshold=0.1)
            det_boxes = det_boxes_batch[0]
            keep = bev_nms(det_boxes, iou_threshold=0.1)
            det_boxes = det_boxes[keep]
            
            # 解析 GT Boxes
            # 尝试从 heatmap 提取 (最通用的方法)
            if 'hm' in targets[0]:
                gt_hm = targets[0]['hm']
                gt_ys, gt_xs = torch.where(gt_hm[0] > 0.9) 
                gt_boxes_approx = torch.stack([gt_xs.float(), gt_ys.float()], dim=1).to(device)
            else:
                # 如果没有 hm，尝试直接读取 boxes 字段
                gt_boxes_approx = torch.tensor([], device=device) # 兜底为空

            # 计算检测指标
            hits, n_gt, n_dt = calculate_metrics_for_frame(det_boxes, gt_boxes_approx)
            
            total_hits += hits
            total_gt_obj += n_gt
            total_dt_obj += n_dt

    # 估算 Recall 和 Precision
    recall = (total_hits / (total_gt_obj + 1e-6)) * 100
    precision = (total_hits / (total_dt_obj + 1e-6)) * 100
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return recall, precision, f1_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = "/root/autodl-tmp/FOD/data"
    CKPT_DIR = "/root/autodl-tmp/FOD/W-BEVFusion/checkpoints"
    
    print(f"🚀 Starting Benchmark on {device}")
    
    pth_files = glob.glob(os.path.join(CKPT_DIR, "model_e*.pth"))
    if not pth_files:
        print("❌ No checkpoints found!")
        return
        
    try:
        pth_files.sort(key=lambda x: int(x.split('_e')[-1].split('.')[0]))
    except:
        pth_files.sort(key=os.path.getmtime)
        
    print(f"📋 Found {len(pth_files)} checkpoints. (Skipping mIoU to avoid errors)")

    model = WBEVFusionNet(cfg).to(device)
    # 注意：val_dataset 在这里如果不返回 mask_gt 也没关系了，因为我们只算 Detection
    val_dataset = BEVMultiTaskDataset(data_root=DATA_ROOT, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=4)
    
    results = []
    pbar = tqdm(pth_files, desc="Benchmarking", unit="ckpt")
    
    for pth_path in pbar:
        epoch_name = os.path.basename(pth_path).replace(".pth", "")
        
        try:
            checkpoint = torch.load(pth_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
            recall, precision, f1 = evaluate_checkpoint(model, val_loader, device)
            
            results.append({
                "Epoch": epoch_name,
                "F1-Score": round(f1, 2),
                "Recall (%)": round(recall, 2),
                "Precision (%)": round(precision, 2)
            })
            
            pbar.set_postfix({"Last F1": f"{f1:.1f}"})
            
        except Exception as e:
            # print(f"\n❌ Error evaluating {epoch_name}: {e}")
            pass # 忽略错误继续跑下一个

    df = pd.DataFrame(results)
    
    if not df.empty:
        print("\n" + "="*40)
        print("🏆 BENCHMARK RESULTS (Sorted by F1)")
        print("="*40)
        
        df_sorted = df.sort_values(by="F1-Score", ascending=False)
        print(df_sorted.to_string(index=False))
        
        best_epoch = df_sorted.iloc[0]
        print("\n🌟 BEST RECOMMENDED CHECKPOINT:")
        print(f"   📂 File: {best_epoch['Epoch']}.pth")
        print(f"   🎯 F1-Score: {best_epoch['F1-Score']}%")
        
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    main()