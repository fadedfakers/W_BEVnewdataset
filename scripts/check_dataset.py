import os
import sys
import torch
import numpy as np
import cv2

# 路径 hack
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset

def check_dataset():
    print("🔬 Checking Dataset Integrity (Fixed Version)...")
    dataset = BEVMultiTaskDataset(split='train')
    
    print(f"📂 Dataset Size: {len(dataset)}")
    
    # 取一个样本进行检查
    # 为了保险，我们循环找一个有物体的样本，以免恰好抽到空样本
    sample_idx = 0
    img, points, targets = dataset[sample_idx]
    
    # 如果第一张图没物体，往后找几张
    while targets['hm'].max() < 0.5 and sample_idx < 10:
        sample_idx += 1
        img, points, targets = dataset[sample_idx]
    
    print(f"👉 Inspecting Sample Index: {sample_idx}")

    # --- 1. 基础形状检查 ---
    print("\n[1] Shape Check:")
    print(f"  - Image: {img.shape} (Expected: 3, 720, 1280)")
    print(f"  - Points: {points.shape} (N, 4)")
    print(f"  - Heatmap: {targets['hm'].shape} (C, 512, 1024)")
    print(f"  - Reg: {targets['reg'].shape} (2, 512, 1024)")
    print(f"  - WH: {targets['wh'].shape} (2, 512, 1024)")
    
    # --- 2. 数值范围检查 (核心修改) ---
    print("\n[2] Value Range Check:")
    
    # 检查 Heatmap
    hm_max = targets['hm'].max().item()
    hm_min = targets['hm'].min().item()
    print(f"  - Heatmap Range: [{hm_min:.4f}, {hm_max:.4f}]")
    if hm_max > 1.0001:
        print("    ❌ ERROR: Heatmap > 1.0! Check gaussian generation.")
    elif hm_max < 0.99:
        print("    ⚠️ WARNING: Heatmap max < 1.0. No valid objects found in this sample?")
    else:
        print("    ✅ Heatmap normalized correctly.")
        
    # 检查 WH (尺寸) - 修复逻辑
    # 逻辑修改：不再依赖 Heatmap > 0.1，而是直接找 WH 张量里非零的点
    # 因为 CenterNet 只在中心点写 WH，其他地方是 0。 exp(0)=1，会导致误报。
    valid_obj_mask = (targets['wh'][0] != 0) | (targets['wh'][1] != 0)
    num_objs = valid_obj_mask.sum().item()
    
    print(f"  - Valid Objects Found (based on WH matrix): {num_objs}")

    if num_objs > 0:
        # 提取这些点的 WH 值
        wh_vals = targets['wh'][:, valid_obj_mask].permute(1, 0) # [N, 2]
        wh_exp = torch.exp(wh_vals) # 还原回线性尺寸 (Grid Units)
        
        # 打印统计信息
        max_w = wh_exp[:, 0].max().item()
        max_l = wh_exp[:, 1].max().item()
        min_w = wh_exp[:, 0].min().item()
        
        print(f"  - Object Sizes (Recovered from Log Space):")
        print(f"    👉 Max Size Found: W={max_w:.2f}, L={max_l:.2f} (Grid Units)")
        
        # 逐个打印前几个
        for i in range(min(5, len(wh_exp))):
            w, l = wh_exp[i].tolist()
            print(f"    Object {i}: W={w:.2f}, L={l:.2f}")
            
            # 智能判断
            if w < 2.0 and l < 2.0:
                 print("      ⚠️ SUSPICIOUS: Extremely small (< 2.0). Noise?")
            elif w < 5.0 and l < 5.0:
                 print("      ℹ️ Note: Small object (2-5 grid units). Likely Pedestrian/Obstacle.")
            elif w > 10.0 or l > 10.0:
                 print("      ✅ Size looks like a Vehicle/Train (Large Grid Units).")
                 
        if max_l > 10.0:
            print("\n    ✅ CONCLUSION: Size units are likely CORRECT (Grid Units).")
        else:
            print("\n    ⚠️ WARNING: Max size is still small. Check if Voxel Size matches Object Size.")
            
    else:
        print("  ⚠️ No objects found in this sample (WH are all zeros).")

    # --- 3. 轨道掩膜检查 ---
    print("\n[3] Rail Mask Check:")
    rail_mask = targets['mask'][0] # [512, 1024]
    rail_pixels = (rail_mask > 0.1).sum().item()
    rail_ratio = rail_pixels / (rail_mask.shape[0] * rail_mask.shape[1])
    print(f"  - Rail Pixels: {rail_pixels}")
    print(f"  - Rail Ratio: {rail_ratio:.2%}")
    
    if rail_ratio < 0.001:
        print("  ❌ ERROR: Rail mask almost empty! Check 'dataset.py' polylines logic.")
    elif rail_ratio > 0.5:
        print("  ⚠️ WARNING: Rail mask covers > 50% of image. Check logic.")
    else:
        print("  ✅ Rail mask looks normal.")

    print("\nDone.")

if __name__ == "__main__":
    check_dataset()