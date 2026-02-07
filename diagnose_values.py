import os
import re
import numpy as np
from configs.config import BEVConfig as cfg

print("🔍 --- 1. File Content Inspection ---")
file_path = 'data/dataset.py'
with open(file_path, 'r') as f:
    content = f.read()

# 检查 WH 计算代码
wh_pattern = r"wh\[:, cty, ctx\] = \[np\.log\(max\(w_grid, 1e-3\)\), np\.log\(max\(l_grid, 1e-3\)\)\]"
div_pattern = r"w_grid = obj\['size'\]\[0\] / cfg\.VOXEL_SIZE"

if re.search(div_pattern, content):
    print("✅ CODE CHECK: 'w_grid = size / VOXEL_SIZE' found in file.")
else:
    print("❌ CODE CHECK: Division logic NOT found! File is still old.")

if re.search(wh_pattern, content):
    print("✅ CODE CHECK: 'wh[...] = log(w_grid)' found in file.")
else:
    print("❌ CODE CHECK: WH assignment logic mismatch.")

print("\n🔍 --- 2. Runtime Value Inspection ---")
try:
    from data.dataset import BEVMultiTaskDataset
    dataset = BEVMultiTaskDataset(split='train')
    
    # 强制注入 Print 调试（猴子补丁）
    # 我们没法直接 hook 局部变量，只能看输出结果
    print(f"📂 Loading Sample 0...")
    img, points, targets = dataset[0]
    
    # 反推逻辑
    wh_map = targets['wh']
    mask = targets['hm'].max(dim=0)[0] > 0.1
    
    if mask.sum() > 0:
        # 获取第一个物体的 wh 值
        wh_val = wh_map[:, mask][:, 0] # [2]
        w_log, l_log = wh_val.tolist()
        w_grid_rec = np.exp(w_log)
        l_grid_rec = np.exp(l_log)
        
        print(f"\n📊 Recovered Values form Tensor:")
        print(f"   Log(W): {w_log:.4f}")
        print(f"   Recovered Grid W (exp): {w_grid_rec:.4f}")
        print(f"   Config Voxel Size: {cfg.VOXEL_SIZE}")
        
        # 逆推物理尺寸
        phy_w = w_grid_rec * cfg.VOXEL_SIZE
        print(f"   Implied Physical W: {phy_w:.4f} meters")
        
        if abs(w_grid_rec - 1.0) < 0.01:
            print("\n🚨 DIAGNOSIS: W is exactly 1.0.")
            print("   Possibility A: Code is OLD (using physical size) AND object is 1.0m wide.")
            print("   Possibility B: Code is NEW (using grid size) AND object is 0.1m wide.")
            print("   Possibility C: Heatmap/WH map was not written to (default 0 -> exp(0)=1).")
    else:
        print("⚠️ Sample 0 has no objects.")

except Exception as e:
    print(f"❌ Error during runtime check: {e}")