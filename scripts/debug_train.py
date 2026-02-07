import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# 路径 hack
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.losses import WBEVLoss
from utils.box_ops import decode_boxes

def plot_bev_debug(hm_gt, hm_pred, box_gt, box_pred, step, save_dir):
    """
    绘制 BEV 调试图：左边是 GT，右边是 Pred
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- 1. GT 可视化 ---
    # 热图背景
    vis_gt = (hm_gt * 255).astype(np.uint8)
    vis_gt = cv2.applyColorMap(vis_gt, cv2.COLORMAP_JET)
    
    # 绘制 GT 框 (绿色)
    # box_gt: [x, y, w, l] (Grid Units)
    for box in box_gt:
        x, y, w, l = box[:4]
        # 简单绘制矩形 (忽略旋转，只看位置和大小是否大致对)
        x1, y1 = int(x - w/2), int(y - l/2)
        x2, y2 = int(x + w/2), int(y + l/2)
        cv2.rectangle(vis_gt, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
        
    axes[0].imshow(cv2.cvtColor(vis_gt, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ground Truth (Green Boxes)")
    axes[0].axis('off')

    # --- 2. Pred 可视化 ---
    # 热图背景
    vis_pred = (hm_pred * 255).astype(np.uint8)
    vis_pred = cv2.applyColorMap(vis_pred, cv2.COLORMAP_JET)
    
    # 绘制 Pred 框 (红色)
    # box_pred: [x, y, w, l, rot, score, class]
    for box in box_pred:
        x, y, w, l, rot, score = box[:6]
        if score < 0.3: continue # 只画置信度高的
        
        x1, y1 = int(x - w/2), int(y - l/2)
        x2, y2 = int(x + w/2), int(y + l/2)
        cv2.rectangle(vis_pred, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red
        # 标出置信度
        cv2.putText(vis_pred, f"{score:.2f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    axes[1].imshow(cv2.cvtColor(vis_pred, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Prediction (Red Boxes) | Step {step}")
    axes[1].axis('off')
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"step_{step:04d}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️  Saved debug visualization to {save_path}")

def debug_train():
    print("🚀 Starting Overfit Debugging (Single Batch)...")
    
    # 1. 准备单样本数据
    full_dataset = BEVMultiTaskDataset(split='train')
    # 只取第 0 个样本，重复 1000 次构成一个 Batch
    # 这样模型每次看到的都是完全相同的数据
    debug_subset = Subset(full_dataset, [0]) 
    dataloader = DataLoader(debug_subset, batch_size=1, shuffle=False, collate_fn=full_dataset.collate_fn)
    
    # 获取这个固定的 Batch
    fixed_batch = next(iter(dataloader))
    images, points, targets = fixed_batch
    
    # 2. 模型与优化器
    device = torch.device('cuda')
    model = WBEVFusionNet().to(device)
    model.train()
    
    criterion = WBEVLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # 用较大的 LR 加速过拟合
    
    # 搬运数据到 GPU
    images = images.to(device)
    points = [p.to(device) for p in points]
    targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    print(f"🎯 Target: Overfit 1 sample. Watch Loss -> 0.")
    
    # 3. 训练循环 (100 Steps)
    for step in range(101):
        optimizer.zero_grad()
        
        # Forward
        preds = model(images, points)
        loss_dict = criterion(preds, targets_gpu)
        loss = loss_dict['loss']
        
        # Backward
        loss.backward()
        
        # 梯度裁剪 (防止初期爆炸)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()
        
        # 统计指标
        with torch.no_grad():
            cls_prob = torch.sigmoid(preds['cls_pred'])
            max_score = cls_prob.max().item()
            mean_score = cls_prob.mean().item()
            
        # 打印日志
        if step % 10 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.6f} (Cls: {loss_dict['l_cls']:.4f} Reg: {loss_dict['l_reg']:.4f}) | "
                  f"MaxScore: {max_score:.4f} | Grad: {grad_norm:.4f}")
            
        # 可视化 (每 20 步)
        if step % 20 == 0:
            with torch.no_grad():
                # 解码预测框
                det_boxes = decode_boxes(preds, threshold=0.1)[0].cpu().numpy()
                
                # 获取 GT 信息
                hm_gt = targets[0]['hm'].max(dim=0)[0].numpy() # 合并所有类别
                hm_pred = cls_prob[0].max(dim=0)[0].cpu().numpy()
                gt_boxes = targets[0]['boxes'].numpy()
                
                plot_bev_debug(hm_gt, hm_pred, gt_boxes, det_boxes, step, save_dir="debug_vis")

    print("\n✅ Debugging Finished. Check 'debug_vis/' folder.")
    print("If Loss is low and Red boxes match Green boxes -> Logic is correct.")
    print("If Boxes are tiny dots -> Unit confusion (Grid vs Meter).")
    print("If Loss high -> Learning rate or Loss function issue.")

if __name__ == "__main__":
    debug_train()