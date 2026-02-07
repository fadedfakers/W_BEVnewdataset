import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保引用正确的模块
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet  
from utils.losses import WBEVLoss           
from configs.config import BEVConfig as cfg

def train_phase2():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Phase 2 Resume Training on {device}...")

    # 2. 初始化模型
    model = WBEVFusionNet(cfg).to(device)

    # 3. 加载黄金存档 e60 (MAE 2.3m)，不要用 e70 继续练
    checkpoint_path = './checkpoints/phase2_resumed_e60.pth'
    start_epoch = 60

    if os.path.exists(checkpoint_path):
        print(f"📦 Resuming from golden checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取 state_dict
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        else:
            sd = checkpoint
            
        # 移除可能存在的 module. 前缀
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        
        # 严格加载 (strict=True)，因为我们要接续训练，结构必须完全一致
        model.load_state_dict(sd, strict=True)
        print("✅ Healthy weights loaded successfully.")
    else:
        print(f"❌ Error: {checkpoint_path} not found! Cannot resume.")
        return

    # 4. 数据准备
    train_dataset = BEVMultiTaskDataset(cfg.DATA_ROOT, split='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,  # 已经稳定，可以开启多进程
        collate_fn=BEVMultiTaskDataset.collate_fn
    )

    # 5. 优化器、调度器与损失
    # 极度保守：1e-6 像绣花一样微调（多项式系数极敏感）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    criterion = WBEVLoss()

    # 6. 训练循环
    total_epochs = 100 # 在 e50 基础上再跑 50 轮
    model.train()
    
    print(f"📈 Resuming from Golden Checkpoint e60. Target: Stabilize MAE < 2m.")
    
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for imgs, points, targets in pbar:
            # 数据移至 GPU
            imgs = imgs.to(device)
            points = [p.to(device) for p in points]

            # 处理 targets：collate_fn 返回 list of dicts
            if isinstance(targets, list):
                new_targets = []
                for item in targets:
                    if isinstance(item, dict):
                        new_targets.append({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()})
                    else:
                        new_targets.append(item.to(device) if isinstance(item, torch.Tensor) else item)
                targets = new_targets
            elif isinstance(targets, dict):
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        targets[k] = v.to(device)

            optimizer.zero_grad()

            # 前向传播
            preds = model(imgs, points)

            # 计算损失
            loss_dict = criterion(preds, targets)

            # 根据 WBEVLoss 的实际返回键名取值，兼容 total_loss / loss
            loss = loss_dict.get('total_loss', loss_dict.get('loss'))
            if loss is None:
                raise KeyError(f"无法在 loss_dict 中找到损失键。当前键名为: {list(loss_dict.keys())}")

            if torch.isnan(loss):
                print("❌ NaN Loss detected! Skipping batch.")
                continue

            loss.backward()

            # 梯度裁剪更严格 (0.5)，防止坏数据时系数被突然踢飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            epoch_loss += loss.item()

            # 动态监控：键名与 loss_dict 对应
            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "Poly": f"{loss_dict.get('poly_loss', loss_dict.get('l_poly', 0)):.4f}",
                "Seg": f"{loss_dict.get('l_seg', loss_dict.get('seg_loss', 0)):.3f}"
            })

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 每 5 轮保存一次，更多机会捕捉好模型
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            save_path = f'checkpoints/phase2_resumed_e{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, save_path)
            print(f"💾 Saved checkpoint: {save_path} (avg_loss={avg_loss:.4f})")

if __name__ == "__main__":
    train_phase2()