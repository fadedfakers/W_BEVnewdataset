import os
import torch
import glob
import re
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.losses import WBEVLoss
from utils.logger import setup_logger

# === ⚙️ 用户配置区 ===
# 🔥 Stage 5 配置：从 e80 开始，微调近处分割
SPECIFIC_CHECKPOINT = "model_e80.pth" 
TOTAL_EPOCHS = 100                     # 再微调 20 个 Epoch (80 -> 100)
FREEZE_SEG_EPOCHS = 0                  # 这里的逻辑不再适用，我们下面手动控制

# 冻结/解冻工具函数
def set_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def find_checkpoint(ckpt_dir, specific_name=None):
    if specific_name:
        target_path = os.path.join(ckpt_dir, specific_name)
        if os.path.exists(target_path):
            try:
                epoch_str = re.findall(r'model_e(\d+).pth', specific_name)[0]
                return target_path, int(epoch_str)
            except IndexError:
                return target_path, 0
    if not os.path.exists(ckpt_dir): return None, 0
    files = glob.glob(os.path.join(ckpt_dir, "model_e*.pth"))
    if not files: return None, 0
    try:
        files.sort(key=lambda x: int(re.findall(r'model_e(\d+).pth', x)[0]))
        return files[-1], int(re.findall(r'model_e(\d+).pth', files[-1])[0])
    except: return None, 0

def train():
    logger = setup_logger('./logs')
    logger.info("="*60)
    logger.info(f"🚀 W-BEVFusion Stage 5: Ignore Region Fine-tuning")
    logger.info("="*60)
    
    dataset = BEVMultiTaskDataset(data_root='/root/autodl-tmp/FOD/data', split='train')
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WBEVFusionNet().to(device)
    criterion = WBEVLoss().to(device)
    
    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    
    latest_ckpt, start_epoch = find_checkpoint(ckpt_dir, SPECIFIC_CHECKPOINT)
    
    if latest_ckpt:
        logger.info(f"♻️  Resuming from: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
    else:
        logger.error("❌ Need model_e80.pth to start Stage 5!")
        return

    # === 🔥 Stage 5 冻结策略: 保护检测，只练分割 🔥 ===
    # 1. 冻结所有
    set_grad(model, False)
    
    # 2. 只解冻分割头
    set_grad(model.head.seg_head, True)
    
    # 验证
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"🔒 [Stage 5 Frozen] Only SegHead is trainable: {trainable}/{total} params")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    for param_group in optimizer.param_groups: param_group['initial_lr'] = cfg.LEARNING_RATE

    def lr_lambda(epoch):
        return 0.1 ** (epoch // 30)
    
    # 重置 LR
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Ep {epoch+1}", ncols=140)
        stats = {'loss':0, 'cls':0, 'seg':0}
        
        for i, (images, points, targets) in pbar:
            images = images.to(device)
            points = [p.to(device) for p in points]
            
            with torch.amp.autocast('cuda'):
                preds = model(images, points)
                loss_dict = criterion(preds, targets)
                loss = loss_dict['loss'] / cfg.GRAD_ACCUM
            
            scaler.scale(loss).backward()
            
            stats['loss'] += loss_dict['loss'].item()
            stats['cls'] += loss_dict['l_cls'].item()
            stats['seg'] += loss_dict['l_seg'].item()
            
            if (i+1) % cfg.GRAD_ACCUM == 0 or (i+1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            pbar.set_postfix({'L': f"{loss_dict['loss']:.3f}", 'Seg': f"{loss_dict['l_seg']:.3f}"})
            
        logger.info(f"Summary Ep {epoch+1} | Loss: {stats['loss']/len(dataloader):.4f} | Seg: {stats['seg']/len(dataloader):.4f}")
        scheduler.step()
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_e{epoch+1}.pth'))

if __name__ == "__main__":
    train()