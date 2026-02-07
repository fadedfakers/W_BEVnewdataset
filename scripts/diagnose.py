"""
诊断脚本：检查模型和数据的各个环节
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet

def diagnose():
    print("="*60)
    print("🔬 W-BEVFusion 诊断脚本")
    print("="*60)
    
    # 1. 检查数据集
    print("\n📊 步骤 1: 检查数据集")
    print("-"*60)
    dataset = BEVMultiTaskDataset(data_root='/root/autodl-tmp/FOD/data', split='val')
    print(f"✅ 数据集大小: {len(dataset)} 个样本")
    
    # 检查第一个样本
    img, points, targets = dataset[0]
    print(f"✅ 图像形状: {img.shape}")
    print(f"✅ 点云形状: {points.shape}")
    print(f"✅ GT Heatmap 形状: {targets['hm'].shape}")
    print(f"✅ GT Rail Mask 形状: {targets['mask'].shape}")
    print(f"✅ GT Boxes 数量: {len(targets['boxes'])}")
    
    # 检查 GT 统计
    hm_max = targets['hm'].max().item()
    hm_pos = (targets['hm'] > 0.1).sum().item()
    mask_ratio = (targets['mask'] > 0.5).float().mean().item()
    
    print(f"\n📈 GT 统计:")
    print(f"   - Heatmap 最大值: {hm_max:.4f}")
    print(f"   - Heatmap 正样本数: {hm_pos}")
    print(f"   - Rail Mask 占比: {mask_ratio*100:.2f}%")
    print(f"   - 目标框数量: {len(targets['boxes'])}")
    
    if hm_pos == 0:
        print("   ⚠️ 警告: 第一个样本没有目标！")
    if mask_ratio < 0.01:
        print("   ⚠️ 警告: Rail Mask 几乎为空！")
    
    # 2. 检查模型
    print("\n🤖 步骤 2: 检查模型")
    print("-"*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WBEVFusionNet().to(device)
    print(f"✅ 模型已加载到 {device}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数: {total_params:,}")
    print(f"✅ 可训练参数: {trainable_params:,}")
    
    # 3. 检查前向传播
    print("\n🔄 步骤 3: 检查前向传播")
    print("-"*60)
    
    images = img.unsqueeze(0).to(device)
    points_list = [points.to(device)]
    
    model.eval()
    with torch.no_grad():
        outputs = model(images, points_list)
    
    print(f"✅ cls_pred 形状: {outputs['cls_pred'].shape}")
    print(f"✅ box_pred 形状: {outputs['box_pred'].shape}")
    print(f"✅ mask_pred 形状: {outputs['mask_pred'].shape}")
    
    # 检查输出统计
    cls_mean = outputs['cls_pred'].mean().item()
    cls_std = outputs['cls_pred'].std().item()
    cls_max = outputs['cls_pred'].max().item()
    cls_min = outputs['cls_pred'].min().item()
    
    print(f"\n📊 cls_pred 统计 (logits):")
    print(f"   - 均值: {cls_mean:.4f}")
    print(f"   - 标准差: {cls_std:.4f}")
    print(f"   - 最大值: {cls_max:.4f}")
    print(f"   - 最小值: {cls_min:.4f}")
    
    cls_prob = torch.sigmoid(outputs['cls_pred'])
    prob_mean = cls_prob.mean().item()
    prob_max = cls_prob.max().item()
    print(f"\n📊 cls_pred 统计 (概率):")
    print(f"   - 均值: {prob_mean:.4f}")
    print(f"   - 最大值: {prob_max:.4f}")
    
    if abs(prob_mean - 0.5) < 0.05:
        print("   ⚠️ 警告: 概率集中在 0.5 附近，可能是双重 sigmoid 或未训练！")
    
    # 检查 mask_pred
    mask_mean = torch.sigmoid(outputs['mask_pred']).mean().item()
    print(f"\n📊 mask_pred 统计:")
    print(f"   - 预测均值: {mask_mean:.4f}")
    
    # 4. 检查权重文件
    print("\n💾 步骤 4: 检查最新权重")
    print("-"*60)
    
    ckpt_dir = '/root/autodl-tmp/FOD/W-BEVFusion/checkpoints'
    if os.path.exists(ckpt_dir):
        import glob
        pth_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
        if pth_files:
            latest_ckpt = max(pth_files, key=os.path.getctime)
            ckpt_time = os.path.getctime(latest_ckpt)
            
            from datetime import datetime
            ckpt_datetime = datetime.fromtimestamp(ckpt_time)
            
            print(f"✅ 最新权重: {os.path.basename(latest_ckpt)}")
            print(f"✅ 修改时间: {ckpt_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 加载权重并检查
            checkpoint = torch.load(latest_ckpt, map_location='cpu')
            
            # 检查是否有 optimizer state（说明是完整训练保存）
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                print("✅ 权重包含优化器状态（完整训练保存）")
            
            # 加载权重到模型
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("✅ 权重已加载")
            
            # 重新检查输出
            model.eval()
            with torch.no_grad():
                outputs_trained = model(images, points_list)
            
            cls_trained_mean = torch.sigmoid(outputs_trained['cls_pred']).mean().item()
            mask_trained_mean = torch.sigmoid(outputs_trained['mask_pred']).mean().item()
            
            print(f"\n📊 加载权重后的输出:")
            print(f"   - cls_pred 均值: {cls_trained_mean:.4f}")
            print(f"   - mask_pred 均值: {mask_trained_mean:.4f}")
            
            if abs(cls_trained_mean - 0.5) < 0.05:
                print("   ❌ 严重问题: 分类输出仍然接近 0.5，模型未学习！")
            else:
                print("   ✅ 分类输出正常")
        else:
            print("⚠️ 未找到权重文件")
    
    # 5. 总结
    print("\n"+"="*60)
    print("📋 诊断总结")
    print("="*60)
    
    issues = []
    if hm_pos == 0:
        issues.append("❌ 数据集中没有正样本标注")
    if mask_ratio < 0.01:
        issues.append("❌ Rail Mask 几乎为空")
    if abs(prob_mean - 0.5) < 0.05:
        issues.append("❌ 模型输出接近随机（可能是双重sigmoid或未训练）")
    if 'cls_trained_mean' in locals() and abs(cls_trained_mean - 0.5) < 0.05:
        issues.append("❌ 加载权重后仍然输出随机值")
    
    if issues:
        print("⚠️ 发现以下问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ 所有检查通过！")
    
    print("\n💡 建议:")
    if len(issues) == 0:
        print("   - 模型和数据看起来正常，可能需要更长时间训练")
    elif "数据集中没有正样本标注" in str(issues):
        print("   - 检查数据集标注解析逻辑 (data/dataset.py)")
        print("   - 确认 raillabel 解析是否正确")
    elif "模型输出接近随机" in str(issues):
        print("   - 检查训练日志中 Cls Loss 是否为 0.0000")
        print("   - 如果是，说明训练使用的是旧代码，需要重新训练")
    
    print("\n"+"="*60)

if __name__ == "__main__":
    diagnose()
