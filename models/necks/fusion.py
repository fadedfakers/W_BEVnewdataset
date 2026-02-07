import torch
import torch.nn as nn
import torch.nn.functional as F

class CPCA(nn.Module):
    """Stage 1: Channel-prior Convolutional Attention (Distillation)"""
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 提取通道先验掩码，过滤背景噪声
        att = self.sigmoid(self.pw_conv(self.dw_conv(x)))
        return x * att

class CMGF(nn.Module):
    """Stage 2: Cross-Modal Gated Fusion (Fermentation)"""
    def __init__(self, channels):
        super().__init__()
        self.gate_img = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.gate_lidar = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.fusion_conv = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, img_feat, lidar_feat):
        concat = torch.cat([img_feat, lidar_feat], dim=1)
        g_i = self.gate_img(concat)
        g_l = self.gate_lidar(concat)
        # 满足权重动态调整逻辑
        fused = self.fusion_conv(concat)
        return fused * g_i + lidar_feat * g_l

class CrossModalAttention(nn.Module):
    """
    DFNet Edition Neck: DCF + CMGF + ASA
    """
    def __init__(self, lidar_channels=256, img_channels=2048, embed_dims=256):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Conv2d(img_channels, embed_dims, 1), nn.BatchNorm2d(embed_dims), nn.ReLU(inplace=True))
        
        # Stage 1: Distillation-Concentration Filter
        self.dcf_lidar = CPCA(lidar_channels)
        self.dcf_img = CPCA(embed_dims)
        
        # Stage 2: Fermentation-Conditioning Fusion
        self.cmgf = CMGF(embed_dims)
        
        # Stage 3: Adaptive Semantic Aggregation (ASA) 
        # 接收来自 ResNet 的浅层特征补偿 (假设后续接入 Layer1)
        self.asa_conv = nn.Sequential(nn.Conv2d(embed_dims, embed_dims, 3, padding=1), nn.BatchNorm2d(embed_dims), nn.ReLU(inplace=True))

    def forward(self, lidar_bev, img_features):
        # 空间对齐
        img_feat = F.interpolate(self.img_proj(img_features), size=lidar_bev.shape[2:], mode='bilinear', align_corners=False)
        
        # Stage 1: DCF 粗过滤
        lidar_distilled = self.dcf_lidar(lidar_bev)
        img_distilled = self.dcf_img(img_feat)
        
        # Stage 2: CMGF 精融合
        fused = self.cmgf(img_distilled, lidar_distilled)
        
        # Stage 3: ASA (残差聚合)
        out = self.asa_conv(fused + lidar_bev)
        return out
