import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint # 引入梯度检查点
from typing import Dict, Any, List

from models.backbones.wavelet_img import WaveletResNet
from models.backbones.pillar_lidar import PillarEncoder
from models.necks.fusion import CrossModalAttention
from models.heads.multitask_head import BEVMultiHead
from configs.config import BEVConfig as cfg

class WBEVFusionNet(nn.Module):
    """
    Wavelet-Enhanced BEV Fusion Network for Railway Obstacle Detection.
    
    Architecture:
    1. Image: WaveletResNet -> 2D Features
    2. Lidar: PillarEncoder -> BEV Features
    3. Fusion: CrossModalAttention (Implicit Geometry Learning)
       - Note: This architecture uses Attention to implicitly learn the mapping 
         between Image and LiDAR features, instead of using explicit Calibration 
         Matrix projection.
    4. Head: BEVMultiHead (Dense Prediction)
    """
    def __init__(self, config: Any = cfg):
        super().__init__()
        
        # 1. Image Backbone (Wavelet-Enhanced)
        self.img_backbone = WaveletResNet(pretrained=True)
        
        # 2. LiDAR Backbone (Pillar-based)
        # Input: (x, y, z, i), Output: 256 channels BEV map
        self.lidar_backbone = PillarEncoder(in_channels=4, out_channels=256)
        
        # 3. Fusion Neck (Cross-Modal Attention)
        # Lidar BEV acts as Query, Image Features act as Key/Value
        self.fusion_neck = CrossModalAttention(
            lidar_channels=256, 
            img_channels=2048, 
            embed_dims=256
        )
        
        # 4. Multi-Task Heads
        self.head = BEVMultiHead(in_channels=256, num_classes=config.NUM_CLASSES)

    def forward(self, images: torch.Tensor, points_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: (B, 3, H_img, W_img)
            points_list: List of (N, 4) Lidar points [x, y, z, i]
        """
        # 1. Extract Image Features
        img_feats = self.img_backbone(images)
        # Use Layer 4 features (strong semantic, lower resolution)
        f4 = img_feats[-1] 
        
        # 2. Extract LiDAR BEV Features
        # Output: (B, 256, H_bev, W_bev)
        lidar_bev = self.lidar_backbone(points_list)
        
        # 3. Cross-Modal Fusion
        # 使用 Gradient Checkpoint 节省显存，支持更大 Batch Size
        def fusion_wrapper(l_bev, i_feat):
            return self.fusion_neck(l_bev, i_feat)
            
        fused_bev = checkpoint(fusion_wrapper, lidar_bev, f4, use_reentrant=False)
        
        # 4. Multi-Task Prediction
        predictions = self.head(fused_bev)
        
        return predictions