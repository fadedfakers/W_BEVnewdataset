import torch
import torch.nn as nn
from typing import Dict
import numpy as np
from configs.config import BEVConfig as cfg

class BEVMultiHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        
        # 1. 检测分支 (Heatmap + Box)
        self.det_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # 包含旋转: [dx, dy, dz, log_w, log_l, log_h, sin, cos]
        self.reg_head = nn.Conv2d(in_channels, cfg.BOX_CODE_SIZE, kernel_size=1)
        
        # === 🔥 [新增] IoU-aware 分支 🔥 ===
        # 预测当前 Anchor 与 GT 的 IoU (0~1)
        self.iou_head = nn.Conv2d(in_channels, 1, kernel_size=1)

        # 2. 多项式回归头 (输出 a, b, c: y = ax^2 + bx + c)
        self.poly_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # 3. 轨道分割分支
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # --- 初始化 ---
        # 修正：初始化为 -2.19，使 sigmoid(x) 初始输出约为 0.1
        self.cls_head.bias.data.fill_(-2.19) 
        # IoU 输出也是 0~1 (Sigmoid)，初始化为 0 (Sigmoid(0)=0.5) 避免初期 Loss 震荡
        self.iou_head.bias.data.fill_(0.)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        det_feats = self.det_conv(x)
        poly_out = self.poly_head(x)
        return {
            'cls_pred': self.cls_head(det_feats),
            'box_pred': self.reg_head(det_feats),
            'iou_pred': self.iou_head(det_feats),
            'mask_pred': self.seg_head(x),
            'poly_pred': poly_out
        }