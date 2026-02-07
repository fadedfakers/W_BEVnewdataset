import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, Tuple

def decode_boxes(predictions: Dict[str, torch.Tensor], K: int = 50, threshold: float = 0.3) -> torch.Tensor:
    cls_pred = predictions['cls_pred']
    box_pred = predictions['box_pred']
    B, C, H, W = cls_pred.shape
    
    # 1. 峰值提取
    scores = torch.sigmoid(cls_pred)
    keep = (F.max_pool2d(scores, kernel_size=3, stride=1, padding=1) == scores)
    scores = scores * keep.float()
    
    # 2. 提取 Top-K
    scores_flat = scores.view(B, -1)
    topk_scores, topk_indices = torch.topk(scores_flat, K)
    
    topk_classes = (topk_indices // (H * W)).int()
    topk_indices_spatial = topk_indices % (H * W)
    topk_ys = (topk_indices_spatial // W).int()
    topk_xs = (topk_indices_spatial % W).int()
    
    # 3. 获取回归值 
    # box_pred shape: [B, 8, H, W] -> [dx, dy, dz, log_w, log_l, log_h, sin, cos]
    box_pred_flat = box_pred.view(B, 8, -1)
    reg_selected = torch.gather(box_pred_flat, 2, topk_indices_spatial.unsqueeze(1).long().expand(-1, 8, -1))
    
    # --- 解码 ---
    dx = torch.sigmoid(reg_selected[:, 0, :])
    dy = torch.sigmoid(reg_selected[:, 1, :])
    
    # 🔥🔥🔥 [新增] Z轴解码 🔥🔥🔥
    # 假设 Z_RANGE=(-3.0, 2.0)，voxel=0.1，这里预测的是相对 Voxel 的偏移
    # 我们直接取 raw value 作为 grid 坐标系下的 z (CenterNet 风格)
    dz = reg_selected[:, 2, :] 
    
    w = torch.exp(torch.clamp(reg_selected[:, 3, :], min=-5.0, max=5.0))
    l = torch.exp(torch.clamp(reg_selected[:, 4, :], min=-5.0, max=5.0))
    
    # 🔥🔥🔥 [新增] 高度 H 解码 🔥🔥🔥
    h = torch.exp(torch.clamp(reg_selected[:, 5, :], min=-5.0, max=5.0))
    
    sin_rot = reg_selected[:, 6, :]
    cos_rot = reg_selected[:, 7, :]
    rot = torch.atan2(sin_rot, cos_rot)
    
    # 4. 计算最终网格坐标
    final_x = topk_xs.float() + dx
    final_y = topk_ys.float() + dy
    # final_z = dz # 暂时直接用输出值
    
    # 组装结果: [x, y, w, l, rot, score, class, z, h] (扩充到 9 维)
    det_boxes = torch.stack([
        final_x, final_y, w, l, rot, topk_scores, topk_classes.float(), dz, h
    ], dim=2)
    
    return det_boxes

def bev_nms(boxes: torch.Tensor, iou_threshold: float = 0.3) -> torch.Tensor:
    """
    NMS 依然基于 BEV 平面 (x, y, w, l)
    boxes: [K, 9] -> [x, y, w, l, rot, score, class, z, h]
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    torch_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = boxes[:, 5]
    
    keep = torchvision.ops.nms(torch_boxes, scores, iou_threshold)
    return keep