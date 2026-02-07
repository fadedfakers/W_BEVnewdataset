import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WBEVLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.register_buffer('iou_mean', torch.tensor(0.5))

    def focal_loss(self, pred, gt):
        pred = pred.float()
        gt = gt.float()
        pred = torch.sigmoid(pred).clamp(min=1e-6, max=1-1e-6)
        
        pos_mask = gt.ge(0.99).float()
        neg_mask = gt.lt(0.99).float()
        num_pos = pos_mask.sum()
        
        if num_pos > 0:
            pos_weight = torch.pow(1 - pred, self.alpha)
            pos_loss = -torch.log(pred) * pos_weight * pos_mask
            pos_loss_sum = pos_loss.sum()
        else:
            pos_loss_sum = torch.tensor(0.0, device=pred.device)
        
        neg_weight = torch.pow(pred, self.alpha) * torch.pow(1 - gt, self.beta)
        neg_loss = -torch.log(1 - pred) * neg_weight * neg_mask
        neg_loss_sum = neg_loss.sum()
        
        if num_pos > 0:
            total_loss = ((pos_loss_sum + neg_loss_sum) / num_pos) * 0.25
        else:
            total_loss = neg_loss_sum * 0.25
        
        if not torch.isfinite(total_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return total_loss

    def wiou_v3_loss(self, pred_boxes, gt_boxes, iou):
        dist = torch.pow(pred_boxes[:, :2] - gt_boxes[:, :2], 2).sum(dim=1).clamp(max=1e5)
        r_wiou = torch.exp(torch.clamp(dist / 10000.0, max=5.0))
        
        beta = (1.0 - iou.detach()) / (1.0 - self.iou_mean + 1e-6)
        beta = torch.clamp(beta, min=0.1, max=3.0)
        non_monotonic_factor = beta / (1.9 * torch.pow(3.0, beta - 1.9) + 1e-6)
        non_monotonic_factor = non_monotonic_factor.clamp(max=10.0)
        
        if self.training:
            self.iou_mean = 0.9 * self.iou_mean + 0.1 * iou.detach().mean()
        
        loss = (non_monotonic_factor * r_wiou * (1.0 - iou)).mean()
        return loss.clamp(max=10.0)

    def get_poly_loss(self, pred, target, has_rail):
        """
        pred: [B, 3] (a, b, c)
        target: [B, 3]
        has_rail: [B]
        """
        if has_rail.sum() == 0:
            return torch.tensor(0.0).to(pred.device)

        mask = has_rail > 0
        p = pred[mask]
        t = target[mask]

        x_coords = torch.linspace(0, 100, 20).to(pred.device)
        y_pred = p[:, 0:1] * x_coords**2 + p[:, 1:2] * x_coords + p[:, 2:3]
        y_true = t[:, 0:1] * x_coords**2 + t[:, 1:2] * x_coords + t[:, 2:3]
        loss = F.l1_loss(y_pred, y_true)
        return loss

    def generate_distance_mask(self, mask_shape, device, max_dist=50.0, voxel_size=0.1):
        B, C, H, W = mask_shape
        xs = torch.arange(W, device=device) * voxel_size
        dist_mask = (xs < max_dist).float().view(1, 1, 1, W).expand(B, C, H, W)
        return dist_mask

    def forward(self, preds, targets):
        device = preds['cls_pred'].device
        hm_gt = torch.stack([t['hm'] for t in targets]).to(device)
        reg_gt = torch.stack([t['reg'] for t in targets]).to(device)
        wh_gt = torch.stack([t['wh'] for t in targets]).to(device)
        mask_gt = torch.stack([t['mask'] for t in targets]).to(device)

        l_cls = self.focal_loss(preds['cls_pred'], hm_gt)
        
        mask = (hm_gt.max(dim=1, keepdim=True)[0] > 0.99).float()
        pos_mask = mask.bool().squeeze(1)
        
        l_reg = torch.tensor(0.0, device=device)
        l_off = torch.tensor(0.0, device=device)
        l_iou_aware = torch.tensor(0.0, device=device) # 🔥 初始化 IoU Loss
        
        if pos_mask.any():
            B, _, H, W = hm_gt.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            
            p_dxdy_all = preds['box_pred'][:, :2].permute(0, 2, 3, 1)
            p_wh_all = torch.exp(torch.clamp(preds['box_pred'][:, 3:5], max=5.0)).permute(0, 2, 3, 1)
            # 提取 IoU 预测
            p_iou_all = preds['iou_pred'].permute(0, 2, 3, 1) # [B, H, W, 1]

            g_reg_all = reg_gt.permute(0, 2, 3, 1)
            g_wh_all = torch.exp(torch.clamp(wh_gt, max=5.0)).permute(0, 2, 3, 1)
            
            p_dxdy, p_wh = p_dxdy_all[pos_mask], p_wh_all[pos_mask]
            p_iou = p_iou_all[pos_mask] # 🔥 [N_pos, 1]
            g_reg, g_wh = g_reg_all[pos_mask], g_wh_all[pos_mask]
            
            curr_grid_x, curr_grid_y = grid_x.expand(B, -1, -1)[pos_mask], grid_y.expand(B, -1, -1)[pos_mask]
            p_boxes = torch.stack([curr_grid_x + p_dxdy[:, 0], curr_grid_y + p_dxdy[:, 1], p_wh[:, 0], p_wh[:, 1]], dim=1)
            g_boxes = torch.stack([curr_grid_x + g_reg[:, 0], curr_grid_y + g_reg[:, 1], g_wh[:, 0], g_wh[:, 1]], dim=1)
            
            inter = (torch.min(p_boxes[:, 0]+p_boxes[:, 2]/2, g_boxes[:, 0]+g_boxes[:, 2]/2) - 
                     torch.max(p_boxes[:, 0]-p_boxes[:, 2]/2, g_boxes[:, 0]-g_boxes[:, 2]/2)).clamp(0) * \
                    (torch.min(p_boxes[:, 1]+p_boxes[:, 3]/2, g_boxes[:, 1]+g_boxes[:, 3]/2) - 
                     torch.max(p_boxes[:, 1]-p_boxes[:, 3]/2, g_boxes[:, 1]-g_boxes[:, 3]/2)).clamp(0)
            union = p_boxes[:, 2]*p_boxes[:, 3] + g_boxes[:, 2]*g_boxes[:, 3] - inter + 1e-6
            
            # 🔥 计算 IoU Target (必须 detach，不让梯度传回 Box 分支)
            real_iou = (inter / union).detach().clamp(0, 1)
            
            l_reg = self.wiou_v3_loss(p_boxes, g_boxes, inter / union)
            l_off = F.l1_loss(p_dxdy, g_reg)
            
            # 🔥 IoU-aware Loss: 让 p_iou 逼近 real_iou
            l_iou_aware = F.binary_cross_entropy_with_logits(p_iou, real_iou.unsqueeze(1))

        # 多项式回归 Loss (几何 Loss)
        if 'poly_pred' in preds and 'rail_coeffs' in targets[0]:
            rail_coeffs_gt = torch.stack([t['rail_coeffs'] for t in targets]).to(device)
            has_rail_gt = torch.tensor([t['has_rail'] for t in targets], dtype=torch.float32, device=device)
            l_poly = self.get_poly_loss(preds['poly_pred'], rail_coeffs_gt, has_rail_gt)
        else:
            l_poly = torch.tensor(0.0, device=device)

        # 分割 Loss (保持原样)
        pixel_loss = F.binary_cross_entropy_with_logits(preds['mask_pred'], mask_gt, reduction='none')
        loss_mask = self.generate_distance_mask(pixel_loss.shape, device, max_dist=50.0, voxel_size=0.1)
        l_seg = (pixel_loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        
        # 汇总 Loss (IoU 权重 1.0, Poly 权重 1.0)
        total_loss = l_cls * 1.0 + l_reg * 1.0 + l_off * 1.0 + l_seg * 10.0 + l_iou_aware * 1.0 + l_poly * 1.0

        if not torch.isfinite(total_loss):
            safe_loss = (preds['cls_pred'].sum() + preds['box_pred'].sum()) * 0.0
            return {'loss': safe_loss, 'l_cls': l_cls, 'l_reg': l_reg, 'l_off': l_off, 'l_seg': l_seg, 'l_iou': l_iou_aware, 'l_poly': l_poly}

        return {'loss': total_loss, 'l_cls': l_cls, 'l_reg': l_reg, 'l_off': l_off, 'l_seg': l_seg, 'l_iou': l_iou_aware, 'l_poly': l_poly}