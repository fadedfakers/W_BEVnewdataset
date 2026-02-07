import torch
import numpy as np
from typing import Tuple, Union

def world_to_grid(wx: Union[float, torch.Tensor], 
                  wy: Union[float, torch.Tensor], 
                  cfg) -> Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]]:
    """
    将物理坐标（米）转换为网格索引。
    """
    # 计算浮点索引
    gx_f = (wx - cfg.X_RANGE[0]) / cfg.VOXEL_SIZE
    gy_f = (wy - cfg.Y_RANGE[0]) / cfg.VOXEL_SIZE
    
    if isinstance(gx_f, torch.Tensor):
        gx = gx_f.long().clamp(0, cfg.GRID_W - 1)
        gy = gy_f.long().clamp(0, cfg.GRID_H - 1)
    else:
        gx = int(np.clip(np.floor(gx_f), 0, cfg.GRID_W - 1))
        gy = int(np.clip(np.floor(gy_f), 0, cfg.GRID_H - 1))
        
    return gx, gy

def grid_to_world(gx: Union[int, float, torch.Tensor], 
                  gy: Union[int, float, torch.Tensor], 
                  cfg) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    将网格索引转换为物理坐标（米）。
    注意：增加了 0.5 * VOXEL_SIZE 以对齐网格中心。
    """
    if isinstance(gx, torch.Tensor):
        # 确保是浮点数进行计算
        gx = gx.float()
        gy = gy.float()
        
    # 公式：物理坐标 = 索引 * 分辨率 + 起始偏移 + 半个网格补偿
    wx = gx * cfg.VOXEL_SIZE + cfg.X_RANGE[0] + (0.5 * cfg.VOXEL_SIZE)
    wy = gy * cfg.VOXEL_SIZE + cfg.Y_RANGE[0] + (0.5 * cfg.VOXEL_SIZE)
    
    return wx, wy