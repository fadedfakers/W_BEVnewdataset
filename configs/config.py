import numpy as np

class BEVConfig:
    """
    Advanced Config for W-BEVFusion 2.0 - Optimized for OSDaR23 Perception
    """
    # --- Voxelization & Grid ---
    VOXEL_SIZE = 0.1
    X_RANGE = (0.0, 102.4)
    Y_RANGE = (-25.6, 25.6)
    Z_RANGE = (-3.0, 2.0)

    # 1024 x 512
    GRID_W = int((X_RANGE[1] - X_RANGE[0]) / VOXEL_SIZE)
    GRID_H = int((Y_RANGE[1] - Y_RANGE[0]) / VOXEL_SIZE)

    # --- Classes ---
    CLASS_NAMES = [
        'person', 'worker', 'constructionVehicle', 'signal', 
        'road_vehicle', 'rail_vehicle', 'obstacle', 'animal', 'bicycle'
    ]
    NUM_CLASSES = len(CLASS_NAMES)
    
    # --- Rail Geometry ---
    # [优化] 加粗掩码：从 12 (1.2m) 增加到 16 (1.6m)
    # 目的：Label Dilation，让轨道在特征图上更容易被召回，提升 mIoU
    RAIL_MASK_THICKNESS = 16       
    RAIL_GAUSSIAN_SIGMA = 3.0  # 稍微增大高斯模糊半径，使边缘过渡更平滑    
    
    # --- Training ---
    BATCH_SIZE = 2                 
    GRAD_ACCUM = 8                 
    LEARNING_RATE = 5e-5           
    WARMUP_EPOCHS = 3              
    NUM_EPOCHS = 60
    
    # Regression Code: [dx, dy, dz, log_w, log_l, log_h, sin, cos]
    # Matched with Head and Dataset
    BOX_CODE_SIZE = 8
    
    # Image
    IMG_SIZE = (1280, 720)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # Paths (AutoDL)
    CKPT_DIR = "/root/autodl-tmp/FOD/W-BEVFusion/checkpoints"
    DATA_ROOT = "/root/autodl-tmp/FOD/data"