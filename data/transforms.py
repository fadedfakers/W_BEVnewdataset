import torch
import numpy as np
import random

class SynchronizedBEVAug:
    """
    Synchronized augmentation for Point Clouds, Object Geometries, and Tracks.
    Performed in world coordinates before rasterization.
    """
    def __init__(self, flip_prob=0.5, rot_range=[-0.3925, 0.3925], scale_range=[0.95, 1.05]):
        self.flip_prob = flip_prob
        self.rot_range = rot_range
        self.scale_range = scale_range

    def __call__(self, points, obj_geometries, track_geometries):
        # 1. 随机水平翻转 (Y轴翻转)
        if random.random() < self.flip_prob:
            points[:, 1] = -points[:, 1]
            for obj in obj_geometries:
                obj['center'][1] = -obj['center'][1]
            for i in range(len(track_geometries)):
                track_geometries[i][:, 1] = -track_geometries[i][:, 1]

        # 2. 随机旋转 (Z轴旋转)
        rot_angle = random.uniform(self.rot_range[0], self.rot_range[1])
        rot_mat = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle),  np.cos(rot_angle)]
        ], dtype=np.float32)
        
        # 保护：确保 points 合法
        if len(points) > 0 and np.isfinite(points[:, :2]).all():
            points[:, :2] = points[:, :2] @ rot_mat.T
        
        for obj in obj_geometries:
            if np.isfinite(obj['center']).all():
                obj['center'] = obj['center'] @ rot_mat.T
            
        for i in range(len(track_geometries)):
            if np.isfinite(track_geometries[i]).all():
                track_geometries[i] = track_geometries[i] @ rot_mat.T
            
        # 3. 随机缩放
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        points[:, :3] *= scale
        for obj in obj_geometries:
            obj['center'] *= scale
            obj['size'] *= scale
        for i in range(len(track_geometries)):
            track_geometries[i] *= scale
            
        return points, obj_geometries, track_geometries
