import numpy as np
import cv2

class RailKalmanFilter:
    def __init__(self):
        # 状态向量: [a, b, c] 对应方程 y = ax^2 + bx + c
        self.kalman = cv2.KalmanFilter(3, 3) 
        self.kalman.transitionMatrix = np.eye(3, dtype=np.float32) 
        self.kalman.measurementMatrix = np.eye(3, dtype=np.float32) 
        self.kalman.processNoiseCov = np.eye(3, dtype=np.float32) * 0.001 
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1  
        self.kalman.errorCovPost = np.eye(3, dtype=np.float32) * 1.0 
        self.is_initialized = False

    def update(self, coeffs):
        measurement = np.array(coeffs, dtype=np.float32).reshape(3, 1)
        if not self.is_initialized:
            self.kalman.statePre = measurement
            self.kalman.statePost = measurement
            self.is_initialized = True
            return coeffs
        self.kalman.predict()
        corrected = self.kalman.correct(measurement)
        return corrected.flatten()

class IntrusionLogic:
    def __init__(self, roi_width_meters=3.0, voxel_size=0.1, y_range_min=-25.6):
        self.roi_width_px = roi_width_meters / voxel_size
        self.voxel_size = voxel_size
        self.y_range_min = y_range_min
        self.rail_filter = RailKalmanFilter()

        self.CONF_HIGH = 0.50
        self.CONF_LOW = 0.25

    def convert_physical_to_grid_coeffs(self, phys_coeffs):
        """
        将物理坐标系下的多项式系数转换为 Grid 坐标系，供 check_intrusion 使用。
        物理: y_phys = a*x_phys^2 + b*x_phys + c, x_phys=深度(m), y_phys=横向(m)
        Grid: y_grid = a_g*x_grid^2 + b_g*x_grid + c_g, x_grid=col(0~W-1), y_grid=row(0~H-1)
        """
        a_phys, b_phys, c_phys = phys_coeffs
        vs = self.voxel_size
        y_min = self.y_range_min
        a_grid = a_phys * vs
        b_grid = b_phys
        c_grid = (c_phys - y_min) / vs
        return np.array([a_grid, b_grid, c_grid], dtype=np.float32)

    def fit_rail_lines(self, rail_mask):
        """
        拟合轨道中心线: y = ax^2 + bx + c
        Input mask shape: (H, W) -> (512, 1024)
        """
        # H=512 (Y轴/Row), W=1024 (X轴/Col)
        # np.where 返回 (Row_idx, Col_idx) 即 (y, x)
        y_idxs, x_idxs = np.where(rail_mask > 0.5)
        
        if len(x_idxs) < 50:
            return None 

        try:
            # 🔥 关键修正：拟合 y = f(x)
            # 轨道是横向延伸的 (沿着X轴/1024方向)，所以 x 是自变量
            coeffs = np.polyfit(x_idxs, y_idxs, 2)
            
            # 卡尔曼平滑
            smooth_coeffs = self.rail_filter.update(coeffs)
            return smooth_coeffs
        except:
            return None

    def check_intrusion(self, boxes, scores, rail_coeffs, img_shape):
        """
        判定异物入侵
        """
        alarms = []
        H, W = img_shape[:2] # 512, 1024
        
        # 1. 如果没有轨道，只能靠高分强判
        if rail_coeffs is None:
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score > self.CONF_HIGH:
                    alarms.append({
                        "box": box, "score": score, 
                        "level": "CRITICAL", "msg": "High Conf (No Rail)"
                    })
            return alarms

        # 2. 计算轨道中心线 Look-up Table
        # 自变量是 X (0 ~ 1023)
        xs = np.arange(W)
        a, b, c = rail_coeffs
        # 计算对应的 Y (轨道在每一列的高度位置)
        rail_center_ys = a * xs**2 + b * xs + c 
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # Box: x1, y1, x2, y2 (Pixel Coords)
            x1, y1, x2, y2 = box
            
            # 计算 Box 中心
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2 
            
            # 越界保护
            cx_int = int(np.clip(cx, 0, W-1))
            
            # 🔥 核心判定：
            # 在 box 的 x 位置 (cx)，轨道的 y 应该在哪里？
            rail_y_at_cx = rail_center_ys[cx_int]
            
            # 计算 实际y 与 轨道y 的距离
            lateral_dist_px = abs(cy - rail_y_at_cx)
            lateral_dist_m = lateral_dist_px * self.voxel_size
            
            # 判断是否在轨道宽度内
            in_rail = lateral_dist_px < (self.roi_width_px / 2)
            
            # 状态判定
            status = None
            if score > self.CONF_HIGH:
                status = "CRITICAL"
            elif score > self.CONF_LOW and in_rail:
                status = "WARNING"
            
            if status:
                alarms.append({
                    "box": box, 
                    "score": score, 
                    "level": status,
                    "dist_to_rail": lateral_dist_m, 
                    "msg": f"{status}: {score:.2f}"
                })
                
        return alarms