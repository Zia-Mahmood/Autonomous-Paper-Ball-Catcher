import open3d as o3d
import numpy as np
import pandas as pd
import time
import sys
import os
import glob

# ---------- Configuration ----------
BALL_RADIUS = 0.025
RIM_HEIGHT = 0.20  # 20cm Target Height (Red Grid)
GRAVITY = 9.81

# Colors
COLOR_RAW = [1.0, 0.5, 0.0]   # Orange (Actual Detection)
COLOR_LKF = [0.1, 0.1, 0.9]   # Blue (Linear Kalman)
COLOR_EKF = [0.8, 0.0, 0.8]   # Purple (Extended Kalman)
COLOR_RLS = [0.0, 0.8, 0.0]   # Green (Recursive Least Squares)

# Hidden dummy point to prevent Open3D "0 points" warning
DUMMY_POINT = np.array([[0.0, 0.0, -99.0]])

# ---------- Filters ----------

class LinearKalmanFilter:
    def __init__(self, dt=1/30.0):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100.0
        self.Q = np.eye(6) * 0.01
        self.Q[3:, 3:] *= 0.1 
        self.R = np.eye(3) * 0.05
        self.H = np.zeros((3, 6))
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1
        self.last_ts = None

    def predict(self, ts):
        if self.last_ts is None: dt = 1/30.0
        else: dt = ts - self.last_ts
        self.last_ts = ts

        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        
        self.x = F @ self.x
        self.x[5] -= GRAVITY * dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, meas):
        z = np.array(meas)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3]

class ExtendedKalmanFilter:
    def __init__(self):
        self.x = np.zeros(6) 
        self.P = np.eye(6) * 100.0
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(3) * 0.05
        self.H = np.zeros((3, 6))
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1
        self.last_ts = None
        self.drag_coeff = 0.05 

    def predict(self, ts):
        if self.last_ts is None: dt = 1/30.0
        else: dt = ts - self.last_ts
        self.last_ts = ts
        
        x, y, z, vx, vy, vz = self.x
        new_vx = vx + (-self.drag_coeff * vx) * dt
        new_vy = vy + (-self.drag_coeff * vy) * dt
        new_vz = vz + (-GRAVITY - self.drag_coeff * vz) * dt
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_z = z + vz * dt
        
        self.x = np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz])

        F = np.eye(6)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        F[3,3] = 1 - self.drag_coeff * dt
        F[4,4] = 1 - self.drag_coeff * dt
        F[5,5] = 1 - self.drag_coeff * dt
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, meas):
        z = np.array(meas)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3]

class RecursiveLeastSquares:
    def __init__(self):
        self.theta_x = np.zeros((2,1))
        self.P_x = np.eye(2) * 1000
        self.theta_y = np.zeros((2,1))
        self.P_y = np.eye(2) * 1000
        self.theta_z = np.zeros((3,1)) 
        self.P_z = np.eye(3) * 1000
        
        # TUNING: 0.85 for responsive tracking
        self.lambda_factor = 0.85 
        self.start_time = None
        self.path = []
        self.current_state = np.zeros(6)

    def update(self, ts, meas):
        if self.start_time is None: self.start_time = ts
        t = ts - self.start_time
        x, y, z = meas

        # Linear Fit for X, Y
        phi_lin = np.array([[1], [t]])
        self.theta_x, self.P_x = self._rls_step(self.theta_x, self.P_x, phi_lin, x)
        self.theta_y, self.P_y = self._rls_step(self.theta_y, self.P_y, phi_lin, y)
        
        # Quadratic Fit for Z
        phi_quad = np.array([[1], [t], [t**2]])
        self.theta_z, self.P_z = self._rls_step(self.theta_z, self.P_z, phi_quad, z)
        
        # Lookahead compensation
        lag_comp = 0.02
        state = self.get_state(ts + lag_comp)
        
        self.current_state = state
        self.path.append(state[:3].copy())
        return state

    def _rls_step(self, theta, P, phi, val):
        num = P @ phi
        den = self.lambda_factor + phi.T @ P @ phi
        K = num / den
        err = val - phi.T @ theta
        theta_new = theta + K * err
        P_new = (P - K @ phi.T @ P) / self.lambda_factor
        return theta_new, P_new

    def get_state(self, ts):
        if self.start_time is None: return np.zeros(6)
        t = ts - self.start_time
        
        px = (np.array([[1, t]]) @ self.theta_x)[0,0]
        py = (np.array([[1, t]]) @ self.theta_y)[0,0]
        pz = (np.array([[1, t, t**2]]) @ self.theta_z)[0,0]
        
        vx = self.theta_x[1,0]
        vy = self.theta_y[1,0]
        vz = self.theta_z[1,0] + 2*self.theta_z[2,0]*t
        
        return np.array([px, py, pz, vx, vy, vz])
    
    # Helper to push an interpolated state to path
    def interpolate_to_path(self, ts):
        # We query the polynomial at the interpolated time
        lag_comp = 0.02
        s = self.get_state(ts + lag_comp)
        self.path.append(s[:3].copy())
        return s[:3]

# ---------- Helper Functions ----------
def create_grid():
    grid = o3d.geometry.LineSet()
    points = []
    lines = []
    step = 0.5
    size = 3.0
    
    # 1. Floor Grid (Grey) - Z=0
    for i in range(int(-size/step), int(size/step)+1):
        x = i * step
        points.append([x, -size, 0]); points.append([x, size, 0])
        lines.append([len(points)-2, len(points)-1])
        points.append([-size, x, 0]); points.append([size, x, 0])
        lines.append([len(points)-2, len(points)-1])
    
    # 2. Target Height Plane (Red-ish) - Z=RIM_HEIGHT
    # This helps you visualize where the ball lands relative to the rim
    offset = len(points)
    for i in range(int(-size/step), int(size/step)+1):
        x = i * step
        points.append([x, -size, RIM_HEIGHT]); points.append([x, size, RIM_HEIGHT])
        lines.append([len(points)-2, len(points)-1])
        points.append([-size, x, RIM_HEIGHT]); points.append([size, x, RIM_HEIGHT])
        lines.append([len(points)-2, len(points)-1])

    colors = [[0.3, 0.3, 0.3] for _ in range(offset)] + [[0.5, 0.1, 0.1] for _ in range(len(lines)-offset)]
    
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def predict_landing(state_vec, dt_step=0.02):
    x, y, z, vx, vy, vz = state_vec
    traj_points = []
    
    # If already below rim and falling, no valid landing prediction needed
    if z < RIM_HEIGHT and vz < 0:
        return []

    curr_x, curr_y, curr_z = x, y, z
    curr_vx, curr_vy, curr_vz = vx, vy, vz
    
    sim_t = 0
    # Predict for up to 2 seconds into the future
    while curr_z > 0 and sim_t < 2.0: 
        traj_points.append([curr_x, curr_y, curr_z])
        
        # Stop exactly when we hit RIM_HEIGHT while falling
        if curr_z <= RIM_HEIGHT and curr_vz < 0:
            break
            
        curr_x += curr_vx * dt_step
        curr_y += curr_vy * dt_step
        curr_z += curr_vz * dt_step
        curr_vz -= GRAVITY * dt_step
        sim_t += dt_step
        
    return traj_points

# ---------- Main Visualizer ----------

class DetectionVisualizer:
    def __init__(self):
        self.data = self.load_data()
        self.idx = 0
        self.paused = False
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Filter Comparison", 1280, 720)
        
        self.ball = o3d.geometry.TriangleMesh.create_sphere(BALL_RADIUS)
        self.ball.paint_uniform_color(COLOR_RAW)
        self.ball.compute_vertex_normals()
        
        # Initialize Trails
        self.trails = {
            'raw': o3d.geometry.PointCloud(),
            'lkf': o3d.geometry.PointCloud(),
            'ekf': o3d.geometry.PointCloud(),
            'rls': o3d.geometry.PointCloud(),
            'pred_lkf': o3d.geometry.LineSet(),
            'pred_ekf': o3d.geometry.LineSet(),
            'pred_rls': o3d.geometry.LineSet()
        }
        
        # Prevent "0 points" warning by initializing with dummy hidden data
        for k in self.trails:
            self.trails[k].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            if isinstance(self.trails[k], o3d.geometry.LineSet):
                self.trails[k].lines = o3d.utility.Vector2iVector([])

        self.points_history = {'raw': [], 'lkf': [], 'ekf': [], 'rls': []}
        
        self.vis.add_geometry(self.ball)
        self.vis.add_geometry(create_grid())
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.3))
        
        for k in self.trails:
            self.vis.add_geometry(self.trails[k])

        self.lkf = LinearKalmanFilter()
        self.ekf = ExtendedKalmanFilter()
        self.rls = RecursiveLeastSquares()
        self.filters_initialized = False

        self.vis.register_key_callback(ord(' '), self.toggle_pause)
        self.vis.register_animation_callback(self.update)
        
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0.5, 0, 0.5])
        ctr.set_front([-0.5, -1.0, 0.5])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.6)

    def load_data(self):
        # Use glob to find CSVs
        files = glob.glob("../../data/triangulation_logs/log_1763931870.csv")
        if not files: print("No CSV found"); sys.exit()
        latest = max(files, key=os.path.getmtime)
        print(f"Loading {latest}")
        
        # Load and clean data (coerce errors to NaN)
        df = pd.read_csv(latest)
        cols = ['ball_3d_x', 'ball_3d_y', 'ball_3d_z', 'timestamp']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def toggle_pause(self, vis):
        self.paused = not self.paused
        return False

    def update_trail_geom(self, name, points, color):
        # Fix for Open3D Warning: Never send empty array
        if not points or len(points) == 0:
            p_data = DUMMY_POINT
        else:
            p_data = np.array(points)
            
        self.trails[name].points = o3d.utility.Vector3dVector(p_data)
        self.trails[name].paint_uniform_color(color)
        self.vis.update_geometry(self.trails[name])

    def update_pred_line(self, name, points, color):
        # Fix for Open3D Warning: Need at least 2 points for a line
        if len(points) < 2:
            self.trails[name].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            self.trails[name].lines = o3d.utility.Vector2iVector([])
        else:
            lines = [[i, i+1] for i in range(len(points)-1)]
            self.trails[name].points = o3d.utility.Vector3dVector(np.array(points))
            self.trails[name].lines = o3d.utility.Vector2iVector(lines)
            self.trails[name].colors = o3d.utility.Vector3dVector([color for _ in lines])
            
        self.vis.update_geometry(self.trails[name])

    def update(self, vis):
        if self.paused: return False
        
        # Loop logic
        if self.idx >= len(self.data):
            time.sleep(1.0)
            self.idx = 0
            self.points_history = {k: [] for k in self.points_history}
            self.lkf = LinearKalmanFilter()
            self.ekf = ExtendedKalmanFilter()
            self.rls = RecursiveLeastSquares()
            self.filters_initialized = False
            # Reset trails to dummy
            for k in self.trails:
                self.trails[k].points = o3d.utility.Vector3dVector(DUMMY_POINT)
                if isinstance(self.trails[k], o3d.geometry.LineSet):
                    self.trails[k].lines = o3d.utility.Vector2iVector([])
                vis.update_geometry(self.trails[k])
            return True

        row = self.data.iloc[self.idx]
        
        # Check for valid data (NaNs were coerced in load_data)
        if pd.isna(row['ball_3d_x']):
            self.idx += 1
            return True
            
        bx, by, bz = row['ball_3d_x'], row['ball_3d_y'], row['ball_3d_z']
        ts = row['timestamp']
        meas = np.array([bx, by, bz])
        
        # Initialize filters on first valid point
        if not self.filters_initialized:
            self.lkf.x[:3] = meas; self.ekf.x[:3] = meas
            self.rls = RecursiveLeastSquares() 
            self.filters_initialized = True
            self.lkf.last_ts = ts; self.ekf.last_ts = ts

        # 1. Predict
        self.lkf.predict(ts)
        self.ekf.predict(ts)
        
        # 2. Update
        lkf_pos = self.lkf.update(meas)
        ekf_pos = self.ekf.update(meas)
        rls_state = self.rls.update(ts, meas)
        rls_pos = rls_state[:3]

        # 3. Store History for Tracing
        self.points_history['raw'].append(meas)
        self.points_history['lkf'].append(lkf_pos)
        self.points_history['ekf'].append(ekf_pos)
        self.points_history['rls'].append(rls_pos)
        
        # 4. Move Ball
        center = self.ball.get_center()
        self.ball.translate(meas - center)
        vis.update_geometry(self.ball)

        # 5. Future Predictions
        lkf_traj = predict_landing(self.lkf.x)
        ekf_traj = predict_landing(self.ekf.x)
        rls_traj = predict_landing(rls_state)

        self.update_pred_line('pred_lkf', lkf_traj, COLOR_LKF)
        self.update_pred_line('pred_ekf', ekf_traj, COLOR_EKF)
        self.update_pred_line('pred_rls', rls_traj, COLOR_RLS)

        # 6. Draw Trails
        self.update_trail_geom('raw', self.points_history['raw'], COLOR_RAW)
        self.update_trail_geom('lkf', self.points_history['lkf'], COLOR_LKF)
        self.update_trail_geom('ekf', self.points_history['ekf'], COLOR_EKF)
        self.update_trail_geom('rls', self.points_history['rls'], COLOR_RLS)

        self.idx += 1
        time.sleep(0.03) 
        return True

    def run(self):
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    app = DetectionVisualizer()
    app.run()