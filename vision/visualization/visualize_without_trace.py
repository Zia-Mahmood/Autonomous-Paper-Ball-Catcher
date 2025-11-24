import open3d as o3d
import numpy as np
import pandas as pd
import time
import sys
import os
import glob

# ---------- Configuration ----------
BALL_RADIUS = 0.025
RIM_HEIGHT = 0.20  # 20cm Target Height
GRAVITY = 9.81
CAM1_POS = [0.15, -1.4, 2.0]
CAM2_POS = [1.1, 1.8, 2.0]
LOOP_DELAY = 5.0
MIN_MOVE_DIST = 0.0001 
DUMMY_POS = np.array([0.0, 0.0, -100.0])  # Hidden position far below

# Gap Filling Config
GAP_THRESHOLD = 0.04  # If gap > 40ms, start filling
INTERPOLATION_STEP = 0.01 # Generate a point every 10ms during gaps

# ---------- BRIGHT NEON COLORS ----------
COLOR_RAW = [1.0, 0.85, 0.0]   # Neon Gold/Yellow
COLOR_LKF = [0.0, 1.0, 1.0]    # Neon Cyan
COLOR_EKF = [1.0, 0.0, 1.0]    # Neon Magenta
COLOR_RLS = [0.0, 1.0, 0.0]    # Neon Lime Green
COLOR_PRED = [1.0, 0.2, 0.2]   # Bright Red for Landing

# ---------- Filter Logic ----------

class LinearKalmanFilter:
    def __init__(self, dt=1/30.0):
        self.x = np.zeros(6) # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 100.0
        self.Q = np.eye(6) * 0.01
        self.Q[3:, 3:] *= 0.1 
        self.R = np.eye(3) * 0.05
        self.H = np.zeros((3, 6))
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1
        self.last_ts = None
        self.path = []

    def predict(self, ts):
        if self.last_ts is None: dt = 1/30.0
        else: dt = ts - self.last_ts
        
        # Prevent zero division or tiny steps if called redundantly
        if dt <= 1e-6: return

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
        self.path.append(self.x[:3].copy())
        return self.x[:3]
    
    # Helper to just save current state to path (for interpolation)
    def save_state_to_path(self):
        self.path.append(self.x[:3].copy())

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
        self.path = []

    def predict(self, ts):
        if self.last_ts is None: dt = 1/30.0
        else: dt = ts - self.last_ts

        if dt <= 1e-6: return

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
        self.path.append(self.x[:3].copy())
        return self.x[:3]
    
    def save_state_to_path(self):
        self.path.append(self.x[:3].copy())

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
def create_custom_grid(size=4.0, step=0.3):
    lines = []
    points = []
    start = -size
    end = size
    num_steps = int((end - start) / step) + 1
    
    # Floor (Grey)
    for i in range(num_steps):
        y = start + i * step
        points.append([-size, y, 0]); points.append([size, y, 0])
        lines.append([len(points)-2, len(points)-1])
        x = start + i * step
        points.append([x, -size, 0]); points.append([x, size, 0])
        lines.append([len(points)-2, len(points)-1])
        
    # Rim Height (Red)
    offset = len(points)
    for i in range(num_steps):
        y = start + i * step
        points.append([-size, y, RIM_HEIGHT]); points.append([size, y, RIM_HEIGHT])
        lines.append([len(points)-2, len(points)-1])
        x = start + i * step
        points.append([x, -size, RIM_HEIGHT]); points.append([x, size, RIM_HEIGHT])
        lines.append([len(points)-2, len(points)-1])

    colors = [[0.2, 0.2, 0.2] for _ in range(offset)] + [[0.5, 0.1, 0.1] for _ in range(len(lines)-offset)]
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def create_robot_mesh():
    robot_box = o3d.geometry.TriangleMesh.create_box(0.25, 0.25, 0.075)
    robot_box.translate([-0.2, -0.15, 0])
    robot_box.paint_uniform_color([0.1, 0.1, 0.9]) 
    dustbin = o3d.geometry.TriangleMesh.create_cylinder(radius=0.06, height=0.12)
    dustbin.translate([-0.05, 0, 0.075])
    dustbin.paint_uniform_color([0.9, 0.2, 0.2]) 
    robot_mesh = robot_box + dustbin
    robot_mesh.compute_vertex_normals()
    return robot_mesh

def compute_robot_center(row):
    p4 = None
    if not pd.isna(row['tag4_x']) and str(row['tag4_x']) != '':
        p4 = np.array([float(row['tag4_x']), float(row['tag4_y']), float(row['tag4_z'])])
    p5 = None
    if not pd.isna(row['tag5_x']) and str(row['tag5_x']) != '':
        p5 = np.array([float(row['tag5_x']), float(row['tag5_y']), float(row['tag5_z'])])

    shift_m = 0.096
    default_y_axis = np.array([0.0, 1.0, 0.0])
    if p4 is not None and p5 is not None: return 0.5 * (p4 + p5)
    if p4 is not None: return p4 + (-shift_m) * default_y_axis
    if p5 is not None: return p5 + (shift_m) * default_y_axis
    return None

def calculate_landing_point(state_vec):
    x0, y0, z0, vx, vy, vz = state_vec
    if z0 < RIM_HEIGHT: return None
    a = -0.5 * GRAVITY
    b = vz
    c = z0 - RIM_HEIGHT
    disc = b**2 - 4*a*c
    if disc < 0: return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)
    times = [t for t in [t1, t2] if t > 0]
    if not times: return None
    t_impact = min(times)
    pred_x = x0 + vx * t_impact
    pred_y = y0 + vy * t_impact
    return np.array([pred_x, pred_y, RIM_HEIGHT])

def predict_landing(state_vec, dt_step=0.02):
    x, y, z, vx, vy, vz = state_vec
    traj_points = []
    if z < RIM_HEIGHT and vz < 0: return []
    curr_x, curr_y, curr_z = x, y, z
    curr_vx, curr_vy, curr_vz = vx, vy, vz
    sim_t = 0
    while curr_z > 0 and sim_t < 2.0: 
        traj_points.append([curr_x, curr_y, curr_z])
        if curr_z <= RIM_HEIGHT and curr_vz < 0: break
        curr_x += curr_vx * dt_step
        curr_y += curr_vy * dt_step
        curr_z += curr_vz * dt_step
        curr_vz -= GRAVITY * dt_step
        sim_t += dt_step
    return traj_points

# ---------- Main App Logic ----------

class LogVisualizerWithPredictions:
    def __init__(self):
        # Data Loading
        self.data = self.find_and_load_csv()
        self.frame_idx = 0
        self.fps = 30
        self.is_paused = False
        self.ball_path_points = []
        
        # Toggles
        self.show_raw = True
        self.show_lkf = True
        self.show_ekf = True
        self.show_rls = True
        
        # Initialize Filters
        self.lkf = LinearKalmanFilter()
        self.ekf = ExtendedKalmanFilter()
        self.rls = RecursiveLeastSquares()
        self.filters_initialized = False
        self.prev_ts = None
        
        # --- Geometry Setup ---
        
        self.raw_ball = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS)
        self.raw_ball.paint_uniform_color(COLOR_RAW)
        self.raw_ball.compute_vertex_normals()

        self.lkf_ball = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS*0.9)
        self.lkf_ball.paint_uniform_color(COLOR_LKF)
        self.lkf_ball.compute_vertex_normals()

        self.ekf_ball = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS*0.9)
        self.ekf_ball.paint_uniform_color(COLOR_EKF)
        self.ekf_ball.compute_vertex_normals()

        self.rls_ball = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS*0.9)
        self.rls_ball.paint_uniform_color(COLOR_RLS)
        self.rls_ball.compute_vertex_normals()
        
        self.robot_geom = create_robot_mesh()
        
        # Trails
        self.raw_trail = o3d.geometry.PointCloud()
        self.raw_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
        self.raw_trail.paint_uniform_color(COLOR_RAW)

        self.lkf_trail = o3d.geometry.PointCloud()
        self.lkf_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
        self.lkf_trail.paint_uniform_color(COLOR_LKF)

        self.ekf_trail = o3d.geometry.PointCloud()
        self.ekf_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
        self.ekf_trail.paint_uniform_color(COLOR_EKF)

        self.rls_trail = o3d.geometry.PointCloud()
        self.rls_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
        self.rls_trail.paint_uniform_color(COLOR_RLS)

        # Landing Spheres
        self.land_lkf = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS * 0.8)
        self.land_lkf.paint_uniform_color(COLOR_LKF)
        self.land_lkf.compute_vertex_normals()
        
        self.land_ekf = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS * 0.8)
        self.land_ekf.paint_uniform_color(COLOR_EKF)
        self.land_ekf.compute_vertex_normals()
        
        self.land_rls = o3d.geometry.TriangleMesh.create_sphere(radius=BALL_RADIUS * 0.8)
        self.land_rls.paint_uniform_color(COLOR_RLS)
        self.land_rls.compute_vertex_normals()
        
        # Prediction Lines
        self.pred_lines = {
            'lkf': o3d.geometry.LineSet(),
            'ekf': o3d.geometry.LineSet(),
            'rls': o3d.geometry.LineSet()
        }
        for k in self.pred_lines:
            self.pred_lines[k].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            self.pred_lines[k].lines = o3d.utility.Vector2iVector([])

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Ball Filter Playground", width=1280, height=720)

        opt = self.vis.get_render_option()
        opt.point_size = 5.0
        opt.background_color = np.array([0.05, 0.05, 0.05]) 

        geoms = [
            self.raw_ball, self.robot_geom, self.raw_trail,
            self.lkf_ball, self.lkf_trail, self.land_lkf,
            self.ekf_ball, self.ekf_trail, self.land_ekf,
            self.rls_ball, self.rls_trail, self.land_rls,
            create_custom_grid(),
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        ]
        for g in geoms: self.vis.add_geometry(g)
        for k in self.pred_lines: self.vis.add_geometry(self.pred_lines[k])
        
        # Add Cameras
        for i, pos in enumerate([CAM1_POS, CAM2_POS]):
            cam = o3d.geometry.TriangleMesh.create_cone(radius=0.1, height=0.2)
            cam.paint_uniform_color([0,1,0] if i==1 else [1,0,0])
            cam.compute_vertex_normals()
            t = np.eye(4); t[:3,3] = pos
            R = cam.get_rotation_matrix_from_xyz((np.pi,0,0))
            cam.rotate(R, center=(0,0,0))
            cam.transform(t)
            self.vis.add_geometry(cam)

        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_front([0, -1, 1])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)

        self._hide_all_models()

        self.vis.register_key_callback(ord(' '), self.toggle_pause)
        self.vis.register_key_callback(ord('+'), self.increase_fps)
        self.vis.register_key_callback(ord('='), self.increase_fps)
        self.vis.register_key_callback(ord('-'), self.decrease_fps)
        
        self.vis.register_key_callback(ord('0'), self.toggle_raw)
        self.vis.register_key_callback(ord('1'), self.toggle_lkf)
        self.vis.register_key_callback(ord('2'), self.toggle_ekf)
        self.vis.register_key_callback(ord('3'), self.toggle_rls)
        
        self.vis.register_animation_callback(self.update_frame)

    def _translate_to(self, geom, pos):
        center = np.asarray(geom.get_center())
        geom.translate(pos - center)

    def _hide_all_models(self):
        self._translate_to(self.raw_ball, DUMMY_POS)
        self._translate_to(self.lkf_ball, DUMMY_POS)
        self._translate_to(self.ekf_ball, DUMMY_POS)
        self._translate_to(self.rls_ball, DUMMY_POS)
        self._translate_to(self.land_lkf, DUMMY_POS)
        self._translate_to(self.land_ekf, DUMMY_POS)
        self._translate_to(self.land_rls, DUMMY_POS)

    def find_and_load_csv(self):
        files = glob.glob("../../data/triangulation_logs/log_1763931870.csv")
        if not files:
            print("No CSV found!")
            sys.exit()
        latest = max(files, key=os.path.getmtime)
        print(f"Loading: {latest}")
        df = pd.read_csv(latest)
        cols = ['ball_3d_x', 'ball_3d_y', 'ball_3d_z', 'timestamp']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def toggle_pause(self, vis):
        self.is_paused = not self.is_paused
        print(f"Paused: {self.is_paused}")
        return False

    def increase_fps(self, vis):
        self.fps = min(65, self.fps + 5)
        print(f"FPS: {self.fps}")
        return False

    def decrease_fps(self, vis):
        self.fps = max(5, self.fps - 5)
        print(f"FPS: {self.fps}")
        return False

    def toggle_raw(self, vis):
        self.show_raw = not self.show_raw
        if not self.show_raw:
            self._translate_to(self.raw_ball, DUMMY_POS)
            self.raw_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
        self.vis.update_geometry(self.raw_ball)
        self.vis.update_geometry(self.raw_trail)
        return False

    def toggle_lkf(self, vis):
        self.show_lkf = not self.show_lkf
        if not self.show_lkf:
            self._translate_to(self.lkf_ball, DUMMY_POS)
            self._translate_to(self.land_lkf, DUMMY_POS)
            self.lkf_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            self.pred_lines['lkf'].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            self.pred_lines['lkf'].lines = o3d.utility.Vector2iVector([])
        self.vis.update_geometry(self.lkf_ball)
        self.vis.update_geometry(self.land_lkf)
        self.vis.update_geometry(self.lkf_trail)
        self.vis.update_geometry(self.pred_lines['lkf'])
        return False

    def toggle_ekf(self, vis):
        self.show_ekf = not self.show_ekf
        if not self.show_ekf:
            self._translate_to(self.ekf_ball, DUMMY_POS)
            self._translate_to(self.land_ekf, DUMMY_POS)
            self.ekf_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            self.pred_lines['ekf'].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            self.pred_lines['ekf'].lines = o3d.utility.Vector2iVector([])
        self.vis.update_geometry(self.ekf_ball)
        self.vis.update_geometry(self.land_ekf)
        self.vis.update_geometry(self.ekf_trail)
        self.vis.update_geometry(self.pred_lines['ekf'])
        return False

    def toggle_rls(self, vis):
        self.show_rls = not self.show_rls
        if not self.show_rls:
            self._translate_to(self.rls_ball, DUMMY_POS)
            self._translate_to(self.land_rls, DUMMY_POS)
            self.rls_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            self.pred_lines['rls'].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            self.pred_lines['rls'].lines = o3d.utility.Vector2iVector([])
        self.vis.update_geometry(self.rls_ball)
        self.vis.update_geometry(self.land_rls)
        self.vis.update_geometry(self.rls_trail)
        self.vis.update_geometry(self.pred_lines['rls'])
        return False

    def update_frame(self, vis):
        if self.is_paused: return False
        time.sleep(1.0 / self.fps)

        if self.frame_idx >= len(self.data):
            print(f"Looping in {LOOP_DELAY}s...")
            time.sleep(LOOP_DELAY)
            self.frame_idx = 0
            self.ball_path_points = []
            self.prev_ts = None
            self.lkf = LinearKalmanFilter()
            self.ekf = ExtendedKalmanFilter()
            self.rls = RecursiveLeastSquares()
            self.filters_initialized = False
            
            # Reset Geoms
            self.raw_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            self.lkf_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            self.ekf_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            self.rls_trail.points = o3d.utility.Vector3dVector(np.array([DUMMY_POS]))
            for k in self.pred_lines:
                self.pred_lines[k].points = o3d.utility.Vector3dVector(DUMMY_POINT)
                self.pred_lines[k].lines = o3d.utility.Vector2iVector([])
            self._hide_all_models()
            
            geoms = [self.raw_trail, self.lkf_trail, self.ekf_trail, self.rls_trail,
                     self.raw_ball, self.lkf_ball, self.ekf_ball, self.rls_ball,
                     self.land_lkf, self.land_ekf, self.land_rls, 
                     self.pred_lines['lkf'], self.pred_lines['ekf'], self.pred_lines['rls']]
            for g in geoms: vis.update_geometry(g)
            return True

        row = self.data.iloc[self.frame_idx]
        bx, by, bz = row['ball_3d_x'], row['ball_3d_y'], row['ball_3d_z']
        ts = row['timestamp']
        
        valid_ball = (not pd.isna(bx) and not pd.isna(by) and not pd.isna(bz))
        
        if valid_ball:
            pos = np.array([float(bx), float(by), float(bz)])
            
            if not self.filters_initialized:
                self.lkf.x[:3] = pos; self.ekf.x[:3] = pos
                self.lkf.last_ts = ts; self.ekf.last_ts = ts
                self.filters_initialized = True

            if self.prev_ts is not None:
                dt = ts - self.prev_ts
                
                # --- COASTING / INTERPOLATION LOGIC ---
                # If we have a significant gap (> 40ms), fill it with predictions
                if dt > GAP_THRESHOLD:
                    # Generate points every 10ms
                    steps = int(dt / INTERPOLATION_STEP)
                    for i in range(1, steps):
                        interp_ts = self.prev_ts + i * INTERPOLATION_STEP
                        
                        # 1. Kalman Predict (State evolves, but no measurement)
                        self.lkf.predict(interp_ts)
                        self.lkf.save_state_to_path()
                        
                        self.ekf.predict(interp_ts)
                        self.ekf.save_state_to_path()
                        
                        # 2. RLS Query (Polynomial evaluation)
                        self.rls.interpolate_to_path(interp_ts)

                # --- REAL MEASUREMENT UPDATE ---
                self.lkf.predict(ts); self.lkf.update(pos)
                self.ekf.predict(ts); self.ekf.update(pos)
                self.rls.update(ts, pos)
            
            self.prev_ts = ts

            # --- Update VISUALS ---
            
            # 1. RAW
            if self.show_raw:
                self._translate_to(self.raw_ball, pos)
                self.update_trail(self.raw_trail, vis, self.ball_path_points, pos)
            else:
                self._translate_to(self.raw_ball, DUMMY_POS)

            # 2. LKF
            if self.show_lkf:
                self._translate_to(self.lkf_ball, self.lkf.x[:3])
                self.update_trail(self.lkf_trail, vis, self.lkf.path, self.lkf.x[:3])
                pred = calculate_landing_point(self.lkf.x)
                self._update_pred_geom(self.land_lkf, pred, vis)
                self.update_pred_line('lkf', predict_landing(self.lkf.x), COLOR_LKF, vis)
            else:
                self._translate_to(self.lkf_ball, DUMMY_POS)

            # 3. EKF
            if self.show_ekf:
                self._translate_to(self.ekf_ball, self.ekf.x[:3])
                self.update_trail(self.ekf_trail, vis, self.ekf.path, self.ekf.x[:3])
                pred = calculate_landing_point(self.ekf.x)
                self._update_pred_geom(self.land_ekf, pred, vis)
                self.update_pred_line('ekf', predict_landing(self.ekf.x), COLOR_EKF, vis)
            else:
                self._translate_to(self.ekf_ball, DUMMY_POS)

            # 4. RLS
            if self.show_rls:
                self._translate_to(self.rls_ball, self.rls.current_state[:3])
                self.update_trail(self.rls_trail, vis, self.rls.path, self.rls.current_state[:3])
                pred = calculate_landing_point(self.rls.current_state)
                self._update_pred_geom(self.land_rls, pred, vis)
                self.update_pred_line('rls', predict_landing(self.rls.current_state), COLOR_RLS, vis)
            else:
                self._translate_to(self.rls_ball, DUMMY_POS)

            geoms = [self.raw_ball, self.lkf_ball, self.ekf_ball, self.rls_ball]
            for g in geoms: vis.update_geometry(g)

        else:
            self._translate_to(self.raw_ball, DUMMY_POS)
            vis.update_geometry(self.raw_ball)

        # Update Robot
        rpos = compute_robot_center(row)
        if rpos is not None:
            center = self.robot_geom.get_center()
            self.robot_geom.translate(rpos - center)
            vis.update_geometry(self.robot_geom)
        
        self.frame_idx += 1
        return True

    def _update_pred_geom(self, geom, pos, vis):
        if pos is not None: self._translate_to(geom, pos)
        else: self._translate_to(geom, DUMMY_POS)
        vis.update_geometry(geom)

    def update_trail(self, trail_geom, vis, path_list, new_pt):
        # Only update trail if significant move or first point
        if len(path_list) > 0:
            dist = np.linalg.norm(new_pt - path_list[-1])
            if dist < MIN_MOVE_DIST: return
        # Note: path_list is updated inside filters or manually for raw, 
        # but for raw we append here. Filters append internally to handle interpolation.
        # Wait, raw appends here. Filters appends in update().
        # Let's keep logic consistent.
        
        # Actually, for filters, 'path_list' passed here is filter.path
        # The filter class already appended the point (and interpolated points).
        # So we just need to push to geometry.
        
        points = np.array(path_list)
        if len(points) > 0:
            trail_geom.points = o3d.utility.Vector3dVector(points)
            vis.update_geometry(trail_geom)
    
    def update_pred_line(self, name, points, color, vis):
        if len(points) < 2:
            self.pred_lines[name].points = o3d.utility.Vector3dVector(DUMMY_POINT)
            self.pred_lines[name].lines = o3d.utility.Vector2iVector([])
        else:
            lines = [[i, i+1] for i in range(len(points)-1)]
            self.pred_lines[name].points = o3d.utility.Vector3dVector(np.array(points))
            self.pred_lines[name].lines = o3d.utility.Vector2iVector(lines)
            self.pred_lines[name].colors = o3d.utility.Vector3dVector([color for _ in lines])
        vis.update_geometry(self.pred_lines[name])

    def run(self):
        print("Controls: Space=Pause, 0=Raw, 1=LKF, 2=EKF, 3=RLS")
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    app = LogVisualizerWithPredictions()
    app.run()