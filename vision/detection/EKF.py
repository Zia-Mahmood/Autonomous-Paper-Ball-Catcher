#!/usr/bin/env python3
# triangulation.py (patched - EKF tuning, logging, velocity reset)
import cv2, zmq, numpy as np, time, threading, queue, traceback, sys, os, csv
from collections import deque, defaultdict
import math
import json

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]
FPS_WINDOW = 1.0        # seconds for fps moving window
DISPLAY_FPS = 20
VISUALIZE = True     # show tiled view window

STATIC_TAG_IDS = [0,1,2,3]
TAG_POSITIONS = {
    0: np.array([0.9, 0.0, 0.0], dtype=float),
    1: np.array([0.0, 0.0, 0.0], dtype=float),
    2: np.array([0.9, 0.9, 0.0], dtype=float),
    3: np.array([0.0, 1.2, 0.0], dtype=float)
}
TAG_SIZES = {0: 0.099, 1: 0.096, 2: 0.096, 3: 0.096, 4: 0.096, 5: 0.096}
CALIB_FRAMES = 30
CALIB_DIR = "../calibration/"

# ---------- EKF & prediction logging config ----------
g = 9.814  # gravity (m/s^2)

# Robot limits (tune to your robot)
robot_max_speed = 1.0    # m/s
robot_max_accel = 1.0    # m/s^2

# Tag height and floor z
TAG_MOUNT_HEIGHT = 0.075   # 7.5 cm
FLOOR_Z = 0.0

_PRED_LOG_PATH = "predictions_log.txt"

# ---------- Detection params ----------
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11

def create_april_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 2
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 6.0
    params.polygonalApproxAccuracyRate = 0.02
    params.adaptiveThreshConstant = 7
    return cv2.aruco.ArucoDetector(aruco_dict, params)

orange_hsvVals = {'hmin': 0,  'smin': 94,  'vmin': 156, 'hmax': 12,  'smax': 255, 'vmax': 255}
purple_hsvVals = {'hmin': 113,'smin': 78,  'vmin': 3,   'hmax': 129, 'smax': 255, 'vmax': 255}

MIN_AREA = 150
MAX_AREA = 20000
CIRCULARITY_MIN = 0.25
ASPECT_RATIO_MAX = 2.0
MAX_DETECTIONS_PER_CAM = 12

# ---------- Helpers ----------
def fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) + f".{int((ts%1)*1000):03d}"

def recv_latest(sub):
    msg = None
    while True:
        try:
            msg = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return msg

def update_fps(camera, cam_ts):
    dq = fps_windows[camera]
    dq.append(cam_ts)
    while dq and (cam_ts - dq[0]) > FPS_WINDOW:
        dq.popleft()
    fps = len(dq) / FPS_WINDOW
    return fps

def load_camera_calib(cam_name):
    path = os.path.join(CALIB_DIR, f'camera_calibration_{cam_name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    calib = np.load(path)
    camera_matrix = calib["cameraMatrix"]
    dist_coeffs = calib['distCoeffs']
    print("[INFO] Loaded calibrated camera parameters")
    return camera_matrix, dist_coeffs

def build_tag_world_map_from_centers(tag_centers, tag_sizes):
    out = {}
    for tid, center in tag_centers.items():
        size = tag_sizes.get(tid, tag_sizes.get(1))
        half = float(size) / 2.0
        local = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        corners_world = (local + center.reshape(1,3)).astype(np.float64)
        out[tid] = corners_world
    return out

TAG_WORLD_MAP = build_tag_world_map_from_centers(TAG_POSITIONS, TAG_SIZES)

# ---------- EKF, robot-center helpers, prediction logging ----------
class BallEKF:
    def __init__(self, pos_sigma=0.02, vel_sigma=3.0, accel_sigma=3.0, drag_coeff=0.1):
        self.x = None
        self.P = None
        self.last_ts = None
        # Config
        self.pos_sigma = pos_sigma
        self.accel_sigma = accel_sigma
        self.drag = drag_coeff # Simple linear drag: a = -drag * v
        
        # Buffer for two-point initialization
        self.init_buffer = [] 

    def update(self, meas_pos, meas_R=None, meas_ts=None):
        t = meas_ts if meas_ts is not None else time.time()
        z = np.asarray(meas_pos, dtype=np.float64).reshape(3,)

        # 1. Handle Initialization (Two-Point Method)
        if self.x is None:
            self.init_buffer.append((t, z))
            if len(self.init_buffer) < 2:
                return False # Not ready yet
            
            # We have 2 points, calculate initial velocity
            t0, z0 = self.init_buffer[0]
            t1, z1 = self.init_buffer[1]
            dt = t1 - t0
            if dt <= 0: return False
            
            vel = (z1 - z0) / dt
            
            self.x = np.zeros((6,), dtype=np.float64)
            self.x[0:3] = z1 # Current pos
            self.x[3:6] = vel # Calculated vel
            
            # Initialize P with higher velocity uncertainty
            self.P = np.diag([
                self.pos_sigma**2, self.pos_sigma**2, self.pos_sigma**2,
                (self.pos_sigma/dt)**2, (self.pos_sigma/dt)**2, (self.pos_sigma/dt)**2
            ])
            self.last_ts = t1
            print(f"[EKF] Initialized with V={vel}")
            return True

        # 2. Prediction Step
        dt = t - self.last_ts
        if dt<= 0.0001:
            return
        if dt > 0:
            self.predict(dt)

        # 3. Update Step
        H = np.zeros((3,6))
        H[0,0] = 1; H[1,1] = 1; H[2,2] = 1
        
        R = meas_R if meas_R is not None else np.diag([self.pos_sigma**2]*3)

        z_pred = H @ self.x
        y = z - z_pred # Innovation
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.last_ts = t
        return True

    def predict(self, dt):
        # State Transition Matrix (F) with simple drag approximation
        # x_new = x + v*dt
        # v_new = v + (-g - drag*v)*dt  => v*(1 - drag*dt) - g*dt
        
        drag_factor = max(0.0, 1.0 - (self.drag * dt))
        
        F = np.eye(6)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        F[3,3] = drag_factor
        F[4,4] = drag_factor
        F[5,5] = drag_factor
        
        # Predict State
        self.x[0:3] += self.x[3:6] * dt
        self.x[3:6] = self.x[3:6] * drag_factor
        self.x[5]   -= g * dt # Gravity
        
        # Predict Covariance
        Q = np.eye(6) * (self.accel_sigma * dt)**2 
        self.P = F @ self.P @ F.T + Q

    def predict_future(self, tau):
        """
        Project the state 'tau' seconds into the future 
        using the current Drag + Gravity model.
        """
        if self.x is None:
            return None
        
        # 1. Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        
        # 2. Calculate Drag Factor for the horizon 'tau'
        # Same logic as predict(): simple linear decay
        drag_factor = max(0.0, 1.0 - (self.drag * tau))
        
        # 3. Predict Position
        # p_new = p + v*t 
        # For Z, we add the standard gravity term: -0.5 * g * t^2
        pred_pos = pos + vel * tau
        pred_pos[2] -= 0.5 * g * tau * tau 
        
        # 4. Predict Velocity
        # v_new = v * drag_factor
        # For Z, we subtract gravity: v_z = v_z - g*t
        pred_vel = vel * drag_factor
        pred_vel[2] -= g * tau
        
        # Return combined 6-element vector
        return np.concatenate([pred_pos, pred_vel])

def compute_robot_center_from_tags(tag_info):
    t4 = tag_info.get('tag4', None)
    t5 = tag_info.get('tag5', None)
    shift_m = 0.096
    if t4 is not None and t5 is not None:
        p4 = np.asarray(t4.get('world_pos', t4.get('pos')), dtype=np.float64)
        p5 = np.asarray(t5.get('world_pos', t5.get('pos')), dtype=np.float64)
        center = 0.5*(p4 + p5)
        return center
    if t4 is not None:
        p4 = np.asarray(t4.get('world_pos', t4.get('pos')), dtype=np.float64)
        y_axis = np.asarray(t4.get('y_axis_unit', [0.0, 1.0, 0.0]), dtype=np.float64)
        center = p4 + (-shift_m) * y_axis
        return center
    if t5 is not None:
        p5 = np.asarray(t5.get('world_pos', t5.get('pos')), dtype=np.float64)
        y_axis = np.asarray(t5.get('y_axis_unit', [0.0, 1.0, 0.0]), dtype=np.float64)
        center = p5 + (shift_m) * y_axis
        return center
    return None

# ---------- Async CSV Logging Config ----------
_PRED_LOG_PATH = f"predictions_log_{int(time.time())}.csv"
log_queue = queue.Queue()

def logger_worker():
    """
    Background thread that pulls data from the queue and writes to CSV.
    This keeps file I/O off the main camera thread.
    """
    # Define CSV Headers
    headers = [
        "sys_ts", "pred_ts", "pred_x", "pred_y", "pred_z",
        "meas_ts", "meas_x", "meas_y", "meas_z",
        "robot_x", "robot_y", "robot_z", "v_req", "can_reach",
        "ekf_x", "ekf_y", "ekf_z", "ekf_vx", "ekf_vy", "ekf_vz"
    ]

    # Check if file exists (to decide whether to write header)
    file_exists = os.path.isfile(_PRED_LOG_PATH)

    with open(_PRED_LOG_PATH, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow(headers)
        
        while True:
            row = log_queue.get()
            if row is None: # Poison pill to stop thread
                break
            writer.writerow(row)

# Start the logging thread immediately
log_thread = threading.Thread(target=logger_worker, daemon=True)
log_thread.start()

def append_prediction_log(predicted_pos, predicted_t, current_time, robot_pos, 
                          v_req=None, can_reach=None, measured_pos=None, 
                          meas_ts=None, ekf_state=None):
    """
    Flattens data into a list and pushes to the async queue.
    Non-blocking operation (instant return).
    """
    
    # Helper to safely get list indices or return empty string
    def get_idx(arr, i):
        return arr[i] if (arr is not None and len(arr) > i) else ""
    
    def get_val(v):
        return v if v is not None else ""

    # Construct the row matching the headers defined in logger_worker
    row = [
        current_time,                           # sys_ts
        predicted_t,                            # pred_ts
        predicted_pos[0],                       # pred_x
        predicted_pos[1],                       # pred_y
        predicted_pos[2],                       # pred_z                 
        get_val(meas_ts),                       # meas_ts
        get_idx(measured_pos, 0),               # meas_x
        get_idx(measured_pos, 1),               # meas_y
        get_idx(measured_pos, 2),               # meas_z 
        get_idx(robot_pos, 0),                  # robot_x
        get_idx(robot_pos, 1),                  # robot_y
        get_idx(robot_pos, 2),                  # robot_z
        get_val(v_req),                         # v_req
        1 if can_reach else 0,                  # can_reach (1/0 for CSV)
        get_idx(ekf_state, 0),                  # ekf_x
        get_idx(ekf_state, 1),                  # ekf_y
        get_idx(ekf_state, 2),                  # ekf_z
        get_idx(ekf_state, 3),                  # ekf_vx
        get_idx(ekf_state, 4),                  # ekf_vy
        get_idx(ekf_state, 5)                   # ekf_vz
    ]

    # Push to queue (fast!)
    log_queue.put(row)

# ---------- Color masking helpers ----------
def hsv_mask_from_vals(bgr_img, hsvVals):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']], dtype=np.uint8)
    upper = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    return mask

def postprocess_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m

def find_candidate_contours(mask):
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        x,y,w,h = cv2.boundingRect(c)
        aspect = float(w)/float(h) if h>0 else 0.0
        if circularity >= CIRCULARITY_MIN or (0.5*min(w,h) > 5 and area > (MIN_AREA*2)):
            if aspect <= ASPECT_RATIO_MAX:
                candidates.append((c, area, (int(x),int(y),int(w),int(h))))
    candidates.sort(key=lambda d: d[1], reverse=True)
    return candidates

def estimate_pose_apriltag(corners, tag_size, cam_mtx, cam_dist):
    half = tag_size / 2.0
    objp = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float32)

    imgp = corners.reshape(4,2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        objp, imgp, cam_mtx, cam_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

# APRILTAG THREAD
class AprilTagThread(threading.Thread):
    def __init__(self, cam_name, frame_queue, detect_cache, lock, calibrator, shared_robot_poses):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.detect_cache = detect_cache
        self.lock = lock
        self.detector = create_april_detector()
        self.stop_flag = False
        self.calibrator = calibrator
        self.shared_robot_poses = shared_robot_poses

    def run(self):
        print(f"[{self.cam_name}] AprilTagThread started")
        while not self.stop_flag:
            try:
                frame,ts = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)
                det={"corners":corners,"ids":ids,"ts":ts,"det_time":time.time()}

                with self.lock:
                    if self.cam_name not in self.detect_cache:
                        self.detect_cache[self.cam_name]={}
                    self.detect_cache[self.cam_name]["tags"]=det

                if ids is not None:
                    self.calibrator.add_detection(self.cam_name,ids,corners,ts)
                    self.calibrator.try_compute_extrinsic(self.cam_name)

                    if self.cam_name in self.calibrator.extrinsics:
                        try:
                            K, dist = self.calibrator.load_intrinsics(self.cam_name)
                            for i,idarr in enumerate(ids):
                                tid = int(idarr[0])
                                if tid in (4,5):
                                    corners_i = np.array(corners[i]).reshape(4,2)

                                    T_tag_cam = estimate_pose_apriltag(corners_i, TAG_SIZES[tid], K, dist)
                                    R_tag_cam = T_tag_cam[:3,:3]
                                    t_tag_cam = T_tag_cam[:3,3].reshape(3,)

                                    cam_corners = (R_tag_cam @ np.array([
                                        [-TAG_SIZES[tid]/2, TAG_SIZES[tid]/2, 0.0],
                                        [ TAG_SIZES[tid]/2, TAG_SIZES[tid]/2, 0.0],
                                        [ TAG_SIZES[tid]/2, -TAG_SIZES[tid]/2, 0.0],
                                        [-TAG_SIZES[tid]/2, -TAG_SIZES[tid]/2, 0.0]
                                    ], dtype=np.float64).T).T + t_tag_cam.reshape(1,3)

                                    tag_origin_world = self.calibrator.cam_to_world(self.cam_name, t_tag_cam)
                                    world_corners = self.calibrator.cam_to_world(self.cam_name, cam_corners)

                                    mean_corner_z = float(np.mean(world_corners[:,2]))
                                    corrected_tag_origin_world = tag_origin_world.copy()
                                    if mean_corner_z < 0:
                                        corrected_tag_origin_world[2] = abs(corrected_tag_origin_world[2])
                                        print(f"[WARN] {self.cam_name} tag{tid} mean_corner_z negative ({mean_corner_z:.4f}). Flipping Z to {corrected_tag_origin_world[2]:.4f}")

                                    nominal_tag_height = TAG_MOUNT_HEIGHT
                                    if abs(corrected_tag_origin_world[2] - nominal_tag_height) > 0.05:
                                        print(f"[INFO] {self.cam_name} tag{tid} measured z {corrected_tag_origin_world[2]:.3f}, nominal {nominal_tag_height:.3f}")

                                    try:
                                        c0 = world_corners[0]; c1 = world_corners[1]; c3 = world_corners[3]
                                        local_x = c1 - c0
                                        local_minus_y = c3 - c0
                                        local_y = -local_minus_y
                                        y_axis_unit = local_y / (np.linalg.norm(local_y) + 1e-12)
                                    except Exception:
                                        y_axis_unit = np.array([0.0, 1.0, 0.0], dtype=np.float64)

                                    with self.lock:
                                        self.shared_robot_poses[tid] = {
                                            'world_pos': corrected_tag_origin_world,
                                            'cam': self.cam_name,
                                            'ts': ts,
                                            'det_time': time.time(),
                                            'reproj_src': 'estimate_pose_apriltag',
                                            'mean_corner_z': mean_corner_z,
                                            'cam_t': t_tag_cam.copy(),
                                            'y_axis_unit': y_axis_unit
                                        }
                                        # debug
                                        # print(f"tid: {tid}, pos: {corrected_tag_origin_world}")
                        except Exception as e:
                            print(f"[{self.cam_name}] robot pose compute error:", e)
                            traceback.print_exc()

            except Exception as e:
                print(f"[ERROR-{self.cam_name}] AprilTag Detection exception:", e)
                traceback.print_exc()

        print(f"[DETECT-{self.cam_name}] AprilTag Detector thread stopped")

    def stop(self):
        self.stop_flag = True

# BALL DETECTOR THREAD
class BallThread(threading.Thread):
    def __init__(self, cam_name, frame_queue, detect_cache, lock, shared_camera_candidates):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.detect_cache = detect_cache
        self.lock = lock
        self.stop_flag = False
        self.shared_camera_candidates = shared_camera_candidates

    def run(self):
        print(f"[{self.cam_name}] BallThread started")
        while not self.stop_flag:
            try:
                frame,ts=self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                mo=hsv_mask_from_vals(frame,orange_hsvVals)
                mp=hsv_mask_from_vals(frame,purple_hsvVals)

                mc = cv2.bitwise_or(mo,mp)
                mc = postprocess_mask(mc)

                cand = find_candidate_contours(mc)

                dets = []
                for c,area,(x,y,w,h) in cand[:MAX_DETECTIONS_PER_CAM]:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    else:
                        cx = x + w//2; cy = y + h//2

                    s_or = int(np.count_nonzero(mo[y:y+h,x:x+w])) if mo is not None else 0
                    s_pu = int(np.count_nonzero(mp[y:y+h,x:x+w])) if mp is not None else 0

                    if s_or>s_pu and s_or>0:
                        col="orange"
                    elif s_pu>s_or and s_pu>0:
                        col="purple"
                    else:
                        hsv_roi=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
                        mean_h=int(np.mean(hsv_roi[:,:,0]))
                        if orange_hsvVals["hmin"]<=mean_h<=orange_hsvVals["hmax"]:
                            col="orange"
                        elif purple_hsvVals["hmin"]<=mean_h<=purple_hsvVals["hmax"]:
                            col="purple"
                        else:
                            col="unknown"

                    dets.append({
                        "bbox":(x,y,w,h),
                        "centroid":(cx,cy),
                        "area":float(area),
                        "color":col,
                        "ts":ts,
                        "det_time":time.time()
                    })
                with self.lock:
                    if self.cam_name not in self.detect_cache:
                        self.detect_cache[self.cam_name]={}
                    self.detect_cache[self.cam_name]["balls"] = dets
                    self.shared_camera_candidates[self.cam_name] = [{"centroid":d["centroid"], "area":d["area"], "color":d["color"], "ts":d["ts"]} for d in dets]
            except Exception as e:
                print(f"[ERROR-{self.cam_name}] ball detection exception:", e)
                traceback.print_exc()

        print(f"[{self.cam_name}] BallDetectorThread stopped")

    def stop(self):
        self.stop_flag = True

# Calibrator Class (same as before)
class StaticCalibrator:
    def __init__(self, tag_world_map, tag_size_map):
        self.tag_world_map = tag_world_map
        self.tag_size_map = tag_size_map
        self.obs = defaultdict(list)
        self.extrinsics = {}
        self.frame_count = defaultdict(int)
        self.P_cache = {}
        self.K_cache = {}
        self.dist_cache = {}

    def load_intrinsics(self, cam_name):
        if cam_name in self.K_cache:
            return self.K_cache[cam_name], self.dist_cache[cam_name]
        camera_matrix, dist_coeffs = load_camera_calib(cam_name)
        self.K_cache[cam_name] = camera_matrix
        self.dist_cache[cam_name] = dist_coeffs
        return camera_matrix, dist_coeffs

    def add_detection(self, cam_name, ids, corners, ts):
        if ids is None:
            return
        self.frame_count[cam_name] += 1
        for i, idarr in enumerate(ids):
            tid = int(idarr[0])
            if tid in STATIC_TAG_IDS:
                c = np.array(corners[i]).reshape(4,2).astype(np.float64)
                self.obs[cam_name].append((tid, c, ts))

    def try_compute_extrinsic(self, cam_name):
        if cam_name in self.extrinsics:
            return True

        if self.frame_count.get(cam_name, 0) < CALIB_FRAMES:
            return False

        if cam_name == "kreo1":
            target_tag = 2
        elif cam_name == "kreo2":
            target_tag = 1
        else:
            return False

        obs_list = list(reversed(self.obs.get(cam_name, [])))

        use_corners = None
        for (tid, corners, ts) in obs_list:
            if int(tid) == int(target_tag):
                use_corners = corners.reshape(4,2).astype(np.float64)
                break

        if use_corners is None:
            return False

        try:
            K, dist = self.load_intrinsics(cam_name)
        except Exception as e:
            return False

        obj_corners = np.array(self.tag_world_map[target_tag], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_corners,
            use_corners,
            K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            return False

        R, _ = cv2.Rodrigues(rvec)
        tvec = tvec.reshape(3,1)

        self.extrinsics[cam_name] = {
            "rvec": rvec,
            "tvec": tvec,
            "R": R
        }

        P = K @ np.hstack((R, tvec))
        self.P_cache[cam_name] = P

        print(f"[Calib] extrinsics computed for {cam_name}: tvec={tvec.ravel()}")

        return True

    def cam_to_world(self, cam_name, X_cam):
        e = self.extrinsics.get(cam_name)
        if e is None:
            raise RuntimeError("Calibrator: extrinsic not ready for " + cam_name)
        R = e['R']; t = e['tvec']
        X = np.asarray(X_cam, dtype=np.float64)
        if X.ndim == 1 and X.shape[0] == 3:
            Xc = X.reshape(3,1)
            Xw = R.T @ (Xc - t)
            return Xw[:,0]
        if X.ndim == 2 and X.shape[1] == 3:
            Xc = X.T
            Xw = R.T @ (Xc - t)
            return Xw.T
        if X.ndim == 2 and X.shape[0] == 3:
            Xc = X
            Xw = R.T @ (Xc - t)
            return Xw.T
        raise ValueError("Invalid X_cam shape: " + str(X.shape))

    def projection_matrix(self, cam_name):
        return self.P_cache.get(cam_name, None)

# Triangulation and reprojection helpers (same as before)
def undistort_points_to_normalized(K, dist, pts):
    if len(pts) == 0:
        return np.zeros((0,2), dtype=np.float64)
    pts = np.array(pts, dtype=np.float64).reshape(-1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    und = und.reshape(-1,2)
    return und

def triangulate_pair(P1, P2, pt1, pt2):
    pts1 = np.array(pt1, dtype=np.float64).reshape(2,1)
    pts2 = np.array(pt2, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, pts1, pts2)
    X = (Xh[:3,0] / Xh[3,0]).astype(np.float64)
    return X

def reprojection_error(P, X_world, observed_xy):
    Xh = np.zeros((4,1), dtype=np.float64)
    Xh[:3,0] = X_world
    Xh[3,0] = 1.0
    proj = P @ Xh
    proj_xy = proj[:2,0] / proj[2,0]
    return float(np.linalg.norm(proj_xy - np.array(observed_xy, dtype=np.float64)))

def match_and_triangulate(camera_candidates, calibrator, max_pairs=128, reproj_thresh_px=6.0, depth_min=0.03, depth_max=10.0):
    cams = list(camera_candidates.keys())
    if len(cams) < 2:
        return []

    pair = None
    if 'kreo1' in camera_candidates and 'kreo2' in camera_candidates:
        pair = ('kreo1', 'kreo2')
    else:
        pair = (cams[0], cams[1])

    c1, c2 = pair
    cand1 = sorted(camera_candidates[c1], key=lambda x: -x.get('area',1))[:8]
    cand2 = sorted(camera_candidates[c2], key=lambda x: -x.get('area',1))[:8]
    P1 = calibrator.projection_matrix(c1)
    P2 = calibrator.projection_matrix(c2)
    if P1 is None or P2 is None:
        return []

    results = []
    for a in cand1:
        for b in cand2:
            if len(results) >= max_pairs:
                break
            x1 = a['centroid']
            x2 = b['centroid']
            try:
                Xw = triangulate_pair(P1, P2, x1, x2)
            except Exception:
                continue
            e1 = calibrator.extrinsics[c1]
            R1, t1 = e1['R'], e1['tvec']
            Xcam1 = R1 @ Xw.reshape(3,1) + t1
            depth1 = float(Xcam1[2,0])
            e2 = calibrator.extrinsics[c2]
            R2, t2 = e2['R'], e2['tvec']
            Xcam2 = R2 @ Xw.reshape(3,1) + t2
            depth2 = float(Xcam2[2,0])
            if not (depth_min < depth1 < depth_max and depth_min < depth2 < depth_max):
                continue
            err1 = reprojection_error(P1, Xw, x1)
            err2 = reprojection_error(P2, Xw, x2)
            tot = err1 + err2
            if tot > (2*reproj_thresh_px):
                continue
            results.append({'pt': Xw, 'reproj_err': tot, 'pair': (c1,c2), 'cam_pixels': (x1,x2), 'depths':(depth1,depth2), 'meas_ts': a.get('ts', time.time())})
    results = sorted(results, key=lambda r: r['reproj_err'])
    return results

# ---------- ZMQ subscriber ----------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(ZMQ_ADDR)
sub.setsockopt(zmq.RCVHWM, 1)
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt(zmq.LINGER, 0)

flushed = 0
while True:
    try:
        sub.recv_multipart(flags=zmq.NOBLOCK)
        flushed += 1
    except zmq.Again:
        break
if flushed > 0:
    print(f"[Subscriber] Flushed {flushed} stale messages.")
for t in SUB_TOPICS:
    sub.setsockopt(zmq.SUBSCRIBE, t)

frames = {}
fps_windows = defaultdict(lambda: deque())
frame_queues = {t.decode(): queue.Queue(maxsize=1) for t in SUB_TOPICS}
detect_cache = {}
detect_lock = threading.Lock()
tag_threads={}
ball_threads={}

calibrator = StaticCalibrator(TAG_WORLD_MAP, TAG_SIZES)
shared_camera_candidates = defaultdict(list)
shared_robot_poses = {}
last_triangulated_ball = None

# EKF instance (tuned)
ball_ekf = BallEKF(pos_sigma=0.02, vel_sigma=3.0, accel_sigma=3.0)

for t in SUB_TOPICS:
    cam_name = t.decode()
    tag_threads[cam_name] = AprilTagThread(cam_name,frame_queues[cam_name],detect_cache,detect_lock,calibrator,shared_robot_poses)
    ball_threads[cam_name] = BallThread(cam_name,frame_queues[cam_name],detect_cache,detect_lock,shared_camera_candidates)
    tag_threads[cam_name].start()
    ball_threads[cam_name].start()

print("[Subscriber] connected, waiting for frames... (Press ESC to exit)")

last_show = time.time()

try:
    while True:
        parts=recv_latest(sub)
        if parts is None: continue

        topic=parts[0]; cam=topic.decode()
        if len(parts)>=3:
            ts_part=parts[1]; jpg_part=parts[2]
        else:
            ts_part=None; jpg_part=parts[1]

        recv_t=time.time()
        try: cam_ts=float(ts_part.decode()) if ts_part else recv_t
        except: cam_ts=recv_t

        img=cv2.imdecode(np.frombuffer(jpg_part,np.uint8),cv2.IMREAD_COLOR)
        if img is None: continue

        fps=update_fps(cam,cam_ts)
        frames[cam]={"img":img,"cam_ts":cam_ts,"fps":fps}

        fq=frame_queues[cam]
        try: fq.get_nowait()
        except: pass
        try: fq.put_nowait((img.copy(),cam_ts))
        except: pass

        with detect_lock:
            ready = all(c in calibrator.extrinsics for c in [t.decode() for t in SUB_TOPICS])
        if ready:
            with detect_lock:
                tri_results = match_and_triangulate(shared_camera_candidates, calibrator)
            if len(tri_results) > 0:
                best = tri_results[0]
                Xw = best['pt']
                last_triangulated_ball = {'pos': Xw, 'err': best['reproj_err'], 'ts': time.time(), 'pair': best['pair'], 'depths':best['depths']}

                meas_ts = best.get('meas_ts', time.time())
                reproj = best.get('reproj_err', None)
                # new measurement noise mapping (tighter)
                if reproj is not None:
                    sigma = max(0.015, 0.002 * float(reproj))
                    meas_R = np.diag([sigma*sigma, sigma*sigma, sigma*sigma])
                else:
                    meas_R = None

                # EKF update (with internal velocity reset if measurement jumps)
                updated = ball_ekf.update(Xw, meas_R=meas_R, meas_ts=meas_ts)

                # prediction horizon reduced to 15 ms
                tau = 0.015
                pred = ball_ekf.predict_future(tau)
                if pred is not None:
                    p_future = pred[0:3]
                    pred_abs_t = (ball_ekf.last_ts if ball_ekf.last_ts is not None else time.time()) + tau

                    # compute robot center from tag4/tag5
                    tag_info = {}
                    if 4 in shared_robot_poses:
                        tag_info['tag4'] = shared_robot_poses[4]
                    if 5 in shared_robot_poses:
                        tag_info['tag5'] = shared_robot_poses[5]

                    robot_center = compute_robot_center_from_tags(tag_info)
                    if robot_center is None:
                        for k,v in shared_robot_poses.items():
                            if isinstance(v, dict) and 'world_pos' in v:
                                robot_center = np.asarray(v['world_pos'], dtype=np.float64)
                                break

                    v_req = None
                    can_reach = None
                    if robot_center is not None:
                        dist = float(np.linalg.norm(p_future - robot_center))
                        v_req = dist / tau if tau > 0 else float('inf')
                        can_reach = (v_req <= robot_max_speed)

                    # log measured_pos, meas_ts, ekf_state
                    ekf_state = None
                    if ball_ekf.x is not None:
                        ekf_state = ball_ekf.x.copy().tolist()

                    append_prediction_log(predicted_pos=p_future, predicted_t=pred_abs_t,
                                          current_time=time.time(), robot_pos=robot_center,
                                          v_req=v_req, can_reach=can_reach,
                                          measured_pos=Xw, meas_ts=meas_ts, ekf_state=ekf_state)

                    try:
                        with detect_lock:
                            shared_robot_poses['ball_prediction'] = {
                                'pred_pos': p_future.tolist(),
                                'pred_t': pred_abs_t,
                                'meas_ts': meas_ts,
                                'v_req': v_req,
                                'can_reach_in_tau': can_reach
                            }
                    except Exception:
                        shared_robot_poses['ball_prediction'] = {
                            'pred_pos': p_future.tolist(),
                            'pred_t': pred_abs_t,
                            'meas_ts': meas_ts,
                            'v_req': v_req,
                            'can_reach_in_tau': can_reach
                        }

                    print(f"[EKF PRED] t+{tau:.3f}s -> {p_future}  can_reach={can_reach} v_req={v_req} meas_z={Xw[2]:.3f} ekf_z={ball_ekf.x[2]:.3f}")
                else:
                    print("[EKF] prediction not ready (filter uninitialized)")

        # visualization / CLI output (same structure as before)
        if all(k in frames for k in [t.decode() for t in SUB_TOPICS]):
            cams=[t.decode() for t in SUB_TOPICS]
            L=frames[cams[0]]; R=frames[cams[1]]
            drift_ms=abs(L["cam_ts"]-R["cam_ts"])*1000.0

            def overlay(F,cam_name):
                im=F["img"].copy()
                y=20
                cv2.putText(im,f"{cam_name}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,32,20),2)
                cv2.putText(im,f"FPS:{F['fps']:.1f}",(10,y+26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(14,117,5),2)
                cv2.putText(im,f"cam_ts:{fmt_ts(F['cam_ts'])}",(10,y+52),cv2.FONT_HERSHEY_SIMPLEX,0.5,(5,12,117),1)

                with detect_lock:
                    block=detect_cache.get(cam_name,{})

                    if "tags" in block and block["tags"]["ids"] is not None:
                        cs=block["tags"]["corners"]
                        ids=block["tags"]["ids"]
                        cv2.aruco.drawDetectedMarkers(im,cs,ids)

                    if "balls" in block:
                        for i,d in enumerate(block["balls"]):
                            x,y,w,h=d["bbox"]
                            cx,cy=d["centroid"]
                            color=d["color"]
                            if color=="orange": bc=(0,200,255)
                            elif color=="purple": bc=(200,0,200)
                            else: bc=(0,200,200)
                            cv2.rectangle(im,(x,y),(x+w,y+h),bc,2)
                            cv2.circle(im,(cx,cy),4,(0,0,255),-1)
                            cv2.putText(im,f"{color}:{i}",(x,y-6),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,bc,2)

                    text_y = 80
                    if shared_robot_poses:
                        for tid,info in shared_robot_poses.items():
                            pos = info.get('world_pos')
                            if pos is not None and isinstance(pos, (list, tuple, np.ndarray)):
                                tx = f"Tag{tid} world: {pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}"
                                cv2.putText(im, tx, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                                text_y += 18

                    if cam_name == cams[0] and last_triangulated_ball is not None:
                        p = last_triangulated_ball['pos']
                        tx = f"Ball world: {p[0]:.3f},{p[1]:.3f},{p[2]:.3f} err:{last_triangulated_ball['err']:.2f}"
                        cv2.putText(im, tx, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                return im

            if VISUALIZE and (time.time()-last_show)>(1.0/DISPLAY_FPS):
                last_show=time.time()
                left_im=overlay(L,cams[0])
                right_im=overlay(R,cams[1])

                h=max(left_im.shape[0],right_im.shape[0])
                right_res=cv2.resize(right_im,(left_im.shape[1],h))
                tile=np.hstack([left_im,right_res])

                cv2.putText(tile,f"Drift:{drift_ms:.1f}ms",(10,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

                cv2.putText(tile, f"Host now: {fmt_ts(time.time())}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                cv2.imshow("Combined Detection",tile)
            elif not VISUALIZE:
                with detect_lock:
                    tag_summary = []
                    parts_status = []
                    for c in cams:
                        block = detect_cache.get(c,{})
                        if "tags" in block and block["tags"]["ids"] is not None:
                            ids=block["tags"]["ids"]
                            tag_summary.append(f"{c}: {ids.flatten().tolist()}")
                        elif "tags" in block:
                            tag_summary.append(f"{c}: No tags")
                        else:
                            tag_summary.append(f"{c}: No detection data")
                        if "balls" in block:
                            counts = { "orange":0, "purple":0, "unknown":0 }
                            for i,d in enumerate(block["balls"]):
                                counts[d.get("color","unknown")] = counts.get(d.get("color","unknown"),0) + 1
                            parts_status.append(f"{c}: Orange: {counts['orange']} Purple: {counts['purple']}")
                        else:
                            parts_status.append(f"{c}:NoBall")
                status = (
                    f"Drift {drift_ms:.1f} ms | "
                    f"{cams[0]} ts: {fmt_ts(L['cam_ts'])} | "
                    f"{cams[1]} ts: {fmt_ts(R['cam_ts'])} | "
                    f"Host now: {fmt_ts(time.time())} | "
                    f"{cams[0]} FPS: {L['fps']:.1f} | "
                    f"{cams[1]} FPS: {R['fps']:.1f} | "
                    f"Tags:" + ",".join(tag_summary)
                )
                sys.stdout.write("\r" + status + " " * 20 + " | ".join(parts_status) + " " * 20)
                sys.stdout.flush()

        if cv2.waitKey(1)&0xFF==27:
            break

except KeyboardInterrupt:
    pass

finally:
    for t in tag_threads.values(): t.stop()
    for t in ball_threads.values(): t.stop()
    time.sleep(0.1)
    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    log_queue.put(None) 
    log_thread.join()
    print("Logger stopped.")
    print("Exit clean.")
