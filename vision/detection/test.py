import cv2, zmq, numpy as np, time, threading, queue, traceback, os, csv
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]
FPS_WINDOW = 1.0        
DISPLAY_FPS = 20
VISUALIZE = True

# Sync threshold for triangulation (seconds)
# 0.05s = 50ms. Since cams run at ~40-60FPS (16-25ms gap), this allows finding pairs.
MAX_SYNC_DIFF = 0.05 

# Scaling (Same as detection.py)
BALL_DETECTION_SCALE = 0.5
TAG_DETECTION_SCALE = 1.0 

LOG_DIR = "../../data/triangulation_logs"
LOG_FILENAME = f"{LOG_DIR}/triang_log_{int(time.time())}.csv"

# Calibration Paths (Adjust if needed)
CALIB_DIR = "../calibration" 

# Tag Configurations
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11
TAG_SIZES = {0: 0.099, 1: 0.096, 2: 0.096, 3: 0.096, 4: 0.096, 5: 0.096}
# World Positions of Static Tags (0-3) for PnP
TAG_POSITIONS = {
    0: np.array([0.9, 0.0, 0.0], dtype=float),
    1: np.array([0.0, 0.0, 0.0], dtype=float),
    2: np.array([0.9, 0.9, 0.0], dtype=float),
    3: np.array([0.0, 0.9, 0.0], dtype=float)
}

# HSV Config
HSV_CONFIG = {
    "kreo1": { "orange": {'hmin': 0, 'smin': 116, 'vmin': 160, 'hmax': 12, 'smax': 197, 'vmax': 255} },
    "kreo2": { "orange": {'hmin': 0, 'smin': 79, 'vmin': 181, 'hmax': 12, 'smax': 255, 'vmax': 255} }
}
DEFAULT_HSV = {'hmin': 0, 'smin': 100, 'vmin': 100, 'hmax': 25, 'smax': 255, 'vmax': 255}

# Detector Params
BASE_MIN_AREA = 100    
BASE_MAX_AREA = 20000  
CIRCULARITY_MIN = 0.5
ASPECT_RATIO_MIN = 0.6 
ASPECT_RATIO_MAX = 1.6   
MAX_DETECTIONS_PER_CAM = 5    

# ---------- Logging Setup ----------
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_queue = queue.Queue()

def logger_worker():
    try:
        with open(LOG_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", 
                "ball_3d_x", "ball_3d_y", "ball_3d_z",
                "car_3d_x", "car_3d_y", "car_3d_z",
                "cam1_ball_detected", "cam2_ball_detected"
            ])
            while True:
                entry = log_queue.get()
                if entry is None: break
                writer.writerow(entry)
                log_queue.task_done()
    except Exception as e: print(f"[LOGGER ERROR] {e}")

log_thread = threading.Thread(target=logger_worker, daemon=True)
log_thread.start()

# ---------- Global Shared State ----------
# This holds the LATEST processed data from threads for Main Loop to consume
# structure: { 'kreo1': {ts: 0.0, ball: {}, car_center: [], P: []}, ... }
latest_data = {
    "kreo1": {"ts": 0.0, "ball": None, "car_center": None, "P": None, "img": None},
    "kreo2": {"ts": 0.0, "ball": None, "car_center": None, "P": None, "img": None}
}
data_lock = threading.Lock()

# ---------- Helpers: Calibration & Geometry ----------

def load_camera_calib(cam_name):
    path = os.path.join(CALIB_DIR, f'camera_calibration_{cam_name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    calib = np.load(path)
    camera_matrix = calib["cameraMatrix"]
    dist_coeffs = calib['distCoeffs']
    print("[INFO] Loaded calibrated camera parameters")
    return camera_matrix, dist_coeffs

def compute_robot_center_from_tags(tag_info):
    """
    Your provided logic to compute car center from Tag 4 and Tag 5.
    Expects tag_info to have keys 'tag4' and 'tag5', each with 'world_pos'.
    """
    t4 = tag_info.get('tag4', None)
    t5 = tag_info.get('tag5', None)
    shift_m = 0.096
    
    # Helper to safely get pos
    def get_pos(t): return np.asarray(t.get('world_pos'), dtype=np.float64) if t and 'world_pos' in t else None
    def get_y(t): return np.asarray(t.get('y_axis_unit', [0.0, 1.0, 0.0]), dtype=np.float64) if t else np.array([0,1,0])

    p4 = get_pos(t4)
    p5 = get_pos(t5)

    if p4 is not None and p5 is not None:
        center = 0.5 * (p4 + p5)
        return center
    if p4 is not None:
        y_axis = get_y(t4)
        center = p4 + (-shift_m) * y_axis
        return center
    if p5 is not None:
        y_axis = get_y(t5)
        center = p5 + (shift_m) * y_axis
        return center
    return None

def get_orange_mask(bgr_img, hsv_dict):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_dict['hmin'], hsv_dict['smin'], hsv_dict['vmin']], dtype=np.uint8)
    upper = np.array([hsv_dict['hmax'], hsv_dict['smax'], hsv_dict['vmax']], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.GaussianBlur(mask, (5, 5), 0)

def find_ball_contours(mask, min_area, max_area):
    if mask is None: return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area: continue
        x,y,w,h = cv2.boundingRect(c)
        aspect = float(w)/h if h > 0 else 0
        if ASPECT_RATIO_MIN > aspect or aspect > ASPECT_RATIO_MAX: continue
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity >= CIRCULARITY_MIN:
            candidates.append({"bbox": (x,y,w,h), "area": area, "centroid": (x+w//2, y+h//2)})
    candidates.sort(key=lambda d: d["area"], reverse=True)
    return candidates

# ---------- THREAD 1: TAG & EXTRINSICS (Background) ----------
class TagProcessor(threading.Thread):
    def __init__(self, cam_name, frame_queue, K, D):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.K = K
        self.D = D
        
        # AprilTag Setup
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
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
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        self.stop_flag = False
        
        # State persistency
        self.extrinsics_locked = False # Only calc 0-3 once
        self.last_P = None # Projection Matrix (3x4)
        self.last_car_center = None
        self.R_inv = None # World->Cam inverse (Cam->World rotation)
        self.t_inv = None # World->Cam inverse (Cam->World translation)

    def run(self):
        print(f"[{self.cam_name}] Tag Thread Started.")
        while not self.stop_flag:
            try:
                jpg_bytes, cam_ts = self.frame_queue.get(timeout=0.1)
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)

                current_tags = {}
                image_points = []
                object_points = []

                if ids is not None:
                    ids_flat = ids.flatten()
                    for i, tid in enumerate(ids_flat):
                        current_tags[tid] = corners[i]
                        
                        # If it's a static tag (0-3) AND we haven't locked extrinsics yet
                        if not self.extrinsics_locked and tid in TAG_POSITIONS:
                            image_points.append(corners[i][0].mean(axis=0)) # Center of tag
                            object_points.append(TAG_POSITIONS[tid])

                # 1. Calculate Camera Extrinsics (World -> Camera) using Static Tags
                # Only do this if we haven't locked them yet
                if not self.extrinsics_locked and len(image_points) >= 4: 
                    try:
                        _, rvec, tvec = cv2.solvePnP(np.array(object_points), np.array(image_points), self.K, self.D, flags=cv2.SOLVEPNP_EPNP)
                        
                        # Refine
                        rvec, tvec = cv2.solvePnPRefineLM(np.array(object_points), np.array(image_points), self.K, self.D, rvec, tvec)
                        
                        # Rotation Matrix
                        R, _ = cv2.Rodrigues(rvec)
                        
                        # Projection Matrix P = K * [R|t]
                        RT = np.hstack((R, tvec))
                        self.last_P = np.dot(self.K, RT)
                        
                        # Pre-calc Inverse Transform (Camera -> World) for Car Tags
                        # Pos_world = R_inv * (Pos_cam - t)
                        self.R_inv = R.T
                        self.t_inv = -np.dot(self.R_inv, tvec)
                        
                        print(f"[{self.cam_name}] Extrinsics LOCKED.")
                        self.extrinsics_locked = True
                    except Exception as e:
                        print(f"[{self.cam_name}] Extrinsic Calc Failed: {e}")

                # 2. Calculate Dynamic Tags (Car) in World Space
                # We need valid extrinsics (R_inv, t_inv) to do this
                if self.extrinsics_locked and self.R_inv is not None:
                    tag_info_for_car = {}
                    for car_tid in [4, 5]:
                        if car_tid in current_tags:
                            # Get Pose of Tag relative to Camera
                            sz = TAG_SIZES[car_tid]
                            # Define tag corners in Tag's local space
                            obj_corners = np.array([
                                [-sz/2, sz/2, 0], [sz/2, sz/2, 0], [sz/2, -sz/2, 0], [-sz/2, -sz/2, 0]
                            ], dtype=np.float32)
                            
                            # Solve PnP for just this tag
                            _, rvec_t, tvec_t = cv2.solvePnP(obj_corners, current_tags[car_tid][0], self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                            
                            # tvec_t is Tag Position in Camera Frame
                            # Convert to World Frame: P_world = R_inv * tvec_t + t_inv
                            # (Note: t_inv here is already R_inv * -tvec_cam, so logic is P_world = R_inv * (P_cam - tvec_cam))
                            
                            # tvec_t is (3,1), t_inv is (3,1)
                            # R_inv is (3,3)
                            # We want: WorldPos = R_inv * (TagPosCam) + (-R_inv * CamPosWorld) -> wait
                            # Correct: X_c = R * X_w + t  => X_w = R^T * (X_c - t)
                            pos_world = np.dot(self.R_inv, (tvec_t)) + self.t_inv
                            
                            # Get Y-axis for user function logic (orientation)
                            # Y-unit in Tag Space is (0,1,0). Rotate it to Camera Space, then to World Space.
                            # Tag->Cam Rotation is R_tag. Cam->World Rotation is R_inv.
                            R_tag, _ = cv2.Rodrigues(rvec_t)
                            y_axis_cam = np.dot(R_tag, np.array([0,1,0], dtype=float))
                            y_axis_world = np.dot(self.R_inv, y_axis_cam)

                            # Store info
                            tag_info_for_car[f"tag{car_tid}"] = {
                                "world_pos": pos_world.flatten(),
                                "y_axis_unit": y_axis_world.flatten()
                            }
                    
                    # Compute Car Center using User's Logic
                    if tag_info_for_car:
                        self.last_car_center = compute_robot_center_from_tags(tag_info_for_car)

                # Update Global State
                with data_lock:
                    latest_data[self.cam_name]["P"] = self.last_P
                    latest_data[self.cam_name]["car_center"] = self.last_car_center

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[TAG-{self.cam_name}] Error: {e}")
                traceback.print_exc()

    def stop(self):
        self.stop_flag = True

# ---------- THREAD 2: BALL PROCESSOR (Fast) ----------
class BallProcessor(threading.Thread):
    def __init__(self, cam_name, frame_queue):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.stop_flag = False
        
        # Config
        self.min_area = BASE_MIN_AREA * (BALL_DETECTION_SCALE**2)
        self.max_area = BASE_MAX_AREA * (BALL_DETECTION_SCALE**2)
        self.hsv = HSV_CONFIG.get(cam_name, {}).get("orange", DEFAULT_HSV)

    def run(self):
        print(f"[{self.cam_name}] Ball Thread Started.")
        while not self.stop_flag:
            try:
                jpg_bytes, cam_ts = self.frame_queue.get(timeout=0.1)
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue
                
                # Resize
                if BALL_DETECTION_SCALE != 1.0:
                    f_small = cv2.resize(frame, None, fx=BALL_DETECTION_SCALE, fy=BALL_DETECTION_SCALE, interpolation=cv2.INTER_NEAREST)
                else:
                    f_small = frame
                
                # Detect
                mask = get_orange_mask(f_small, self.hsv)
                balls = find_ball_contours(mask, self.min_area, self.max_area)
                
                ball_center = None
                if balls:
                    # Get best ball
                    b = balls[0]
                    scale = 1.0 / BALL_DETECTION_SCALE
                    cx = int(b["centroid"][0] * scale)
                    cy = int(b["centroid"][1] * scale)
                    ball_center = (float(cx), float(cy))
                
                # Update Global State
                with data_lock:
                    latest_data[self.cam_name]["ts"] = cam_ts
                    latest_data[self.cam_name]["ball"] = ball_center
                    if VISUALIZE:
                        latest_data[self.cam_name]["img"] = frame # Store full res for viz

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[BALL-{self.cam_name}] Error: {e}")

    def stop(self):
        self.stop_flag = True

# ---------- Main Triangulation Loop ----------
def main():
    # 1. Setup ZMQ
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.setsockopt(zmq.RCVHWM, 4)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.LINGER, 0)
    for t in SUB_TOPICS: sub.setsockopt(zmq.SUBSCRIBE, t)

    # 2. Load Calib & Init Queues
    K1, D1 = load_camera_calib("kreo1")
    K2, D2 = load_camera_calib("kreo2")
    
    queues = {
        "kreo1": {"ball": queue.Queue(maxsize=1), "tag": queue.Queue(maxsize=1)},
        "kreo2": {"ball": queue.Queue(maxsize=1), "tag": queue.Queue(maxsize=1)}
    }

    # 3. Start Threads
    threads = []
    
    # Kreo 1
    t1_tag = TagProcessor("kreo1", queues["kreo1"]["tag"], K1, D1); t1_tag.start(); threads.append(t1_tag)
    t1_ball = BallProcessor("kreo1", queues["kreo1"]["ball"]); t1_ball.start(); threads.append(t1_ball)
    
    # Kreo 2
    t2_tag = TagProcessor("kreo2", queues["kreo2"]["tag"], K2, D2); t2_tag.start(); threads.append(t2_tag)
    t2_ball = BallProcessor("kreo2", queues["kreo2"]["ball"]); t2_ball.start(); threads.append(t2_ball)

    print(f"[Triangulator] System running. Logging to {LOG_FILENAME}")
    
    last_viz = time.time()

    try:
        while True:
            # --- INGESTION ---
            try:
                parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                cam = parts[0].decode()
                ts_part = parts[1] if len(parts) >= 3 else None
                jpg_part = parts[2] if len(parts) >= 3 else parts[1]
                try: cam_ts = float(ts_part.decode()) if ts_part else time.time()
                except: cam_ts = time.time()

                # Distribute to queues
                qs = queues.get(cam)
                if qs:
                    # LIFO push
                    for qtype in ["ball", "tag"]:
                        try: qs[qtype].put_nowait((jpg_part, cam_ts))
                        except queue.Full:
                            try: qs[qtype].get_nowait(); qs[qtype].put_nowait((jpg_part, cam_ts))
                            except: pass
            except zmq.Again:
                time.sleep(0.0001)

            # --- TRIANGULATION LOGIC ---
            # Run this frequently, but relies on shared state updated by threads
            with data_lock:
                d1 = latest_data["kreo1"]
                d2 = latest_data["kreo2"]
            
            # Check sync
            time_diff = abs(d1["ts"] - d2["ts"])
            
            if time_diff < MAX_SYNC_DIFF:
                # Timestamps aligned. Attempt Triangulation.
                
                # 1. 3D Ball
                ball_3d = [None, None, None]
                if d1["ball"] and d2["ball"] and d1["P"] is not None and d2["P"] is not None:
                    # Convert to format for triangulation: (2, N)
                    pts1 = np.array([d1["ball"]]).T 
                    pts2 = np.array([d2["ball"]]).T
                    
                    # Triangulate (Homogeneous 4D)
                    pts4d = cv2.triangulatePoints(d1["P"], d2["P"], pts1, pts2)
                    
                    # Convert to 3D Euclidean
                    pts3d = pts4d[:3] / pts4d[3]
                    ball_3d = pts3d.flatten().tolist() # [x, y, z]

                # 2. 3D Car Center
                # Average the estimates from both cameras if available
                car_est = []
                if d1["car_center"] is not None: car_est.append(d1["car_center"])
                if d2["car_center"] is not None: car_est.append(d2["car_center"])
                
                car_3d = [None, None, None]
                if car_est:
                    avg_car = np.mean(car_est, axis=0)
                    car_3d = avg_car.flatten().tolist()

                # 3. Log if we have ANY data (even just 2D detections)
                # To prevent log spam, maybe only log if something changed or detected?
                # For now, logging continuously when synced to track trajectory
                if d1["ball"] or d2["ball"] or car_3d[0] is not None:
                    log_queue.put([
                        f"{d1['ts']:.3f}", # Using cam1 ts as reference
                        ball_3d[0], ball_3d[1], ball_3d[2],
                        car_3d[0], car_3d[1], car_3d[2],
                        bool(d1["ball"]), bool(d2["ball"])
                    ])

            # --- VISUALIZATION ---
            if VISUALIZE and (time.time() - last_viz) > (1.0/DISPLAY_FPS):
                with data_lock:
                    if latest_data["kreo1"]["img"] is not None and latest_data["kreo2"]["img"] is not None:
                        imgs = []
                        for c in ["kreo1", "kreo2"]:
                            im = latest_data[c]["img"].copy()
                            # Draw Ball
                            b = latest_data[c]["ball"]
                            if b: cv2.circle(im, (int(b[0]), int(b[1])), 8, (0,255,0), -1)
                            
                            # Draw Car Text
                            cc = latest_data[c]["car_center"]
                            if cc is not None:
                                cv2.putText(im, "Car Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                            else:
                                cv2.putText(im, "No Car", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                
                            imgs.append(im)
                        
                        # Stack
                        l_im, r_im = imgs[0], imgs[1]
                        h = min(l_im.shape[0], r_im.shape[0])
                        if l_im.shape[0] != h: l_im = cv2.resize(l_im, (int(l_im.shape[1]*h/l_im.shape[0]), h))
                        if r_im.shape[0] != h: r_im = cv2.resize(r_im, (int(r_im.shape[1]*h/r_im.shape[0]), h))
                        
                        tile = np.hstack([l_im, r_im])
                        cv2.imshow("Triangulation Debug", tile)
                        last_viz = time.time()
                
                if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        pass
    finally:
        for t in threads: t.stop()
        log_queue.put(None)
        log_thread.join()
        cv2.destroyAllWindows()
        sub.close()
        ctx.term()
        print("\nExiting.")

if __name__ == "__main__":
    main()