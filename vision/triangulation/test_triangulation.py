import cv2, zmq, numpy as np, time, threading, queue, traceback, sys, os, csv
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]
FPS_WINDOW = 1.0        
DISPLAY_FPS = 70        
VISUALIZE = True        

# SCALING CONFIG
BALL_DETECTION_SCALE = 0.5  # Downscale for speed (Ball)
TAG_DETECTION_SCALE = 1.0   # Full res for accuracy (Tags)

LOG_DIR = "../../data/triangulation_logs"
LOG_FILENAME = f"{LOG_DIR}/log_{int(time.time())}.csv"
CALIB_DIR = "../calibration/"

# HSV Config
HSV_CONFIG = {
    "kreo1": { "orange": {'hmin': 0, 'smin': 116, 'vmin': 160, 'hmax': 12, 'smax': 197, 'vmax': 255} },
    "kreo2": { "orange": {'hmin': 0, 'smin': 79, 'vmin': 181, 'hmax': 12, 'smax': 255, 'vmax': 255} }
}
DEFAULT_HSV = {'hmin': 0, 'smin': 100, 'vmin': 100, 'hmax': 25, 'smax': 255, 'vmax': 255}

# Detector parameters
BASE_MIN_AREA = 100    
BASE_MAX_AREA = 20000  
CIRCULARITY_MIN = 0.5
ASPECT_RATIO_MIN = 0.6 
ASPECT_RATIO_MAX = 1.6   
MAX_DETECTIONS_PER_CAM = 5 

# 3D / TRIANGULATION CONFIG
STATIC_TAG_IDS = [0,1,2,3]
TAG_POSITIONS = {
    0: np.array([0.9, 0.0, 0.0], dtype=float),
    1: np.array([0.0, 0.0, 0.0], dtype=float),
    2: np.array([0.9, 0.9, 0.0], dtype=float),
    3: np.array([0.0, 0.9, 0.0], dtype=float)
}
TAG_SIZES = {0: 0.099, 1: 0.096, 2: 0.096, 3: 0.096, 4: 0.096, 5: 0.096}
CALIB_FRAMES = 30
MAX_TIME_DIFF = 0.05  # Max ms diff for triangulation pairing

# AprilTag Config
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

class StaticCalibrator:
    def __init__(self, tag_world_map, tag_size_map):
        self.tag_world_map = tag_world_map
        self.tag_size_map = tag_size_map
        self.obs = defaultdict(list)
        self.extrinsics = {}
        self.frame_count = defaultdict(int)
        self.K_cache = {}
        self.dist_cache = {}

    def load_intrinsics(self, cam_name):
        if cam_name in self.K_cache: return self.K_cache[cam_name], self.dist_cache[cam_name]
        camera_matrix, dist_coeffs = load_camera_calib(cam_name)
        self.K_cache[cam_name] = camera_matrix
        self.dist_cache[cam_name] = dist_coeffs
        return camera_matrix, dist_coeffs

    def add_detection(self, cam_name, ids, corners, ts):
        if ids is None: return
        self.frame_count[cam_name] += 1
        for i, idarr in enumerate(ids):
            tid = int(idarr[0])
            if tid in STATIC_TAG_IDS:
                c = np.array(corners[i]).reshape(4,2).astype(np.float64)
                self.obs[cam_name].append((tid, c, ts))

    def try_compute_extrinsic(self, cam_name):
        if cam_name in self.extrinsics: return True
        if self.frame_count.get(cam_name, 0) < CALIB_FRAMES: return False

        obs_list = list(reversed(self.obs.get(cam_name, [])))
        
        target_tag = None
        use_corners = None

        for (tid, corners, ts) in obs_list:
            if tid in self.tag_world_map:
                target_tag = tid
                use_corners = corners.reshape(4,2).astype(np.float64)
                break 
        
        if use_corners is None: return False 

        try: K, dist = self.load_intrinsics(cam_name)
        except: return False

        obj_corners = np.array(self.tag_world_map[target_tag], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_corners, use_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return False

        R, _ = cv2.Rodrigues(rvec)
        tvec = tvec.reshape(3,1)
        self.extrinsics[cam_name] = {"rvec": rvec, "tvec": tvec, "R": R}
        print(f"[Calib] {cam_name} extrinsics locked using Tag {target_tag}")
        return True

    def cam_to_world(self, cam_name, X_cam):
        e = self.extrinsics.get(cam_name)
        if e is None: raise RuntimeError("Calibrator: extrinsic not ready for " + cam_name)
        R = e['R']; t = e['tvec']
        X = np.asarray(X_cam, dtype=np.float64)
        print(X.ndim)
        if X.ndim == 1: Xc = X.reshape(3,1); Xw = R.T @ (Xc - t); return Xw[:,0]
        else: Xc = X.T; Xw = R.T @ (Xc - t); return Xw.T

    def get_norm_projection_matrix(self, cam_name):
        e = self.extrinsics.get(cam_name)
        if e is None: return None
        return np.hstack((e['R'], e['tvec']))


# ---------- Logging Setup ----------
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_queue = queue.Queue()

def logger_worker():
    try:
        with open(LOG_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","camera", 
                "ball_2d_x", "ball_2d_y", "ball_2d_area",
                "ball_3d_x", "ball_3d_y", "ball_3d_z",
                "tag4_x", "tag4_y", "tag4_z",
                "tag5_x", "tag5_y", "tag5_z",
                "detected_kreo1", "detected_kreo2"
            ])
            while True:
                entry = log_queue.get()
                if entry is None: break
                writer.writerow(entry)
                log_queue.task_done()
    except Exception as e: print(f"[LOGGER ERROR] {e}")

log_thread = threading.Thread(target=logger_worker, daemon=True)
log_thread.start()



# ---------- Helpers ----------

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

def estimate_pose_apriltag(corners, tag_size, cam_mtx, cam_dist):
    half = tag_size / 2.0
    objp = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float32)
    imgp = corners.reshape(4,2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, cam_mtx, cam_dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.reshape(3)
    return T

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
        aspect = float(w)/float(h) if h > 0 else 0
        if ASPECT_RATIO_MIN>aspect or aspect > ASPECT_RATIO_MAX: continue
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity >= CIRCULARITY_MIN:
            candidates.append({"bbox": (x,y,w,h), "area": area, "centroid": (x+w//2, y+h//2)})
    candidates.sort(key=lambda d: d["area"], reverse=True)
    return candidates

# ---------- Shared Data (Thread Safe) ----------
# Stores the latest tag detection result for each camera
shared_tag_data = {
    "kreo1": {"tag4": {"detected": False, "x":"", "y":""}, "tag5": {"detected": False, "x":"", "y":""}, "viz": []},
    "kreo2": {"tag4": {"detected": False, "x":"", "y":""}, "tag5": {"detected": False, "x":"", "y":""}, "viz": []}
}
# 3D Data (Populated by Threads and Main Loop)
shared_3d_poses = {
    "ball": None, # (x, y, z)
    4: None,      # (x, y, z)
    5: None       # (x, y, z)
}

# Candidates for Ball Triangulation (Written by BallThread, Read by Main)
shared_ball_candidates = defaultdict(list)
shared_ball_status = {"kreo1": False, "kreo2": False}
shared_data_lock = threading.Lock()
viz_cache = {}
viz_lock = threading.Lock()
# Global Calibrator Instance
calibrator = StaticCalibrator(TAG_WORLD_MAP, TAG_SIZES)

# ---------- THREAD 1: TAG DETECTOR (Slower, High Accuracy) ----------
class TagDetectorThread(threading.Thread):
    def __init__(self, cam_name, frame_queue):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.aruco_detector = create_april_detector()
        self.stop_flag = False

    def run(self):
        print(f"[{self.cam_name}] Tag Thread started.")
        while not self.stop_flag:
            try:
                # Get raw bytes (Tag thread decodes independently to avoid blocking Ball thread)
                jpg_bytes, cam_ts = self.frame_queue.get(timeout=0.1)
                
                # Decode
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue

                # Detect (Full Res)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
                
                viz_tags = []
                local_tag4_det = False
                local_tag5_det = False

                if ids is not None:
                    calibrator.add_detection(self.cam_name, ids, corners, cam_ts)
                    calibrator.try_compute_extrinsic(self.cam_name)

                    ids_flat = ids.flatten()
                    for i, tag_id in enumerate(ids_flat):
                        viz_tags.append({"id": tag_id, "corners": corners[i]})
                        
                        # 2. Compute 3D for Moving Tags (4 & 5)
                        # Only if extrinsic is locked
                        if tag_id in [4, 5] and self.cam_name in calibrator.extrinsics:
                            try:
                                K, dist = calibrator.load_intrinsics(self.cam_name)
                                c_corners = np.array(corners[i]).reshape(4,2)
                                T_tag_cam = estimate_pose_apriltag(c_corners, TAG_SIZES[tag_id], K, dist)
                                tag_world = calibrator.cam_to_world(self.cam_name, T_tag_cam[:3,3].reshape(3,))
                                
                                with shared_data_lock:
                                    shared_3d_poses[tag_id] = tag_world
                                    if tag_id == 4: local_tag4_det = True
                                    if tag_id == 5: local_tag5_det = True
                            except Exception as e:
                                pass # Pose estimation failed for this frame
                
                # Update 2D Shared
                with shared_data_lock:
                    shared_tag_data[self.cam_name] = {
                        "tag4": {"detected": local_tag4_det},
                        "tag5": {"detected": local_tag5_det},
                        "viz": viz_tags
                    }
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[TAG-ERR-{self.cam_name}]", e)

    def stop(self):
        self.stop_flag = True

# ---------- THREAD 2: BALL DETECTOR (Fast, Low Latency) ----------
class BallDetectorThread(threading.Thread):
    def __init__(self, cam_name, frame_queue):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.min_area_scaled = BASE_MIN_AREA * (BALL_DETECTION_SCALE**2)
        self.max_area_scaled = BASE_MAX_AREA * (BALL_DETECTION_SCALE**2)
        self.hsv_vals = HSV_CONFIG.get(cam_name, {}).get("orange", DEFAULT_HSV)
        self.stop_flag = False
        self.fps_dq = deque()

    def run(self):
        print(f"[{self.cam_name}] Ball Thread started.")
        while not self.stop_flag:
            try:
                jpg_bytes, cam_ts = self.frame_queue.get(timeout=0.1)
                
                # Decode
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue

                # Resize for Speed
                if BALL_DETECTION_SCALE != 1.0:
                    frame_small = cv2.resize(frame, None, fx=BALL_DETECTION_SCALE, fy=BALL_DETECTION_SCALE, interpolation=cv2.INTER_NEAREST)
                else:
                    frame_small = frame

                # Detect Ball
                mask = get_orange_mask(frame_small, self.hsv_vals)
                balls = find_ball_contours(mask, self.min_area_scaled, self.max_area_scaled)
                
                ball_data = {"detected": False, "x": "", "y": "", "area": ""}
                viz_ball = None
                candidates_for_triangulation = []

                if balls:
                    b = balls[0] 
                    scale_inv = 1.0 / BALL_DETECTION_SCALE
                    bx, by, bw, bh = b["bbox"]
                    real_x = int(bx * scale_inv); real_y = int(by * scale_inv)
                    real_w = int(bw * scale_inv); real_h = int(bh * scale_inv)
                    real_area = int(b["area"] * (scale_inv**2))
                    cx, cy = real_x + real_w//2, real_y + real_h//2
                    
                    ball_data = {"detected": True, "x": cx, "y": cy, "area": real_area}
                    viz_ball = {"bbox": (real_x, real_y, real_w, real_h), "centroid": (cx, cy)}
                    
                    # Prepare data for match_and_triangulate
                    candidates_for_triangulation.append({
                        "centroid": (cx, cy),
                        "area": real_area,
                        "color": "orange",
                        "ts": cam_ts
                    })
                # Update Shared Memory
                with shared_data_lock:
                    # Update candidates for this camera
                    shared_ball_candidates[self.cam_name] = candidates_for_triangulation
                    shared_ball_status[self.cam_name] = ball_data["detected"]
                    
                    # Read GLOBAL 3D Data (Calculated by Main Loop or Tag Thread)
                    # Note: There might be a 1-frame delay for Ball 3D, which is acceptable for decoupled logging
                    ball_3d = shared_3d_poses.get("ball")
                    tag4_3d = shared_3d_poses.get(4)
                    tag5_3d = shared_3d_poses.get(5)
                    tags_viz = shared_tag_data[self.cam_name]["viz"]
                    
                    # Read detection status of OTHER camera for logging
                    det_k1 = shared_ball_status["kreo1"]
                    det_k2 = shared_ball_status["kreo2"]
                
                # --- LOGGING (Driven by this thread) ---
                b3d = ball_3d if ball_3d is not None else ["", "", ""]
                t4d = tag4_3d if tag4_3d is not None else ["", "", ""]
                t5d = tag5_3d if tag5_3d is not None else ["", "", ""]

                log_queue.put([
                    f"{cam_ts:.3f}", self.cam_name,
                    ball_data["x"], ball_data["y"], ball_data["area"],
                    b3d[0], b3d[1], b3d[2],
                    t4d[0], t4d[1], t4d[2],
                    t5d[0], t5d[1], t5d[2],
                    det_k1, det_k2
                ])

                # FPS Calc
                self.fps_dq.append(time.time())
                while self.fps_dq and (self.fps_dq[-1] - self.fps_dq[0]) > FPS_WINDOW:
                    self.fps_dq.popleft()
                fps = len(self.fps_dq) / FPS_WINDOW

                if VISUALIZE:
                    with viz_lock:
                        viz_cache[self.cam_name] = {
                            "img": frame,
                            "ball": viz_ball,
                            "tags": tags_viz,
                            "fps": fps
                        }

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[BALL-ERR-{self.cam_name}]", e)
    
    def stop(self):
        self.stop_flag = True

def match_and_triangulate(camera_candidates, calibrator, reproj_thresh_px=15.0):
    cams = list(camera_candidates.keys())
    if len(cams) < 2: return []
    
    pair = ('kreo1', 'kreo2') if 'kreo1' in cams and 'kreo2' in cams else (cams[0], cams[1])
    c1, c2 = pair
    
    # Filter candidates (sort by area)
    cand1 = sorted(camera_candidates[c1], key=lambda x: -x.get('area',1))[:8]
    cand2 = sorted(camera_candidates[c2], key=lambda x: -x.get('area',1))[:8]

    # Get Calibration Data
    if c1 not in calibrator.extrinsics or c2 not in calibrator.extrinsics: return []
    
    K1, D1 = calibrator.load_intrinsics(c1)
    K2, D2 = calibrator.load_intrinsics(c2)
    P1_norm = calibrator.get_norm_projection_matrix(c1) # [R1|t1]
    P2_norm = calibrator.get_norm_projection_matrix(c2) # [R2|t2]

    results = []
    
    for a in cand1:
        for b in cand2:
            if a['color'] != b['color']: continue
            
            dt = abs(a['ts'] - b['ts'])
            if dt > MAX_TIME_DIFF: continue

            pt1_in = np.array(a['centroid'], dtype=float).reshape(-1,1,2)
            pt2_in = np.array(b['centroid'], dtype=float).reshape(-1,1,2)
            
            # Undistort to Normalized Coordinates (x, y) where z=1
            pt1_norm = cv2.undistortPoints(pt1_in, K1, D1, P=None)
            pt2_norm = cv2.undistortPoints(pt2_in, K2, D2, P=None)

            pt1_norm = pt1_norm.reshape(-1, 2).T  
            pt2_norm = pt2_norm.reshape(-1, 2).T 

            # Triangulate in World Frame
            Xh = cv2.triangulatePoints(P1_norm, P2_norm, pt1_norm, pt2_norm)
            w = Xh[3]
            if abs(w) < 1e-6: continue
            Xw = (Xh[:3] / w).flatten()

            # Reproject to verify
            img_pt1, _ = cv2.projectPoints(Xw.reshape(1,3), calibrator.extrinsics[c1]['rvec'], calibrator.extrinsics[c1]['tvec'], K1, D1)
            img_pt2, _ = cv2.projectPoints(Xw.reshape(1,3), calibrator.extrinsics[c2]['rvec'], calibrator.extrinsics[c2]['tvec'], K2, D2)

            err1 = np.linalg.norm(img_pt1.flatten() - np.array(a['centroid']))
            err2 = np.linalg.norm(img_pt2.flatten() - np.array(b['centroid']))
            tot_err = err1 + err2

            if tot_err < reproj_thresh_px:
                results.append({
                    'pt': Xw, 
                    'reproj_err': tot_err, 
                    'ts': max(a['ts'], b['ts']) 
                })

    results.sort(key=lambda r: r['reproj_err'])
    return results

# ---------- Main Execution ----------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(ZMQ_ADDR)
sub.setsockopt(zmq.RCVHWM, 4)
sub.setsockopt(zmq.CONFLATE, 1) 
sub.setsockopt(zmq.LINGER, 0)
for t in SUB_TOPICS: sub.setsockopt(zmq.SUBSCRIBE, t)

# Init Queues (2 per camera: 1 for Ball, 1 for Tag)
queues = {
    "kreo1": {"ball": queue.Queue(maxsize=1), "tag": queue.Queue(maxsize=1)},
    "kreo2": {"ball": queue.Queue(maxsize=1), "tag": queue.Queue(maxsize=1)}
}

threads = []
# Spin up threads
for cam in ["kreo1", "kreo2"]:
    # Ball Thread
    bt = BallDetectorThread(cam, queues[cam]["ball"])
    bt.start()
    threads.append(bt)
    
    # Tag Thread
    tt = TagDetectorThread(cam, queues[cam]["tag"])
    tt.start()
    threads.append(tt)

print(f"[System] Decoupled Detection Started. Logging to {LOG_FILENAME}")

last_show = time.time()

try:
    while True:
        # High Speed Ingestion Loop
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
            topic = parts[0]
            cam = topic.decode()
            ts_part = parts[1] if len(parts) >= 3 else None
            jpg_part = parts[2] if len(parts) >= 3 else parts[1]
            try: cam_ts = float(ts_part.decode()) if ts_part else time.time()
            except: cam_ts = time.time()
            
            qs = queues.get(cam)
            if qs:
                try: qs["ball"].put_nowait((jpg_part, cam_ts))
                except queue.Full: pass
                try: qs["tag"].put_nowait((jpg_part, cam_ts))
                except queue.Full: pass
        except zmq.Again:
            time.sleep(0.0001)

        # ----------------------------------------------------
        #  3D BALL CALCULATION & LOGGING
        # ----------------------------------------------------
        # We periodically check shared candidates to triangulate
        with shared_data_lock:
            current_ball_candidates = dict(shared_ball_candidates)

        # Perform Triangulation
        tri_results = match_and_triangulate(current_ball_candidates, calibrator)
        
        with shared_data_lock:
            if len(tri_results) > 0:
                res = tri_results[0]
                shared_3d_poses["ball"] = res['pt']
            else:
                # Optional: Decay old data or keep last known position? 
                # Keeping last known for now to avoid flickering, or set to None
                res = None
                shared_3d_poses["ball"] = None
                pass 

        # ----------------------------------------------------
        #  VISUALIZATION
        # ----------------------------------------------------
        curr_time = time.time()
        if VISUALIZE and (curr_time - last_viz_time) > (1.0/DISPLAY_FPS):
            last_viz_time = curr_time
            with viz_lock:
                has_data = all(c in viz_cache for c in ["kreo1", "kreo2"])
                if has_data:
                    def draw(cam_key):
                        d = viz_cache[cam_key]
                        im = d["img"].copy()
                        
                        if d["ball"]:
                            bx,by,bw,bh = d["ball"]["bbox"]
                            cx,cy = d["ball"]["centroid"]
                            cv2.rectangle(im, (bx,by), (bx+bw, by+bh), (0,165,255), 2)
                            cv2.circle(im, (cx,cy), 5, (0,0,255), -1)
                        
                        for tag in d["tags"]:
                             cv2.aruco.drawDetectedMarkers(im, [tag["corners"].astype(int)], np.array([[tag["id"]]]))
                        
                        # 3D Text Overlay
                        y_off = 60
                        if shared_3d_poses["ball"] is not None:
                            bp = shared_3d_poses["ball"]
                            cv2.putText(im, f"Ball 3D: {bp[0]:.2f}, {bp[1]:.2f}, {bp[2]:.2f}", (10, y_off), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                            y_off += 25
                        
                        for tid in [4, 5]:
                            if shared_3d_poses[tid] is not None:
                                tp = shared_3d_poses[tid]
                                cv2.putText(im, f"Tag {tid} 3D: {tp[0]:.2f}, {tp[1]:.2f}, {tp[2]:.2f}", (10, y_off), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                y_off += 25
                        
                        if cam_key not in calibrator.extrinsics:
                             cv2.putText(im, f"CALIBRATING... {calibrator.frame_count.get(cam_key,0)}/{CALIB_FRAMES}", 
                                         (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                        cv2.putText(im, f"FPS: {d['fps']:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        return im

                    l_im = draw("kreo1")
                    r_im = draw("kreo2")
                    
                    h = min(l_im.shape[0], r_im.shape[0])
                    if l_im.shape[0] != h: l_im = cv2.resize(l_im, (int(l_im.shape[1]*h/l_im.shape[0]), h))
                    if r_im.shape[0] != h: r_im = cv2.resize(r_im, (int(r_im.shape[1]*h/r_im.shape[0]), h))
                    
                    cv2.imshow("Decoupled 3D", np.hstack([l_im, r_im]))

            if cv2.waitKey(1) & 0xFF == 27:
                break

except KeyboardInterrupt:
    pass
finally:
    for t in threads: t.stop()
    log_queue.put(None)
    log_thread.join()
    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    print("\nClean exit.")