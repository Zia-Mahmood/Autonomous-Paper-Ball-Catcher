#!/usr/bin/env python3
# triangulation.py (Fixed - Stationary Detection + Custom HSV + Optimized)
import cv2, zmq, numpy as np, time, threading, queue, traceback, sys, os, csv, copy
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]
FPS_WINDOW = 1.0        
DISPLAY_FPS = 20
VISUALIZE = True     

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

# Tuning for Matching
MAX_TIME_DIFF = 0.08  # Slightly increased to 80ms to catch more pairs if cameras drift

# --- USER TUNED HSV VALUES ---
HSV_CONFIG = {
    "kreo1": {
        "orange": {'hmin': 0, 'smin': 100, 'vmin': 165, 'hmax': 13, 'smax': 255, 'vmax': 255},
        # Keep purple as backup or ignore if not used
        "purple": {'hmin': 113,'smin': 78,  'vmin': 3,   'hmax': 129, 'smax': 255, 'vmax': 255}
    },
    "kreo2": {
        "orange": {'hmin': 0, 'smin': 100, 'vmin': 200, 'hmax': 13, 'smax': 255, 'vmax': 255},
        "purple": {'hmin': 113,'smin': 78,  'vmin': 3,   'hmax': 129, 'smax': 255, 'vmax': 255}
    }
}

# Contour Filtering (Tuned for Ball vs Arm)
MIN_AREA = 20       # Reduced to catch ball far away
MAX_AREA = 8000
CIRCULARITY_MIN = 0.5  # Strict circle check to reject Arm/Hand
ASPECT_RATIO_MIN = 0.6
ASPECT_RATIO_MAX = 1.6
MAX_DETECTIONS_PER_CAM = 10

# ---------- Async CSV Logging ----------
log_filename = f"data_log_{int(time.time())}.csv"
log_queue = queue.Queue()

def logger_worker():
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["timestamp", "ball_x", "ball_y", "ball_z", "tag4_x", "tag4_y", "tag4_z", "tag5_x", "tag5_y", "tag5_z"]
        writer.writerow(header)
        while True:
            row = log_queue.get()
            if row is None: break
            writer.writerow(row)

log_thread = threading.Thread(target=logger_worker, daemon=True)
log_thread.start()

# ---------------- APRILTAG CONFIG ----------------
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11
def create_april_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    params = cv2.aruco.DetectorParameters()
    # Standard robust parameters
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 2
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)

# ---------- Helpers ----------
def fmt_ts(ts): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) + f".{int((ts%1)*1000):03d}"

def recv_latest(sub):
    msg = None
    while True:
        try: msg = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again: break
    return msg

def update_fps(camera, cam_ts):
    dq = fps_windows[camera]; dq.append(cam_ts)
    while dq and (cam_ts - dq[0]) > FPS_WINDOW: dq.popleft()
    return len(dq) / FPS_WINDOW

def load_camera_calib(cam_name):
    path = os.path.join(CALIB_DIR, f'camera_calibration_{cam_name}.npz')
    if not os.path.exists(path): raise FileNotFoundError(path)
    calib = np.load(path)
    return calib["cameraMatrix"], calib['distCoeffs']

def build_tag_world_map_from_centers(tag_centers, tag_sizes):
    out = {}
    for tid, center in tag_centers.items():
        size = tag_sizes.get(tid, tag_sizes.get(1))
        half = float(size) / 2.0
        local = np.array([[-half,  half, 0.0], [ half,  half, 0.0], [ half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)
        corners_world = (local + center.reshape(1,3)).astype(np.float64)
        out[tid] = corners_world
    return out

TAG_WORLD_MAP = build_tag_world_map_from_centers(TAG_POSITIONS, TAG_SIZES)

# ---------- Color masking helpers ----------
def hsv_mask_from_vals(hsv_img, hsvVals):
    lower = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']], dtype=np.uint8)
    upper = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower, upper)
    return mask

def postprocess_mask(mask):
    # Open to remove noise (white dots), Close to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    return m

def find_candidate_contours(mask):
    if mask is None: return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA: continue
        
        # Geometric filters to distinguish Ball vs Arm
        x,y,w,h = cv2.boundingRect(c)
        aspect = float(w)/float(h) if h>0 else 0.0
        
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        circularity = 4 * np.pi * area / (perim * perim)

        # Ball = High Circularity (~0.7-0.9), Arm = Low Circularity
        if circularity < CIRCULARITY_MIN: continue 
        
        # Ball = Square-ish aspect ratio
        if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX: continue
        
        candidates.append((c, area, (int(x),int(y),int(w),int(h))))
        
    candidates.sort(key=lambda d: d[1], reverse=True)
    return candidates

def estimate_pose_apriltag(corners, tag_size, cam_mtx, cam_dist):
    half = tag_size / 2.0
    objp = np.array([[-half,  half, 0.0], [ half,  half, 0.0], [ half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)
    imgp = corners.reshape(4,2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, cam_mtx, cam_dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.reshape(3)
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
        while not self.stop_flag:
            try: frame,ts = self.frame_queue.get(timeout=0.1)
            except queue.Empty: continue
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)
                det={"corners":corners,"ids":ids,"ts":ts,"det_time":time.time()}

                with self.lock:
                    if self.cam_name not in self.detect_cache: self.detect_cache[self.cam_name]={}
                    self.detect_cache[self.cam_name]["tags"]=det

                if ids is not None:
                    self.calibrator.add_detection(self.cam_name,ids,corners,ts)
                    self.calibrator.try_compute_extrinsic(self.cam_name)
                    if self.cam_name in self.calibrator.extrinsics:
                        K, dist = self.calibrator.load_intrinsics(self.cam_name)
                        for i,idarr in enumerate(ids):
                            tid = int(idarr[0])
                            if tid in (4,5):
                                corners_i = np.array(corners[i]).reshape(4,2)
                                T_tag_cam = estimate_pose_apriltag(corners_i, TAG_SIZES[tid], K, dist)
                                tag_origin_world = self.calibrator.cam_to_world(self.cam_name, T_tag_cam[:3,3].reshape(3,))
                                with self.lock:
                                    self.shared_robot_poses[tid] = {'world_pos': tag_origin_world, 'cam': self.cam_name, 'ts': ts}
            except Exception: traceback.print_exc()

    def stop(self): self.stop_flag = True

# BALL THREAD - FIXED: REMOVED BACKGROUND SUBTRACTION
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
        print(f"[{self.cam_name}] BallThread started (HSV Only)")
        
        # Get camera specific HSV config
        cam_hsv = HSV_CONFIG.get(self.cam_name, HSV_CONFIG.get("kreo1")) 
        
        while not self.stop_flag:
            try: frame,ts=self.frame_queue.get(timeout=0.1)
            except queue.Empty: continue
            try:
                # Direct HSV Masking (No Motion Requirement)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                mask_orange = hsv_mask_from_vals(hsv, cam_hsv["orange"])
                mask_orange = postprocess_mask(mask_orange)
                
                # mask_purple = hsv_mask_from_vals(hsv, cam_hsv["purple"]) # Disable purple if not used

                cand_o = find_candidate_contours(mask_orange)
                
                dets = []
                for c, area, (x,y,w,h) in cand_o[:MAX_DETECTIONS_PER_CAM]:
                     M = cv2.moments(c)
                     if M["m00"] != 0: cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                     else: cx = x + w//2; cy = y + h//2
                     dets.append({
                        "bbox":(x,y,w,h), "centroid":(cx,cy), "area":float(area),
                        "color":"orange", "ts":ts
                     })

                with self.lock:
                    if self.cam_name not in self.detect_cache: self.detect_cache[self.cam_name]={}
                    self.detect_cache[self.cam_name]["balls"] = dets
                    self.shared_camera_candidates[self.cam_name] = dets
                    
            except Exception: traceback.print_exc()
    def stop(self): self.stop_flag = True

# --- CALIBRATOR ---
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
        target_tag = None; use_corners = None
        for (tid, corners, ts) in obs_list:
            if tid in self.tag_world_map:
                target_tag = tid; use_corners = corners.reshape(4,2).astype(np.float64); break
        if use_corners is None: return False
        try: K, dist = self.load_intrinsics(cam_name)
        except: return False
        obj_corners = np.array(self.tag_world_map[target_tag], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_corners, use_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return False
        R, _ = cv2.Rodrigues(rvec)
        self.extrinsics[cam_name] = {"rvec": rvec, "tvec": tvec.reshape(3,1), "R": R}
        print(f"[Calib] {cam_name} extrinsics computed using Tag {target_tag}")
        return True
    def cam_to_world(self, cam_name, X_cam):
        e = self.extrinsics.get(cam_name)
        if e is None: raise RuntimeError("Calibrator: extrinsic not ready for " + cam_name)
        R = e['R']; t = e['tvec']
        X = np.asarray(X_cam, dtype=np.float64)
        if X.ndim == 1: return (R.T @ (X.reshape(3,1) - t))[:,0]
        return (R.T @ (X.T - t)).T
    def get_norm_projection_matrix(self, cam_name):
        e = self.extrinsics.get(cam_name)
        return np.hstack((e['R'], e['tvec'])) if e else None

def match_and_triangulate(camera_candidates, calibrator, reproj_thresh_px=15.0):
    cams = list(camera_candidates.keys())
    if len(cams) < 2: return []
    pair = ('kreo1', 'kreo2') if 'kreo1' in cams and 'kreo2' in cams else (cams[0], cams[1])
    c1, c2 = pair
    cand1 = sorted(camera_candidates[c1], key=lambda x: -x.get('area',1))[:8]
    cand2 = sorted(camera_candidates[c2], key=lambda x: -x.get('area',1))[:8]
    
    if c1 not in calibrator.extrinsics or c2 not in calibrator.extrinsics: return []
    K1, D1 = calibrator.load_intrinsics(c1)
    K2, D2 = calibrator.load_intrinsics(c2)
    P1_norm = calibrator.get_norm_projection_matrix(c1)
    P2_norm = calibrator.get_norm_projection_matrix(c2)
    
    results = []
    for a in cand1:
        for b in cand2:
            if a['color'] != b['color']: continue
            dt = abs(a['ts'] - b['ts'])
            if dt > MAX_TIME_DIFF: continue
            
            # Correct reshaping for undistortPoints: Input (N,1,2) -> Output (N,1,2)
            pt1_in = np.array(a['centroid'],dtype=float).reshape(-1,1,2)
            pt2_in = np.array(b['centroid'],dtype=float).reshape(-1,1,2)
            
            pt1_norm = cv2.undistortPoints(pt1_in, K1, D1, P=None)
            pt2_norm = cv2.undistortPoints(pt2_in, K2, D2, P=None)

            # TriangulatePoints needs (2, N)
            pt1_norm = pt1_norm.reshape(-1,2).T
            pt2_norm = pt2_norm.reshape(-1,2).T

            Xh = cv2.triangulatePoints(P1_norm, P2_norm, pt1_norm, pt2_norm)
            w = Xh[3]
            if abs(w) < 1e-6: continue
            Xw = (Xh[:3] / w).flatten()
            
            # Reprojection Check
            img_pt1, _ = cv2.projectPoints(Xw.reshape(1,3), calibrator.extrinsics[c1]['rvec'], calibrator.extrinsics[c1]['tvec'], K1, D1)
            img_pt2, _ = cv2.projectPoints(Xw.reshape(1,3), calibrator.extrinsics[c2]['rvec'], calibrator.extrinsics[c2]['tvec'], K2, D2)
            
            tot_err = np.linalg.norm(img_pt1.flatten()-np.array(a['centroid'])) + np.linalg.norm(img_pt2.flatten()-np.array(b['centroid']))
            
            if tot_err < reproj_thresh_px:
                results.append({'pt': Xw, 'reproj_err': tot_err, 'pair': (c1,c2), 'depths': (float(Xw[2]), float(Xw[2])), 'ts': max(a['ts'], b['ts'])})
    results.sort(key=lambda r: r['reproj_err'])
    return results

# ---------- Main Loop ----------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(ZMQ_ADDR)
sub.setsockopt(zmq.RCVHWM, 1); sub.setsockopt(zmq.CONFLATE, 1); sub.setsockopt(zmq.LINGER, 0)
for t in SUB_TOPICS: sub.setsockopt(zmq.SUBSCRIBE, t)

frames = {}; fps_windows = defaultdict(lambda: deque())
frame_queues = {t.decode(): queue.Queue(maxsize=1) for t in SUB_TOPICS}
detect_cache = {}; detect_lock = threading.Lock()
tag_threads={}; ball_threads={}

calibrator = StaticCalibrator(TAG_WORLD_MAP, TAG_SIZES)
shared_camera_candidates = defaultdict(list); shared_robot_poses = {}
last_triangulated_ball = None; last_ekf_ts = -1.0

for t in SUB_TOPICS:
    cam = t.decode()
    tag_threads[cam] = AprilTagThread(cam,frame_queues[cam],detect_cache,detect_lock,calibrator,shared_robot_poses)
    ball_threads[cam] = BallThread(cam,frame_queues[cam],detect_cache,detect_lock,shared_camera_candidates)
    tag_threads[cam].start(); ball_threads[cam].start()

print("[Main] Connected.")
last_show = time.time()

try:
    while True:
        try: parts = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again: continue
        
        topic=parts[0]; cam=topic.decode()
        jpg=parts[2] if len(parts)>=3 else parts[1]
        ts_part=parts[1] if len(parts)>=3 else None
        cam_ts = float(ts_part.decode()) if ts_part else time.time()
        
        img = cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        
        frames[cam]={"img":img, "cam_ts":cam_ts, "fps":update_fps(cam,cam_ts)}
        
        fq=frame_queues[cam]
        try: fq.get_nowait()
        except: pass
        try: fq.put_nowait((img.copy(),cam_ts))
        except: pass

        # Triangulation
        candidates_copy = None; calib_ready = False
        with detect_lock:
            if all(c in calibrator.extrinsics for c in [t.decode() for t in SUB_TOPICS]):
                calib_ready = True; candidates_copy = copy.deepcopy(shared_camera_candidates)
        
        if calib_ready and candidates_copy:
            tri = match_and_triangulate(candidates_copy, calibrator)
            if tri:
                best = tri[0]
                meas_ts = best['ts']
                # Always log high speed, let EKF filter downstream
                Xw = best['pt']
                last_triangulated_ball = {'pos': Xw, 'err': best['reproj_err'], 'ts': meas_ts}
                log_queue.put([meas_ts, Xw[0], Xw[1], Xw[2], "","","","","",""])
        
        # Visuals
        if VISUALIZE and (time.time()-last_show > 1.0/DISPLAY_FPS) and len(frames)>=2:
            last_show = time.time()
            cams = sorted(frames.keys())
            L = frames[cams[0]]; R = frames[cams[1]]
            
            # Simple Horizontal Stack
            imL = cv2.resize(L['img'], (640,360)); imR = cv2.resize(R['img'], (640,360))
            combo = np.hstack([imL, imR])
            cv2.putText(combo, f"Drift: {abs(L['cam_ts']-R['cam_ts'])*1000:.1f}ms", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Stereo", combo)
            if cv2.waitKey(1)==27: break

except KeyboardInterrupt: pass
finally:
    log_queue.put(None); log_thread.join()
    for t in tag_threads.values(): t.stop()
    for t in ball_threads.values(): t.stop()
    cv2.destroyAllWindows()