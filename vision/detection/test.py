#!/usr/bin/env python3
# triangulation_fixed.py
import cv2, zmq, numpy as np, time, threading, queue, traceback, sys, os, csv
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
MAX_TIME_DIFF = 0.05  # 50ms max difference between camera frames to allow triangulation

# ---------- Async CSV Logging ----------
log_filename = f"data_log_{int(time.time())}.csv"
log_queue = queue.Queue()

def logger_worker():
    """Background thread to handle file writes without blocking the camera loop."""
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "timestamp", 
            "ball_x", "ball_y", "ball_z", 
            "tag4_x", "tag4_y", "tag4_z", 
            "tag5_x", "tag5_y", "tag5_z"
        ]
        writer.writerow(header)
        print(f"[Logging] Started logging to {log_filename}")
        
        while True:
            row = log_queue.get()
            if row is None: break
            writer.writerow(row)
            # f.flush() # Optional: Un-comment if you need real-time safety at cost of speed

log_thread = threading.Thread(target=logger_worker, daemon=True)
log_thread.start()

# ---------------- APRILTAG CONFIG ----------------
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11
def create_april_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    params = cv2.aruco.DetectorParameters()
    # Tuned params from your script
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
    if mask is None: return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA: continue
        perim = cv2.arcLength(c, True)
        if perim <= 0: continue
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
        print(f"[{self.cam_name}] AprilTagThread started")
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
                    # Try to compute extrinsic immediately if we see a known tag
                    self.calibrator.try_compute_extrinsic(self.cam_name)

                    # (Existing robot tag logic maintained...)
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
                                    
                                    tag_origin_world = self.calibrator.cam_to_world(self.cam_name, t_tag_cam)
                                    
                                    with self.lock:
                                        self.shared_robot_poses[tid] = {
                                            'world_pos': tag_origin_world,
                                            'cam': self.cam_name,
                                            'ts': ts
                                        }
                        except Exception as e:
                            traceback.print_exc()
            except Exception:
                traceback.print_exc()

    def stop(self): self.stop_flag = True

# BALL THREAD
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
            try: frame,ts=self.frame_queue.get(timeout=0.1)
            except queue.Empty: continue
            try:
                mo=hsv_mask_from_vals(frame,orange_hsvVals)
                mp=hsv_mask_from_vals(frame,purple_hsvVals)
                mc = cv2.bitwise_or(mo,mp)
                mc = postprocess_mask(mc)
                cand = find_candidate_contours(mc)

                dets = []
                for c,area,(x,y,w,h) in cand[:MAX_DETECTIONS_PER_CAM]:
                    M = cv2.moments(c)
                    if M["m00"] != 0: cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    else: cx = x + w//2; cy = y + h//2
                    
                    s_or = int(np.count_nonzero(mo[y:y+h,x:x+w])) if mo is not None else 0
                    s_pu = int(np.count_nonzero(mp[y:y+h,x:x+w])) if mp is not None else 0

                    if s_or>s_pu and s_or>0: col="orange"
                    elif s_pu>s_or and s_pu>0: col="purple"
                    else: col="unknown"

                    dets.append({
                        "bbox":(x,y,w,h),
                        "centroid":(cx,cy),
                        "area":float(area),
                        "color":col,
                        "ts":ts
                    })
                with self.lock:
                    if self.cam_name not in self.detect_cache: self.detect_cache[self.cam_name]={}
                    self.detect_cache[self.cam_name]["balls"] = dets
                    self.shared_camera_candidates[self.cam_name] = dets
            except Exception:
                traceback.print_exc()
    def stop(self): self.stop_flag = True


# --- IMPROVED CALIBRATOR ---
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

        # FIX 1: Removed hardcoded tag IDs. Use ANY visible tag defined in world map.
        obs_list = list(reversed(self.obs.get(cam_name, [])))
        
        target_tag = None
        use_corners = None

        for (tid, corners, ts) in obs_list:
            if tid in self.tag_world_map:
                target_tag = tid
                use_corners = corners.reshape(4,2).astype(np.float64)
                break # Use the most recent valid tag
        
        if use_corners is None: return False # No known tags seen

        try: K, dist = self.load_intrinsics(cam_name)
        except: return False

        obj_corners = np.array(self.tag_world_map[target_tag], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_corners, use_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return False

        R, _ = cv2.Rodrigues(rvec)
        tvec = tvec.reshape(3,1)
        self.extrinsics[cam_name] = {"rvec": rvec, "tvec": tvec, "R": R}
        print(f"[Calib] {cam_name} extrinsics computed using Tag {target_tag}")
        return True

    def cam_to_world(self, cam_name, X_cam):
        e = self.extrinsics.get(cam_name)
        if e is None: raise RuntimeError("Calibrator: extrinsic not ready for " + cam_name)
        R = e['R']; t = e['tvec']
        X = np.asarray(X_cam, dtype=np.float64)
        if X.ndim == 1: Xc = X.reshape(3,1); Xw = R.T @ (Xc - t); return Xw[:,0]
        else: Xc = X.T; Xw = R.T @ (Xc - t); return Xw.T

    def get_norm_projection_matrix(self, cam_name):
        # Returns [R|t] (3x4)
        e = self.extrinsics.get(cam_name)
        if e is None: return None
        return np.hstack((e['R'], e['tvec']))


# --- IMPROVED TRIANGULATION ---
def match_and_triangulate(camera_candidates, calibrator, reproj_thresh_px=10.0):
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
            # FIX 3: Check Color and Timestamp
            if a['color'] != 'unknown' and b['color'] != 'unknown':
                if a['color'] != b['color']: continue
            
            dt = abs(a['ts'] - b['ts'])
            if dt > MAX_TIME_DIFF: continue

            # FIX 2: Undistort Points before Triangulation
            pt1 = np.array(a['centroid'], dtype=float).reshape(1,1,2)
            pt2 = np.array(b['centroid'], dtype=float).reshape(1,1,2)
            
            # Undistort to Normalized Coordinates (x, y) where z=1
            pt1_norm = cv2.undistortPoints(pt1, K1, D1, P=None)
            pt2_norm = cv2.undistortPoints(pt2, K2, D2, P=None)

            # Triangulate in World Frame using Normalized Projection Matrices
            Xh = cv2.triangulatePoints(P1_norm, P2_norm, pt1_norm, pt2_norm)
            Xw = (Xh[:3] / Xh[3]).flatten()

            # Reprojection Check (Project back to distorted pixel space to compare with original detection)
            # 1. World -> Cam
            X_cam1 = calibrator.extrinsics[c1]['R'] @ Xw.reshape(3,1) + calibrator.extrinsics[c1]['tvec']
            X_cam2 = calibrator.extrinsics[c2]['R'] @ Xw.reshape(3,1) + calibrator.extrinsics[c2]['tvec']

            # 2. Project (using K and Dist)
            img_pt1, _ = cv2.projectPoints(Xw.reshape(1,3), calibrator.extrinsics[c1]['rvec'], calibrator.extrinsics[c1]['tvec'], K1, D1)
            img_pt2, _ = cv2.projectPoints(Xw.reshape(1,3), calibrator.extrinsics[c2]['rvec'], calibrator.extrinsics[c2]['tvec'], K2, D2)

            err1 = np.linalg.norm(img_pt1.flatten() - np.array(a['centroid']))
            err2 = np.linalg.norm(img_pt2.flatten() - np.array(b['centroid']))
            tot_err = err1 + err2

            if tot_err < reproj_thresh_px:
                results.append({
                    'pt': Xw, 
                    'reproj_err': tot_err, 
                    'pair': (c1,c2), 
                    'depths': (float(X_cam1[2]), float(X_cam2[2])),
                    'ts': max(a['ts'], b['ts']) # Use latest ts
                })

    results.sort(key=lambda r: r['reproj_err'])
    return results

# ---------- ZMQ subscriber ----------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(ZMQ_ADDR)
sub.setsockopt(zmq.RCVHWM, 1)
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt(zmq.LINGER, 0)

# Active flush
flushed = 0
while True:
    try: sub.recv_multipart(flags=zmq.NOBLOCK); flushed += 1
    except zmq.Again: break
if flushed > 0: print(f"[Subscriber] Flushed {flushed} stale messages.")
for t in SUB_TOPICS: sub.setsockopt(zmq.SUBSCRIBE, t)

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

for t in SUB_TOPICS:
    cam_name = t.decode()
    tag_threads[cam_name] = AprilTagThread(cam_name,frame_queues[cam_name],detect_cache,detect_lock,calibrator,shared_robot_poses)
    ball_threads[cam_name] = BallThread(cam_name,frame_queues[cam_name],detect_cache,detect_lock,shared_camera_candidates)
    tag_threads[cam_name].start()
    ball_threads[cam_name].start()

print("[Subscriber] connected.")

last_show = time.time()

try:
    while True:
        parts=recv_latest(sub)
        if parts is None: continue

        topic=parts[0]; cam=topic.decode()
        jpg_part=parts[2] if len(parts)>=3 else parts[1]
        ts_part=parts[1] if len(parts)>=3 else None

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

        # Check calibration ready
        with detect_lock:
            ready = all(c in calibrator.extrinsics for c in [t.decode() for t in SUB_TOPICS])

        if ready:
            with detect_lock:
                # Using the NEW robust triangulation function
                tri_results = match_and_triangulate(shared_camera_candidates, calibrator)
            
            if len(tri_results) > 0:
                best = tri_results[0]
                Xw = best['pt']
                last_triangulated_ball = {'pos': Xw, 'err': best['reproj_err'], 'ts': best['ts'], 'pair': best['pair'], 'depths':best['depths']}

            # Logging (Now Async)
            current_time = time.time()
            b_x, b_y, b_z = "", "", ""
            if last_triangulated_ball and (current_time - last_triangulated_ball['ts'] < 0.5):
                pos = last_triangulated_ball['pos']
                b_x, b_y, b_z = pos[0], pos[1], pos[2]
            
            t4_x, t4_y, t4_z = "", "", ""
            if 4 in shared_robot_poses:
                pos4 = shared_robot_poses[4]['world_pos']
                t4_x, t4_y, t4_z = pos4[0], pos4[1], pos4[2]
            
            t5_x, t5_y, t5_z = "", "", ""
            if 5 in shared_robot_poses:
                pos5 = shared_robot_poses[5]['world_pos']
                t5_x, t5_y, t5_z = pos5[0], pos5[1], pos5[2]

            log_queue.put([current_time, b_x, b_y, b_z, t4_x, t4_y, t4_z, t5_x, t5_y, t5_z])

        # Visualization (Same as before)
        if all(k in frames for k in [t.decode() for t in SUB_TOPICS]):
            if VISUALIZE and (time.time()-last_show)>(1.0/DISPLAY_FPS):
                last_show=time.time()
                cams=[t.decode() for t in SUB_TOPICS]
                L=frames[cams[0]]; R=frames[cams[1]]
                
                # Simple overlay logic
                # ... (omitted for brevity, assume same overlay function as original)
                # (You can copy the overlay function block from your original script here if needed)
                
        if cv2.waitKey(1)&0xFF==27: break

except KeyboardInterrupt: pass
finally:
    log_queue.put(None)
    log_thread.join()
    for t in tag_threads.values(): t.stop()
    for t in ball_threads.values(): t.stop()
    time.sleep(0.1)
    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    print("Exit clean.")