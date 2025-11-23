import cv2, zmq, numpy as np, time, threading, queue, traceback, os, csv
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]

FPS_WINDOW = 1.0        
DISPLAY_FPS = 30        
VISUALIZE = True        

# OPTIMIZATION: Downscale frame for BOTH Ball and Tag detection
# 0.5 scale = 360p. This is 4x faster than 720p.
DETECTION_SCALE = 0.5 

LOG_DIR = "../../data/detection_logs"
LOG_FILENAME = f"{LOG_DIR}/log_{int(time.time())}.csv"

# HSV Config
HSV_CONFIG = {
    "kreo1": {
        "orange": {'hmin': 0, 'smin': 120, 'vmin': 175, 'hmax': 12, 'smax': 255, 'vmax': 255},
    },
    "kreo2": {
        "orange": {'hmin': 0, 'smin': 150, 'vmin': 150, 'hmax': 12, 'smax': 255, 'vmax': 255}
    }
}
DEFAULT_HSV = {'hmin': 0, 'smin': 100, 'vmin': 100, 'hmax': 25, 'smax': 255, 'vmax': 255}

# Detector parameters
BASE_MIN_AREA = 100    
BASE_MAX_AREA = 20000  
CIRCULARITY_MIN = 0.5
ASPECT_RATIO_MIN = 0.6 
ASPECT_RATIO_MAX = 1.6   
MAX_DETECTIONS_PER_CAM = 5 

# AprilTag Config
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11

def create_april_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    params = cv2.aruco.DetectorParameters()
    
    # TUNING FOR SPEED:
    # Reduce window size and increase step size to check fewer pixels
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23  # Reduced from 35
    params.adaptiveThreshWinSizeStep = 10 # Increased from 2 (checks fewer threshold levels)
    
    # Faster corner refinement (SUBPIX is very slow, APRILTAG is balanced)
    # If you need raw speed, comment out cornerRefinementMethod lines entirely
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG 
    
    return cv2.aruco.ArucoDetector(aruco_dict, params)

# ---------- Logging Setup ----------
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_queue = queue.Queue()

def logger_worker():
    """
    Logs: ts, cam, ball_found, ball_x, ball_y, tag4_found, tag4_x, tag4_y, tag5_found, tag5_x, tag5_y
    """
    try:
        with open(LOG_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "timestamp", "camera", 
                "ball_detected", "ball_x", "ball_y", "ball_area",
                "tag4_detected", "tag4_x", "tag4_y",
                "tag5_detected", "tag5_x", "tag5_y"
            ])
            while True:
                entry = log_queue.get()
                if entry is None: break
                writer.writerow(entry)
                log_queue.task_done()
    except Exception as e: print(f"[LOGGER ERROR] {e}")

log_thread = threading.Thread(target=logger_worker, daemon=True)
log_thread.start()

# ---------- Helper Functions ----------
def update_fps(camera, cam_ts, fps_windows):
    dq = fps_windows[camera]
    dq.append(cam_ts)
    while dq and (cam_ts - dq[0]) > FPS_WINDOW: dq.popleft()
    return len(dq) / FPS_WINDOW

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
        if ASPECT_RATIO_MIN>aspect or aspect > ASPECT_RATIO_MAX:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity >= CIRCULARITY_MIN:
            candidates.append({"bbox": (x,y,w,h), "area": area, "centroid": (x+w//2, y+h//2)})
    # Return largest ball first
    candidates.sort(key=lambda d: d["area"], reverse=True)
    return candidates

# ---------- Unified Detector Thread ----------
class DetectorThread(threading.Thread):
    def __init__(self, cam_name, frame_queue, detect_cache, lock):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.detect_cache = detect_cache
        self.lock = lock
        self.stop_flag = False
        
        # Ball Setup
        self.min_area_scaled = BASE_MIN_AREA * (DETECTION_SCALE**2)
        self.max_area_scaled = BASE_MAX_AREA * (DETECTION_SCALE**2)
        self.hsv_vals = HSV_CONFIG.get(cam_name, {}).get("orange", DEFAULT_HSV)
        
        # Tag Setup
        self.aruco_detector = create_april_detector()
        self.scale_inv = 1.0 / DETECTION_SCALE

    def run(self):
        print(f"[{self.cam_name}] Unified Detector Thread started.")
        while not self.stop_flag:
            try:
                # 1. Get Raw Data
                jpg_bytes, cam_ts = self.frame_queue.get(timeout=0.1)
                
                # 2. Decode
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue

                # 3. Create SCALED Image for BOTH detections (Massive speedup)
                if DETECTION_SCALE != 1.0:
                    frame_small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE, interpolation=cv2.INTER_NEAREST)
                else:
                    frame_small = frame
                
                # --- PART A: BALL DETECTION (Scaled) ---
                mask = get_orange_mask(frame_small, self.hsv_vals)
                balls = find_ball_contours(mask, self.min_area_scaled, self.max_area_scaled)
                
                ball_data = {"detected": False, "x": "", "y": "", "area": ""}
                best_ball = None 
                
                if balls:
                    b = balls[0] 
                    # Scale back to original resolution
                    bx, by, bw, bh = b["bbox"]
                    real_x = int(bx * self.scale_inv)
                    real_y = int(by * self.scale_inv)
                    real_w = int(bw * self.scale_inv)
                    real_h = int(bh * self.scale_inv)
                    real_area = int(b["area"] * (self.scale_inv**2))
                    cx, cy = real_x + real_w//2, real_y + real_h//2
                    
                    ball_data = {"detected": True, "x": cx, "y": cy, "area": real_area}
                    best_ball = {"bbox": (real_x, real_y, real_w, real_h), "centroid": (cx, cy)}

                # --- PART B: APRILTAG DETECTION (Scaled) ---
                # CRITICAL FIX: Detect on frame_small, not frame
                gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = self.aruco_detector.detectMarkers(gray_small)
                
                tag4_data = {"detected": False, "x": "", "y": ""}
                tag5_data = {"detected": False, "x": "", "y": ""}
                
                found_tags_viz = [] 
                
                if ids is not None:
                    ids_flat = ids.flatten()
                    for i, tag_id in enumerate(ids_flat):
                        # Get center on SMALL image
                        c_small = corners[i][0]
                        cx_small = np.mean(c_small[:, 0])
                        cy_small = np.mean(c_small[:, 1])
                        
                        # Scale back to ORIGINAL resolution
                        cx = int(cx_small * self.scale_inv)
                        cy = int(cy_small * self.scale_inv)
                        
                        # Scale corners for visualization
                        viz_corners = corners[i] * self.scale_inv
                        found_tags_viz.append({"id": tag_id, "corners": viz_corners})

                        if tag_id == 4:
                            tag4_data = {"detected": True, "x": cx, "y": cy}
                        elif tag_id == 5:
                            tag5_data = {"detected": True, "x": cx, "y": cy}

                # --- PART C: LOGGING ---
                log_row = [
                    f"{cam_ts:.3f}", self.cam_name,
                    ball_data["detected"], ball_data["x"], ball_data["y"], ball_data["area"],
                    tag4_data["detected"], tag4_data["x"], tag4_data["y"],
                    tag5_data["detected"], tag5_data["x"], tag5_data["y"]
                ]
                log_queue.put(log_row)

                # --- PART D: UPDATE CACHE (For Viz) ---
                with self.lock:
                    self.detect_cache[self.cam_name] = {
                        "ball": best_ball,
                        "tags": found_tags_viz,
                        "img": frame, 
                        "fps": 0 
                    }

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[ERROR-{self.cam_name}]", e)
                traceback.print_exc()

    def stop(self):
        self.stop_flag = True


# ---------- Main Execution ----------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(ZMQ_ADDR)
sub.setsockopt(zmq.RCVHWM, 4)
sub.setsockopt(zmq.CONFLATE, 1) 
sub.setsockopt(zmq.LINGER, 0)
for t in SUB_TOPICS: sub.setsockopt(zmq.SUBSCRIBE, t)

# Flush init
while True:
    try:
        sub.recv_multipart(flags=zmq.NOBLOCK)
    except zmq.Again:
        break

frame_queues = {t.decode(): queue.Queue(maxsize=1) for t in SUB_TOPICS}
detect_cache = {}        
detect_lock = threading.Lock()
detectors = {}
fps_state = defaultdict(lambda: deque())

# Start Threads
for t in SUB_TOPICS:
    cam_name = t.decode()
    dt = DetectorThread(cam_name, frame_queues[cam_name], detect_cache, detect_lock)
    dt.start()
    detectors[cam_name] = dt

print(f"[Subscriber] Connected to {ZMQ_ADDR}")
print(f"[Logging] Saving to {LOG_FILENAME}")

last_show = time.time()

try:
    while True:
        # ZMQ Receiver Loop
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
            topic = parts[0]
            cam = topic.decode()
            
            ts_part = parts[1] if len(parts) >= 3 else None
            jpg_part = parts[2] if len(parts) >= 3 else parts[1]
            try:
                cam_ts = float(ts_part.decode()) if ts_part else time.time()
            except:
                cam_ts = time.time()

            # Pass raw bytes to thread
            fq = frame_queues.get(cam)
            if fq:
                try:
                    fq.put_nowait((jpg_part, cam_ts))
                except queue.Full:
                    # Drop old frame to keep current
                    try:
                        fq.get_nowait()
                        fq.put_nowait((jpg_part, cam_ts))
                    except: pass
            
            # FPS tracking for display
            cur_fps = update_fps(cam, cam_ts, fps_state)
            # Inject FPS into cache for Viz to see
            with detect_lock:
                if cam in detect_cache:
                    detect_cache[cam]["fps"] = cur_fps

        except zmq.Again:
            time.sleep(0.001)

        # Visualization
        if VISUALIZE and (time.time() - last_show) > (1.0/DISPLAY_FPS):
            with detect_lock:
                # Check if we have data for both cams
                has_data = all(c in detect_cache and "img" in detect_cache[c] for c in ["kreo1", "kreo2"])
                
                if has_data:
                    def draw_overlay(cam_key):
                        data = detect_cache[cam_key]
                        im = data["img"].copy()
                        fps = data.get("fps", 0)
                        
                        # Info
                        cv2.putText(im, f"{cam_key} FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        
                        # Draw Ball
                        if data.get("ball"):
                            bx, by, bw, bh = data["ball"]["bbox"]
                            cx, cy = data["ball"]["centroid"]
                            cv2.rectangle(im, (bx,by), (bx+bw, by+bh), (0,165,255), 2)
                            cv2.circle(im, (cx,cy), 5, (0,0,255), -1)
                        
                        # Draw Tags
                        if data.get("tags"):
                            for tag in data["tags"]:
                                cv2.aruco.drawDetectedMarkers(im, [tag["corners"].astype(int)], np.array([[tag["id"]]]))
                        
                        return im

                    l_im = draw_overlay("kreo1")
                    r_im = draw_overlay("kreo2")

                    # Stack
                    h = min(l_im.shape[0], r_im.shape[0])
                    if l_im.shape[0] != h: l_im = cv2.resize(l_im, (int(l_im.shape[1]*h/l_im.shape[0]), h))
                    if r_im.shape[0] != h: r_im = cv2.resize(r_im, (int(r_im.shape[1]*h/r_im.shape[0]), h))
                    
                    tile = np.hstack([l_im, r_im])
                    cv2.imshow("Stereo Tracking", tile)
                    last_show = time.time()
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

except KeyboardInterrupt:
    pass
finally:
    for d in detectors.values(): d.stop()
    log_queue.put(None)
    log_thread.join()
    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    print("\nExiting.")