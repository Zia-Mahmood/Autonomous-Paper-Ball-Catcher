import cv2
import zmq
import numpy as np
import time
import threading
import queue
import traceback
import sys
import os
import csv
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]

FPS_WINDOW = 1.0        
DISPLAY_FPS = 30        
VISUALIZE = True        

# SCALING CONFIG
BALL_DETECTION_SCALE = 0.5  # Downscale for speed (Ball)
TAG_DETECTION_SCALE = 1.0   # Full res for accuracy (Tags)

LOG_DIR = "../../data/detection_logs"
LOG_FILENAME = f"{LOG_DIR}/log_{int(time.time())}.csv"

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

# ---------- Logging Setup ----------
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_queue = queue.Queue()

def logger_worker():
    try:
        with open(LOG_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
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

# ---------- Shared Data (Thread Safe) ----------
# Stores the latest tag detection result for each camera
shared_tag_data = {
    "kreo1": {"tag4": {"detected": False, "x":"", "y":""}, "tag5": {"detected": False, "x":"", "y":""}, "viz": []},
    "kreo2": {"tag4": {"detected": False, "x":"", "y":""}, "tag5": {"detected": False, "x":"", "y":""}, "viz": []}
}
shared_data_lock = threading.Lock()

# Stores the latest image/ball for visualization
viz_cache = {}
viz_lock = threading.Lock()

# ---------- Helpers ----------
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
                
                tag4 = {"detected": False, "x": "", "y": ""}
                tag5 = {"detected": False, "x": "", "y": ""}
                viz_tags = []

                if ids is not None:
                    ids_flat = ids.flatten()
                    for i, tag_id in enumerate(ids_flat):
                        c = corners[i][0]
                        cx = int(np.mean(c[:, 0]))
                        cy = int(np.mean(c[:, 1]))
                        
                        viz_tags.append({"id": tag_id, "corners": corners[i]})

                        if tag_id == 4:
                            tag4 = {"detected": True, "x": cx, "y": cy}
                        elif tag_id == 5:
                            tag5 = {"detected": True, "x": cx, "y": cy}
                
                # Update Shared Memory
                with shared_data_lock:
                    shared_tag_data[self.cam_name] = {
                        "tag4": tag4,
                        "tag5": tag5,
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

                # Read latest Tag Data (Non-blocking read)
                with shared_data_lock:
                    tags = shared_tag_data[self.cam_name]
                
                # LOGGING: Driven by the fast ball thread
                log_queue.put([
                    f"{cam_ts:.3f}", self.cam_name,
                    ball_data["detected"], ball_data["x"], ball_data["y"], ball_data["area"],
                    tags["tag4"]["detected"], tags["tag4"]["x"], tags["tag4"]["y"],
                    tags["tag5"]["detected"], tags["tag5"]["x"], tags["tag5"]["y"]
                ])

                # FPS Calc
                self.fps_dq.append(time.time())
                while self.fps_dq and (self.fps_dq[-1] - self.fps_dq[0]) > FPS_WINDOW:
                    self.fps_dq.popleft()
                fps = len(self.fps_dq) / FPS_WINDOW

                # Update Viz Cache
                if VISUALIZE:
                    with viz_lock:
                        viz_cache[self.cam_name] = {
                            "img": frame,
                            "ball": viz_ball,
                            "tags": tags["viz"],
                            "fps": fps
                        }

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[BALL-ERR-{self.cam_name}]", e)
    
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
            try:
                cam_ts = float(ts_part.decode()) if ts_part else time.time()
            except:
                cam_ts = time.time()
            
            # Fan-out to both queues (LIFO logic)
            qs = queues.get(cam)
            if qs:
                # Push to Ball Queue
                try:
                    qs["ball"].put_nowait((jpg_part, cam_ts))
                except queue.Full:
                    try: qs["ball"].get_nowait(); qs["ball"].put_nowait((jpg_part, cam_ts))
                    except: pass
                
                # Push to Tag Queue
                try:
                    qs["tag"].put_nowait((jpg_part, cam_ts))
                except queue.Full:
                    try: qs["tag"].get_nowait(); qs["tag"].put_nowait((jpg_part, cam_ts))
                    except: pass

        except zmq.Again:
            time.sleep(0.0001)

        # Viz Loop
        if VISUALIZE and (time.time() - last_show) > (1.0/DISPLAY_FPS):
            with viz_lock:
                has_data = all(c in viz_cache for c in ["kreo1", "kreo2"])
                if has_data:
                    def draw(cam_key):
                        d = viz_cache[cam_key]
                        im = d["img"].copy()
                        
                        # Ball
                        if d["ball"]:
                            bx,by,bw,bh = d["ball"]["bbox"]
                            cx,cy = d["ball"]["centroid"]
                            cv2.rectangle(im, (bx,by), (bx+bw, by+bh), (0,165,255), 2)
                            cv2.circle(im, (cx,cy), 5, (0,0,255), -1)
                        
                        # Tags
                        for tag in d["tags"]:
                             cv2.aruco.drawDetectedMarkers(im, [tag["corners"].astype(int)], np.array([[tag["id"]]]))
                        
                        cv2.putText(im, f"FPS: {d['fps']:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        return im

                    l_im = draw("kreo1")
                    r_im = draw("kreo2")
                    
                    # Stack
                    h = min(l_im.shape[0], r_im.shape[0])
                    if l_im.shape[0] != h: l_im = cv2.resize(l_im, (int(l_im.shape[1]*h/l_im.shape[0]), h))
                    if r_im.shape[0] != h: r_im = cv2.resize(r_im, (int(r_im.shape[1]*h/r_im.shape[0]), h))
                    
                    cv2.imshow("Decoupled Tracking", np.hstack([l_im, r_im]))
                    last_show = time.time()

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