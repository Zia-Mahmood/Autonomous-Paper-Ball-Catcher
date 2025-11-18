#!/usr/bin/env python3
# subscriber.py
import cv2, zmq, numpy as np, time, threading, queue, traceback, sys
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]
FPS_WINDOW = 1.0        # seconds for fps moving window
DISPLAY_FPS = 20
VISUALIZE = True     # show tiled view window

# ---------------- APRILTAG CONFIG ----------------
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11
def create_april_detector():
    """Setup AprilTag detector with tuned parameters."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 2
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 100
    params.cornerRefinementMinAccuracy = 0.01
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 6.0
    params.polygonalApproxAccuracyRate = 0.02
    params.adaptiveThreshConstant = 7
    return cv2.aruco.ArucoDetector(aruco_dict, params)

# color thresholds (you gave these)
orange_hsvVals = {'hmin': 0,  'smin': 94,  'vmin': 156, 'hmax': 12,  'smax': 255, 'vmax': 255}
purple_hsvVals = {'hmin': 113,'smin': 78,  'vmin': 3,   'hmax': 129, 'smax': 255, 'vmax': 255}

# detector parameters
MIN_AREA = 100    # min contour area to accept (tune if needed)
MAX_AREA = 20000  # max area (avoid very large blobs)
CIRCULARITY_MIN = 0.25  # min circularity to accept (lower because paper balls can deform)
ASPECT_RATIO_MAX = 2.0   # reject extremely elongated blobs
MAX_DETECTIONS_PER_CAM = 12  # safety limit

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
    # pop older than window
    while dq and (cam_ts - dq[0]) > FPS_WINDOW:
        dq.popleft()
    fps = len(dq) / FPS_WINDOW
    return fps

# ---------- Color masking helpers ----------
def hsv_mask_from_vals(bgr_img, hsvVals):
    """Return binary mask from HSV thresholds dict."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']], dtype=np.uint8)
    upper = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # small blur to reduce speckle
    mask = cv2.medianBlur(mask, 5)
    return mask

def postprocess_mask(mask):
    """Morphological clean-up."""
    # open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m

def find_candidate_contours(mask):
    """Return list of contours filtered by area and shape heuristics."""
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
        # accept if roughly circular-ish or moderate area even if circularity low
        if circularity >= CIRCULARITY_MIN or (0.5*min(w,h) > 5 and area > (MIN_AREA*2)):
            if aspect <= ASPECT_RATIO_MAX:
                candidates.append((c, area, (int(x),int(y),int(w),int(h))))
    # sort by area desc
    candidates.sort(key=lambda d: d[1], reverse=True)
    return candidates


# APRILTAG DETECTOR THREAD

class AprilTagThread(threading.Thread):
    def __init__(self, cam_name, frame_queue, detect_cache, lock):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.detect_cache = detect_cache
        self.lock = lock
        self.detector = create_april_detector()
        self.stop_flag = False
    
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

            except Exception as e:
                print(f"[ERROR-{self.cam_name}] AprilTag Detection exception:", e)
                traceback.print_exc()

        print(f"[DETECT-{self.cam_name}] AprilTag Detector thread stopped")

    def stop(self):
        self.stop_flag = True


# BALL DETECTOR THREAD
class BallThread(threading.Thread):
    def __init__(self, cam_name, frame_queue, detect_cache, lock):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.detect_cache = detect_cache
        self.lock = lock
        self.stop_flag = False
    
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
            except Exception as e:
                print(f"[ERROR-{self.cam_name}] ball detection exception:", e)
                traceback.print_exc()

        print(f"[{self.cam_name}] BallDetectorThread stopped")

    def stop(self):
        self.stop_flag = True


# ---------- ZMQ subscriber ----------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(ZMQ_ADDR)
sub.setsockopt(zmq.RCVHWM, 1)
sub.setsockopt(zmq.CONFLATE, 1)  # keep only last message
sub.setsockopt(zmq.LINGER, 0)

# ---- ACTIVE FLUSH ----
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

# per-camera structures
frames = {}
fps_windows = defaultdict(lambda: deque())
frame_queues = {t.decode(): queue.Queue(maxsize=1) for t in SUB_TOPICS}
detect_cache = {}        # cam -> detection dict
detect_lock = threading.Lock()
tag_threads={}
ball_threads={}

for t in SUB_TOPICS:
    cam_name = t.decode()
    tag_threads[cam_name] = AprilTagThread(cam_name,frame_queues[cam_name],detect_cache,detect_lock)
    ball_threads[cam_name] = BallThread(cam_name,frame_queues[cam_name],detect_cache,detect_lock)

    tag_threads[cam_name].start()
    ball_threads[cam_name].start()

print("[Subscriber] connected, waiting for frames... (Press ESC to exit)")

last_show = time.time()
# ---------- Main loop ----------
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

        # push frame to both detector queues
        fq=frame_queues[cam]
        try: fq.get_nowait()
        except: pass
        try: fq.put_nowait((img.copy(),cam_ts))
        except: pass

        # visual output if both cams available
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

                    # draw AprilTags
                    if "tags" in block and block["tags"]["ids"] is not None:
                        cs=block["tags"]["corners"]
                        ids=block["tags"]["ids"]
                        cv2.aruco.drawDetectedMarkers(im,cs,ids)

                    # draw balls
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
    print("Exit clean.")

