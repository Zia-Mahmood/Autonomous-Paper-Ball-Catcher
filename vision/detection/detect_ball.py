import cv2, zmq, numpy as np, time, threading, queue, traceback, sys
from collections import deque, defaultdict

# ---------- Config ----------
ZMQ_ADDR = "tcp://localhost:5555"
SUB_TOPICS = [b"kreo1", b"kreo2"]
FPS_WINDOW = 1.0        # seconds for fps moving window
DISPLAY_FPS = 20
VISUALIZE = True     # show tiled view window

# color thresholds (you gave these)
orange_hsvVals = {'hmin': 0, 'smin': 100, 'vmin': 100, 'hmax': 25, 'smax': 255, 'vmax': 255}
purple_hsvVals = {'hmin': 149, 'smin': 69, 'vmin': 82, 'hmax': 177, 'smax': 229, 'vmax': 252}

# detector parameters
MIN_AREA = 100    # min contour area to accept (tune if needed)
MAX_AREA = 20000  # max area (avoid very large blobs)
CIRCULARITY_MIN = 0.25  # min circularity to accept (lower because paper balls can deform)
ASPECT_RATIO_MAX = 2.0   # reject extremely elongated blobs
MAX_DETECTIONS_PER_CAM = 12  # safety limi


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

def find_candidate_contours(mask, min_area=MIN_AREA, max_area=MAX_AREA):
    """Return list of contours filtered by area and shape heuristics."""
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        x,y,w,h = cv2.boundingRect(c)
        aspect = float(w)/float(h) if h>0 else 0.0
        # accept if roughly circular-ish or moderate area even if circularity low
        if circularity >= CIRCULARITY_MIN or (0.5*min(w,h) > 5 and area > (min_area*2)):
            if aspect <= ASPECT_RATIO_MAX:
                candidates.append({
                    "contour": c,
                    "area": area,
                    "perimeter": perim,
                    "circularity": circularity,
                    "bbox": (int(x),int(y),int(w),int(h))
                })
    # sort by area desc
    candidates.sort(key=lambda d: d["area"], reverse=True)
    return candidates


# ---------- Detector thread (per-camera) ----------
class BallDetectorThread(threading.Thread):
    def __init__(self, cam_name, frame_queue, detect_cache, lock):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.frame_queue = frame_queue
        self.detect_cache = detect_cache
        self.lock = lock
        self.stop_flag = False

    def run(self):
        print(f"[{self.cam_name}] Detector thread started.")
        while not self.stop_flag:
            try:
                frame,cam_ts = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # build masks for both colors
                mask_orange = hsv_mask_from_vals(frame, orange_hsvVals)
                mask_purple = hsv_mask_from_vals(frame, purple_hsvVals)

                # combine & clean
                combined_mask = cv2.bitwise_or(mask_orange, mask_purple)
                combined_mask = postprocess_mask(combined_mask)

                # find contours
                candidates = find_candidate_contours(combined_mask)

                detections = []
                for cand in candidates[:MAX_DETECTIONS_PER_CAM]:
                    c = cand["contour"]
                    x,y,w,h = cand["bbox"]
                    area = cand["area"]
                    circ = cand["circularity"]

                    # centroid
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    else:
                        cx = x + w//2; cy = y + h//2

                    # classify color by sampling the masks inside bbox
                    s_orange = int(np.count_nonzero(mask_orange[y:y+h, x:x+w])) if mask_orange is not None else 0
                    s_purple = int(np.count_nonzero(mask_purple[y:y+h, x:x+w])) if mask_purple is not None else 0

                    # small area leads to ambiguity; prefer stronger mask
                    color = "unknown"
                    if s_orange > s_purple and s_orange > 0:
                        color = "orange"
                    elif s_purple > s_orange and s_purple > 0:
                        color = "purple"
                    else:
                        # fallback: mean hue in bbox
                        try:
                            hsv_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                            mean_h = int(np.mean(hsv_roi[:,:,0]))
                            if orange_hsvVals['hmin'] <= mean_h <= orange_hsvVals['hmax']:
                                color = "orange"
                            elif purple_hsvVals['hmin'] <= mean_h <= purple_hsvVals['hmax']:
                                color = "purple"
                        except Exception:
                            color = "unknown"

                    det = {
                        "bbox": (int(x),int(y),int(w),int(h)),
                        "centroid": (int(cx),int(cy)),
                        "area": float(area),
                        "circularity": float(circ),
                        "color": color,
                        "ts": float(cam_ts),
                        "det_time": time.time()
                    }
                    detections.append(det)

                with self.lock:
                    self.detect_cache[self.cam_name] = detections

            except Exception as e:
                print(f"[ERROR-{self.cam_name}] detection exception:", e)
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
det_threads = {}

for t in SUB_TOPICS:
    cam_name = t.decode()
    dt = BallDetectorThread(cam_name, frame_queues[cam_name], detect_cache, detect_lock)
    dt.start()
    det_threads[cam_name] = dt


print("[Subscriber] connected, waiting for frames... (Press ESC to exit)")

last_show = time.time()
try:
    while True:
        parts = recv_latest(sub)
        if parts is None:
            continue

        # unpack message
        topic = parts[0]; cam = topic.decode()
        if len(parts) >= 3:
            ts_part = parts[1]; jpg_part = parts[2]
        else:
            ts_part = None; jpg_part = parts[1]

        recv_time = time.time()
        try:
            cam_ts = float(ts_part.decode()) if ts_part else recv_time
        except:
            cam_ts = recv_time

        img = cv2.imdecode(np.frombuffer(jpg_part, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        fps = update_fps(cam, cam_ts)
        frames[cam] = {"img": img, "cam_ts": cam_ts, "fps": fps}

        # push latest frame to detector queue (maxsize=1)
        fq = frame_queues[cam]
        try:
            fq.get_nowait()
        except queue.Empty:
            pass
        try:
            fq.put_nowait((img.copy(), cam_ts))
        except queue.Full:
            pass

        # build tiled view if both cams available
        if VISUALIZE and all(k in frames for k in [t.decode() for t in SUB_TOPICS]):
            cams = [t.decode() for t in SUB_TOPICS]
            left = frames[cams[0]]; right = frames[cams[1]]
            drift_ms = abs(left["cam_ts"] - right["cam_ts"]) * 1000.0

            def overlay(frame_info, cam_name):
                im = frame_info["img"].copy()
                y = 20
                cv2.putText(im, f"{cam_name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,32,20), 2)
                cv2.putText(im, f"FPS: {frame_info['fps']:.1f}", (10, y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (14,117,5), 2)
                cv2.putText(im, f"cam_ts: {fmt_ts(frame_info['cam_ts'])}", (10, y+52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5,12,117), 1)

                # draw ALL cached detections for this camera
                with detect_lock:
                    dets = detect_cache.get(cam_name, [])
                    for i, d in enumerate(dets):
                        x,y,w,h = d["bbox"]
                        cx,cy = d["centroid"]
                        color = d.get("color", "unknown")
                        box_color = (0,200,200)  # default
                        if color == "orange":
                            box_color = (0,200,255)  # orange-ish
                        elif color == "purple":
                            box_color = (200,0,200)  # purple-ish
                        # draw bounding box and centroid
                        cv2.rectangle(im, (x,y), (x+w, y+h), box_color, 2)
                        cv2.circle(im, (cx,cy), 4, (0,0,255), -1)
                        cv2.putText(im, f"{color}:{i}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                return im

            left_im = overlay(left, cams[0])
            right_im = overlay(right, cams[1])

            # tile horizontally
            h = max(left_im.shape[0], right_im.shape[0])
            right_resized = cv2.resize(right_im, (left_im.shape[1], h))
            tile = np.hstack([left_im, right_resized])

            # overlays
            cv2.putText(tile, f"Drift: {drift_ms:.1f} ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(tile, f"Host now: {fmt_ts(time.time())}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # display throttled
            if VISUALIZE and (time.time() - last_show) > (1.0/DISPLAY_FPS):
                last_show = time.time()
                cv2.imshow("Both Cameras (tiled)", tile)
        elif not VISUALIZE:
            with detect_lock:
                parts_status = []
                for c in [t.decode() for t in SUB_TOPICS]:
                    dets = detect_cache.get(c, [])
                    if dets:
                        counts = { "orange":0, "purple":0, "unknown":0 }
                        for d in dets:
                            counts[d.get("color","unknown")] = counts.get(d.get("color","unknown"),0) + 1
                        parts_status.append(f"{c}: Orange: {counts['orange']} Purple: {counts['purple']}")
                    else:
                        parts_status.append(f"{c}:NoBall")
                sys.stdout.write("\r" + " | ".join(parts_status) + " " * 20)
                sys.stdout.flush()

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    # stop threads
    for d in det_threads.values():
        d.stop()
    # allow threads to exit
    time.sleep(0.1)
    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    print("Exit clean.")