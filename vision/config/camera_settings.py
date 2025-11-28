import cv2, time, threading, subprocess, json, os, numpy as np, sys

# =============== CONFIG ===============
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11

CAM_SOURCES = {
    "mobile":"http://192.168.137.110:8080/video",
    "kreo1": 4,
    "kreo2": 2
}

# parameter grids. Tweak if needed
EXPOSURES = [80, 120, 160, 200, 240, 300]
FOCUSES = [80, 120, 160, 200, 240, 280, 320, 360]
GAINS = [0, 6, 12, 18]
BRIGHTNESSES = [0, 8, 16, 24]

# evaluation parameters
EVAL_SECONDS = 1
SAMPLE_SLEEP = 0.02

OUTPUT_DIR = "./camera_tune_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AprilTag detector setup
def create_detector():
    """Setup AprilTag detector with tuned parameters."""
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

class CameraWorker(threading.Thread):
    def __init__(self, name, src, detector):
        super().__init__(daemon=True)
        self.name = name
        self.src = src
        self.cap = cv2.VideoCapture(src,cv2.CAP_V4L2)

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.detector = detector
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0
        self.running = True
        self.opened = self.cap.isOpened()
        if not self.opened:
            print(f"[{self.name}] Error cannot open source {src}")

    def run(self):
        while self.running and self.opened:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()
            with self.lock:
                self.latest_frame = frame
                self.latest_ts = ts
            time.sleep(0.001)

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_ts
    
    def stop(self):
        self.running = False
        try: self.cap.release()
        except: pass

# v4l2 control helper
def v4l2_set(dev_idx, control, value):
    dev = f"/dev/video{dev_idx}"
    cmd = ["v4l2-ctl", "-d", dev, "-c", f"{control}={value}"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def v4l2_get(dev_idx, control):
    dev = f"/dev/video{dev_idx}"
    cmd = ["v4l2-ctl", "-d", dev, "--get-ctrl", control]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out
    except subprocess.CalledProcessError:
        return None
    
def evaluate_setting(worker, sample_duration=EVAL_SECONDS):
    start = time.time()
    end = start + sample_duration
    tag_counts = []
    frames = 0
    t0 = time.time()
    while time.time()<end:
        frame, ts = worker.read_latest()
        if frame is None:
            time.sleep(SAMPLE_SLEEP)
            continue
        frames += 1
        corners, ids, _ = worker.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        n = 0 if ids is None else len(ids)
        tag_counts.append(n)
        time.sleep(SAMPLE_SLEEP)
    elapsed = time.time()- t0
    avg_tags = float(np.mean(tag_counts)) if tag_counts else 0.0
    std_tags = float(np.std(tag_counts)) if tag_counts else 0.0
    fps = frames / elapsed if elapsed>0 else 0.0
    return { "avg_tags": avg_tags, "std_tags": std_tags, "fps": fps, "samples": len(tag_counts)}

def tune_camera(dev_idx, worker, brief_name):
    if not isinstance(worker.src, int):
        print(f"[{brief_name}] Skipping tuning for non-local source {worker.src}")
        return None

    print(f"[{brief_name}] Starting adaptive tuning (~5 minutes)...")

    # ---------- Disable auto controls ----------
    try:
        v4l2_set(dev_idx, "focus_automatic_continuous", 0)
        v4l2_set(dev_idx, "auto_exposure", 1)
        v4l2_set(dev_idx, "exposure_dynamic_framerate", 0)
    except Exception:
        pass

    # ---------- Utility ----------
    def neighbors(val, lst, radius=1):
        try:
            i = lst.index(val)
        except ValueError:
            return [val]
        lo = max(0, i-radius)
        hi = min(len(lst)-1, i+radius)
        return lst[lo:hi+1]

    def set_all(exp, foc, g, b):
        v4l2_set(dev_idx, "exposure_time_absolute", int(exp))
        time.sleep(0.04)
        v4l2_set(dev_idx, "focus_absolute", int(foc))
        time.sleep(0.03)
        v4l2_set(dev_idx, "gain", int(g))
        time.sleep(0.02)
        v4l2_set(dev_idx, "brightness", int(b))
        time.sleep(0.05)

    # ---------- Stage 1: Coarse grid ----------
    def pick_coarse(lst, n=3):
        if len(lst) <= n:
            return lst
        idxs = [int(round(i*(len(lst)-1)/(n-1))) for i in range(n)]
        return [lst[i] for i in idxs]

    exp_c = pick_coarse(EXPOSURES, 3)
    foc_c = pick_coarse(FOCUSES, 3)
    gain_c = pick_coarse(GAINS, 3)
    bright_c = pick_coarse(BRIGHTNESSES, 3)

    coarse_combos = [(e,f,g,b) for e in exp_c for f in foc_c for g in gain_c for b in bright_c]

    print(f"[{brief_name}] Coarse grid: {len(coarse_combos)} combos")

    coarse_results = []
    for idx, (e, f, g, b) in enumerate(coarse_combos):
        print(f"[{brief_name}] [Coarse {idx+1}/{len(coarse_combos)}] e={e}, f={f}, g={g}, b={b}     ", end="\r")
        set_all(e, f, g, b)
        stats = evaluate_setting(worker)

        score = stats["avg_tags"] * 100.0 + stats["fps"] * 0.2 - stats["std_tags"] * 10.0
        coarse_results.append((score, (e, f, g, b), stats))

    print("")  # newline

    coarse_results.sort(reverse=True, key=lambda x: x[0])
    TOP_K = 4
    topk = coarse_results[:TOP_K]

    print(f"[{brief_name}] Top coarse candidates:")
    for score, combo, stats in topk:
        print("   ", combo, f"score={score:.1f}", stats)

    # ---------- Stage 2: refinement ----------
    refined_results = []
    for score0, (exp0, foc0, g0, b0), stats0 in topk:
        exp_n = neighbors(exp0, EXPOSURES, radius=1)
        foc_n = neighbors(foc0, FOCUSES, radius=1)
        g_n   = neighbors(g0,  GAINS, radius=1)
        b_n   = neighbors(b0,  BRIGHTNESSES, radius=1)

        local = [(e,f,g,b) for e in exp_n for f in foc_n for g in g_n for b in b_n]

        print(f"[{brief_name}] Refining around {exp0, foc0, g0, b0} → {len(local)} combos")
        for idx, combo in enumerate(local):
            e,f,g,b = combo
            print(f"[{brief_name}] [Refine {idx+1}/{len(local)}] e={e}, f={f}, g={g}, b={b}     ", end="\r")
            set_all(e,f,g,b)
            stats = evaluate_setting(worker)
            score = stats["avg_tags"] * 100.0 + stats["fps"] * 0.2 - stats["std_tags"] * 10.0
            refined_results.append((score, combo, stats))
        print("")

    # ---------- Decide best ----------
    all_results = coarse_results + refined_results
    all_results.sort(reverse=True, key=lambda x: x[0])

    best_score, best_combo, best_stats = all_results[0]
    e,f,g,b = best_combo

    print("")
    print(f"[{brief_name}] BEST SETTINGS:")
    print(f"   exposure={e}, focus={f}, gain={g}, brightness={b}")
    print(f"   score={best_score:.1f}, stats={best_stats}")

    # ---------- Apply final ----------
    set_all(e,f,g,b)

    # ---------- Save ----------
    out = {
        "exp": e, "focus": f, "gain": g, "brightness": b,
        "score": best_score,
        "stats": best_stats
    }
    out_file = os.path.join(OUTPUT_DIR, f"best_camera_settings_{brief_name}.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{brief_name}] Saved → {out_file}")

    return out

def get_camera_selection():
    print("\n=== Multi-Camera Live View Setup ===")
    print("Select cameras to open (comma separated):")
    print("1. Kreo Webcam #1")
    print("2. Kreo Webcam #2")
    print("3. Mobile IP Webcam")
    print("Example: 1,2 or 1,3 or 1,2,3")
    user_in = input("Cameras to open: ").strip()
    choices = [x.strip() for x in user_in.split(",") if x.strip()]
    selected = []
    for c in choices:
        if c == "1":
            selected.append(("kreo1", CAM_SOURCES["kreo1"]))
        elif c == "2":
            selected.append(("kreo2", CAM_SOURCES["kreo2"]))
        elif c == "3":
            selected.append(("mobile", CAM_SOURCES["mobile"]))
        else:
            print(f"[WARN] Ignoring invalid entry: {c}")
    if not selected:
        print("[ERROR] No valid cameras selected. Exiting.")
        sys.exit(1)
    return selected

# ==================== MAIN ====================
if __name__ == "__main__":
    selected = get_camera_selection()

    workers = {}
    for name, src in selected:
        w = CameraWorker(name, src, create_detector())
        w.start()
        workers[name] = w
        time.sleep(0.05)

    # small warmup
    print("[INFO] Warmup for 1.2 seconds to let cameras settle...")
    time.sleep(1.2)
    
    for name, w in list(workers.items()):
        if isinstance(w.src,int):
            try:
                dev_idx = int(w.src)
            except Exception:
                print(f"[{name}] Invalid device index for tuning: {w.src}")
                continue
            best = tune_camera(dev_idx, w, name)
        else:
            print(f"[{name}] Not a local device; skipping tuning.")

    # After tuning show live preview with settings applied
    print("[INFO] Tuning complete. Press ESC to exit preview windows.")
    last_ts = {name: 0.0 for name in workers}
    fps_counter = {name: 0 for name in workers}
    fps = {name: 0.0 for name in workers}
    last_fps_update = time.time()

    try:
        while True:
            now = time.time()
            timestamps = {}

            # --- Collect frames from all cameras ---
            for name, w in workers.items():
                frame, ts = w.read_latest()
                if frame is None:
                    continue
                timestamps[name] = ts

                # --- FPS update ---
                if ts != last_ts[name]:
                    fps_counter[name] += 1
                    last_ts[name] = ts

                if now - last_fps_update >= 1.0:
                    fps[name] = fps_counter[name]
                    fps_counter[name] = 0

            if now - last_fps_update >= 1.0:
                last_fps_update = now

            # --- Drift calculation ---
            if len(timestamps) > 1:
                tvals = np.array(list(timestamps.values()))
                drift_ms = (tvals.max() - tvals.min()) * 1000.0
            else:
                drift_ms = 0.0

            # --- Draw every camera independently ---
            for name, w in workers.items():
                frame, ts = w.read_latest()
                if frame is None:
                    continue
                # AprilTag overlay
                corners, ids, _ = w.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # resolution text
                h, wid = frame.shape[:2]
                cv2.putText(frame, f"{name} {wid}x{h}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

                cv2.putText(frame, f"FPS: {fps[name]:.0f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

                cv2.putText(frame, f"Drift: {drift_ms:.1f} ms", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                cv2.imshow(f"Tuned - {name}", frame)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        for w in workers.values():
            w.stop()
        cv2.destroyAllWindows()
