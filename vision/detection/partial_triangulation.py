import cv2, time, json, os, threading, queue, numpy as np, sys
import zmq, msgpack, msgpack_numpy as m
m.patch()
from collections import defaultdict, deque

# =============== CONFIG ===============
DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11

CAM_SOURCES = {
    "Mobile":"http://192.168.137.110:8080/video",
    "Kreo1": 0,
    "Kreo2": 4
}

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
            selected.append(("kreo1", CAM_SOURCES["Kreo1"], 0.096))
        elif c == "2":
            selected.append(("kreo2", CAM_SOURCES["Kreo2"], 0.096))
        elif c == "3":
            selected.append(("mobile", CAM_SOURCES["Mobile"], 0.091))
        else:
            print(f"[WARN] Ignoring invalid entry: {c}")
    if not selected:
        print("[ERROR] No valid cameras selected. Exiting.")
        sys.exit(1)
    return selected

TAG4_MOBILE_ID = 4
TAG4_HEIGHT_M = 0.075  # meters
CAPTURE_TARGET_FPS = 60.0
PROCESSING_TARGET_FPS = 40.0
FRAME_QUEUE_MAX = 8
PROCESS_QUEUE_MAX = 8
DEBUG_PRINT_DETECTIONS = False
APPLY_CAMERA_SETTINGS = True
CALIB_DIR = '../calibration/'  # folder holding camera_calibration_{cam_name}.npz and best_camera_settings_{cam_name}.json
SETTINGS_DIR = './camera_tune_results/'

def load_camera_calib(cam_name):
    path = os.path.join(CALIB_DIR, f'camera_calibration_{cam_name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    calib = np.load(path)
    camera_matrix = calib["cameraMatrix"]
    dist_coeffs = calib['distCoeffs']
    print("[INFO] Loaded calibrated camera parameters")
    return camera_matrix, dist_coeffs

CAMPROP_MAP = {
    'frame_width': cv2.CAP_PROP_FRAME_WIDTH,
    'frame_height': cv2.CAP_PROP_FRAME_HEIGHT,
    'fps': cv2.CAP_PROP_FPS,
    'exposure': cv2.CAP_PROP_EXPOSURE,
    'gain': cv2.CAP_PROP_GAIN,
    'focus': cv2.CAP_PROP_FOCUS,
    'brightness': cv2.CAP_PROP_BRIGHTNESS,
    'contrast': cv2.CAP_PROP_CONTRAST,
    'saturation': cv2.CAP_PROP_SATURATION,
}

def apply_settings_to_capture(cap, settings):
    for k, v in settings.items():
        if k in CAMPROP_MAP:
            try:
                cap.set(CAMPROP_MAP[k], float(v))
            except Exception:
                pass

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

    # debug
    # reprojection
    # imgpts, _ = cv2.projectPoints(objp, rvec, tvec, cam_mtx, cam_dist)
    # imgpts = imgpts.reshape(-1,2)

    # # compute reprojection error against detected corners 'img_corners' (4x2)
    # reproj_err = np.linalg.norm(imgpts - corners, axis=1).mean()

    # # compute camera-frame coords of each object corner
    # # X_cam = R @ X_tag + t
    # cam_pts = (R @ objp.T).T + tvec.reshape(1,3)

    # print("=== REPROJ DEBUG ===")
    # print("rvec:", rvec.ravel())
    # print("tvec:", tvec.ravel())
    # print("R det:", np.linalg.det(R))
    # print("reproj_err px:", reproj_err)
    # print("camera-frame corners (x,y,z):")
    # for i,p in enumerate(cam_pts):
    #     print(f"  corner {i}: {p}")
    return T
    

class CameraWorker:
    def __init__(self, name, device, tag_size):
        self.name = name
        self.device = device
        self.tag_size = float(tag_size)
        self.cap = None
        self.running = threading.Event()
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.process_queue = queue.Queue(maxsize=PROCESS_QUEUE_MAX)
        self.capture_thread = None
        self.process_thread = None
        self.detector = create_detector()
        self.cam_mtx = None
        self.cam_dist = None
        self.settings = {}
        self.stats = {'captured': 0, 'processed': 0, 'last_proc_time': 0.0}

    def open_capture(self):
        if isinstance(self.device, int):
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(int(self.device),cv2.CAP_V4L2)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device)
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_TARGET_FPS)
        return cap
    
    def load_calib_and_settings(self):
        try:
            self.cam_mtx, self.cam_dist = load_camera_calib(self.name)
        except Exception as e:
            print(f'[{self.name}] camera calibration load failed: {e}')
            raise
        sfile = os.path.join(SETTINGS_DIR, f'best_camera_settings_{self.name}.json')
        if os.path.exists(sfile):
            try:
                with open(sfile, 'r') as f:
                    self.settings = json.load(f)
            except Exception:
                self.settings = {}
        else:
            self.settings = {}

    def start(self):
        self.running.set()
        self.load_calib_and_settings()
        self.cap = self.open_capture()
        if APPLY_CAMERA_SETTINGS and self.settings:
            apply_settings_to_capture(self.cap, self.settings)
        self.capture_thread = threading.Thread(target=self._capture_loop, name=f'cap-{self.name}', daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, name=f'proc-{self.name}', daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
    
    def stop(self):
        self.running.clear()
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def _capture_loop(self):
        target_dt = 1.0 / CAPTURE_TARGET_FPS
        while self.running.is_set():
            t0 = time.time()
            ret, frame = self.cap.read()
            t1 = time.time()
            print(f"[{self.name}] cap read time: {t1-t0:.3f}s")
            if not ret:
                time.sleep(0.005)
                continue
            try:
                self.frame_queue.put_nowait((time.time(), frame))
                self.stats['captured'] += 1
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((time.time(), frame))
                except Exception:
                    pass
            dt = time.time() - t0
            sl = max(0.0, target_dt - dt)
            if sl > 0:
                time.sleep(sl)

    def _process_loop(self):
        target_dt = 1.0 / PROCESSING_TARGET_FPS
        while self.running.is_set():
            try:
                ts, frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            results = []

            if ids is not None:
                for i, tid in enumerate(ids.flatten()):
                    c = corners[i][0]  # 4x2

                    if DEBUG_PRINT_DETECTIONS:
                        print(f"[{self.name}] tag {tid}, corners: {c}")

                    try:
                        t_before = time.time()
                        pose = estimate_pose_apriltag(c, self.tag_size, self.cam_mtx, self.cam_dist)
                        t_after = time.time()
                        print("tid: ",tid,"pose time:", t_after - t_before," Pose: ",pose[:,-1])
                        results.append({'id': int(tid), 'pose_cam_tag': pose, 'timestamp': ts})
                    except:
                        continue
            try:
                self.process_queue.put_nowait((ts, frame, results))
            except queue.Full:
                _ = self.process_queue.get_nowait()
                self.process_queue.put_nowait((ts, frame, results))

            self.stats['processed'] += len(results)

            time.sleep(max(0, target_dt - (time.time() - ts)))

# central mapper
class WorldMapper:
    def __init__(self):
        self.lock = threading.RLock()
        self.tag_world_poses = {} # tag_id -> 4x4 pose in world (tag -> world)
        self.camera_world_poses = {}  # cam_name -> 4x4 pose in world (camera -> world)
        self.history = defaultdict(deque)

    @staticmethod
    def invert_pose(T):
        R = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4)
        Tinv[:3, :3] = R.T
        Tinv[:3, 3] = -R.T.dot(t)
        return Tinv
    
    def reset(self):
        with self.lock:
            self.tag_world_poses.clear()
            self.camera_world_poses.clear()
            self.history.clear()

    
    def _recenter_on_tag1(self):
        """
        If tag 1 exists, transform all stored poses so that tag1 becomes the world origin.
        That is, compute S = inv(T_world_tag1) and replace every pose by S.dot(pose).
        After this, T_world_tag1 will be identity.
        """
        TAG1 = 1
        if TAG1 not in self.tag_world_poses:
            return

        T_world_tag1 = self.tag_world_poses[TAG1]
        S = self.invert_pose(T_world_tag1)  # S = inv(T_world_tag1)

        # apply S to all tags and cameras
        for tid in list(self.tag_world_poses.keys()):
            self.tag_world_poses[tid] = S.dot(self.tag_world_poses[tid])

        for cname in list(self.camera_world_poses.keys()):
            self.camera_world_poses[cname] = S.dot(self.camera_world_poses[cname])

    def add_observation(self, cam_name, cam_T_tag, tag_id, timestamp):
        with self.lock:
            cam_T_tag = np.array(cam_T_tag, dtype=float)
            tag_T_cam = self.invert_pose(cam_T_tag)  # tag_T_cam maps camera -> tag

            # Case A: we already know this tag's world pose -> compute camera world
            if tag_id in self.tag_world_poses:
                T_world_tag = self.tag_world_poses[tag_id]
                # T_world_cam = T_world_tag . T_tag_cam
                T_world_cam = T_world_tag.dot(tag_T_cam)
                self.camera_world_poses[cam_name] = T_world_cam

            # Case B: we know this camera's world pose -> compute tag world
            elif cam_name in self.camera_world_poses:
                T_world_cam = self.camera_world_poses[cam_name]
                # T_world_tag = T_world_cam . T_cam_tag
                T_world_tag = T_world_cam.dot(cam_T_tag)

                # If this is TAG4 (mobile), force its height to TAG4_HEIGHT_M (z)
                if tag_id == TAG4_MOBILE_ID:
                    T_world_tag = np.array(T_world_tag, dtype=float)
                    T_world_tag[2, 3] = TAG4_HEIGHT_M

                self.tag_world_poses[tag_id] = T_world_tag

            # Case C: neither tag nor camera is known yet
            else:
                # If this is tag 1, make it the world origin
                if tag_id == 1:
                    T_world_tag = np.eye(4)
                    # camera world = T_world_tag . T_tag_cam = I . T_tag_cam = T_tag_cam
                    T_world_cam = T_world_tag.dot(tag_T_cam)
                    self.tag_world_poses[tag_id] = T_world_tag
                    self.camera_world_poses[cam_name] = T_world_cam

                # If this is TAG4 (mobile) and first observation, place tag relative to camera
                # then set its z to TAG4_HEIGHT_M
                elif tag_id == TAG4_MOBILE_ID:
                    # choose camera as temporary world (i.e., set T_world_cam = I)
                    # So tag_world = T_world_cam . T_cam_tag = I . T_cam_tag = cam_T_tag
                    T_world_tag = cam_T_tag.copy()
                    T_world_tag = np.array(T_world_tag, dtype=float)
                    T_world_tag[2, 3] = TAG4_HEIGHT_M
                    self.tag_world_poses[tag_id] = T_world_tag
                    # camera world is then T_world_cam = T_world_tag . T_tag_cam
                    T_world_cam = T_world_tag.dot(tag_T_cam)
                    self.camera_world_poses[cam_name] = T_world_cam

                else:
                    # Default: choose camera as temporary world origin.
                    # So T_world_cam = Identity, and T_world_tag = I . cam_T_tag = cam_T_tag
                    T_world_cam = np.eye(4)
                    T_world_tag = cam_T_tag.copy()
                    # store both
                    self.camera_world_poses[cam_name] = T_world_cam
                    self.tag_world_poses[tag_id] = T_world_tag

            # record history
            self.history[tag_id].append((timestamp, cam_name))
            # keep limited history
            if len(self.history[tag_id]) > 200:
                self.history[tag_id].popleft()

            # IMPORTANT: if tag 1 appears at any time, recenter the whole map so tag1 is origin
            if 1 in self.tag_world_poses:
                # Recenter only if tag1 is not already identity
                T_world_tag1 = self.tag_world_poses[1]
                if not np.allclose(T_world_tag1, np.eye(4), atol=1e-6):
                    self._recenter_on_tag1()

    def get_map_snapshot(self):
        with self.lock:
            tags = {tid: T.copy() for tid, T in self.tag_world_poses.items()}
            cams = {c: T.copy() for c, T in self.camera_world_poses.items()}
        return {'tags': tags, 'cameras': cams}

class SnapshotPublisher:
    def __init__(self, port=5557):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
    
    def send(self, snapshot):
        # snapshot = {'tags': {tid: np.array(4x4)}, 'cameras': {name: np.array(4x4)}}
        # convert to pure dict of lists
        msg = {
            "tags": {tid: snapshot['tags'][tid] for tid in snapshot['tags']},
            "cameras": {c: snapshot['cameras'][c] for c in snapshot['cameras']}
        }
        if "frames" in snapshot:
            msg["frames"] = snapshot["frames"] # dict of cam_name -> (jpg bytes)
        packed = msgpack.packb(msg, default=m.encode)
        self.sock.send(packed)



def main():

    workers = []
    for name, dev, tag_size in get_camera_selection():
        w = CameraWorker(name, dev, tag_size)
        try:
            w.start()
        except Exception as e:
            print(f'Failed to start camera {name}: {e}')
            continue
        workers.append(w)
    
    mapper = WorldMapper()
    mapper.reset()
    pub = SnapshotPublisher(port=5557)
    map_thread_stop = threading.Event()

    def mapper_loop():
        while not map_thread_stop.is_set():
            for w in workers:
                try:
                    ts, frame, dets = w.process_queue.get_nowait()
                    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        frame_jpg = jpg.tobytes()
                    else:
                        frame_jpg = None
                except queue.Empty:
                    continue

                for d in dets:
                    mapper.add_observation(w.name, d['pose_cam_tag'], d['id'], d['timestamp'])
                    snapshot = mapper.get_map_snapshot()
                    snapshot['frames'] = {w.name: frame_jpg} if frame_jpg is not None else {}
                    pub.send(snapshot)
            time.sleep(0.002)
    
    mt = threading.Thread(target=mapper_loop, name='mapper', daemon=True)
    mt.start()

    try:
        last_print = time.time()
        while True:
            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                snap = mapper.get_map_snapshot()
                print('--- status ---')
                for w in workers:
                    print(f'{w.name}: cap {w.stats["captured"]} proc {w.stats["processed"]}')
                print('tags known:', list(snap['tags'].keys()))
            # time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        map_thread_stop.set()
        for w in workers:
            w.stop()
        print('exiting main')

if __name__ == '__main__':
    main()