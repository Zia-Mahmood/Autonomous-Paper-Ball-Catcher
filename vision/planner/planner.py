"""
planner_sim_open3d.py

Simulate free-fall planner + visualization:
 - Subscribe to predictor PUB at tcp://localhost:5566
 - Ball: sphere + trail
 - Car: cuboid (0.25 x 0.25 x 0.075 m) with cylinder on top
 - Car moves at constant speed (0.5 m/s) toward predicted landing XY (free-fall -> current XY)
 - Detect landing at z <= 0.20 m (transition from >0.20 -> <=0.20). Mark landing and check if within car footprint.
"""

import zmq, time, math, numpy as np, open3d as o3d, threading
from collections import deque

# ---------- CONFIG ----------
ZMQ_ADDR = "tcp://localhost:5566"   # predictor publishes here by default. See predictor.py.
TARGET_Z = 0.20                     # 20 cm intercept height
CAR_START = np.array([0.30, 0.30, 0.0])  # meters (x,y,z)
CAR_L = 0.25   # length (m)
CAR_W = 0.25   # width  (m)
CAR_H = 0.075  # height (m)
TURRET_H = 0.125   # cylinder height (m)
TURRET_D = 0.11    # cylinder diameter (m)
CAR_SPEED = 0.5    # m/s (constant velocity for planner simulation)
SIM_DT = 0.03      # seconds per simulation tick (~33Hz)

# Visualization colors
COLOR_CAR = [0.2, 0.6, 0.2]
COLOR_TURRET = [0.8, 0.7, 0.2]
COLOR_BALL = [1.0, 0.4, 0.0]
COLOR_TRAIL = [0.9, 0.2, 0.2]
COLOR_LANDING_OK = [0.0, 0.8, 0.0]
COLOR_LANDING_FAIL = [0.8, 0.0, 0.0]

# ---------- Helper functions ----------
def solve_time_to_z(z0, vz0, target_z=TARGET_Z, g=9.81):
    a = -0.5 * g
    b = vz0
    c = z0 - target_z
    disc = b*b - 4*a*c
    if disc < 0: return None
    r1 = (-b + math.sqrt(disc)) / (2*a)
    r2 = (-b - math.sqrt(disc)) / (2*a)
    ts = [t for t in (r1, r2) if t > 0]
    return min(ts) if ts else None

def is_safe_catch(car_center, ball_xy):
    # Dimensions
    turret_radius = TURRET_D / 2.0  # 0.055 m
    ball_radius = 0.025             # 0.025 m (Matches mesh radius)
    
    # Calculate distance between Car Center and Ball Center
    dx = ball_xy[0] - car_center[0]
    dy = ball_xy[1] - car_center[1]
    dist = math.sqrt(dx*dx + dy*dy)

    # STRICT CHECK: Ball must be fully inside the turret (not hitting the rim)
    # Distance + BallRadius <= TurretRadius
    safe_margin = turret_radius - ball_radius  # 0.055 - 0.025 = 0.03 m
    
    return dist <= safe_margin

# ---------- ZMQ Subscriber Thread ----------
class PredictorSubscriber(threading.Thread):
    def __init__(self, zmq_addr=ZMQ_ADDR):
        super().__init__(daemon=True)
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(zmq_addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.latest = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            try:
                msg = self.sub.recv_json(flags=0)
            except Exception:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest = msg

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False
        try: self.sub.close()
        except: pass
        try: self.ctx.term()
        except: pass

# ---------- Open3D Visualizer / Simulator ----------
class PlannerSim:
    def __init__(self):
        # state
        self.ball_pos = None      # [x,y,z]
        self.ball_prev_z = None
        self.ball_trail = deque(maxlen=200)
        self.predicted_landing = None
        self.landed = False
        self.landing_point = None
        self.landing_inside = False

        # car state
        self.car_center = CAR_START.copy()
        self.car_vel = np.array([0.0, 0.0, 0.0])

        # tracking for optimized updates
        self.last_ball_pos = None
        self.last_car_center = None

        # ZMQ subscriber
        self.sub = PredictorSubscriber(ZMQ_ADDR)
        self.sub.start()

        # open3d setup
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Planner Simulation (Free-Fall)", width=1280, height=720)
        self._setup_scene()
        self.vis.register_animation_callback(self.update)

        # timing
        self.last_time = time.time()

    def _setup_scene(self):
        # floor grid and axes
        grid = o3d.geometry.LineSet()
        pts = []
        lines = []
        size = 1.5
        step = 0.25
        for i in np.arange(-size, size+1e-6, step):
            pts.append([i, -size, 0]); pts.append([i, size, 0]); lines.append([len(pts)-2, len(pts)-1])
            pts.append([-size, i, 0]); pts.append([size, i, 0]); lines.append([len(pts)-2, len(pts)-1])
        grid.points = o3d.utility.Vector3dVector(np.array(pts))
        grid.lines = o3d.utility.Vector2iVector(np.array(lines))
        grid.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for _ in lines])
        self.vis.add_geometry(grid)

        # ball (sphere)
        self.ball_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        self.ball_mesh.paint_uniform_color(COLOR_BALL)
        self.ball_mesh.compute_vertex_normals()
        self.vis.add_geometry(self.ball_mesh)

        # trail point cloud
        self.trail_pcd = o3d.geometry.PointCloud()
        self.trail_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.trail_pcd.paint_uniform_color(COLOR_TRAIL)
        self.vis.add_geometry(self.trail_pcd)

        # car box
        box = o3d.geometry.TriangleMesh.create_box(width=CAR_L, height=CAR_W, depth=CAR_H)
        # Open3D box origin at (0,0,0) with x in width dir; we want center at car_center and z base at 0.
        box.translate([-CAR_L/2.0, -CAR_W/2.0, 0.0])
        box.paint_uniform_color(COLOR_CAR)
        box.compute_vertex_normals()
        self.car_mesh = box
        self.vis.add_geometry(self.car_mesh)

        # turret cylinder
        self.turret_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=TURRET_D / 2.0, height=TURRET_H)
        self.turret_mesh.paint_uniform_color(COLOR_TURRET)
        self.turret_mesh.compute_vertex_normals()
        # initial position: bottom at z=0
        self.turret_mesh.translate([0, 0, CAR_H])
        self.vis.add_geometry(self.turret_mesh)

        # landing marker (sphere) - hidden initially
        self.land_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        self.land_marker.paint_uniform_color(COLOR_LANDING_FAIL)
        self.land_marker.compute_vertex_normals()
        # hide initially
        lm_cur = self.land_marker.get_center()
        hidden_pos = np.array([0.0, 0.0, -10.0])
        self.land_marker.translate(hidden_pos - lm_cur, relative=True)
        self.land_marker_visible = False
        self.vis.add_geometry(self.land_marker)

        # camera
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0.45, 0.45, 0.1])
        ctr.set_front([-0.5, -0.8, -0.3])
        ctr.set_up([0,0,1])
        ctr.set_zoom(0.6)

    def update(self, vis):
        # dt
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = SIM_DT
        self.last_time = now

        msg = self.sub.get()
        ball_updated = False
        if msg is not None:
            # payload format per predictor/planner: "ball": {x,y,z,valid}, "tag4":..., "tag5":...
            b = msg.get("ball", {})
            bx = b.get("x"); by = b.get("y"); bz = b.get("z")
            valid = b.get("valid", False)
            if valid and (bx is not None) and (by is not None) and (bz is not None):
                new_pos = np.array([float(bx), float(by), float(bz)])
                # check if significantly different to avoid redundant updates
                if self.ball_pos is None or np.linalg.norm(new_pos - self.ball_pos) > 1e-6:
                    self.ball_pos = new_pos
                    self.ball_trail.append(self.ball_pos.copy())
                    ball_updated = True

                    # compute free-fall predicted landing XY: for free fall we aim directly under current xy
                    # Compute time to TARGET_Z with vz=0 (free-fall)
                    vz_assumed = 0.0
                    t_hit = solve_time_to_z(self.ball_pos[2], vz_assumed, TARGET_Z)
                    if t_hit is None: 
                        self.predicted_landing = None
                    else:
                        # free-fall landing xy = current xy (no horizontal motion predicted)
                        self.predicted_landing = np.array([self.ball_pos[0], self.ball_pos[1]])
            else:
                # no valid ball -> keep previous
                pass

        # Move car toward predicted landing XY at CAR_SPEED
        car_moved = False
        if self.predicted_landing is not None and not self.landed:
            target = np.array([self.predicted_landing[0], self.predicted_landing[1], 0.0])
            vec = target - self.car_center
            dist = np.linalg.norm(vec[:2])
            if dist > 1e-6:
                dir2 = vec[:2] / dist
                step = CAR_SPEED * dt
                old_car_center = self.car_center.copy()
                if step >= dist:
                    self.car_center[:2] = target[:2]
                else:
                    self.car_center[:2] += dir2 * step
                # check if moved
                if self.last_car_center is None or np.linalg.norm(self.car_center - self.last_car_center) > 1e-6:
                    car_moved = True

        # update ball and trail only if updated
        if ball_updated:
            # ball
            b_center = self.ball_mesh.get_center()
            self.ball_mesh.translate(self.ball_pos - b_center, relative=True)
            self.vis.update_geometry(self.ball_mesh)

            # trail update
            if len(self.ball_trail) > 0:
                pts = np.array(self.ball_trail)
                self.trail_pcd.points = o3d.utility.Vector3dVector(pts)
                self.trail_pcd.paint_uniform_color(COLOR_TRAIL)
                self.vis.update_geometry(self.trail_pcd)

        # update car and turret only if moved
        if car_moved:
            # car mesh update: we created box with base at z=0 and origin shifted to center; simply translate to car_center
            cur_center = self.car_mesh.get_center()
            desired_center = np.array([self.car_center[0], self.car_center[1], CAR_H/2.0])
            self.car_mesh.translate(desired_center - cur_center, relative=True)
            self.vis.update_geometry(self.car_mesh)

            # turret update (place at car top center)
            t_cur_center = self.turret_mesh.get_center()
            turret_des = np.array([self.car_center[0], self.car_center[1], CAR_H + TURRET_H/2.0])
            self.turret_mesh.translate(turret_des - t_cur_center, relative=True)
            self.vis.update_geometry(self.turret_mesh)

            self.last_car_center = self.car_center.copy()

        # landing detection: when ball crosses TARGET_Z from above to <=TARGET_Z
        # MULTI-CATCH LOGIC (safe version)
        if self.ball_pos is not None:
            prev_z = self.ball_prev_z if self.ball_prev_z is not None else self.ball_pos[2]

            # Reset landing when ball rises above catch height again
            if self.landed and self.ball_pos[2] > TARGET_Z + 0.01:
                self.landed = False
                self.landing_point = None
                self.landing_inside = False

                # hide landing marker
                lm_cur = self.land_marker.get_center()
                hidden_pos = np.array([0.0, 0.0, -10.0])
                self.land_marker.translate(hidden_pos - lm_cur, relative=True)
                self.vis.update_geometry(self.land_marker)

            # Detect NEW landing event
            if (not self.landed) and (prev_z > TARGET_Z) and (self.ball_pos[2] <= TARGET_Z):
                self.landed = True
                self.landing_point = np.array([self.ball_pos[0], self.ball_pos[1], TARGET_Z])
                self.landing_inside = is_safe_catch(self.car_center, self.landing_point[:2])
                print(f"[PlannerSim] Ball landed at {self.landing_point[:2]} inside_car={self.landing_inside}")

                # show marker
                self.land_marker.paint_uniform_color(
                    COLOR_LANDING_OK if self.landing_inside else COLOR_LANDING_FAIL
                )
                lm_cur = self.land_marker.get_center()
                lm_des = self.landing_point
                self.land_marker.translate(lm_des - lm_cur, relative=True)
                self.vis.update_geometry(self.land_marker)

            # save for next frame
            self.ball_prev_z = float(self.ball_pos[2])

        # pause small time implicitly done by Open3D callback; request redraw
        return False

    def run(self):
        try:
            self.vis.run()
        finally:
            self.sub.stop()
            self.vis.destroy_window()

if __name__ == "__main__":
    print("Starting Planner + Open3D Simulation (Free-Fall). Subscribe:", ZMQ_ADDR)
    sim = PlannerSim()
    sim.run()