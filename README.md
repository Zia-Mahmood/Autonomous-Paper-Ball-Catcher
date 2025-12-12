<!-- # Autonomous Paper Ball Catcher

## Overview
The **Autonomous Paper Ball Catcher** is a mobile robotics project that integrates **multi-camera vision**, **object tracking**, and **robot motion planning**.  
The goal is for a mobile dustbin robot to autonomously catch a thrown paper ball by predicting its landing point using multiple stationary cameras.

Unlike conventional approaches that rely on an onboard camera, this system uses **environment-mounted RGB cameras** that observe the entire arena to track both the ball and the robot in a shared world coordinate frame.

---

## Current Status
- ✅ Robot hardware (omnidirectional mecanum base) assembled.  
- 🕓 Motion control testing pending.  
- 🛠️ Camera-based perception and trajectory estimation not started yet.  
- ⚙️ ROS 2 + OpenCV setup and calibration tools under preparation.

---

## Planned Modules
| Module | Description | Status |
|--------|--------------|--------|
| **Robot Motion** | Control of mecanum drive via ESP32 | ⚙️ In Progress |
| **Multi-Camera Setup** | Calibration + synchronization | ⏳ Pending |
| **Perception** | Ball + robot detection and 3D triangulation | ⏳ Pending |
| **Trajectory Estimation** | EKF-based 3D tracking under gravity | ⏳ Pending |
| **Planning & Control** | Path planning to intercept predicted landing point | ⏳ Pending |

---

## Hardware Stack
- **Robot Base:** Custom omnidirectional mecanum platform  
- **Motor Drivers:** TB6612FNG  
- **Controller:** ESP32 (motor control and communication)  
- **Cameras:** Multiple stationary RGB webcams (Kreo Owl Full HD 60 FPS planned)  
- **Processing:** Laptop running ROS 2 and OpenCV  
- **Power:** 12 V Li-ion 3S battery  
- **Hub:** Powered USB 3.0 hub with 5 V 2–3 A adapter

---

## Directory Structure
```

hardware/     → Robot electronics, motor control
vision/       → Camera calibration, detection, triangulation
control/      → Motion planner, robot control algorithms
ros2_ws/      → ROS 2 workspace for camera + robot integration
data/         → Images, videos, and calibration logs
docs/         → Reports, figures, and project documentation

````

---

## Getting Started
```bash
# Clone repository
git clone https://github.com/Zia-Mahmood/Autonomous-Paper-Ball-Catcher.git
cd Autonomous-Paper-Ball-Catcher

# (Future) Install dependencies
pip install -r requirements.txt
```` -->

# **Autonomous Paper Ball Catcher**

### *A Multi-Camera System for Real-Time 3D Ball Tracking & Robotic Interception*

Team **Slam-Dunk** — *Zia Mahmood Hussain & Nikhil Singh*

---

# **1. Overview**

This project implements a **real-time multi-camera perception and prediction pipeline** that detects a thrown paper ball, reconstructs its 3D trajectory, predicts its future path, and commands a mobile robot (mecanum drive) to move to the interception point.

The system achieves:

* **60 FPS ball detection**
* **40–45 FPS triangulation**
* **Accurate 3D reconstruction of the ball and robot**
* **Fast, low-latency prediction using RLS (Recursive Least Squares)**
* **Open3D-based interception simulation**
* **Full end-to-end pipeline working in real time**

**Current limitation:**
Projectile-motion prediction is **not yet perfectly refined** — it performs excellently for **free-fall / slow arcs**, but accuracy drops for **fast, angled projectile throws**. This is the next target for improvement.

Everything else works end-to-end.

---

# **2. System Pipeline**

```
Multi-Camera Capture (60 FPS)
        ↓
HSV Ball Detection + AprilTag Robot Pose
        ↓
Stereo Triangulation (40–45 FPS)
        ↓
Trajectory Estimation (RLS / EKF / LKF)
        ↓
Intercept Prediction (landing point + intercept time)
        ↓
Robot Controller (real robot + simulation)
```

---

# **3. Features**

### **✔ Multi-Camera Calibration**

* Intrinsics & distortion correction
* World-frame alignment using AprilTags
* Automatic exposure/gain tuning for reliable detection

### **✔ AprilTag Robot Localization**

* Stable 6-DoF robot pose in the shared world frame
* Used for intercept planning

### **✔ High-Speed Ball Detection**

* HSV thresholding
* Contour extraction
* Noise filtering
* 60 FPS sustained across both cameras

### **✔ Stereo Triangulation**

* Epipolar gating for view association
* `triangulatePoints()`
* 3D output at ~40–45 FPS
* Good depth stability for practical throwing ranges

### **✔ Prediction Models**

Implemented models:

* Linear Kalman Filter
* Extended Kalman Filter
* Sliding Window Regression
* **Recursive Least Squares (RLS) — best performing overall**

RLS gives:

* Smooth trajectories
* Very low RMSE for free-fall
* Strong real-time performance

### **✔ Intercept Planner**

* Computes whether the robot can reach the intercept point in time
* Uses velocity + kinematic constraints
* Supports both simulation and real control

### **✔ Open3D Simulation**

* Real-time 3D playback
* Two modes:

  1. **Full trace mode** (past + future trajectory)
  2. **Prediction-only mode** (future trajectory only)

---

# **4. Project Structure**

```
.
├── vision/
│   ├── calibration/          # Intrinsics, extrinsics
│   ├── config/               # Auto tuning camera lighting settigns
│   ├── detection/            # Ball + AprilTag detection
│   ├── triangulation/        # Multi-camera 3D reconstruction
│   ├── predictor/            # RLS, EKF, LKF trajectory models
│   ├── planner/              # Intercept solver
│   ├── publisher/            # Publishes raw images using zmq + stack datastructure
│   └── visualization/        # Open3D + plotting tools
│
├── hardware/
│   ├── Motion_scripts/ 
│   ├── schematics/            
│   └── test_motion/          
│
├── data/
│   ├── april_tags/
│   ├── ball_detection_logs/
│   ├── triangulation_logs/
│   ├── prediction_logs/
│   └── simulation/
│
├── results/                  # images/GIFs
└── README.md
```

---

# **5. How to Run Everything**

## **5.1 Calibration**

```
python vision/calibration/calibrate_camera.py
```

Outputs:

```
camera_calibration_kreo1.npz
camera_calibration_kreo2.npz
camera_calibration_mobile.npz
```

---

## **5.2 Run Detection**

### **AprilTags**

```
python vision/detection/detect_apriltags.py
```

### **Ball Detection**

```
python vision/detection/detect_ball.py
```

Outputs detection overlays and logs.

---

## **5.3 Triangulation**

```
python vision/triangulation/triangulation.py
```

Produces 3D points at ~45 FPS.

---

## **5.4 Prediction (RLS recommended)**

```
python vision/predictor/predictor.py
```

Outputs:

* Predicted trajectory
* Landing/intercept point
* RMSE logs

---

## **5.5 Interception Planner**

```
python vision/planner/planner.py
```

Inputs:

* predicted bivariate polynomial from RLS
* robot max speed
* current robot pose

Outputs:

* intercept point
* can-catch / cannot-catch flag

---

## **5.6 3D Simulation (Open3D)**

### **Full trace mode:**

```
python vision/visualization/visualize_with_trace.py
```

### **Prediction-only mode:**

```
python vision/visualization/visualize_without_trace.py
```

Produces 3D visualization of:

* ball trajectory
* predicted trajectory
* robot motion

---

# **6. Performance Summary**

### **FPS**

| Stage              | FPS   |
| ------------------ | ----- |
| Capture            | 60    |
| Ball Detection     | 60    |
| AprilTag Detection | 40    |
| Triangulation      | 38–45 |
| RLS Predictor      | 30–40 |

### **Prediction Performance**

* **Free-fall:** extremely accurate
* **Slow arcs:** accurate
* **Fast projectile throws:** works, but not yet refined → next improvement target

---

# **7. Media Placeholders**

<!-- ### **7.1 Ball Detection**

```
results/detection_1.png
results/detection_2.png
```

### **7.2 Triangulation Frames**

```
results/triangulation_view_1.png
results/triangulation_view_2.png
results/triangulation_3d_plot.png
```

### **7.3 Prediction Visualizations**

```
results/rls_fit_front_view.png
results/rls_fit_side_view.png
results/rls_fit_top_view.png
results/prediction_error_curve.png
```

### **7.4 Simulation Videos**

```
results/sim_full_trace.mp4
results/sim_predicted_only.mp4
```

### **7.5 Real-World Demonstration GIFs**

```
results/real_detection.gif
results/real_triangulation.gif
results/real_robot_intercept.gif
``` -->

### Google Drive Link 

[Google Drive Link to Result Videos and Images](https://drive.google.com/drive/folders/1lEcQoplVQeGwwzSsM_GPhqJEsyI0ENAF?usp=drive_link)

---

# **8. Future Work**

### 🔧 **1. Improve projectile-motion prediction**

Current predictor is excellent for free-fall and short arcs, but for high-speed projectile throws:

* Early-trajectory noise magnifies velocity estimation error
* Drag affects trajectories in nontrivial ways
* Some throws require more sophisticated modeling

**Planned fixes:**

* Add drag model or hybrid physics + RLS
* Use multi-frame batch fitting (better initial velocity estimate)
* Smooth 2D detections temporally before triangulation
* Optional: migrate predictor to C++ for higher FPS

### 🚀 **2. Multi-camera fusion beyond stereo**

Adding a **third camera** improves depth accuracy dramatically.

### 🤖 **3. Full end-to-end real-time interception**

The perception → planner → robot control loop is ready for tighter integration and full-speed live demos.

---

# **9. Authors**

* **Zia Mahmood Hussain** 
* **Nikhil Singh** 
