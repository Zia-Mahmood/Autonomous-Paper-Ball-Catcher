# Autonomous Paper Ball Catcher

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
````

---

## Pending Work

* Implement 2-camera calibration and synchronization
* Develop color-based ball detection and triangulation
* Integrate robot control with world coordinates
* Tune intercept planner for real-time motion

---

