import cv2
import numpy as np
import os

# ---------------- CONFIG ----------------
FIELD_W, FIELD_H = 1.219, 1.219  # meters
TAG_SIZE_M = 0.06  # 6 cm
# TAG_IDS = [0, 1, 2, 3, 4, 5]  # 4 field corners + 1 robot
TAG_IDS = [6, 7, 8, 9]
OUTPUT_DIR = "D:/IIITH/SEM 3/MR/Project/Autonomous-Paper-Ball-Catcher/data/april_tags"

# ---------------- TAG GENERATION ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
for tid in TAG_IDS:
    tag_img = cv2.aruco.generateImageMarker(aruco_dict, tid, 600)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"tag_{tid}.png"), tag_img)
print(f"✅ Generated {len(TAG_IDS)} AprilTags in '{OUTPUT_DIR}'")

