import cv2
import json
import shutil
from pathlib import Path

DATASET = "dataset_box_h1"

IMG_DIR = Path(f"data/{DATASET}/images")
ROI_JSON = Path(f"data/{DATASET}/roi_boxes.json")
OUT_DIR = Path(f"data/{DATASET}/images_roi")

SCALE = 5   # ROI expand scale

# =============================
# AUTO CLEAR OLD ROI FOLDER
# =============================

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir()

print("Cleared images_roi folder")

# =============================
# LOAD ROI JSON
# =============================

with open(ROI_JSON) as f:
    rois = json.load(f)

print("Total ROI:", len(rois))

# =============================
# CROP LOOP
# =============================

for name, box in rois.items():

    img_path = IMG_DIR / name
    img = cv2.imread(str(img_path))

    if img is None:
        print("WARNING: Cannot read image:", name)
        continue

    H, W = img.shape[:2]

    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    cx = x + w // 2
    cy = y + h // 2

    new_w = int(w * SCALE)
    new_h = int(h * SCALE)

    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)

    x2 = min(W, cx + new_w // 2)
    y2 = min(H, cy + new_h // 2)

    crop = img[y1:y2, x1:x2]

    out_path = OUT_DIR / name
    cv2.imwrite(str(out_path), crop)

    print("Saved:", name)

print("\nExpanded ROI dataset created successfully")