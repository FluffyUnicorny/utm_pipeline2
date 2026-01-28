import cv2
from pathlib import Path
import json

# =============================
# PATH CONFIG
# =============================

DATASET = "dataset_umb_h3"

IMG_DIR = Path(f"data/{DATASET}/images")
SAVE_PATH = Path(f"data/{DATASET}/roi_boxes.json")

image_paths = sorted(list(IMG_DIR.glob("*.jpg")))

roi_data = {}

print("Total images:", len(image_paths))

for img_path in image_paths:

    print("\nLoading:", img_path)

    img = cv2.imread(str(img_path))

    if img is None:
        print("ERROR: Cannot load image")
        continue

    h, w, _ = img.shape
    print("Image size:", w, "x", h)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

    print("Draw ROI then press ENTER or SPACE (ESC to skip)")
    bbox = cv2.selectROI("ROI", img, False, False)

    cv2.destroyAllWindows()

    x, y, bw, bh = bbox

    # ถ้ากด ESC ไม่เลือกอะไร
    if bw == 0 or bh == 0:
        print("Skipped:", img_path.name)
        continue

    roi_data[img_path.name] = {
        "x": int(x),
        "y": int(y),
        "w": int(bw),
        "h": int(bh)
    }

    print("Saved ROI:", roi_data[img_path.name])

# =============================
# SAVE ROI JSON
# =============================

with open(SAVE_PATH, "w") as f:
    json.dump(roi_data, f, indent=4)

print("\nROI annotation saved to:", SAVE_PATH)
