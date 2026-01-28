import json
import numpy as np
from pathlib import Path

DATASET = "dataset_box_h2"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / DATASET
ROI_DIR = DATA / "images_roi"

K_full = np.load(DATA / "K_full.npy")

with open(ROI_DIR / "roi_meta.json") as f:
    roi_meta = json.load(f)

for name, m in roi_meta.items():

    x = m["x_offset"]
    y = m["y_offset"]

    K_roi = K_full.copy()
    K_roi[0,2] -= x
    K_roi[1,2] -= y

    out = ROI_DIR / f"{Path(name).stem}_K.npy"
    np.save(out, K_roi)

    print("Saved:", out.name)

print("DONE")
