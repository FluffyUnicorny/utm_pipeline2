import numpy as np
from pathlib import Path

def load_camera_intrinsic(dataset_name):

    ROOT = Path(__file__).resolve().parents[1]   # project_new
    CALIB_DIR = ROOT / "data" / dataset_name / "calib"

    K_path = CALIB_DIR / "K.npy"
    dist_path = CALIB_DIR / "dist.npy"

    if not K_path.exists() or not dist_path.exists():
        raise FileNotFoundError(f"Calibration not found for dataset: {dataset_name}")

    K = np.load(K_path)
    dist = np.load(dist_path)

    return K, dist
