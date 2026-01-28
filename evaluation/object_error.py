import numpy as np
from pathlib import Path

# ==================================================
# Object Position Evaluation (UTM World Frame)
# ==================================================

def run_object_position_eval(dataset, GT_OBJECT):

    print("\n==============================")
    print("Object Position Evaluation")
    print("Dataset:", dataset)
    print("==============================")

    ROOT = Path(__file__).resolve().parents[1]

    sfm_dir = ROOT / "data" / dataset / "sfm"

    pts_file = sfm_dir / "points3d_world.npy"

    if not pts_file.exists():
        raise FileNotFoundError("points3d_world.npy not found")

    pts_world = np.load(pts_file)

    # ===============================
    # Simple centroid (mean)
    # ===============================

    centroid = np.mean(pts_world, axis=0)

    # ===============================
    # Position Error
    # ===============================

    error = np.linalg.norm(centroid - GT_OBJECT)

    print("\nEstimated object centroid (UTM):")
    print("E:", centroid[0])
    print("N:", centroid[1])
    print("H:", centroid[2])

    print("\nGT object (UTM):")
    print("E:", GT_OBJECT[0])
    print("N:", GT_OBJECT[1])
    print("H:", GT_OBJECT[2])

    print("\nPosition error (meters):", error)

    return centroid, error
