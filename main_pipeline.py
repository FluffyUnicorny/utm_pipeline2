import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import sys
import utm

# ==================================================
# CONFIG
# ==================================================

DATASET = "dataset_box_h3"

# ---- GT OBJECT (Lat/Lon) ----
GT_LAT = 13.84682465
GT_LON = 100.5653152
GT_ALT = -5.409000397

E_gt, N_gt, zone, letter = utm.from_latlon(GT_LAT, GT_LON)
GT_OBJECT = np.array([E_gt, N_gt, GT_ALT])

print("GT OBJECT (UTM):", GT_OBJECT)
print("UTM Zone:", zone, letter)

ROOT = Path(__file__).resolve().parent

# ==================================================
# STEP 1 — Load Calibration
# ==================================================

print("\n===== STEP 1: Load Calibration =====")

calib_dir = ROOT / "calibration" / "params"

K = np.load(calib_dir / "K.npy")
dist = np.load(calib_dir / "dist.npy")

print("Loaded calibration from:", calib_dir)

# ==================================================
# CLEAN OLD OUTPUT
# ==================================================

sfm_dir = ROOT / "data" / DATASET / "sfm"

if sfm_dir.exists():
    shutil.rmtree(sfm_dir)

sfm_dir.mkdir(parents=True, exist_ok=True)

# ==================================================
# STEP 2 — SfM
# ==================================================

from sfm.sfm import run_sfm

print("\n===== STEP 2: SfM =====")

ok = run_sfm(DATASET)

if not ok:
    print("SfM FAILED")
    sys.exit(1)

# ==================================================
# STEP 3 — Alignment (SfM → UTM)
# ==================================================

from alignment.align_world import run_umeyama_alignment

print("\n===== STEP 3: Alignment (UTM) =====")

s, R, t, cam_rmse = run_umeyama_alignment(DATASET)

print("Alignment RMSE:", cam_rmse)

# ==================================================
# STEP 4 — Refinement (UTM SPACE)
# ==================================================

from refinement.refine_camera import robust_refine_camera

print("\n===== STEP 4: Refinement (UTM) =====")

sfm_cam = np.load(sfm_dir / "camera_centers.npy")

gps_file = ROOT / "data" / DATASET / "camera_world.csv"
gps = pd.read_csv(gps_file)

# ---- Convert CSV Lon/Lat → UTM ----
utm_list = []

for _, row in gps.iterrows():

    lon = row["E"]    # YOUR FILE: E = longitude
    lat = row["N"]    # YOUR FILE: N = latitude
    h = row["H"]

    E_utm, N_utm, zone, letter = utm.from_latlon(lat, lon)

    utm_list.append([E_utm, N_utm, h])

gps_xyz = np.array(utm_list)[:len(sfm_cam)]

print("Refinement UTM Zone:", zone, letter)

# ---- Refine transform in METRIC space ----

R_ref, t_ref = robust_refine_camera(
    s, R, t,
    sfm_cam,
    gps_xyz,
    fix_scale=True,
    anchor_translation=False
)

R = R_ref
t = t_ref

# ==================================================
# STEP 4.2 — SfM Point Cloud Cleaning
# ==================================================

print("\n===== STEP 4.2: Clean SfM Points =====")

sfm_pts = np.load(sfm_dir / "points3d.npy")

# ---------- Radius filter ----------

center = np.mean(sfm_pts, axis=0)

dist = np.linalg.norm(sfm_pts - center, axis=1)

r_th = np.percentile(dist, 90)   # keep closest 90%

mask_r = dist < r_th

# ---------- Depth filter ----------

z = np.abs(sfm_pts[:,2])

z_th = np.percentile(z, 90)

mask_z = z < z_th

# ---------- Combine ----------

mask = mask_r & mask_z

sfm_pts_clean = sfm_pts[mask]

print("SfM points before:", len(sfm_pts))
print("SfM points after :", len(sfm_pts_clean))

# overwrite
np.save(sfm_dir / "points3d_clean.npy", sfm_pts_clean)


# ==================================================
# STEP 4.5 — Transform SfM Points → UTM
# ==================================================

print("\n===== STEP 4.5: Generate World Points =====")

sfm_pts = np.load(sfm_dir / "points3d_clean.npy")

sfm_pts_map = np.zeros_like(sfm_pts)

# SfM → World axis mapping
sfm_pts_map[:, 0] = sfm_pts[:, 0]     # X → East
sfm_pts_map[:, 1] = sfm_pts[:, 2]     # Z → North
sfm_pts_map[:, 2] = -sfm_pts[:, 1]    # -Y → Up

pts_world = (s * (R @ sfm_pts_map.T)).T + t

# =====================================
# Weak Object Anchor (Soft Constraint)
# =====================================
'''
alpha = 0.4   # 0.1–0.3 recommended

pts_world = (1 - alpha) * pts_world + alpha * GT_OBJECT


print("Applied weak object anchoring, alpha =", alpha)
'''
np.save(sfm_dir / "points3d_world.npy", pts_world)

print("Saved points3d_world.npy")

# ==================================================
# STEP 5 — Object Evaluation
# ==================================================

from evaluation.object_error import run_object_position_eval

print("\n===== STEP 5: Evaluation =====")

centroid, error = run_object_position_eval(DATASET, GT_OBJECT)

# ==================================================
# SUMMARY
# ==================================================

print("\n===================================")
print("PIPELINE FINISHED (UTM WORLD FRAME)")
print("Dataset:", DATASET)
print("Scale factor:", s)
print("Alignment RMSE (m):", cam_rmse)
print("Object position error (m):", error)
print("===================================")
