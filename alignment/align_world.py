import numpy as np
import pandas as pd
from pathlib import Path
import utm

# ==================================================
# Umeyama Alignment
# ==================================================

def align_umeyama(X, Y):

    n = X.shape[0]

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)

    Xc = X - muX
    Yc = Y - muY

    Sigma = (Yc.T @ Xc) / n

    U, S, Vt = np.linalg.svd(Sigma)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    var_X = np.mean(np.sum(Xc ** 2, axis=1))
    s = np.sum(S) / var_X

    t = muY - s * R @ muX

    return s, R, t


# ==================================================
# MAIN
# ==================================================

def run_umeyama_alignment(dataset):

    print("\n==============================")
    print("Running Umeyama Alignment (UTM)")
    print("==============================")

    ROOT = Path(__file__).resolve().parents[1]

    DATASET_DIR = ROOT / "data" / dataset
    SFM_DIR = DATASET_DIR / "sfm"
    GPS_FILE = DATASET_DIR / "camera_world.csv"

    cam_file = SFM_DIR / "camera_centers.npy"
    pts_file = SFM_DIR / "points3d.npy"

    sfm_cam = np.load(cam_file)
    sfm_pts = np.load(pts_file)

    gps = pd.read_csv(GPS_FILE)

    # ===============================
    # Convert CSV (Lon,Lat,H) → UTM
    # ===============================

    utm_list = []

    for _, row in gps.iterrows():

        lon = row["E"]   # YOUR CSV: E = Longitude
        lat = row["N"]   # YOUR CSV: N = Latitude
        h = row["H"]

        E_utm, N_utm, zone, letter = utm.from_latlon(lat, lon)

        utm_list.append([E_utm, N_utm, h])

    gps_xyz = np.array(utm_list)[:len(sfm_cam)]

    print("UTM Zone:", zone, letter)

    # ===============================
    # SfM Axis Mapping
    # ===============================

    sfm_cam_map = np.zeros_like(sfm_cam)

    sfm_cam_map[:, 0] = sfm_cam[:, 0]      # X → East
    sfm_cam_map[:, 1] = sfm_cam[:, 2]      # Z → North
    sfm_cam_map[:, 2] = -sfm_cam[:, 1]     # -Y → Up

    # ===============================
    # Align
    # ===============================

    s, R, t = align_umeyama(sfm_cam_map, gps_xyz)

    sfm_cam_aligned = (s * (R @ sfm_cam_map.T)).T + t

    cam_rmse = np.sqrt(
        np.mean(np.sum((sfm_cam_aligned - gps_xyz) ** 2, axis=1))
    )

    print("Camera RMSE (m):", cam_rmse)

    # ===============================
    # Transform SfM Points
    # ===============================

    sfm_pts_map = np.zeros_like(sfm_pts)

    sfm_pts_map[:, 0] = sfm_pts[:, 0]
    sfm_pts_map[:, 1] = sfm_pts[:, 2]
    sfm_pts_map[:, 2] = -sfm_pts[:, 1]

    pts_world = (s * (R @ sfm_pts_map.T)).T + t

    np.save(SFM_DIR / "points3d_world.npy", pts_world)

    return s, R, t, cam_rmse
