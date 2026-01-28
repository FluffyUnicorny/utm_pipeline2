import numpy as np


# ============================
# Projection
# ============================

def project_points(points_3d, R, t, K):

    pts = (R @ points_3d.T + t).T
    pts = pts / pts[:, 2:3]

    proj = (K @ pts.T).T

    return proj[:, :2]


# ============================
# BA Outlier Filter
# ============================

def bundle_adjustment_filter(points_3d,
                             pts2d,
                             R, t,
                             K,
                             thresh=2.5):

    proj = project_points(points_3d, R, t, K)

    error = np.linalg.norm(proj - pts2d, axis=1)

    mask = error < thresh

    print(f"[BA] inliers: {np.sum(mask)} / {len(mask)}")

    return points_3d[mask], pts2d[mask], error
