import numpy as np


def robust_refine_camera(
    s, R, t,
    sfm_cam,
    gps_xyz,
    fix_scale=True,
    anchor_translation=True
):

    # Map SfM axes
    sfm_map = np.zeros_like(sfm_cam)
    sfm_map[:, 0] = sfm_cam[:, 0]
    sfm_map[:, 1] = sfm_cam[:, 2]
    sfm_map[:, 2] = -sfm_cam[:, 1]

    # Initial transform
    pred = (s * (R @ sfm_map.T)).T + t

    # Residuals
    diff = gps_xyz - pred

    # Remove outliers (median based)
    dist = np.linalg.norm(diff, axis=1)
    med = np.median(dist)

    mask = dist < 2.5 * med

    pred_f = pred[mask]
    gps_f = gps_xyz[mask]

    # Translation refinement
    delta_t = np.mean(gps_f - pred_f, axis=0)

    t_new = t + delta_t

    # Optional scale correction (usually OFF)
    if not fix_scale:
        num = np.sum((gps_f - t_new) * (R @ sfm_map[mask].T).T)
        den = np.sum((R @ sfm_map[mask].T).T ** 2)
        s = num / den

    # Anchor translation to camera centroid (very important)
    if anchor_translation:
        pred_new = (s * (R @ sfm_map.T)).T + t_new
        shift = np.mean(gps_xyz, axis=0) - np.mean(pred_new, axis=0)
        t_new += shift

    return R, t_new
