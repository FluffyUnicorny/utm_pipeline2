import cv2
import numpy as np
from pathlib import Path

def run_sfm_reconstruction(dataset_name, K):

    ROOT = Path(__file__).resolve().parents[1]
    DATASET_DIR = ROOT / "data" / dataset_name

    IMG_DIR = DATASET_DIR / "images_roi"
    SFM_OUT = DATASET_DIR / "sfm"

    SFM_OUT.mkdir(parents=True, exist_ok=True)

    images = sorted(IMG_DIR.glob("*.png"))

    if len(images) < 2:
        raise RuntimeError("Need at least 2 images")

    clahe = cv2.createCLAHE(2.0, (8, 8))
    sift = cv2.SIFT_create(nfeatures=8000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    all_points_3d = []

    for i in range(len(images) - 1):

        img1 = cv2.imread(str(images[i]))
        img2 = cv2.imread(str(images[i + 1]))

        if img1 is None or img2 is None:
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            continue

        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.85 * n.distance]

        if len(good) < 30:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

        if E is None:
            continue

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        all_points_3d.append(pts3d)

    if len(all_points_3d) == 0:
        raise RuntimeError("No 3D points reconstructed")

    all_points_3d = np.vstack(all_points_3d)

    np.save(SFM_OUT / "points3d.npy", all_points_3d)

    print("SfM reconstruction finished")
    print("Points:", all_points_3d.shape)

    return all_points_3d
