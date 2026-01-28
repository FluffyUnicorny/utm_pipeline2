import cv2
import numpy as np
from pathlib import Path


# ===================================
# OpenCV Triangulation
# ===================================

def cv_triangulate(P1, P2, pts1, pts2):

    pts1 = pts1.T.astype(np.float64)
    pts2 = pts2.T.astype(np.float64)

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    pts3d = pts4d[:3] / pts4d[3]

    return pts3d.T


# ===================================
# MAIN
# ===================================

def run_sfm(dataset):

    ROOT = Path(__file__).resolve().parents[1]

    DATASET = ROOT / "data" / dataset
    IMG_DIR = DATASET / "images_roi"
    CALIB = ROOT / "calibration/params/K.npy"

    OUT = DATASET / "sfm"
    OUT.mkdir(exist_ok=True)

    K = np.load(CALIB)

    imgs = sorted(IMG_DIR.glob("*.jpg"))

    sift = cv2.SIFT_create(6000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # ===== GLOBAL MAP =====
    map_3d = []
    map_kp_idx = []   # keypoint index of LAST frame

    camera_centers = []

    R_prev = None
    t_prev = None

    kp_prev = None
    des_prev = None

    initialized = False


    for i, img_path in enumerate(imgs):

        print("\nFrame", i)

        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray, None)

        if i == 0:
            kp_prev = kp
            des_prev = des
            continue


        # ========== MATCH ==========
        matches = bf.knnMatch(des_prev, des, 2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print("Matches:", len(good))

        if len(good) < 100:
            kp_prev = kp
            des_prev = des
            continue


        pts1 = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])


        # ======================================
        # BOOTSTRAP
        # ======================================

        if not initialized:

            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                cv2.RANSAC, 0.999, 1.5
            )

            if E is None:
                continue

            inl = mask.ravel() == 1

            pts1 = pts1[inl]
            pts2 = pts2[inl]

            good = np.array(good)[inl]

            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

            # ---- OpenCV triangulation ----
            P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
            P2 = K @ np.hstack([R, t])

            pts3d = cv_triangulate(P1, P2, pts1, pts2)

            # cheirality filter (remove points behind camera)
            mask_z = pts3d[:,2] > 0
            pts3d = pts3d[mask_z]
            good = good[mask_z]

            if len(pts3d) < 150:
                continue

            map_3d = pts3d.tolist()

            map_kp_idx = [m.trainIdx for m in good]

            R_prev = R.copy()
            t_prev = t.copy()

            C = -R.T @ t
            camera_centers.append(C.reshape(3))

            initialized = True

            print("BOOTSTRAP OK")


        # ======================================
        # INCREMENTAL PNP
        # ======================================

        else:

            pts3d = []
            pts2d = []

            kp_map = dict(zip(map_kp_idx, map_3d))

            for m in good:
                if m.queryIdx in kp_map:
                    pts3d.append(kp_map[m.queryIdx])
                    pts2d.append(kp[m.trainIdx].pt)

            pts3d = np.float32(pts3d)
            pts2d = np.float32(pts2d)

            print("PnP correspondences:", len(pts3d))

            if len(pts3d) < 30:
                kp_prev = kp
                des_prev = des
                continue


            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d,
                K, None,
                reprojectionError=4,
                confidence=0.999,
                iterationsCount=300
            )

            if not ok or inliers is None:
                print("PnP failed")
                kp_prev = kp
                des_prev = des
                continue


            R, _ = cv2.Rodrigues(rvec)

            # ---- triangulate new points ----

            P_prev = K @ np.hstack([R_prev, t_prev])
            P_cur  = K @ np.hstack([R, tvec])

            new_pts3d = cv_triangulate(P_prev, P_cur, pts1, pts2)

            mask_z = new_pts3d[:,2] > 0
            new_pts3d = new_pts3d[mask_z]

            if len(new_pts3d) > 50:
                map_3d.extend(new_pts3d.tolist())
                map_kp_idx.extend([m.trainIdx for m in good])

            C = -R.T @ tvec
            camera_centers.append(C.reshape(3))

            R_prev = R.copy()
            t_prev = tvec.copy()

            print("Frame added")


        kp_prev = kp
        des_prev = des


    # =============================

    if not initialized or len(camera_centers) < 2:
        print("\n❌ SfM FAILED")
        return False


    np.save(OUT / "points3d.npy", np.array(map_3d))
    np.save(OUT / "camera_centers.npy", np.array(camera_centers))

    print("\n✅ SfM DONE")
    print("Points:", len(map_3d))
    print("Cameras:", len(camera_centers))

    return True