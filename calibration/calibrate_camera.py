import cv2
import numpy as np
import glob
from pathlib import Path

# =========================
# CONFIG
# =========================

CHESSBOARD_SIZE = (8, 5)   # inner corners (BoofCV 9x6 board)
SQUARE_SIZE = 0.025       # meters (25 mm)

IMAGE_DIR = "data/calibration_images"
SAVE_DIR = Path("calibration/params")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PREPARE OBJECT POINTS
# =========================

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                       0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# =========================
# LOAD IMAGES
# =========================

images = glob.glob(f"{IMAGE_DIR}/*.jpg")

if len(images) == 0:
    raise ValueError("No calibration images found!")

print("Found images:", len(images))

# =========================
# FIND CHESSBOARD CORNERS
# =========================

valid_count = 0

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        valid_count += 1

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS +
             cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(150)

    else:
        print("Corner not found:", fname)

cv2.destroyAllWindows()

print("Valid calibration images:", valid_count)

if valid_count < 10:
    print("WARNING: Low number of valid images. Recommend >15.")

# =========================
# CAMERA CALIBRATION
# =========================

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n========== CALIBRATION RESULT ==========")
print("RMS reprojection error:", ret)
print("\nCamera matrix K:\n", K)
print("\nDistortion coefficients:\n", dist)

# =========================
# COMPUTE MEAN REPROJECTION ERROR
# =========================

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K, dist
    )

    error = cv2.norm(imgpoints[i], imgpoints2,
                     cv2.NORM_L2) / len(imgpoints2)

    mean_error += error

mean_error /= len(objpoints)

print("\nMean reprojection error (pixel):", mean_error)

# =========================
# SAVE PARAMETERS
# =========================

np.save(SAVE_DIR / "K.npy", K)
np.save(SAVE_DIR / "dist.npy", dist)

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]

np.savetxt(
    SAVE_DIR / "intrinsics.txt",
    np.array([fx, fy, cx, cy]),
    header="fx fy cx cy"
)

# =========================
# SAVE UNDISTORTED SAMPLE
# =========================

test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]

newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1)

undistorted = cv2.undistort(test_img, K, dist, None, newK)

cv2.imwrite(str(SAVE_DIR / "undistorted_sample.jpg"), undistorted)

print("\nSaved files:")
print(" - K.npy")
print(" - dist.npy")
print(" - intrinsics.txt")
print(" - undistorted_sample.jpg")

print("\nCalibration completed successfully")