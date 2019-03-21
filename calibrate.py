import numpy as np
import cv2
import pipeline
import argparse
import logging
import constants

parser = argparse.ArgumentParser()
parser.add_argument("--fisheye", help="Indicates that the camera has a fisheye lens", action="store_true")
parser.add_argument("--fast", help="Continuously take pictures every second without waiting for the spacebar to be pressed", action="store_true")
parser.add_argument("--num_images", help="The number of calibration images to take", action="store", default=13, type=int)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

num_left = args.num_images
if args.fast:
    logger.info("going fast!")

dim = (9, 6)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((dim[0] * dim[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dim[0], 0:dim[1]].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture(constants.CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
logging.info("Initialized camera capture stream")
logger.info("Press spacebar to take an image")
while len(imgpoints) < args.num_images:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    cv2.imshow("cam", img)
    key = cv2.waitKey(1000 // 30)
    if args.fast or ' ' == chr(key & 255):
        ret, corners = cv2.findChessboardCorners(gray, dim, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, dim, corners2, ret)
            cv2.imshow('img', img)
            num_left -= 1
            logger.info("Stored a point. {} of {} left".format(num_left, args.num_images))
            if args.fast:
                cv2.waitKey(1000)

if args.fisheye:
    fisheye_objp = [x.reshape(1, -1, 3) for x in objpoints]
    fisheye_imjp = [x.reshape(-1, 1, 2) for x in imgpoints]
    objpoints = np.array(objpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(fisheye_objp, fisheye_imjp, gray.shape[::-1], None, None)
else:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
pipeline.save_calibration_results(mtx, dist, rvecs, tvecs, args.fisheye, constants.CALIBRATION_FILE_LOCATION)

cv2.destroyAllWindows()

logger.info("Finished calibrating")
logger.debug("Camera matrix: ")
logger.debug(str(mtx))
logger.debug("Distortion coeffs:")
logger.debug(str(dist))
