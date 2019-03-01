import cv2
import time

import constants
from pipeline import load_calibration_results

cap = cv2.VideoCapture(constants.CAMERA_ID)
calib_info = load_calibration_results("testtest.pickle")

print(calib_info.dist_coeffs)
while True:
    start = time.time()
    ret, frame = cap.read()
    undistort = cv2.fisheye.undistortImage(frame, calib_info.camera_matrix, calib_info.dist_coeffs)
    cv2.imshow("orig", frame)
    cv2.imshow("new", undistort)
    cv2.imshow("avg", (frame + undistort) / (255 * 2))
    cv2.waitKey(1000 // 30)

