import cv2
import time

import constants

cap = cv2.VideoCapture(constants.CAMERA_ID)

while True:
    start = time.time()
    ret, frame = cap.read()
    print("fps:", 1/(time.time() - start))

