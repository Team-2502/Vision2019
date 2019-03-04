import cv2
import numpy as np

import constants
import pipeline

vision_pipeline = pipeline.VisionPipeline(True)

for angle in range(1, 180, 4):
    # Read image from camera
    a = constants.AUTOGEN_FORMAT_STRING.format(angle)
    print(a)
    image = cv2.imread(a)
    cv2.imshow("cam", image)
    print("a")
    # Invert image (assuming that tapes are black and background is white)
    cv2.imshow("img", image)

    # Process image
    pipeline_result = vision_pipeline.process_image(image)
    contours = pipeline_result.contours
    pose_estimation = pipeline_result.pose_estimation
    tvecs = None if pose_estimation is None else (pose_estimation.left_tvec + pose_estimation.right_tvec) / 2
    rvecs =  None if pose_estimation is None else (pose_estimation.left_rvec + pose_estimation.right_rvec) / 2
    euler_angles = None if pose_estimation is None else (pipeline_result.euler_angles.left + pipeline_result.euler_angles.right) / 2
    dist = None if pose_estimation is None else np.linalg.norm(tvecs)

    contours_img = cv2.drawContours(image, contours, -1, (255, 255, 0), thickness=3)
    contours_img = cv2.drawContours(image, contours[:1], -1, (255, 0, 0), thickness=3)
    contours_img = cv2.drawContours(image, pipeline_result.trash, -1, (0, 0, 255), thickness=3)
    cv2.imshow("contours", contours_img)

    center = np.array([
        [0, 0, 0],
    ], dtype=np.float32)
    print("c")
    print(pose_estimation)
    if pose_estimation is not None:
        for corner in pipeline_result.corners[0]:
            corner_img = cv2.circle(image, tuple(corner[0].astype(np.int32)), 3, (255, 0, 0), thickness=3)

        cv2.imshow("corner_img", corner_img)
        print("dist: {0:0.2f} | angle (rad): {1:0.2f}".format(dist, euler_angles[1]))

    cv2.waitKey(0)
