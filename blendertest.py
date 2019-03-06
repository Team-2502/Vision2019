import cv2
import numpy as np

import constants
import pipeline

vision_pipeline = pipeline.VisionPipeline(True)
errors_squared=[]

for angle in range(0, 360):
    angle /= 2
    # Read image from camera
    a = constants.AUTOGEN_FORMAT_STRING.format(angle)
    image = cv2.imread(a)
    cv2.imshow("cam", image)
    # Invert image (assuming that tapes are black and background is white)
    cv2.imshow("img", image)

    # Process image
    pipeline_result = vision_pipeline.process_image(image)
    contours = pipeline_result.contours
    pose_estimation = pipeline_result.pose_estimation
    tvecs = None if pose_estimation is None else (pose_estimation.left_tvec + pose_estimation.right_tvec) / 2
    rvecs = None if pose_estimation is None else (pose_estimation.left_rvec)
    euler_angles = None if pose_estimation is None else (pipeline_result.euler_angles.left)
    dist = None if pose_estimation is None else np.linalg.norm(tvecs)

    contours_img = cv2.drawContours(image, contours, -1, (255, 255, 0), thickness=3)
    contours_img = cv2.drawContours(image, contours[:1], -1, (255, 0, 0), thickness=3)
    contours_img = cv2.drawContours(image, pipeline_result.trash, -1, (0, 0, 255), thickness=3)
    cv2.imshow("contours", contours_img)

    bitmask = pipeline_result.bitmask
    blank = np.zeros(bitmask.shape).astype(np.uint8)


    cv2.drawContours(blank, contours[:1], -1, (255,), thickness=cv2.FILLED)

    # kernel = np.ones((5, 5), np.float32) / 1
    # blank = cv2.filter2D(blank, -1, kernel)

    dst = cv2.goodFeaturesToTrack(image=blank, maxCorners=5, qualityLevel=0.16, minDistance=15)
    print("dst", dst)

    blank = cv2.cvtColor(blank.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if dst is not None:
        for corner in dst.reshape(-1, 2):
            cv2.circle(blank, tuple(corner.astype(np.int32)), 3, (0, 255, 255), thickness=1)
    cv2.imshow('blank', blank)


    center = np.array([
        [0, 0, 0],
    ], dtype=np.float32)

    if pose_estimation is not None:
        print(tvecs.shape)
        image = cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  pose_estimation.left_rvec, tvecs, 1)
        image = cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  pose_estimation.left_rvec, pose_estimation.left_tvec, 1)
        image = cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  pose_estimation.right_rvec, pose_estimation.right_tvec, 1)
        for corner in pipeline_result.corners[0]:
            image = cv2.circle(image, tuple(corner[0].astype(np.int32)), 3, (0, 255, 0), thickness=1)
        for corner in pipeline_result.corners[1]:
            image = cv2.circle(image, tuple(corner[0].astype(np.int32)), 3, (0, 255, 0), thickness=1)

        cv2.imshow("corner_img", image)
        print("estimated angle", np.degrees(euler_angles[1]))
        print("actual", 90 - angle)
        error = np.degrees(euler_angles[1]) + angle - 90
        errors_squared.append(np.math.fabs(error * error))
        print('error', error)

    cv2.waitKey(0)

print("rmse", np.math.sqrt(sum(errors_squared) / len(errors_squared)))
