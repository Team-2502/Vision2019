import atexit

import pipeline
import cv2
import numpy as np

if __name__ == '__main__':
    # TODO: Store calib_fname in environment variable or something
    vision_pipeline = pipeline.VisionPipeline(False, calib_fname="ritik_webcam2.pickle")
    cap = cv2.VideoCapture(0)


    def exit():
        cv2.destroyAllWindows()
        cap.release()


    atexit.register(exit)

    while True:
        # Read image from camera
        _, image = cap.read()
        cv2.imshow("cam", image)

        # Invert image (assuming that tapes are black and background is white)
        # TODO: Remove inversion
        image = cv2.bitwise_not(image)
        cv2.imshow("img", image)

        # Process image
        contours, corners_subpixel, rvecs, tvecs, dist, euler_angles = vision_pipeline.process_image(image)

        contours_img = cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=3)
        cv2.imshow("contours", contours_img)

        center = np.array([
            [0, 0, 0],
        ], dtype=np.float32)

        imagePoints, jacobian = cv2.projectPoints(center, rvecs, tvecs, vision_pipeline.calibration_info.camera_matrix,
                                                  vision_pipeline.calibration_info.dist_coeffs)
        imagePoints = imagePoints.reshape(-1, 2)
        image = cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  rvecs, tvecs, 1)

        corner_img = cv2.circle(image, tuple(imagePoints[0].astype(np.int32)), 3, (66, 244, 113), thickness=3)

        for corner in corners_subpixel:
            corner_img = cv2.circle(corner_img, tuple(corner), 3, (255, 0, 0), thickness=3)

        cv2.imshow("corner_img", corner_img)

        # print(euler_angles)
        print("dist: {0:0.2f} | angle (rad): {1:0.2f}".format(dist, euler_angles[1]))
        cv2.waitKey(1000 // 30)
