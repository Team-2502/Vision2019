import atexit

import pipeline
import cv2
import numpy as np

if __name__ == '__main__':
    vision_pipeline = pipeline.VisionPipeline(False, calib_fname="ritik_webcam.pickle")
    cap = cv2.VideoCapture(0)


    def exit():
        cv2.destroyAllWindows()
        cap.release()


    atexit.register(exit)

    while True:
        _, image = cap.read()
        cv2.imshow("cam", image)
        all_white = np.ones(image.shape) * 255
        cv2.imshow("white", all_white)

        image = cv2.bitwise_not(image)
        cv2.imwrite("cam_image.png", image)

        cv2.imshow("img", image)

        contours, corners_subpixel, rvecs, tvecs  = vision_pipeline.process_image(image)

        contours_img = cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=3)
        cv2.imshow("contours", contours_img)

        corner_img = image
        for corner in corners_subpixel:
            corner_img = cv2.circle(corner_img, tuple(corner), 3, (0, 255, 0), thickness=3)

        cv2.imshow("corner_img", corner_img)

        cv2.waitKey(1000 // 30)
