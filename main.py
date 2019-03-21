import atexit
import time
import pipeline
import cv2
import numpy as np
import argparse
import constants
from networktables import NetworkTables
import os

parser = argparse.ArgumentParser()
parser.add_argument("--no_sockets", help="Does not attempt to transmit data via network tables", action="store_true")
parser.add_argument("--invert", help="invert camera image", action="store_true")
parser.add_argument("--yes_gui", help="invert camera image", action="store_false")
args = parser.parse_args()


if __name__ == '__main__':
    sockets_on = True

    if args.no_sockets:
        sockets_on = False

    use_gui = not args.yes_gui

    cap = cv2.VideoCapture(constants.CAMERA_ID)
#    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=19")
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 20)

    vision_pipeline = pipeline.VisionPipeline(False, calib_fname=constants.CALIBRATION_FILE_LOCATION)

    if sockets_on:
        NetworkTables.initialize(server='10.25.2.2')
        vision_table = NetworkTables.getTable('Vision2019')


    def exit():
        cv2.destroyAllWindows()
        cap.release()

    atexit.register(exit)

    # TODO Wait for network tables to initalize

    while True:
        # Read image from camera
        _, image = cap.read()
        start = time.time()
        if use_gui:
            cv2.imshow("cam", image)
        print("a")
        # Invert image (assuming that tapes are black and background is white)
        if args.invert:
            image = cv2.bitwise_not(image)
        if use_gui:
            cv2.imshow("img", image)

        # Process image
        contours, corners_subpixel, rvecs, tvecs, dist, euler_angles = vision_pipeline.process_image(image)

        if use_gui:
            print("b")
            
            contours_img = cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=3)
            contours_img = cv2.drawContours(image, contours[:1], -1, (255, 0, 0), thickness=3)
            cv2.imshow("contours", contours_img)

        center = np.array([
            [0, 0, 0],
        ], dtype=np.float32)
        print("c")
        if rvecs is not None:
            if use_gui:
                imagePoints, jacobian = cv2.projectPoints(center, rvecs, tvecs, vision_pipeline.calibration_info.camera_matrix,
                                                      vision_pipeline.calibration_info.dist_coeffs)
                imagePoints = imagePoints.reshape(-1, 2)
                image = cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                      vision_pipeline.calibration_info.dist_coeffs,
                                      rvecs, tvecs, 1)
                print("d")
                corner_img = cv2.circle(image, tuple(imagePoints[0].astype(np.int32)), 3, (66, 244, 113), thickness=3)

                for corner in corners_subpixel:
                    corner_img = cv2.circle(corner_img, tuple(corner), 3, (255, 0, 0), thickness=3)

                cv2.imshow("corner_img", corner_img)
                print("e")
            # print(euler_angles)
            print("dist: {0:0.2f} | angle (rad): {1:0.2f}".format(dist, euler_angles[1]))

            # TODO Just use a number array
            if sockets_on:
                vision_table.putNumber("tvecs1", tvecs[0][0])
                vision_table.putNumber("tvecs2", tvecs[2][0])
                vision_table.putNumber("angle", euler_angles[1])

        elif rvecs is None and sockets_on:
            vision_table.putNumber("tvecs1", -9001)
            vision_table.putNumber("tvecs2", -9001)
            vision_table.putNumber("angle", -9001)

        print("loop")
        print("fps: ", (1 / (time.time() - start)))
        if use_gui:
            cv2.waitKey(1000 // 30)
