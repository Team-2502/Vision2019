import atexit
import time
import pipeline
import cv2
import numpy as np
import socket
import argparse
import constants
import os

parser = argparse.ArgumentParser()
parser.add_argument("--no_sockets", help="Does not attempt to transmit data via sockets", action="store_true")
parser.add_argument("--yes_gui", help="invert camera image", action="store_false")
args = parser.parse_args()


def generate_socket_msg(x, y, angle):
    return bytes(str(x), 'utf-8') + b',' + \
           bytes(str(y), 'utf-8') + b',' + \
           bytes(str(angle), 'utf-8') + b'\n'


if __name__ == '__main__':
    sockets_on = True

    if args.no_sockets:
        sockets_on = False

    send_sock_msg = None

    use_gui = not args.yes_gui

    cap = cv2.VideoCapture(constants.CAMERA_ID)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(constants.IM_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(constants.IM_HEIGHT))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl=exposure_auto=1".format(constants.CAMERA_ID))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl=exposure_absolute=19".format(constants.CAMERA_ID))

    vision_pipeline = pipeline.VisionPipeline(False, calib_fname=constants.CALIBRATION_FILE_LOCATION)

    if sockets_on:
        HOST, PORT = "", constants.PORT
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        def send_sock_msg(x, y, angle):
            conn.sendall(generate_socket_msg(x, y, angle))
    else:
        def send_sock_msg(x, y, angle):
            print(f"x: {x:0.2f}, y: {y:0.2f}, theta: {angle:0.2f}")

    def exit():
        cv2.destroyAllWindows()
        cap.release()
        if sockets_on:
            s.close()


    atexit.register(exit)
    if sockets_on:
        s.listen(100)
        print("Waiting for socket connection on port {} . . .".format(constants.PORT))
        conn, addr = s.accept()
        print(addr)

    while True:
        # Read image from camera
        _, image = cap.read()
        start = time.time()

        # Process image
        pipeline_result = vision_pipeline.process_image(image)
        contours = pipeline_result.contours
        pose_estimation = pipeline_result.pose_estimation

        tvecs = None
        rvecs = None
        euler_angles = None
        if pose_estimation is not None:
            tvecs =  (pose_estimation.left_tvec + pose_estimation.right_tvec) / 2  # Between 2 tapes
            rvecs = pose_estimation.left_rvec
            euler_angles = (pipeline_result.euler_angles.left + pipeline_result.euler_angles.right) / 2

            try:
                send_sock_msg(tvecs[0][0], tvecs[2][0], euler_angles[1])
            except (ConnectionResetError, BrokenPipeError):
                s.listen(100)
                print("Waiting for socket connection on port {} . . .".format(constants.PORT))
                conn, addr = s.accept()
                print(addr)
            print("f")
        elif rvecs is None and sockets_on:
            try:
                send_sock_msg(-9001, -9001, -9001)

            except (ConnectionResetError, BrokenPipeError):
                s.listen(100)
                print("Waiting for socket connection on port {} . . .".format(constants.PORT))
                conn, addr = s.accept()
                print(addr)

        print("loop")
        print("fps: ", (1 / (time.time() - start)))

        if use_gui:
            # Show pre-processing/localization images
            cv2.imshow("cam", image)
            cv2.imshow("img", image)
            cv2.imshow("bitmask", pipeline_result.bitmask)

            # Draw contours. Incorrect contours are red, primary contour is blue,
            cv2.drawContours(image, contours, -1, (255, 255, 0), thickness=3)
            cv2.drawContours(image, contours[:1], -1, (255, 0, 0), thickness=3)
            cv2.drawContours(image, pipeline_result.trash, -1, (0, 0, 255), thickness=2)
            cv2.imshow("contours", image)

            if pose_estimation is not None:
                cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  rvecs, tvecs, 1)
                cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  pose_estimation.left_rvec, tvecs, 1)
                cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  pose_estimation.left_rvec, pose_estimation.left_tvec, 1)
                cv2.drawFrameAxes(image, vision_pipeline.calibration_info.camera_matrix,
                                  vision_pipeline.calibration_info.dist_coeffs,
                                  pose_estimation.right_rvec, pose_estimation.right_tvec, 1)

            if pipeline_result.corners is not None and len(pipeline_result.corners) > 0:
                for corner in pipeline_result.corners[0]:
                    corner_img = cv2.circle(image, tuple(corner[0].astype(np.int32)), 3, (255, 0, 0), thickness=3)

            cv2.imshow("corner_img", image)

            cv2.waitKey(1000 // 30)
