import atexit

import pipeline
import cv2
import numpy as np
import socket
import argparse
import constants


def generate_socket_msg(x, y, angle):
    return bytes(str(x), 'utf-8') + b',' + \
           bytes(str(y), 'utf-8') + b',' + \
           bytes(str(angle), 'utf-8') + b'\n'


if __name__ == '__main__':
    sockets_on = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_sockets", help="Does not attempt to transmit data via sockets", action="store_true")
    args = parser.parse_args()
    if args.no_sockets:
        sockets_on = False

    cap = cv2.VideoCapture(constants.CAMERA_ID)
    vision_pipeline = pipeline.VisionPipeline(False, calib_fname=constants.CALIBRATION_FILE_LOCATION)

    if sockets_on:
        HOST, PORT = "", constants.PORT
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))


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

        if rvecs is not None:
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

            if sockets_on:
                try:
                    conn.sendall(generate_socket_msg(tvecs[0][0], tvecs[2][0],  euler_angles[1]))
                except (ConnectionResetError, BrokenPipeError):
                    s.listen(100)
                    print("Waiting for socket connection on port {} . . .".format(constants.PORT))
                    conn, addr = s.accept()
                    print(addr)

        elif rvecs is None and sockets_on:
            try:
                conn.sendall(generate_socket_msg(-9001, -9001, -9001))

            except (ConnectionResetError, BrokenPipeError):
                s.listen(100)
                print("Waiting for socket connection on port {} . . .".format(constants.PORT))
                conn, addr = s.accept()
                print(addr)

        cv2.waitKey(1000 // 30)
