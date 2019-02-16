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
parser.add_argument("--invert", help="invert camera image", action="store_true")
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

    use_gui = not args.yes_gui

    cap = cv2.VideoCapture(constants.CAMERA_ID)
#    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=19")
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 20)

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

            if sockets_on:
                try:
                    conn.sendall(generate_socket_msg(tvecs[0][0], tvecs[2][0],  euler_angles[1]))
                except (ConnectionResetError, BrokenPipeError):
                    s.listen(100)
                    print("Waiting for socket connection on port {} . . .".format(constants.PORT))
                    conn, addr = s.accept()
                    print(addr)
            print("f")
        elif rvecs is None and sockets_on:
            try:
                conn.sendall(generate_socket_msg(-9001, -9001, -9001))

            except (ConnectionResetError, BrokenPipeError):
                s.listen(100)
                print("Waiting for socket connection on port {} . . .".format(constants.PORT))
                conn, addr = s.accept()
                print(addr)
        print("loop")
        print("fps: ", (1 / (time.time() - start)))
        if use_gui:
            cv2.waitKey(1000 // 30)
