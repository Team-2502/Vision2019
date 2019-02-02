import math

import cv2
import numpy as np
import scipy.ndimage
import scipy.optimize
import os
import constants
import pipeline

AUTOGEN_IMAGE_FOLDER = "./autogen_images"

# REAL_HEIGHT_FT = (5.5 * math.sin(math.radians(75.5)) + 2 * math.sin(math.radians(14.5))) / 12.0

VISION_TAPE_LENGTH_IN = 5.5
VISION_TAPE_LENGTH_FT = VISION_TAPE_LENGTH_IN / 12

VISION_TAPE_WIDTH_IN = 2
VISION_TAPE_WIDTH_FT = 2 / 12

VISION_TAPE_MIN_SEPARATION_IN = 8
VISION_TAPE_MIN_SEPARATION_FT = VISION_TAPE_MIN_SEPARATION_IN / 12

VISION_TAPE_ANGLE_FROM_VERT_DEG = 14.5
VISION_TAPE_ANGLE_FROM_VERT_RAD = math.radians(VISION_TAPE_ANGLE_FROM_VERT_DEG)

VISION_TAPE_ANGLE_FROM_HORIZONTAL_DEG = 90 - VISION_TAPE_ANGLE_FROM_VERT_DEG
VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD = math.radians(VISION_TAPE_ANGLE_FROM_HORIZONTAL_DEG)

REAL_HEIGHT_FT = VISION_TAPE_LENGTH_FT * math.sin(VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD)
TOP_WIDTH_FT = (2 * VISION_TAPE_WIDTH_FT * math.sin(VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD) + VISION_TAPE_MIN_SEPARATION_FT)

BOTTOM_WIDTH_FT = (2 * VISION_TAPE_LENGTH_FT * math.sin(VISION_TAPE_ANGLE_FROM_VERT_RAD) + VISION_TAPE_MIN_SEPARATION_FT)

MID_WIDTH_FT = (TOP_WIDTH_FT + BOTTOM_WIDTH_FT) / 2

# Distance from top left corner to bottom right corner
DIAG_WIDTH_FT = np.linalg.norm(
    np.array([BOTTOM_WIDTH_FT, 0]) +
    np.array([VISION_TAPE_WIDTH_FT * math.cos(VISION_TAPE_ANGLE_FROM_VERT_RAD), VISION_TAPE_WIDTH_FT * math.sin(VISION_TAPE_ANGLE_FROM_VERT_RAD)]) +
    np.array([-VISION_TAPE_LENGTH_FT * math.sin(math.radians(VISION_TAPE_ANGLE_FROM_VERT_DEG)), VISION_TAPE_LENGTH_FT * math.cos(VISION_TAPE_ANGLE_FROM_VERT_RAD)])
)


# The focal length of the camera in Blender (used to create autogen_images).
# Calculated with https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
BLENDER_FLEN_PIXELS = 800.6028523694872235460223543952119609884957052979378391303  # (211 * 1.83283) / REAL_HEIGHT_FT

BLENDER_FLEN_MM = 35

BLENDER_SENSOR_WIDTH_MM = 32
BLENDER_SENSOR_WIDTH_FT = BLENDER_SENSOR_WIDTH_MM * 0.00328084

BLENDER_SENSOR_HEIGHT_MM = 18
BLENDER_SENSOR_HEIGHT_FT = BLENDER_SENSOR_HEIGHT_MM * 0.00328084

BLENDER_FLEN_FT = BLENDER_FLEN_MM * 0.00328084

BLENDER_PIXEL_WIDTH = 1440
BLENDER_PIXEL_HEIGHT = 720


def get_dist_ft(height_px):
    """
    
    :param height_px: pixel distance from the lowest to the highest point on the vision target
    :return: the distance in feet
    """
    return (REAL_HEIGHT_FT * BLENDER_FLEN_PIXELS) / height_px


def estimate_angle_deg(left_height_px, right_height_px):
    if math.fabs(left_height_px - right_height_px) <= 1e-6:
        return 90.0

    left_height_ft = get_dist_ft(left_height_px)
    right_height_ft = get_dist_ft(right_height_px)

    c = MID_WIDTH_FT
    a = min(left_height_ft, right_height_ft)
    b = max(left_height_ft, right_height_ft)

    cos_angle = (b * b + c * c - a * a) / (2 * b * c) # TODO: figure out derivation which I need a notebook for
    angle = math.degrees(math.acos(cos_angle))
    return angle


def harris_test():
    """
    harris is an algorithm to detect corners. However, we decided to use contours, instead.
    :return:
    """
    base_image = cv2.imread("./2019_vision_sample.png")  # TODO: fix

    base_img_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(src=base_img_gray, blockSize=2, ksize=3, k=0.15)
    harrisy = np.zeros(dst.shape)
    print(harrisy.shape)

    print(np.transpose(np.where(dst > 0.01 * dst.max())))

    base_image[dst > 0.01 * dst.max()] = [0, 0, 255]

    harrisy[dst > 0.01 * dst.max()] = 255

    # for coord in np.transpose(np.where(dst > 0.01 * dst.max())):
    #     print(coord)
    # base_image = cv2.circle(base_image, tuple(np.flip(coord)), 3, (0,0,255))

    cv2.imshow('dst', base_image)
    cv2.imshow("harris", harrisy)
    cv2.waitKey()

    print(dst.max())


def harris_test2():
    """
    harris is an algorithm to detect corners. However, we decided to use contours, instead.
    :return:
    """
    format_string = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")

    angle = 30.0
    delta = 0.5

    print("top, bottom, angle")
    while angle <= 150.0:
        image = cv2.imread(format_string.format(angle))

        tops, bottoms = get_contours(image)
        for a, b in (tops,):
            image = cv2.line(image, a, b, (255, 0, 0))
            dy = -b[1] + a[1]
            dx = b[0] - a[0]
            deg = np.math.degrees(np.math.atan2(dy, dx))
            if deg > 90:
                deg -= 180
            print(deg, end=", ")
            # print((- b[1] + a[1])/(b[0] - a[0]), end=", ")

        for c, d in (bottoms,):
            image = cv2.line(image, c, d, (255, 0, 0))

            print((- b[1] + a[1]) / (b[0] - a[0]), end=", ")

        # cv2.imshow("a", image)
        # cv2.waitKey()
        print(angle)
        angle += delta


def get_contours(image):
    try:
        bitmask = cv2.inRange(image, (30, 30, 30), (255, 255, 255))
    except cv2.error:
        bitmask = image

    # harris = cv2.cornerHarris(bitmask, 2, 3, 0.04)
    # transpose = np.transpose(np.where(harris > 0.01 * harris.max()))

    contours, hierarchy = cv2.findContours(bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=3)
    # cv2.imshow("b", image)
    # cv2.waitKey()

    # cv2.imshow("a", bitmask)
    # cv2.waitKey()
    # print(contours[0].shape)
    # print(len(contours))
    for i in range(len(contours)):
        contours[i] = contours[i].reshape((contours[i].shape[0], 2))
    # print(contours[0].shape)
    return contours


def get_landmark_points(image):
    """
    Take all contours and map to both their lowest and heightest (y-value) points

    Optimally will be four points
    :param image:
    :return:
    """
    contours = get_contours(image)
    tops = [min(contour, key=lambda x: x[1]) for contour in contours]
    bots = [max(contour, key=lambda x: x[1]) for contour in contours]

    tops.sort(key= lambda x: x[0])
    bots.sort(key= lambda x: x[0])

    return tops + bots


def calculate_angle(image):
    contours = get_contours(image)

    tops = [min(cnt, key=lambda c: c[1]) for cnt in contours]
    bottoms = [max(cnt, key=lambda c: c[1]) for cnt in contours]

    # image = cv2.drawContours(image, contours, -1, (0, 255, 0))
    #
    # cv2.imshow("a", image)
    # cv2.waitKey()

    tops.sort(key=lambda x: x[0])
    bottoms.sort(key=lambda x: x[0])

    left_height = bottoms[0][1] - tops[0][1]
    right_height = bottoms[1][1] - tops[1][1]
    print("Tops: ", tops)
    print("Bottoms: ", bottoms)
    # print("Height: ", )
    # print("Reference: 211")
    angle = estimate_angle_deg(left_height, right_height)
    # print("Estimated angle: ", angle)
    return angle


def test_height_error():
    string_to_format = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")

    angle = 45.0  # The start angle
    delta = 0.5

    while angle <= 150.0:
        image = cv2.imread(string_to_format.format(angle))

        estimated_angle = calculate_angle(image)

        # cv2.imshow("a", image)
        # cv2.waitKey()
        print("Actual angle: {0}, Estimated angle: {1:.02f}, Error: {2:.02f}".format(angle, estimated_angle,
                                                                                     angle - estimated_angle))
        angle += delta


def test_straighten_image():
    """
    takes a picture of the vision target splits it in half, rotates both pieces of tape so they are vert (and parallel)
    and merges them back

    kinda useless
    :return:
    """

    # Converts to bitmap (I think)
    image = cv2.inRange(cv2.imread("2019_frc/2019_vision_sample.png"), (30, 30, 30), (255, 255, 255))

    middle = image.shape[1] // 2 # (row_num, column_num, channel_num ... if colored)

    img_left_half = image.T[:middle].T
    img_right_half = image.T[middle:].T

    # Rotates each image half so the tape should be pointing straight up
    img_left_half = scipy.ndimage.rotate(img_left_half, VISION_TAPE_ANGLE_FROM_VERT_DEG)
    img_right_half = scipy.ndimage.rotate(img_right_half, -VISION_TAPE_ANGLE_FROM_VERT_DEG)

    # stitches both images back together
    # note axis=1 corresponds to y-axis
    fixed_image = np.concatenate((img_left_half, img_right_half), axis=1)

    print(fixed_image.shape)

    # We stitched two images together. This creates a jagged line (which OpenCV can think are contours!)
    # To avoid this, we add a blur
    fixed_image = cv2.medianBlur(fixed_image, 11)

    # display the image in a window
    cv2.imshow("straightened image", fixed_image)

    # waits until a key is pressed
    cv2.waitKey()

    contours = get_contours(fixed_image)
    print(contours[0].shape)

    # Each contour is a list of pixels (x,y) on the contour. We are attempting
    # to find the min and max y (a[1]) values of the contours
    # The axis system is dum so we (top is lower value than bottom)
    tops = [min(contour, key=lambda a: a[1]) for contour in contours]
    bots = [max(contour, key=lambda a: a[1]) for contour in contours]

    # The axis system is dum so we (top is lower value than bottom)
    left_height = bots[0][1] - tops[0][1]  # TODO: is this left tape?

    print(get_dist_ft(left_height))


def denoising_test():
    """
    removes noise from images
    :return:
    """
    noisy_image = cv2.imread("2019_vision_sample_noisy.png")  # TODO: the amount of noise in this image is unrealistically high

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # opening is useful for removing noise (erodes and then dilates)
    # TODO: how does this even work if we don't have a bit mask yet...
    opening = cv2.morphologyEx(noisy_image, cv2.MORPH_OPEN, np.ones((5, 5))) # TODO: wot

    # need to dilate because the image has holes in it
    # TODO: why dilate ==> bitmask ==> dilate when can just bitmask ==> dilate
    dilation = cv2.dilate(opening, np.ones((6, 6)))

    # bitmask anything that remains
    bitmask = cv2.inRange(dilation, (1, 1, 1), (255, 255, 255))

    dilation2 = cv2.dilate(bitmask, np.ones((6, 6)), iterations=2)

    cv2.imshow('dilation 2', dilation2)
    cv2.waitKey()
    cv2.imwrite("abcd.png", dilation) # TODO: this can be better...


def pose_estimator(): # TODO: fix
    """
    Broken right now.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.9567&rep=rep1&type=pdf
    :return: None
    """
    image = cv2.inRange(
        cv2.imread("2019_vision_sample.png"),
        (30, 30, 30),
        (255, 255, 255)
    )

    focal_length = BLENDER_FLEN_FT # feet
    sensor_width = BLENDER_SENSOR_WIDTH_FT # feet
    sensor_height = BLENDER_SENSOR_HEIGHT_FT  # feet

    pixel_width = BLENDER_PIXEL_HEIGHT
    pixel_height = BLENDER_PIXEL_WIDTH

    landmark_points_pixels = get_landmark_points(image)  # Points A 1-4
    landmark_points = [np.array([sensor_width * wid / pixel_width, sensor_height * height / pixel_height, focal_length])
                       for wid, height in landmark_points_pixels] # TODO: I'm lazy... will look over/doc later

    landmark_points_dict = dict()

    landmark_points.sort(key=lambda x: np.linalg.norm(x[:2]))
    landmark_points_pixels.sort(key=lambda x: np.linalg.norm(x[:2]))

    landmark_points_dict["top_left"] = landmark_points[0]
    landmark_points_dict["bottom_right"] = landmark_points[-1]

    landmark_points = landmark_points[1:3]
    landmark_points.sort(key=lambda x: x[0])

    landmark_points_dict["bottom_left"] = landmark_points[0]
    landmark_points_dict["top_right"] = landmark_points[1]

    i_to_index = ["top_left", "top_right", "bottom_left", "bottom_right"]

    u = np.array([landmark_points_dict[index] / np.linalg.norm(landmark_points_dict[index]) for index in i_to_index])


    # lengths = np.zeros((4,))

    A = [
        np.array([0, 0, 0]),
        np.array([0, TOP_WIDTH_FT, 0]),
        np.array([REAL_HEIGHT_FT, 0, 0]),
        np.array([REAL_HEIGHT_FT, BOTTOM_WIDTH_FT, 0]),
    ]

    def a_delta(i, j):
        return A[i] - A[j]

    def error(i, j, lengths):
        return (lengths[i] * lengths[i] + lengths[j] * lengths[j] - 2 * lengths[i] * lengths[j] *
                (u[i].dot(u[j]))) - np.linalg.norm(a_delta(i, j)) ** 2

    def error_deriv(i, j, lengths):
        return 2 * lengths[i] + 2 * lengths[j] * (u[i].dot(u[j]))

    def g(lengths):
        estimate = np.cross((lengths[1] * u[1] - lengths[2] * u[2]), (lengths[3] * u[3] - lengths[2] * u[2])) \
            .dot((lengths[0] * u[0] - lengths[2] * u[2]))
        actual = np.cross(a_delta(1, 2), a_delta(3, 2)).dot(a_delta(0, 2))
        # print("actual", np.cross(a_delta(1, 2), a_delta(3, 2)))
        # print("acgtual", actual)
        return estimate - actual

    def get_error_vec(lengths):
        return np.array([
            error(0, 1, lengths), error(0, 2, lengths), error(0, 3, lengths), error(1, 2, lengths),
            error(1, 3, lengths), error(2, 3, lengths), g(lengths)
        ]).T

    def get_jacobian_row(a, b, lengths):
        row = np.zeros((4,))
        row[a] = error_deriv(a, b, lengths)
        row[b] = error_deriv(b, a, lengths)
        return row.T

    def get_error_jacobian(lengths):
        ret = np.zeros((7, 4))
        ret[0] = get_jacobian_row(0, 1, lengths)
        ret[1] = get_jacobian_row(0, 2, lengths)
        ret[2] = get_jacobian_row(0, 3, lengths)
        ret[3] = get_jacobian_row(1, 2, lengths)
        ret[4] = get_jacobian_row(1, 3, lengths)
        ret[5] = get_jacobian_row(2, 3, lengths)

        def g_prime():
            f1_prime = u[1] - u[2]
            f2_prime = u[3] - u[2]

            f_prime = np.cross(f1_prime, (lengths[3] * u[3] - lengths[2] * u[2])) + np.cross(
                (lengths[1] * u[1] - lengths[2] * u[2]), f2_prime)
            f = np.cross((lengths[1] * u[1] - lengths[2] * u[2]), (lengths[3] * u[3] - lengths[2] * u[2]))

            g1_prime = u[0] - u[2]
            g1 = (lengths[0] * u[0] - lengths[2] * u[2])

            lhs = f_prime.dot(g1)
            rhs = f.dot(g1_prime)

            return lhs + rhs

        ret[6] = g_prime()

        return ret

    result = scipy.optimize.least_squares(get_error_vec, np.ones((4,)) / 3, get_error_jacobian)
    print(result.x)
    return result


def pose_estimator_test():
    image = cv2.inRange(
        cv2.imread("2019_vision_sample.png"),
        (30, 30, 30),
        (255, 255, 255)
    )

    format_string = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")

    angle = 30.0
    delta = 0.5

    print("top, bottom, angle")
    while angle <= 150.0:
        image = cv2.inRange(
            cv2.imread(format_string.format(angle)),
            (30, 30, 30),
            (255, 255, 255)
        )

        pose_estimator(image)

        angle += delta


def pnp_test():
    format_string = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")

    angle = 30
    delta = 1

    ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                           [0, 0.5, 2], [1, 0.5, 2]]) / 10
    ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
                (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (720, 360))

    while angle < 150:
        image = cv2.imread(format_string.format(angle))
        image = cv2.inRange(image, (30, 30, 30),
                            (255, 255, 255)
                            )

        corners = np.array(get_landmark_points(image), dtype=np.float32).reshape(-1, 1, 2)
        if False:  # shows corners
            image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            for corner in corners:
                image2 = cv2.circle(image2,
                                    tuple(corner),
                                    6,
                                    (0, 255, 0),
                                    thickness=5)
            cv2.imshow("a", image2)
            cv2.waitKey()

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        down_left_vec = np.array([-5.5 * math.cos(math.radians(75.5)), -5.5 * math.sin(math.radians(75.5))])
        down_right_vec = np.array([2 * math.cos(math.radians(14.5)), -2 * math.sin(math.radians(14.5))])
        right_vec = np.array([0, BOTTOM_WIDTH_FT])
        objp = np.array([
            [0, 0, 0],
            [REAL_HEIGHT_FT, 0, 0],
            list(down_left_vec + down_right_vec) + [0],
            list(down_left_vec + down_right_vec + right_vec) + [0],
        ])

        print(objp)
        print(constants.VISION_TAPE_OBJECT_POINTS)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        corners2 = cv2.cornerSubPix(image,
                                    corners,
                                    (5, 5), (-1, -1),
                                    criteria)

        if True:  # shows corners
            image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            for corner in corners2.reshape(-1, 2):
                image2 = cv2.circle(image2,
                                    tuple(corner),
                                    6,
                                    (0, 255, 0),
                                    thickness=1)
            cv2.imshow("a", image2)
            cv2.waitKey()

        camera_matrix = np.zeros((3, 3))
        camera_matrix[0, 0] = 787.5
        camera_matrix[0, 2] = 360
        camera_matrix[1, 1] = 700
        camera_matrix[1, 2] = 180
        camera_matrix[2, 2] = 1

        rt = np.zeros((3, 4))
        rt[0, 1] = 1
        rt[1, 2] = -1
        rt[2, 0] = -1
        rt[2, 3] = 1.8328

        p = np.array([(-359.9998, 787.5001, -0.0000, 659.8187),
                      (-179.9999, 0.0000, -700.0001, 329.9093),
                      (-1.0000, 0.0000, -0.0000, 1.8328)])

        distortion = np.array([0, 0, 0, 0]).reshape(-1, 1)
        # Find the rotation and translation vectors.
        retval, rvecs, tvecs = cv2.solvePnPRansac(objp, corners2, camera_matrix, distortion)
        # project 3D points to image plane

        imgpts, jac = cv2.projectPoints(np.array([[math.cos(math.radians(75.5)), math.sin(math.radians(75.5)), 0], ]), rvecs, tvecs, camera_matrix, None)
        imgpts = imgpts.reshape(-1, 2)
        def draw(img, imgpts):
            for pt in imgpts:
                img = cv2.circle(image2, tuple(int(x) for x in pt), 6, (0, 255, 0), thickness=30)
            # for i, j in ar_edges:
            #     print(imgpts[i].ravel())
            #     print(imgpts[j])
            #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 5)
            # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
            return img

        img = cv2.drawFrameAxes(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), camera_matrix, None, rvecs, tvecs, 1, 3)
        # img = draw(img, imgpts)
        cv2.imshow('img', img)
        out.write(img)
        angle += delta
    out.release()

