import math

import cv2
import numpy as np
import scipy.ndimage
import scipy.optimize
import os

AUTOGEN_IMAGE_FOLDER = "./autogen_images"

# REAL_HEIGHT_FT = (5.5 * math.sin(math.radians(75.5)) + 2 * math.sin(math.radians(14.5))) / 12.0

VISION_TAPE_LENGTH_IN = 5.5
VISION_TAPE_LENGTH_FT = VISION_TAPE_LENGTH_IN / 12

VISION_TAPE_WIDTH_IN = 2
VISION_TAPE_WIDTH_FT = 2 / 12

VISION_TAPE_MIN_SEPARATION_IN = 8
VISION_TAPE_MIN_SEPARATION_FT = VISION_TAPE_MIN_SEPARATION_IN / 12

VISION_TAPE_ANGLE_FROM_VERT_DEG = 14.5

VISION_TAPE_ANGLE_FROM_HORIZONTAL_DEG = 90 - VISION_TAPE_ANGLE_FROM_VERT_DEG

VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD = math.radians(VISION_TAPE_ANGLE_FROM_HORIZONTAL_DEG)

REAL_HEIGHT_FT = VISION_TAPE_LENGTH_FT * math.sin(VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD)
TOP_WIDTH_FT = (2 * VISION_TAPE_WIDTH_FT * math.sin(VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD) +
                VISION_TAPE_MIN_SEPARATION_FT)

BOTTOM_WIDTH_FT = (2 * VISION_TAPE_LENGTH_FT * math.sin(math.radians(VISION_TAPE_ANGLE_FROM_VERT_DEG)) +
                   VISION_TAPE_MIN_SEPARATION_FT)

MID_WIDTH_FT = (TOP_WIDTH_FT + BOTTOM_WIDTH_FT) / 2

# TODO: please explain and add variable constant
DIAG_WIDTH_FT = np.linalg.norm(
    np.array([BOTTOM_WIDTH_FT, 0]).T +
    np.array(
        [VISION_TAPE_WIDTH_FT * math.cos(math.radians(14.5)), VISION_TAPE_WIDTH_FT * math.sin(math.radians(14.5))]) +
    np.array(
        [-VISION_TAPE_LENGTH_FT * math.sin(math.radians(14.5)), VISION_TAPE_LENGTH_FT * math.cos(math.radians(14.5))])
)

BLENDER_FLEN = 800.6028523694872235460223543952119609884957052979378391303  # (211 * 1.83283) / REAL_HEIGHT_FT

print("BLENDER_FLEN=", BLENDER_FLEN)


def get_dist(height):
    return (REAL_HEIGHT_FT * BLENDER_FLEN) / height


def estimate_angle(left_height, right_height):
    if math.fabs(left_height - right_height) <= 1e-6:
        return 90.0

    left_height = get_dist(left_height)
    right_height = get_dist(right_height)

    c = MID_WIDTH_FT
    a = min(left_height, right_height)
    b = max(left_height, right_height)

    cos_angle = (b * b + c * c - a * a) / (2 * b * c)
    angle = math.degrees(math.acos(cos_angle))
    return angle


def harris_test():
    base_image = cv2.imread("./2019_vision_sample.png")  # TODO: fix

    base_img_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(base_img_gray, 2, 3, 0.15)
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
    contours = get_contours(image)
    tops = [min(contour, key=lambda x: x[1]) for contour in contours]
    bots = [max(contour, key=lambda x: x[1]) for contour in contours]

    landmark_points = tops + bots
    landmark_points_dict = {}

    landmark_points.sort(key=lambda x: np.linalg.norm(x))

    landmark_points_dict["top_left"] = landmark_points[0]
    landmark_points_dict["bottom_right"] = landmark_points[-1]

    landmark_points = landmark_points[1:3]
    landmark_points.sort(key=lambda x: x[0])

    landmark_points_dict["bottom_left"] = landmark_points[0]
    landmark_points_dict["top_right"] = landmark_points[1]

    i_to_index = ["top_left", "top_right", "bottom_left", "bottom_right"]

    u = np.array([landmark_points_dict[index] for index in i_to_index])

    return u


def calculate_angle(image):
    # img_path = "/home/ritikm/Pictures/2019_frc/autogen_images/2019_vision_angle_91.50.png"

    # image = cv2.imread(img_path)

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
    angle = estimate_angle(left_height, right_height)
    # print("Estimated angle: ", angle)
    return angle


def get_height():
    format_string = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")

    angle = 45.0  # * 2
    delta = 0.5
    #
    # print("top, bottom, angle")
    while angle <= 150.0:
        image = cv2.imread(format_string.format(angle))

        estimated_angle = calculate_angle(image)

        # cv2.imshow("a", image)
        # cv2.waitKey()
        print("Actual angle: {0}, Estimated angle: {1:.02f}, Error: {2:.02f}".format(angle, estimated_angle,
                                                                                     angle - estimated_angle))
        angle += delta


def straighten_image():
    image = cv2.inRange(
        cv2.imread("2019_frc/2019_vision_sample.png"),
        (30, 30, 30),
        (255, 255, 255)
    )

    middle = image.shape[1] // 2
    left_half = image.T[:middle].T
    right_half = image.T[middle:].T
    left_half = scipy.ndimage.rotate(left_half, 14.5)
    right_half = scipy.ndimage.rotate(right_half, -14.5)

    fixed_image = np.concatenate((left_half, right_half), axis=1)

    print(fixed_image.shape)
    fixed_image = cv2.medianBlur(fixed_image, 11)
    cv2.imshow("a", fixed_image)
    cv2.waitKey()
    contours = get_contours(fixed_image)
    print(contours[0].shape)

    tops = [min(contour, key=lambda x: x[1]) for contour in contours]
    bots = [max(contour, key=lambda x: x[1]) for contour in contours]

    left_height = bots[0][1] - tops[0][1]

    print(get_dist(left_height))


def denoising_test():
    noisy_image = cv2.imread("2019_vision_sample_noisy.png")  # TODO: fix
    opening = cv2.morphologyEx(noisy_image, cv2.MORPH_OPEN, np.ones((5, 5)))
    dilation = cv2.dilate(opening, np.ones((6, 6)))
    bitmask = cv2.inRange(dilation, (1, 1, 1), (255, 255, 255))
    dilation2 = cv2.dilate(bitmask, np.ones((6, 6)), iterations=2)
    cv2.imshow('dst', dilation2)
    cv2.waitKey()
    cv2.imwrite("abcd.png", dilation)


def pose_estimator(image):
    focal_length = 35 * 0.00328084  # feet
    sensor_width = 32.0 * 0.00328084  # feet
    sensor_height = 18.0 * 0.00328084  # feet

    pixel_width = image.shape[1]
    pixel_height = image.shape[0]
    landmark_points_pixels = get_landmark_points(image)  # Points A 1-4
    landmark_points = [np.array([sensor_width * wid / pixel_width, sensor_height * height / pixel_height, focal_length])
                       for wid, height in landmark_points_pixels]

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
            list(down_left_vec + down_right_vec) + [0],
            [REAL_HEIGHT_FT, 0, 0],
            list(down_left_vec + down_right_vec + right_vec) + [0],
        ])

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
        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, distortion)
        # project 3D points to image plane

        imgpts, jac = cv2.projectPoints(ar_verts, rvecs, tvecs, camera_matrix, None)
        imgpts = imgpts.reshape(-1, 2)
        def draw(img, imgpts):
            for i, j in ar_edges:
                print(imgpts[i].ravel())
                print(imgpts[j])
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 5)
            # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
            return img

        img = cv2.drawFrameAxes(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), camera_matrix, None, rvecs, tvecs, 1, 3)
        # img = draw(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), imgpts)
        cv2.imshow('img', img)
        out.write(img)
        angle += delta
    out.release()


pnp_test()