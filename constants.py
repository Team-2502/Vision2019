import os
import math

import cv2
import numpy as np


def _rot_mat(rad):
    return np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad), math.cos(rad)]
    ])

def linear_interpolate(point_a, point_b, num_points):
    """
    Linearly interpolates between two points
    :param point_a: The starting point
    :param point_b: The ending point
    :param num_points: The number of points
    :return: A list of points on the line, including the end points
    """
    num_points = int(num_points)
    def line_func(t):
        return (1 - t) * point_a + t * point_b
    return [line_func(t/num_points) for t in range(0, num_points + 1)]



# autogen_images folder
AUTOGEN_IMAGE_FOLDER = "./autogen_images"
"""Location of all of the autogenerated images of the vision target"""

AUTOGEN_FORMAT_STRING = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")
"""Format string to get an image of the vision target when it is at an angle to the optical axis"""

# Calibration file info
BLENDER_CALIBRATION_INFO_LOCATION = "./blender_calib_info.pickle"
"""
The location of the pickled CalibrationResults that contains the calibration information for the blender camera. 
It is used for processing the autogen images.
"""

# Vision tape dimensions
VISION_TAPE_LENGTH_IN = 5.5
"""Length of the vision tape (inches)"""

VISION_TAPE_LENGTH_FT = VISION_TAPE_LENGTH_IN / 12
"""Length of the vision tape (feet)"""

VISION_TAPE_WIDTH_IN = 2
"""Width of the vision tape (inches)"""

VISION_TAPE_WIDTH_FT = 2 / 12
"""Width of the vision tape (feet)"""

# Vision tape angles
VISION_TAPE_ANGLE_FROM_VERT_DEG = 14.5
"""Angle between the vision tape and the vertical axis (degrees)"""

VISION_TAPE_ANGLE_FROM_VERT_RAD = math.radians(VISION_TAPE_ANGLE_FROM_VERT_DEG)
"""Angle between the vision tape and the vertical axis (radians)"""

VISION_TAPE_ANGLE_FROM_HORIZONTAL_DEG = 90 - VISION_TAPE_ANGLE_FROM_VERT_DEG
"""Angle between the vision tape and the horizontal axis (degrees)"""

VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD = math.radians(VISION_TAPE_ANGLE_FROM_HORIZONTAL_DEG)
"""Angle between the vision tape and the horizontal axis (radians)"""

# Vision tape relative geometry
VISION_TAPE_MIN_SEPARATION_IN = 8
"""Distance between the two pieces of vision tape at their closest point (inches)"""

VISION_TAPE_MIN_SEPARATION_FT = VISION_TAPE_MIN_SEPARATION_IN / 12
"""Distance between the two pieces of vision tape at their closest point (feet)"""

VISION_TAPE_TOP_SEPARATION_IN = 2 * VISION_TAPE_WIDTH_IN * math.sin(VISION_TAPE_ANGLE_FROM_HORIZONTAL_RAD) + \
                                VISION_TAPE_MIN_SEPARATION_IN
"""Distance between the top point on the left rectangle and the top point on the right rectangle (inches)"""

VISION_TAPE_TOP_SEPARATION_FT = VISION_TAPE_TOP_SEPARATION_IN / 12
"""Distance between the top point on the left rectangle and the top point on the right rectangle (feet)"""

VISION_TAPE_BOTTOM_SEPARATION_IN = 2 * VISION_TAPE_LENGTH_IN * math.sin(VISION_TAPE_ANGLE_FROM_VERT_RAD) + \
                                   VISION_TAPE_MIN_SEPARATION_IN
"""Distance between the bottom point on the left rectangle and the bottom point on the right rectangle (inches)"""

VISION_TAPE_BOTTOM_SEPARATION_FT = VISION_TAPE_BOTTOM_SEPARATION_IN / 12
"""Distance between the bottom point on the left rectangle and the bottom point on the right rectangle (feet)"""

VISION_TAPE_ROTATED_HEIGHT_FT = np.matmul(_rot_mat(-VISION_TAPE_ANGLE_FROM_VERT_RAD),
                                          np.array([VISION_TAPE_WIDTH_FT, -VISION_TAPE_LENGTH_FT]))[1]

VISION_TAPE_ROTATED_WIDTH_FT = np.matmul(_rot_mat(-VISION_TAPE_ANGLE_FROM_VERT_RAD),
                                         np.array([VISION_TAPE_WIDTH_FT, VISION_TAPE_LENGTH_FT]))[0]

CENTER_LOC_FT = np.array([VISION_TAPE_TOP_SEPARATION_FT / 2, VISION_TAPE_ROTATED_HEIGHT_FT / 2])

# Vision tape coordinates
TOP_LEFT_LOCATION_FT = np.array([0, 0])
"""The location of the top point on the left rectangle in feet. Used in cv2.solvePnP"""

BOTTOM_LEFT_LOCATION_FT = np.array([0, -VISION_TAPE_LENGTH_FT])
"""The location of the bottom point on the left rectangle in feet. Used in cv2.solvePnP"""

TOP_RIGHT_LOCATION_FT = np.array([VISION_TAPE_WIDTH_FT, 0])
"""The location of the top point on the right rectangle in feet. Used in cv2.solvePnP"""

BOTTOM_RIGHT_LOCATION_FT = np.array([VISION_TAPE_WIDTH_FT, -VISION_TAPE_LENGTH_FT])
"""The location of the bottom point on the right rectangle in feet. Used in cv2.solvePnP"""

_two_to_three = np.array([
    [1, 0],
    [0, 1],
    [0, 0]
])

_reflect_across_y_axis = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

def tape_corners_to_obj_points(top_left, top_right, bottom_left, bottom_right, k = 5):
    return np.array(
        # linear_interpolate(top_left, top_right, 2 * k) +
        # # linear_interpolate(top_right, bottom_right, 5.5 * k)  +
        # # linear_interpolate(bottom_right, bottom_left, 2 * k) +
        # linear_interpolate(bottom_left, top_left, 5.5 * k)
        [top_left, top_right, bottom_left, bottom_right]
    )

def line_intersect(*args) -> np.array:
    """
    :param a1: Starting point of first line segment
    :param a2: Ending point of first line segment
    :param b1: Starting point of second line segment
    :param b2: Ending point of second line segment
    :return: Where the two line segments intersect, even if it is not in between the points.
    """
    assert len(args) == 4, "Needs exactly 4 points"
    a1, a2, b1, b2 = [p.reshape((2,)) for p in args]

    t, s = np.linalg.solve(np.array([a2 - a1, b2 - b1]).T, b1 - a1)

    return a1 * (1-t) + a2 * t



VISION_TAPE_OBJECT_POINTS_LEFT_SIDE = np.array([
    np.matmul(_two_to_three, np.matmul(_rot_mat(-VISION_TAPE_ANGLE_FROM_VERT_RAD), TOP_LEFT_LOCATION_FT)),
    np.matmul(_two_to_three, np.matmul(_rot_mat(-VISION_TAPE_ANGLE_FROM_VERT_RAD), TOP_RIGHT_LOCATION_FT)),
    np.matmul(_two_to_three, np.matmul(_rot_mat(-VISION_TAPE_ANGLE_FROM_VERT_RAD), BOTTOM_LEFT_LOCATION_FT)),
    np.matmul(_two_to_three, np.matmul(_rot_mat(-VISION_TAPE_ANGLE_FROM_VERT_RAD), BOTTOM_RIGHT_LOCATION_FT))
])
"""Parameter to cv2.solvePnP and cv2.solvePnPRansac"""

print(VISION_TAPE_OBJECT_POINTS_LEFT_SIDE * 12)
VISION_TAPE_OBJECT_POINTS_RIGHT_SIDE = np.array([
    np.matmul(_reflect_across_y_axis, objp) for objp in VISION_TAPE_OBJECT_POINTS_LEFT_SIDE
])
"""Parameter to cv2.solvePnP and cv2.solvePnPRansac"""

CAMERA_ID = int(os.getenv("V19_CAMERA_ID") or 0)
"""The id of the camera"""

PORT = int(os.getenv("V19_PORT") or 5800)
"""The port to send data over"""

CALIBRATION_FILE_LOCATION = os.getenv("V19_CALIBRATION_FILE_LOCATION") or "prod_camera_calib.pickle"
"""The path to the pickle containing the calibration information"""


SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
