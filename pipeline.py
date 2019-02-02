import collections
from typing import Tuple, List, Optional

import constants
import numpy as np
import logging
import cv2
import pickle

CalibrationResults = collections.namedtuple("CalibrationResults",
                                            ["camera_matrix", "dist_coeffs", "rvecs", "tvecs", "fisheye"])
CalibrationResults.__new__.__defaults__ = (False,)  # Default for rightmost constructor argument is now False


class VisionPipeline:
    """Contains methods and fields necessary to estimate the pose of the vision target relative to the camera"""

    def __init__(self, synthetic: bool = False, calib_fname: str = None):
        """
        Create a VisionPipeline
        :param synthetic: (bool) Whether or not the synthetic Blender images are being used
        :param calib_fname: (str) The file location of the pickled CalibrationResults instance
        """
        self.logger = logging.getLogger("VisionPipeline")
        self.synthetic = synthetic
        if synthetic and calib_fname is None:
            calib_fname = constants.BLENDER_CALIBRATION_INFO_LOCATION
        if calib_fname is None:
            raise TypeError("calib_fname (argument 2) must be str, not None")

        self.calibration_info = load_calibration_results(calib_fname)
        self.logger.debug("Loaded calibration results")
        self.logger.debug("Synthetic: " + str(synthetic))
        self.logger.debug("Fisheye: " + str(self.calibration_info.fisheye))

    def process_image(self, image: np.array) -> Tuple[List[np.array], np.array, Optional[np.array], Optional[np.array], Optional[float], Optional[np.array]]:
        """
        Find the rotation and translation vectors that convert
        from the world coordinate system into the camera coordinate system
        :param image: The image to process (3 channel)
        :return: The return tuple contains the following:
            - The list of contours
            - The list of the pixel locations of the corners of the vision tape
            - The Rodrigues' rotation vector to transform into the model coordinates
            - The translation vector to transform into the model coordinates
            - The distance to the center of the vision tape
            - The euler (really tait-bryan) angles (x-y-z) that represent the rotation of the model coordinates.
                - A y-angle of 0 degrees means that the vision tapes are parallel to the camera
        """
        if not self.synthetic:
            bitmask = self._generate_bitmask_camera(image)
        else:
            bitmask = self._generate_bitmask_synthetic(image)

        contours = self._get_contours(bitmask)
        corners_subpixel = self._get_corners(contours, bitmask)

        try:
            rvec, tvec, dist = self._estimate_pose(corners_subpixel)
            euler_angles = self._rodrigues_to_euler_angles(rvec)
        except cv2.error:
            rvec, tvec, dist, euler_angles = None, None, None, None

        return contours, corners_subpixel.reshape(-1, 2), rvec, tvec, dist, euler_angles

    def _generate_bitmask_camera(self, image: np.array) -> np.array:
        """
        Generate a bitmask where the vision tapes are white from a real image
        :param image: The image (3 channel)
        :return: The bitmask where the stripes of vision tape are white (1 channel)
        """
        self.logger.debug("Generating bitmask (camera)")
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        im = cv2.inRange(  # TODO: Make constants for lower and upper bounds
            hsv_image,
            (0, 0, 236),
            (103, 255, 255)
        )
        closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((3, 3)))
        return closing

    def _generate_bitmask_synthetic(self, image: np.array) -> np.array:
        """
        Generate a bitmask where the vision tapes are white from a synthetic image
        :param image: The image (3 channel)
        :return: The bitmask where the stripes of vision tape are white (1 channel)
        """
        self.logger.debug("Generated bitmask (synthetic)")
        return cv2.inRange(  # TODO: Make constants for lower and upper bounds
            image,
            (30, 30, 30),
            (255, 255, 255)
        )

    def _get_contours(self, bitmask: np.array) -> List[np.array]:
        """
        Get the contours that represent the vision tape
        :param bitmask:
        :return:
        """
        self.logger.debug("Finding contours")
        contours, hierarchy = cv2.findContours(bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_hull_areas = [cv2.contourArea(cv2.convexHull(contour)) for contour in contours]
        is_candidate = []
        for contour, contour_hull_area in zip(contours, contour_hull_areas):
            if contour_hull_area > 10:
                area = cv2.contourArea(contour)
                if area/contour_hull_area > 0.85:
                    _, _, w, h = cv2.boundingRect(contour)
                    ratio = -constants.VISION_TAPE_ROTATED_WIDTH_FT/constants.VISION_TAPE_ROTATED_HEIGHT_FT
                    if 0.5 * ratio <= w/h <= 1.5 * ratio:
                        is_candidate.append(True)
                        continue
            is_candidate.append(False)

        def _approximate_contour(cnt):
            epsilon = 0.02 * cv2.arcLength(cnt, True)  # Tolerance (decrease for a tighter fit if needed)
            approx = (cv2.approxPolyDP(cnt, epsilon, True))
            return approx

        candidates = [_approximate_contour(contour) for i, contour in enumerate(contours) if is_candidate[i]]
        candidates.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
        return candidates

    def _get_corners(self, contours: List[np.array], bitmask: np.array) -> np.array:
        """
        Find the image coordinates of the corners of the vision target
        :param contours: The list of contours, sorted by area in descending order
        :param bitmask: The bitmask of the image
        :return: Subpixel-accurate corner locations
        """
        self.logger.debug("Finding corners on image")

        contours = [x.reshape(-1, 2) for x in contours[:2]]

        # TODO: Rewrite to use np.sort/np,array instead of Python lists and list.sort
        tops = [min(contour, key=lambda x: x[1]) for contour in contours]
        bots = [max(contour, key=lambda x: x[1]) for contour in contours]

        tops.sort(key=lambda x: x[0])
        bots.sort(key=lambda x: x[0])

        pixel_corners = np.array(tops + bots, dtype=np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners_subpixel = cv2.cornerSubPix(bitmask,
                                            pixel_corners,
                                            (5, 5), (-1, -1),
                                            criteria)

        return corners_subpixel

    def _estimate_pose(self, corners_subpixel: np.array) -> Tuple[np.array, np.array, float]:
        """
        Estimate the pose of the vision target
        :param corners_subpixel:
        :return: The rotation vectors and translation vectors representing the vision target's pose
        """

        self.logger.debug("Running solvePnPRansac")

        # NOTE: If using solvePnPRansac, retvals are retval, rvec, tvec, inliers
        if self.calibration_info.fisheye:
            undistorted_points = cv2.fisheye.undistortPoints(corners_subpixel, self.calibration_info.camera_matrix,
                                                             self.calibration_info.dist_coeffs)
            retval, rvec, tvec = cv2.solvePnP(constants.VISION_TAPE_OBJECT_POINTS,
                                              undistorted_points,
                                              self.calibration_info.camera_matrix,
                                              None)  # Distortion vector is none because we already undistorted the image
        else:
            retval, rvec, tvec = cv2.solvePnP(constants.VISION_TAPE_OBJECT_POINTS,
                                              corners_subpixel,
                                              self.calibration_info.camera_matrix,
                                              self.calibration_info.dist_coeffs)

        dist = np.linalg.norm(tvec)
        return rvec, tvec, dist

    def _rodrigues_to_euler_angles(self, rvec):
        """
        Given the Rodrigues' rotation vector given by cv2.solvePnP,
        :param rvec: The Rodrigues' rotation vectpor
        :return: The euler (x-y-z) angles (in radians) of the vision tape relative to the camera plane
        """
        mat, jac = cv2.Rodrigues(rvec)

        sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.math.atan2(mat[2, 1], mat[2, 2])
            y = np.math.atan2(-mat[2, 0], sy)
            z = np.math.atan2(mat[1, 0], mat[0, 0])

        else:
            x = np.math.atan2(-mat[1, 2], mat[1, 1])
            y = np.math.atan2(-mat[2, 0], sy)
            z = 0

        return np.array([x, y, z])


def save_calibration_results(camera_matrix: np.array,
                             dist_coeffs: np.array,
                             rvecs: np.array,
                             tvecs: np.array,
                             fisheye: bool,
                             fname: str = "calibration_info.pickle"):
    """
    Save calibration results to a pickle
    :param camera_matrix: The camera matrix
    :param dist_coeffs: The distortion coefficients
    :param rvecs: The rotation vectors
    :param tvecs: The translation vectors
    :param fname: The filename to save it to (optional, ideally ends in .pickle)
    :return: None
    """
    results = CalibrationResults(camera_matrix, dist_coeffs, rvecs, tvecs, fisheye)
    with open(fname, "wb") as f:
        pickle.dump(results, f)


def load_calibration_results(fname: str) -> CalibrationResults:
    """
    Load calibration results from a file
    :param fname: The filename of the file
    :return: A CalibrationResults named tuple
    """
    with open(fname, "rb") as f:
        return pickle.load(f)
