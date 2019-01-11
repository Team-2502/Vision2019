import collections
from typing import Tuple, List

import constants
import numpy as np
import logging
import cv2
import pickle

CalibrationResults = collections.namedtuple("CalibrationResults", ["camera_matrix", "dist_coeffs", "rvecs", "tvecs"])


class VisionPipeline:
    """Contains methods and fields necessary to estimate the pose of the vision target relative to the camera"""

    def __init__(self, synthetic: bool = False, calib_fname: str = None):
        """
        Create a VisionPipeline
        :param synthetic: (bool) Whether or not the synthetic Blender images are being used
        :param calib_fname: (str) The file location of the pickled CalibrationResults instance
        """
        self.logger: logging.Logger = logging.getLogger("VisionPipeline")
        self.synthetic: bool = synthetic
        if synthetic and calib_fname is None:
            calib_fname = constants.BLENDER_CALIBRATION_INFO_LOCATION
        if calib_fname is None:
            raise TypeError("calib_fname (argument 2) must be str, not None")

        self.calibration_info = load_calibration_results(calib_fname)
        self.logger.debug("Loaded calibration results")
        self.logger.debug("Synthetic: " + str(synthetic))

    def process_image(self, image: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Find the rotation and translation vectors that convert
        from the world coordinate system into the camera coordinate system
        :param image: The image to process (3 channel)
        :return: The contours, corner locations, rotation vector and translation vector in a 4-tuple
        """
        if not self.synthetic:
            bitmask = self._generate_bitmask_camera(image)
        else:
            bitmask = self._generate_bitmask_synthetic(image)

        contours = self._get_contours(bitmask)
        corners_subpixel = self._get_corners(contours, bitmask)
        rvecs, tvecs = self._estimate_pose(corners_subpixel)
        return contours, corners_subpixel.reshape(-1, 2), rvecs, tvecs

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
            (0, 0, 181),
            (200, 40, 255)
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

    def _get_contours(self, bitmask: np.array) -> np.array:
        """
        Get the contours that represent the vision tape
        :param bitmask:
        :return:
        """
        self.logger.debug("Finding contours")
        contours, hierarchy = cv2.findContours(bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
        return contours

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

    def _estimate_pose(self, corners_subpixel: np.array) -> Tuple[np.array, np.array]:  # TODO: Fix
        """
        Estimate the pose of the vision target
        :param corners_subpixel:
        :return: The rotation vectors and translation vectors representing the vision target's pose
        """

        self.logger.debug("Running solvePnPRansac")

        # TODO: enable useExtrinsicGuess and save last iterations rvec/tvec
        # TODO: Do something useful with rvecs/tvecs
        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(constants.VISION_TAPE_OBJECT_POINTS,
                                                           corners_subpixel,
                                                           self.calibration_info.camera_matrix,
                                                           self.calibration_info.dist_coeffs)
        return rvecs, tvecs


def save_calibration_results(camera_matrix: np.array,
                             dist_coeffs: np.array,
                             rvecs: np.array,
                             tvecs: np.array,
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
    results = CalibrationResults(camera_matrix, dist_coeffs, rvecs, tvecs)
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
