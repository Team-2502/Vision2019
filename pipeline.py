import collections
from typing import Tuple, List, Optional, Dict

import constants
import numpy as np
import logging
import cv2
import pickle

CalibrationResults = collections.namedtuple("CalibrationResults",
                                            ["camera_matrix", "dist_coeffs", "rvecs", "tvecs", "fisheye"])
CalibrationResults.__new__.__defaults__ = (False,)  # Default for rightmost constructor argument is now False

PipelineResults = collections.namedtuple("PipelineResults",
                                         ['bitmask', 'trash', 'contours', 'corners', 'pose_estimation', 'euler_angles'])

PoseEstimation = collections.namedtuple("PoseEstimation", ['left_rvec', 'left_tvec', 'right_rvec', 'right_tvec'])
EulerAngles = collections.namedtuple("EulerAngles", ['left', 'right'])

PipelineResults._field_types = {'bitmask': np.array, 'contours': List[np.array], 'corners': List[np.array],
                                'pose_estimation': PoseEstimation, 'euler_angles': EulerAngles}

def avg(iter):
    try:
        return sum(iter)/len(iter)
    except ZeroDivisionError:
        return 0

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
        self.last_centroid_x = []
        if synthetic and calib_fname is None:
            calib_fname = constants.BLENDER_CALIBRATION_INFO_LOCATION
        if calib_fname is None:
            raise TypeError("calib_fname (argument 2) must be str, not None")

        self.calibration_info = load_calibration_results(calib_fname)
        self.logger.debug("Loaded calibration results")
        self.logger.debug("Synthetic: " + str(synthetic))
        self.logger.debug("Fisheye: " + str(self.calibration_info.fisheye))

    def process_image(self, image: np.array) -> PipelineResults:
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

        contours, trash_contours = self._get_contours(bitmask)
        corners_subpixel = self._get_corners(contours, bitmask)

        try:
            result = self._estimate_pose(corners_subpixel)
            euler_angles = EulerAngles(
                self._rodrigues_to_euler_angles(result.left_rvec),
                self._rodrigues_to_euler_angles(result.right_rvec),
            )

        except (cv2.error, AttributeError):
            result, euler_angles = None, None

        return PipelineResults(bitmask, trash_contours, contours, corners_subpixel, result, euler_angles)

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
            # (25, 0, 221),
            # (279, 255, 255)
            (0, 0, 146),
            (360, 255, 255)

        )
        # closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((3, 3)))
        return im

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

    def _get_contours(self, bitmask: np.array) -> Tuple[List[np.array], List[np.array]]:
        """
        Get the contours that represent the vision tape
        :param bitmask:
        :return:
        """
        self.logger.debug("Finding contours")
        trash = []
        contours, hierarchy = cv2.findContours(bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convex_hulls = [cv2.convexHull(contour) for contour in contours]
        contour_hull_areas = [cv2.contourArea(hull) for hull in convex_hulls]

        height = bitmask.shape[0]
        width = bitmask.shape[1]
        print(height, width)

        def not_touching_edge(cnt):
            cnt = cnt.reshape((-1, 2))
            top_index = cnt[:, 1].argmin()
            bottom_index = cnt[:, 1].argmax()
            left_index = cnt[:, 0].argmin()
            right_index = cnt[:, 0].argmax()

            top_y = cnt[top_index][1]
            bot_y = cnt[bottom_index][1]
            left_x = cnt[left_index][0]
            right_x = cnt[right_index][0]

            return top_y > 10 and bot_y < bitmask.shape[0] - 10 and left_x > 10 and right_x < bitmask.shape[1] - 10

        # Filtering contours
        is_candidate = []
        for contour, contour_hull_area in zip(contours, contour_hull_areas):
            if contour_hull_area > 10:
                area = cv2.contourArea(contour)
                if area / contour_hull_area > 0.85:
                    _, _, w, h = cv2.boundingRect(contour)
                    ratio = -constants.VISION_TAPE_ROTATED_WIDTH_FT / constants.VISION_TAPE_ROTATED_HEIGHT_FT
                    if 0.5 * ratio <= w / h <= 1.5 * ratio:
                        if not_touching_edge(contour):
                            is_candidate.append(True)
                            continue
                        else:
                            print("contour cut off")
                    else:
                        print("contour has bad proportions")
                else:
                    print("contour is not full")
            else:
                print("contour is smolboi")
            is_candidate.append(False)
            trash.append(contour)

        candidates = [convex_hulls[i] for i, contour in enumerate(contours) if is_candidate[i]]

        def get_centroid_x(cnt: np.array) -> int:
            all_x = cnt.reshape((-1, 2))[:, 0]
            return int(np.sum(all_x)/all_x.shape)
            # M = cv2.moments(cnt)
            # return int(M["m10"] / M["m00"])

        def is_tape_on_left_side(cnt):
            min_area_box = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))

            left_box_point = min(min_area_box, key=lambda row: row[0])
            right_box_point = max(min_area_box, key=lambda row: row[0])

            return left_box_point[1] > right_box_point[1]  # left point is below right point

        candidates.sort(key=get_centroid_x)

        if len(candidates) > 0:
            try:
                if not is_tape_on_left_side(candidates[0]):  # pointing to right
                    trash.append(candidates[0])
                    del candidates[0]  # left-most one should point to left
                    # print("removed leftmost for pointing to right")
                if is_tape_on_left_side(candidates[-1]):
                    trash.append(candidates[-1])
                    del candidates[-1]
                    # print("removed rightmost for pointing to left")
            except Exception as e:
                # print("whoops 4", e)
                #
                pass

        if len(candidates) > 1:
            contour_pair_centroids = {}

            # Iterates through contours, two at a time, while counting number of pairs already iterated through
            for i, left_cnt, right_cnt in zip(range(len(candidates)), candidates[::2], candidates[1::2]):
                # Join the two contours and get the centroid of _that_ and stick it in as the key of contour_pair_centroids 
                centroid = get_centroid_x(np.concatenate((left_cnt, right_cnt)))
                contour_pair_centroids[centroid] = i

            if len(contour_pair_centroids) > 0:  # hmmmmm how could this happen?
                if len(self.last_centroid_x) == 0:
                    self.last_centroid_x.append(min(contour_pair_centroids.keys(), key=lambda x: 320 / 2 - x))
                    avg_X = avg(self.last_centroid_x)
                else:
                    if len(self.last_centroid_x) > 5:
                        del self.last_centroid_x[0]
                    avg_X = avg(self.last_centroid_x)
                    self.last_centroid_x.append(min(contour_pair_centroids.keys(),
                                               key=lambda x: np.math.fabs(avg_X - x)))  # hmmm pt 2

                pair_num = contour_pair_centroids[self.last_centroid_x[-1]]

                left_index = pair_num * 2  # maybe hmm but probably correct
                right_index = left_index + 1

                trash.extend(candidates[:left_index])
                trash.extend(candidates[right_index + 1:])
                candidates = [candidates[left_index], candidates[right_index]]
                # scandidates.sort(key=get_centroid_x)
                print(is_tape_on_left_side(candidates[0]))

                return candidates, trash  # left guaranteed to be first

        self.last_centroid_x = []
        return [], candidates + trash


    def _get_corners(self, contours: List[np.array], bitmask: np.array) -> List[np.array]:
        """
        Find the image coordinates of the corners of the vision target
        :param contours: The list of contours, sorted by area in descending order
        :param bitmask: The bitmask of the image
        :return: Subpixel-accurate corner locations
        """
        self.logger.debug("Finding corners on image")

        contours = [x.reshape(-1, 2) for x in contours[:2]]

        def get_corners_intpixel_alternate(cnt):
            def removearray(L, arr):
                ind = 0
                size = len(L)
                while ind != size and not np.array_equal(L[ind], arr):
                    ind += 1
                if ind != size:
                    L.pop(ind)
                else:
                    raise ValueError('array not found in list.')

            blank = np.zeros(bitmask.shape).astype(np.uint8)
            cv2.drawContours(blank, [cnt], -1, (255,), thickness=cv2.FILLED)
            dst = cv2.goodFeaturesToTrack(image=blank, maxCorners=5, qualityLevel=0.16, minDistance=15).reshape(-1, 2)
            if len(dst) < 5:
                return get_corners_intpixel(cnt)

            points = list(dst)

            top_point = min(points, key=lambda x: x[1])
            removearray(points, top_point)

            fake_bottom_point = max(points, key=lambda x: x[1])
            removearray(points, fake_bottom_point)

            left_point = min(points, key=lambda x: x[0])
            removearray(points, left_point)

            right_point = max(points, key=lambda x: x[0])
            removearray(points, right_point)

            leftover_point = points[0]

            top_point, inner_pt, outer_pt, _ = get_corners_intpixel(cnt)

            if left_point[1] > right_point[1]:  # left lower than right
                inner_pt, outer_pt = right_point, left_point
            else:
                inner_pt, outer_pt = left_point, right_point

            bot_point = constants.line_intersect(inner_pt, leftover_point, outer_pt, fake_bottom_point)

            return top_point, inner_pt, outer_pt, bot_point

        def get_corners_intpixel(cnt):
            top_index = cnt[:, 1].argmin()
            bottom_index = cnt[:, 1].argmax()
            left_index = cnt[:, 0].argmin()
            right_index = cnt[:, 0].argmax()

            top_point = cnt[top_index]
            bot_point = cnt[bottom_index]
            left_point = cnt[left_index]
            right_point = cnt[right_index]

            if left_point[1] > right_point[1]:  # left lower than right
                return top_point, right_point, left_point, bot_point
            else:
                return top_point, left_point, right_point, bot_point

        corners = [np.array(get_corners_intpixel(cnt)).reshape((-1, 1, 2)) for cnt in contours]  # left is 0, right is 1

        corners_subpixel = [constants.tape_corners_to_obj_points(*cv2.cornerSubPix(bitmask,
                                                                                   corner.astype(np.float32),
                                                                                   (5, 5), (-1, -1),
                                                                                   constants.SUBPIXEL_CRITERIA)) for corner
                            in corners]

        return corners_subpixel


    def _estimate_pose(self, corners_subpixel: List[np.array]) -> PoseEstimation:
        """
        Estimate the pose of the vision target
        :param corners_subpixel:
        :return: The rotation vectors and translation vectors representing the vision target's pose
        """

        self.logger.debug("Running solvePnPRansac")

        result = {"left": None, "right": None}

        for name, corners, objp in zip(result.keys(), corners_subpixel, (
        constants.VISION_TAPE_OBJECT_POINTS_LEFT_SIDE, constants.VISION_TAPE_OBJECT_POINTS_RIGHT_SIDE)):
            # NOTE: If using solvePnPRansac, retvals are retval, rvec, tvec, inliers
            if self.calibration_info.fisheye:
                undistorted_points = cv2.fisheye.undistortPoints(corners, self.calibration_info.camera_matrix,
                                                                 self.calibration_info.dist_coeffs)[1:]
                result[name] = cv2.solvePnP(objp,
                                            undistorted_points,
                                            self.calibration_info.camera_matrix,
                                            None)  # Distortion vector is none because we already undistorted the image
            else:
                result[name] = cv2.solvePnP(objp,
                                            corners,
                                            self.calibration_info.camera_matrix,
                                            self.calibration_info.dist_coeffs)[1:]  # exclude retval, just rvec and tvec

        try:
            return PoseEstimation(result['left'][0], result['left'][1], result['right'][0], result['right'][1])
        except TypeError:
            return None


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
