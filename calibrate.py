import numpy as np
import quaternion
import argparse
import json
import os
from cv2 import aruco
import cv2
from scipy.spatial.transform import Rotation as R


"""
data:
    - images should named like `numbers.png`, and index from 0.
    - robot_pose_txt file should name with `pose.txt`: trans(3)+quaternion(4)
usage:
    # c = IntrinsicCalibrate.factory()
    c = HandInEyeChessCalibrate.factory()
    # c = HandInEyeCalibrate.factory()
    c.calibrate()
"""


class Utility:
    @staticmethod
    def skew(mat):
        """
        Copy from: https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array
        This function returns a numpy array with the skew symmetric cross product matrix for vector.
        The skew symmetric cross product matrix is defined such that
        np.cross(a, b) = np.dot(skew(a), b)
        :param mat: An array like vector to create the skew symmetric cross product matrix for
        :return: A numpy array of the skew symmetric cross product vector
        """
        if mat.ndim == 1:
            return np.array([[0, -mat[2], mat[1]],
                             [mat[2], 0, -mat[0]],
                             [-mat[1], mat[0], 0]])
        else:
            shape = mat.shape[0:-1]
            zeros = np.zeros(shape, dtype=mat.dtype)
            skew_mat = np.stack([zeros, -mat[..., 2], mat[..., 1],
                                 mat[..., 2], zeros, -mat[..., 0],
                                 -mat[..., 1], mat[..., 0], zeros], axis=-1)
            new_shape = list(mat.shape) + [3]
            skew_mat = skew_mat.reshape(new_shape)
            return skew_mat

    @staticmethod
    def rodrigues_to_rotation_matrix(rvec):
        """
        :param rvec: Nx1x3-d or 3x1-d numpy array representing rotation vector(s)
        :return: Nx3x3-d or 3x3-d numpy array representing the corresponding rotation matrix
        """
        rvec = np.squeeze(rvec)
        if len(rvec.shape) == 1:
            rvec = np.expand_dims(rvec, axis=0)  # 1x3
        num_vec = rvec.shape[0]
        theta = np.linalg.norm(rvec, axis=1, keepdims=True)  # Nx1
        r = rvec / theta  # Nx3
        theta = theta.reshape((num_vec, 1, 1))
        zero_vec = np.zeros_like(r[:, 0])
        rot = np.cos(theta) * np.eye(3).reshape(1, 3, 3) +\
              (1 - np.cos(theta)) * r.reshape(num_vec, 3, 1) @ r.reshape(num_vec, 1, 3) +\
              np.sin(theta) * np.array([[zero_vec, -r[:, 2], r[:, 1]],
                                        [r[:, 2], zero_vec, -r[:, 0]],
                                        [-r[:, 1], r[:, 0], zero_vec]]).transpose((2, 0, 1))
        return rot


class IntrinsicCalibrate:
    @staticmethod
    def find_chessboard_corners(bgr, width, height, criteria):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret:
            # corners = np.squeeze(corners, axis=1)
            img_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return ret, np.squeeze(corners, axis=1), np.squeeze(img_corners, axis=1).astype(np.float32)
        else:
            return ret, corners, None

    @classmethod
    def factory(cls):
        parser = argparse.ArgumentParser(description='camera calibration')
        parser.add_argument('--num_img',
                            type=int,
                            default=30,
                            help='number of images used for calibration')
        parser.add_argument('--img_height',
                            type=int,
                            default=720,
                            help='pixel height of image')
        parser.add_argument('--img_width',
                            type=int,
                            default=1280,
                            help='pixel width of image')
        parser.add_argument('--height',
                            type=int,
                            default=7,
                            help='height of the chessboard')
        parser.add_argument('--width',
                            type=int,
                            default=9,
                            help='width of the chessboard')
        parser.add_argument('--size',
                            type=float,
                            default=0.03,
                            help='square size for each square of the chessboard')
        parser.add_argument('--dir',
                            type=str,
                            default='/home/zhao/git_ws/bullet_simulation/simulation/output/camera_calibration/',
                            help='directory to load image and save result')
        args = parser.parse_args()
        return cls(args)

    def __init__(self, args):
        self.args = args

    def calibrate(self):
        # termination criteria
        args = self.args
        img_h = args.img_height
        img_w = args.img_width
        save_dir = args.dir
        path = os.path.dirname(save_dir+'result/')
        if not os.path.exists(path):
            os.makedirs(path)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corner_pts = np.zeros((args.height * args.width, 3), dtype=np.float32)
        corner_pts[:, :2] = np.mgrid[0:args.width, 0:args.height].T.reshape(-1, 2)
        corner_pts = corner_pts * args.size
        corner_pts_list = list()
        img_pts_list = list()

        i = 0
        while i < args.num_img:
            # capturing chessboard with various distance will achieve more precise estimation
            rgbPath = save_dir + str(i) + '.png'
            bgr = cv2.imread(rgbPath)
            ret, corners, img_corners = self.find_chessboard_corners(bgr, args.width, args.height, criteria)
            if ret:
                print('chessboard detection succeeded.')
                img = cv2.drawChessboardCorners(bgr.copy(), (args.width, args.height), img_corners, ret)
                cv2.imwrite(save_dir + 'result' + '/{:04d}_display.png'.format(i), img)
                corner_pts_list.append(corner_pts)
                img_pts_list.append(img_corners)
                i += 1
            else:
                print('chessboard detection failed, please try other viewpoints ... ')
                i += 1
                continue
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(corner_pts_list, img_pts_list, (img_w, img_h), None, None)
        if ret:
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
            print('cameraMatrix: {}'.format(mtx))
            print('distCoeffs: {}'.format(dist))
            print('calibrated intrinsic: {}'.format(new_mtx))
            np.savetxt(save_dir + '/mtx.txt', mtx)
            np.savetxt(save_dir + '/dist.txt', dist)
            np.savetxt(save_dir + '/new_mtx.txt', new_mtx)
        else:
            print('fail to calibrate the camera, exit and try again.')
            exit(0)


class HandInEyeCalibrate:
    """
    Hand In Eye Calibration by Aruco
    """
    @classmethod
    def factory(cls):
        parser = argparse.ArgumentParser(description='hand-eye calibration in PyBullet.')
        parser.add_argument('--board_name',
                            type=str,
                            default='ArUco_DICT_7X7_50_5_7_50_10_0',
                            help='the name of aruco board.')
        parser.add_argument('--num_frame',
                            type=int,
                            default=30,
                            help='The required number of frames captured from the camera.')
        parser.add_argument('--a',
                            type=int,
                            default=15,
                            help='one of the motions for which '
                                 'the angle of rotation axes is in [a, pi-a] will be discarded.')
        parser.add_argument('--sc',
                            type=float,
                            default=0.0002,
                            help='the scalar threshold for the difference '
                                 'between two scalar parts of the quaternions of hand and eye motions.')
        parser.add_argument('--dir',
                            type=str,
                            default='/home/zhao/git_ws/bullet_simulation/simulation/output/hand_eye_calibration/',
                            help='directory to load data and save result')
        args = parser.parse_args()
        return cls(args)

    ARUCO_DICT = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11
    }

    @staticmethod
    def detect_markers(bgr, aruco_dict, board, params, mtx, dist):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=params,
                                                                cameraMatrix=mtx,
                                                                distCoeff=dist)
        aruco.refineDetectedMarkers(gray, board, corners, ids, rejected_img_points)
        return corners, ids, rejected_img_points

    @staticmethod
    def pre_selection(ma, mb, scalar_th, angle_th):
        """
        Pre-selection includes
        1) selecting motions that the difference between the scalar values of hand and eye motions is small, and
        2) filter out the motion pairs in which the rotation axes are similar to the chosen candidates.
        :param ma: An Nx4x4-d numpy array including N camera motions.
        :param mb: An Nx4x4-d numpy array including N hand motions.
        :param scalar_th: the scalar threshold, see argparse for more detail.
        :param angle_th: the angle threshold, see argparse for more detail.
        :return: the selected camera and hand motions.
        """
        print('{} motions in total before pre-selection.'.format(ma.shape[0]))
        qa = quaternion.from_rotation_matrix(ma[:, 0:3, 0:3])
        qb = quaternion.from_rotation_matrix(mb[:, 0:3, 0:3])
        diff_scalar_ab = np.abs(quaternion.as_float_array(qa)[:, 0] - quaternion.as_float_array(qb)[:, 0])
        flag_ab = diff_scalar_ab < scalar_th
        ma, mb = ma[flag_ab], mb[flag_ab]
        print('{} motions remained after removing motions with large scalar differences.'.format(ma.shape[0]))
        num_motion = ma.shape[0]
        assert num_motion > 0
        qa = quaternion.from_rotation_matrix(ma[:, 0:3, 0:3])
        rot_vec_a = quaternion.as_rotation_vector(qa)
        rot_vec_a = rot_vec_a / np.linalg.norm(rot_vec_a, axis=1, keepdims=True)
        # pre-selection, remove similar motions
        max_discard_angle = np.deg2rad(angle_th)
        cos = np.cos(max_discard_angle)
        similarity_matrix = np.matmul(rot_vec_a, rot_vec_a.T)
        selection_matrix = np.logical_and(similarity_matrix >= -cos, similarity_matrix <= cos)
        selected_motion_ids = []
        remaining_ids = np.arange(num_motion)
        while selection_matrix.shape[0] != 0:
            selected_motion_ids.append(remaining_ids[0])
            distinct_flag = selection_matrix[0]
            remaining_ids = remaining_ids[distinct_flag]
            selection_matrix = selection_matrix[distinct_flag][:, distinct_flag]
        selected_motion_ids = np.array(selected_motion_ids)
        ma, mb = ma[selected_motion_ids], mb[selected_motion_ids]
        print('{} motions are selected for pose estimation.'.format(ma.shape[0]))
        return ma, mb

    @staticmethod
    def dual_quaternion_method(ma, mb):
        """
        The hand-eye calibration method based on dual quaternions.
        This implementation is based on "Hand-Eye Calibration Using Dual Quaternions" by Konstantinos Daniilidis
        :param ma: An Nx4x4-d numpy array including N camera motions.
        :param mb: An Nx4x4-d numpy array including N hand motions.
        :return: A 4x4-d numpy array representing the pose from camera to gripper.
        """
        n = ma.shape[0]
        ta, tb = np.zeros((n, 4)), np.zeros((n, 4))
        ta[:, 1:], tb[:, 1:] = ma[:, 0:3, 3], mb[:, 0:3, 3]
        ta, tb = quaternion.from_float_array(ta), quaternion.from_float_array(tb)
        quat_a = quaternion.from_rotation_matrix(ma[:, 0:3, 0:3])
        quat_b = quaternion.from_rotation_matrix(mb[:, 0:3, 0:3])
        quat_a_p = 0.5 * ta * quat_a
        quat_b_p = 0.5 * tb * quat_b
        vec_a = quaternion.as_float_array(quat_a)[:, 1:]  # N * 3
        vec_a_p = quaternion.as_float_array(quat_a_p)[:, 1:]
        vec_b = quaternion.as_float_array(quat_b)[:, 1:]
        vec_b_p = quaternion.as_float_array(quat_b_p)[:, 1:]
        skew_ab = Utility.skew(vec_a+vec_b)  # N * 3 * 3
        skew_ab_p = Utility.skew(vec_a_p+vec_b_p)  # N * 3 * 3
        zero_vec, zero_mat = np.zeros_like(vec_a), np.zeros_like(skew_ab)
        eq1 = np.concatenate([(vec_a - vec_b).reshape(n, 3, 1),
                              skew_ab,
                              zero_vec.reshape(n, 3, 1),
                              zero_mat], axis=2).reshape(-1, 8)
        eq2 = np.concatenate([(vec_a_p - vec_b_p).reshape(n, 3, 1),
                              skew_ab_p,
                              (vec_a - vec_b).reshape(n, 3, 1),
                              skew_ab], axis=2).reshape(-1, 8)
        eqs = np.concatenate([eq1, eq2], axis=0)
        _, s, vh = np.linalg.svd(eqs)
        print('the singular values of the equation matrix: {}'.format(s))
        if s[-1] > 0.01 or s[-2] > 0.01:
            print('WARNING!!! The last two singular values are too large, the estimated result might be erroneous!')
        v7, v8 = vh[-1], vh[-2]
        u1, v1, u2, v2 = v7[:4], v7[4:], v8[:4], v8[4:]
        # s^2 * u1v1 + s * (u1v2+u2v1) + u2v2 = 0
        a = u1 @ v1
        b = u1 @ v2 + u2 @ v1
        c = u2 @ v2
        discriminant = b * b - 4 * a * c
        s1 = (-b + np.sqrt(discriminant)) / (2 * a)
        s2 = (-b - np.sqrt(discriminant)) / (2 * a)
        x1 = s1 ** 2 * u1 @ u1 + 2 * s1 * u1 @ u2 + u2 @ u2
        x2 = s2 ** 2 * u1 @ u1 + 2 * s2 * u1 @ u2 + u2 @ u2
        (s, x) = (s1, x1) if x1 > x2 else (s2, x2)
        lambda2 = np.sqrt(1 / x)
        lambda1 = s * lambda2
        q_check = lambda1 * v7 + lambda2 * v8
        q, q_p = quaternion.from_float_array(q_check[:4]), quaternion.from_float_array(q_check[4:])
        t_c2t = 2 * q_p * q.conj()
        rot_c2t = quaternion.as_rotation_matrix(q)
        t_c2t = quaternion.as_float_array(t_c2t)[1:]
        p_c2t = np.eye(4)
        p_c2t[0:3, 0:3] = rot_c2t
        p_c2t[0:3, 3] = t_c2t
        return p_c2t

    def __init__(self, args):
        self.args = args

    def calibrate(self):
        args = self.args
        save_dir = args.dir
        path = os.path.dirname(save_dir+'result/')
        if not os.path.exists(path):
            os.makedirs(path)

        board_name = args.board_name
        dict_str = '_'.join(board_name.split('_')[1:4])
        col, row, length, sep, start_id = map(lambda x: int(x), board_name.split('_')[4:])
        aruco_dict = aruco.Dictionary_get(self.ARUCO_DICT[dict_str])
        board = aruco.GridBoard_create(col, row, length / 1000, sep / 1000, aruco_dict, start_id)
        board_params = aruco.DetectorParameters_create()

        mtx = np.loadtxt(save_dir+'mtx.txt')
        dist = np.loadtxt(save_dir+'dist.txt')

        pose = np.loadtxt(save_dir+'pose.txt')
        # r = R.from_euler('xyz', pose[:, 3:], degrees=True)
        r = R.from_quat(pose[:, 3:])
        matrix = r.as_matrix()

        # collect camera and TCP motions
        i = 0
        p_c2w_e = []  # list of poses from camera to world Estimated by detecting aruco board
        p_b2t_g = []  # list of Ground truth poses from base to TCP
        while i < args.num_frame:
            rgbPath = save_dir + str(i) + '.png'
            bgr = cv2.imread(rgbPath)
            corners, ids, rejected_img_points = self.detect_markers(bgr, aruco_dict, board, board_params, mtx, dist)
            if ids is not None:
                rvec, tvec, marker_points = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)
                # rots = rodrigues_to_rotation_matrix(rvec)
                draw_frame = bgr.copy()
                aruco.drawDetectedMarkers(draw_frame, corners)
                for j in range(rvec.shape[0]):
                    aruco.drawAxis(draw_frame, mtx, dist, rvec[j], tvec[j], 0.02)
                retval, b_rvec, b_tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist, rvec, tvec)
                t_c2w = b_tvec[:, 0]
                rot_c2w = Utility.rodrigues_to_rotation_matrix(b_rvec)
                curr_p_c2w = np.eye(4)
                curr_p_c2w[0:3, 0:3], curr_p_c2w[0:3, 3] = rot_c2w, t_c2w
                p_c2w_e.append(curr_p_c2w)
                aruco.drawAxis(draw_frame, mtx, dist, b_rvec, b_tvec, 0.2)
                cv2.imwrite(save_dir + 'result' + '/{:04d}.png'.format(i), draw_frame)

                curr_p_b2t = np.eye(4)
                curr_p_b2t[:3, 3] = pose[i, :3]
                curr_p_b2t[:3, :3] = matrix[i]
                p_b2t_g.append(curr_p_b2t)

                i += 1
            else:
                print('chessboard detection failed, please try other viewpoints ... ')
                i += 1
        # motion selection
        p_c2w_e = np.stack(p_c2w_e, axis=0)
        p_b2t_g = np.stack(p_b2t_g, axis=0)
        frame_ids = np.array([[i, j] for i in range(args.num_frame - 1) for j in range(i + 1, args.num_frame)])
        ai, aj = p_c2w_e[frame_ids[:, 0]], p_c2w_e[frame_ids[:, 1]]
        bi, bj = p_b2t_g[frame_ids[:, 0]], p_b2t_g[frame_ids[:, 1]]
        ma = np.matmul(ai, np.linalg.inv(aj))  # candidate camera motions
        mb = np.matmul(np.linalg.inv(bi), bj)
        sma, smb = self.pre_selection(ma, mb, args.sc, args.a)
        p_c2t_e = self.dual_quaternion_method(sma, smb)
        print('estimated hand-eye pose \n{}'.format(p_c2t_e))
        p_t_e2c = np.linalg.inv(p_c2t_e)
        print('estimated camera frame w.r.t tcp frame \n{}'.format(p_t_e2c))
        np.savetxt(save_dir + 'tcT.txt', p_t_e2c)


class HandInEyeChessCalibrate(HandInEyeCalibrate):
    """
    Camera Intrinsic and Hand In Eye Calibration by ChessBoard
    """
    @classmethod
    def factory(cls):
        parser = argparse.ArgumentParser(description='camera calibration')
        parser.add_argument('--num_frame',
                            type=int,
                            default=30,
                            help='number of images used for calibration')
        parser.add_argument('--height',
                            type=int,
                            default=7,
                            help='height of the chessboard')
        parser.add_argument('--width',
                            type=int,
                            default=9,
                            help='width of the chessboard')
        parser.add_argument('--img_height',
                            type=int,
                            default=720,
                            help='pixel height of image')
        parser.add_argument('--img_width',
                            type=int,
                            default=1280,
                            help='pixel width of image')
        parser.add_argument('--size',
                            type=float,
                            default=0.03,
                            help='square size for each square of the chessboard')
        parser.add_argument('--dir',
                            type=str,
                            default='/home/zhao/git_ws/bullet_simulation/simulation/output_sim/camera_calibration/',
                            help='directory to load image and save result')
        parser.add_argument('--a',
                            type=int,
                            default=15,
                            help='one of the motions for which '
                                 'the angle of rotation axes is in [a, pi-a] will be discarded.')
        parser.add_argument('--sc',
                            type=float,
                            default=0.0002,
                            help='the scalar threshold for the difference '
                                 'between two scalar parts of the quaternions of hand and eye motions.')
        args = parser.parse_args()
        return cls(args)

    def calibrate(self):
        args = self.args
        img_h = args.img_height
        img_w = args.img_width
        save_dir = args.dir
        path = os.path.dirname(save_dir+'result/')
        if not os.path.exists(path):
            os.makedirs(path)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corner_pts = np.zeros((args.height * args.width, 3), dtype=np.float32)
        corner_pts[:, :2] = np.mgrid[0:args.width, 0:args.height].T.reshape(-1, 2)
        corner_pts = corner_pts * args.size
        corner_pts_list = list()
        img_pts_list = list()

        pose = np.loadtxt(save_dir+'pose.txt')
        # r = R.from_euler('xyz', pose[:, 3:], degrees=True)
        r = R.from_quat(pose[:, 3:])
        matrix = r.as_matrix()

        # collect camera and TCP motions
        i = 0
        p_c2w_e = []  # list of poses from camera to world Estimated by detecting aruco board
        p_b2t_g = []  # list of Ground truth poses from base to TCP
        while i < args.num_frame:
            # capturing chessboard with various distance will achieve more precise estimation
            rgbPath = save_dir + str(i) + '.png'
            bgr = cv2.imread(rgbPath)
            ret, corners, img_corners = IntrinsicCalibrate.find_chessboard_corners(bgr, args.width, args.height, criteria)
            if ret:
                print('chessboard detection succeeded.')
                img = cv2.drawChessboardCorners(bgr.copy(), (args.width, args.height), img_corners, ret)
                cv2.imwrite(save_dir + 'result' + '/{:04d}_display.png'.format(i), img)
                corner_pts_list.append(corner_pts)
                img_pts_list.append(img_corners)

                curr_p_b2t = np.eye(4)
                curr_p_b2t[:3, 3] = pose[i, :3]
                curr_p_b2t[:3, :3] = matrix[i]
                p_b2t_g.append(curr_p_b2t)

                i += 1
            else:
                print('chessboard detection failed, please try other viewpoints ... ')
                i += 1
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(corner_pts_list, img_pts_list, (img_w, img_h), None, None)
        if ret:
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
            print('cameraMatrix: {}'.format(mtx))
            print('distCoeffs: {}'.format(dist))
            print('calibrated intrinsic: {}'.format(new_mtx))
            np.savetxt(save_dir + '/mtx.txt', mtx, fmt='%0.8f')
            np.savetxt(save_dir + '/dist.txt', dist, fmt='%0.8f')
            np.savetxt(save_dir + '/new_mtx.txt', new_mtx, fmt='%0.8f')
        else:
            print('fail to calibrate the camera, exit and try again.')
            exit(0)
        for rvec, tvec in zip(rvecs, tvecs):
            t_c2w = tvec[:, 0]
            rot_c2w = Utility.rodrigues_to_rotation_matrix(rvec)
            curr_p_c2w = np.eye(4)
            curr_p_c2w[0:3, 0:3], curr_p_c2w[0:3, 3] = rot_c2w, t_c2w
            p_c2w_e.append(curr_p_c2w)
        valid_num_frame = len(rvecs)
        # motion selection
        p_c2w_e = np.stack(p_c2w_e, axis=0)
        p_b2t_g = np.stack(p_b2t_g, axis=0)
        frame_ids = np.array([[i, j] for i in range(valid_num_frame - 1) for j in range(i + 1, valid_num_frame)])
        ai, aj = p_c2w_e[frame_ids[:, 0]], p_c2w_e[frame_ids[:, 1]]
        bi, bj = p_b2t_g[frame_ids[:, 0]], p_b2t_g[frame_ids[:, 1]]
        ma = np.matmul(ai, np.linalg.inv(aj))  # candidate camera motions
        mb = np.matmul(np.linalg.inv(bi), bj)
        sma, smb = self.pre_selection(ma, mb, args.sc, args.a)
        p_c2t_e = self.dual_quaternion_method(sma, smb)
        print('estimated hand-eye pose \n{}'.format(p_c2t_e))
        p_t_e2c = np.linalg.inv(p_c2t_e)
        print('estimated camera frame w.r.t tcp frame \n{}'.format(p_t_e2c))
        np.savetxt(save_dir + 'tcT.txt', p_t_e2c, fmt='%0.8f')

