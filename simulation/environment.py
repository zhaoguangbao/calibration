import os

import cv2
import numpy as np
import open3d as o3d
import math
import time
import threading
from calibrate import HandInEyeCalibrate, IntrinsicCalibrate, HandInEyeChessCalibrate

import json

import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from camera import Camera, CameraIntrinsic, Frame
from ur5 import UR5
from panda_sim_grasp import PandaSim
from scene import Scene

coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1, [0, 0, 0])


def setup_target_pose_params(initial_xyz, initial_rpy):
    initial_x, initial_y, initial_z = initial_xyz
    initial_roll, initial_pitch, initial_yaw = initial_rpy

    param_ids = [
        p.addUserDebugParameter('x', -1, 1, initial_x),
        p.addUserDebugParameter('y', -1, 1, initial_y),
        p.addUserDebugParameter('z', 0, 1, initial_z),
        p.addUserDebugParameter('roll', -math.pi, math.pi, initial_roll),
        p.addUserDebugParameter('pitch', -math.pi, math.pi, initial_pitch),
        p.addUserDebugParameter('yaw', -math.pi, math.pi, initial_yaw),
        p.addUserDebugParameter('finger openness', 0, 1, 1)
    ]

    return param_ids


def read_user_params(param_ids):
    return [p.readUserDebugParameter(param_id) for param_id in param_ids]


class DebugAxes(object):
    """
    可视化某个局部坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos)
        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


class Environment(object):
    def __init__(self, gui=True):
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(1.674, 70, -50.8, [0, 0, 0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.scene = Scene()
        self.arm = UR5("./urdf/real_arm.urdf", "./setup.json")
        # self.arm = PandaSim('/home/zhao/git_ws/panda_grasp_sim_2-main/franka_panda/panda.urdf', "./setup.json")
        self.other_ids = [self.arm.uid] + self.scene.other_id  # save urdf id in scene exclude model urdf id

        self.end_axes = DebugAxes()  # 机械臂末端的局部坐标系
        self.camera_axes = DebugAxes()  # 相机坐标系

        # thread for updatig debug axes
        self.update_debug_axes_thread = threading.Thread(target=self.update_debug_axes)
        self.update_debug_axes_thread.setDaemon(True)
        self.update_debug_axes_thread.start()

        # thread for updating camera image
        # self.update_camera_image_thread = threading.Thread(target=self.update_camera_image)
        # self.update_camera_image_thread.setDaemon(True)
        # self.update_camera_image_thread.start()

        init_xyz = [0.08, -0.20, 0.6]
        init_rpy = [0, math.pi / 2., 0]
        self.param_ids = setup_target_pose_params(init_xyz, init_rpy)

    def setup_scene(self, urdfs_lists, poses_lists=None):
        self.urdfs_lists = urdfs_lists
        self.poses_lists = poses_lists

    def reset_scene(self):
        self.scene.removeObjsInURDF()
        self.scene.loadObjsInURDF(self.urdfs_lists, self.poses_lists)
        print('scene settle')

    """
    Thread Callback
    """
    def update_debug_axes(self):
        while True:
            # update debug axes and camera position
            end_pos, end_orn = self.arm.get_end_state()
            self.end_axes.update(end_pos, end_orn)

            wcT = self.arm.get_camera_states()
            self.camera_axes.update(
                pos=wcT[:3, 3],
                orn=R.from_matrix(wcT[:3, :3]).as_quat()
            )
    
    def update_camera_image(self):
        cv2.namedWindow("image")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pc = o3d.geometry.PointCloud()
        vis.add_geometry(pc)
        vis.add_geometry(coordinate)
        while True:
            frame = self.arm.render(align=True)
            assert isinstance(frame, Frame)

            rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel
            cv2.imshow("image", bgr)
            key = cv2.waitKey(1)
            pc_ = frame.point_cloud()
            pc.points = pc_.points
            pc.colors = pc_.colors
            vis.update_geometry(pc)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)

    """
    Callback
    """
    def pre_grasp(self):
        target_pose = [0.08, -0.20, 0.6, 0, math.pi / 2., 0]
        self.arm.move_to(np.asarray(target_pose[:3]), p.getQuaternionFromEuler(target_pose[3:6]))
        self.arm.relax()

    def start_manual_control(self):
        while True:
            keys = p.getKeyboardEvents()
            if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
                print('pre grasp')
                self.pre_grasp()

            if ord('6') in keys and keys[ord('6')] & p.KEY_WAS_TRIGGERED:
                print('reset scene')
                self.reset_scene()

            if ord('7') in keys and keys[ord('7')] & p.KEY_WAS_TRIGGERED:
                target_pose = read_user_params(self.param_ids)  # [x, y, z, roll, pitch, yaw, finger openness]
                print('param control: ', target_pose)
                self.arm.move_to(target_pose[:3], p.getQuaternionFromEuler(target_pose[3:6]))
                time.sleep(0.02)

            time.sleep(0.1)


def loadCameraPose(root, sceneId, camera='kinect', annId=0):
    """
    return Transform matrix where camera relative world
    """
    camera_poses = np.load(
        os.path.join(root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
    camera_pose = camera_poses[annId]
    align_mat = np.load(
        os.path.join(root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
    camera_pose = np.matmul(align_mat, camera_pose)
    return camera_pose


def generator():
    from util import save_image, check_path
    root = '/media/zhao/Newsmy/graspnet_dataset'
    scene_id = 0
    camera = 'kinect'
    env = Environment()
    # for camera intrinsic
    # urdfs_lists = ['/home/zhao/git_ws/bullet_simulation/simulation/assets/chessboard_8_10_30_15.urdf']
    # save_root = './outputs/camera_calibration/'
    # for eye in hand
    urdfs_lists = ['/home/zhao/git_ws/bullet_simulation/simulation/assets/ArUco_DICT_7X7_50_5_7_50_10_0.urdf']
    save_root = './output/hand_eye_calibration/'
    check_path(save_root)
    poses_lists = [np.eye(4)]
    env.setup_scene(urdfs_lists, poses_lists)
    env.reset_scene()

    env.pre_grasp()
    time.sleep(1)
    data = []
    for annId in range(0, 30):
        wcT = loadCameraPose(root, scene_id, camera, annId)
        env.arm.set_camera_states(wcT)
        time.sleep(1)
        frame = env.arm.render(align=False)
        frame.color_image()
        img = frame.color_image()
        rgbPath = os.path.join(save_root, str(annId) + '.png')
        save_image(img, rgbPath)
        end_pos, end_orn = env.arm.get_end_state()
        d = np.asarray(end_pos + end_orn)
        data.append(d)
    data = np.asarray(data)
    posePath = os.path.join(save_root, 'pose.txt')
    np.savetxt(posePath, data, fmt='%.5f')
    print("[INFO] Start manual control!")
    env.start_manual_control()


def calibration():
    # c = IntrinsicCalibrate.factory()
    c = HandInEyeChessCalibrate.factory()
    # c = HandInEyeCalibrate.factory()
    c.calibrate()


if __name__ == "__main__":
    calibration()
    # generator()
