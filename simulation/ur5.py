import copy

import numpy as np
import pybullet as p
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from camera import Camera, CameraIntrinsic, Frame
import math
import time
import json


class UR5(object):
    def __init__(self, urdf_file, camera_config):
        self.file_name = urdf_file
        self.base_pos = [0.0, -0.5, -0.1]
        self.base_orn = [0, 0, 0, 1]  # quaternion (x, y, z, w)

        self.arm_joints_name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.gripper_joints_name = ['bh_j11_joint', 'bh_j21_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint',
                                    'bh_j33_joint', 'bh_j13_joint', 'bh_j23_joint']
        self.arm_zero_joints_states = [0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994, 0.19581299859677043]
        self.gripper_zero_joints_states = [0.0, 0.0, 1.0, 1.0, 1.0, 0.33, 0.33, 0.33]

        self.uid = p.loadURDF(self.file_name, self.base_pos, self.base_orn,
                              flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True)
        self.end_effector_id = 7
        self.joints, self.control_joints_name = self.setup()
        self.arm_control_joints_id = [self.joints[name].id for name in self.arm_joints_name]
        self.arm_control_joints_maxF = [self.joints[name].maxForce for name in self.arm_joints_name]

        self.gripper_control_joints_id = [self.joints[name].id for name in self.gripper_joints_name]
        self.gripper_control_joints_maxF = [self.joints[name].maxForce for name in self.gripper_joints_name]

        self.reset_joints_pose()

        with open(camera_config, "r") as j:
            config = json.load(j)
        camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        # camera_intrinsic = CameraIntrinsic.from_param(60, 1280, 720)
        self.camera = Camera(camera_intrinsic)
        # camera pose in tcp frame
        self.camera_to_tcp = np.array([[0, 0, 1, -0.05],
                                      [-1, 0, 0, 0],
                                      [0, -1, 0, 0.2],
                                      [0, 0, 0, 1]])

    def arm_action(self, cmds):
        """
        - cmds: len=6, target angles for arm controllable joints
        """
        n_joints = len(cmds)
        assert len(self.arm_control_joints_id) >= len(cmds)
        p.setJointMotorControlArray(self.uid, self.arm_control_joints_id[:n_joints], p.POSITION_CONTROL,
                                    targetPositions=cmds, targetVelocities=[0] * n_joints,
                                    positionGains=[0.03] * n_joints, forces=self.arm_control_joints_maxF[:n_joints])

    def gripper_action(self, cmds):
        n_joints = len(cmds)
        assert len(self.gripper_control_joints_id) >= len(cmds)
        p.setJointMotorControlArray(self.uid, self.gripper_control_joints_id[:n_joints], p.POSITION_CONTROL,
                                    targetPositions=cmds, targetVelocities=[0] * n_joints,
                                    positionGains=[0.03] * n_joints, forces=self.gripper_control_joints_maxF[:n_joints])

    # def get_camera_to_tcp(self):
    #     """
    #     get camera pose in tcp frame
    #     """
    #     return self.camera_to_tcp
    #
    # def set_camera_to_tcp(self, trans):
    #     """
    #     set camera pose in tcp frame
    #     """
    #     self.camera_to_tcp = trans

    def set_camera_states(self, wcT):
        """
        set camera pose in world frame

        wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
        """
        w_tcpT = np.matmul(wcT, np.linalg.inv(self.camera_to_tcp))  # tcp pose in world frame
        pos = w_tcpT[:3, 3]
        r = R.from_matrix(w_tcpT[:3, :3])
        orn = r.as_quat()
        self.move_to(pos, orn)

    def get_camera_states(self):
        """设置相机坐标系与末端坐标系的相对位置

        Arguments:
        - end_pos: len=3, end effector position
        - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

        Returns:
        - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
        """
        end_pos, end_orn = self.get_end_state()
        # relative_offset = [-0.05, 0, 0.1]  # 相机原点相对于末端执行器局部坐标系的偏移量
        # relative_offset = [-0.05, 0, 0.2]
        # end_orn = R.from_quat(end_orn).as_matrix()
        # end_x_axis, end_y_axis, end_z_axis = end_orn.T
        #
        # wcT = np.eye(4)  # w: world, c: camera, ^w_c T
        # wcT[:3, 0] = -end_y_axis  # camera x axis
        # wcT[:3, 1] = -end_z_axis  # camera y axis
        # wcT[:3, 2] = end_x_axis  # camera z axis
        # wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position

        end_orn = R.from_quat(end_orn).as_matrix()
        wcT = copy.copy(self.camera_to_tcp)
        wcT[:3, :3] = np.matmul(end_orn, wcT[:3, :3])
        relative_offset = self.camera_to_tcp[:3, 3]
        wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
        return wcT

    def render(self, align):
        wcT = self.get_camera_states()
        cwT = np.linalg.inv(wcT)
        return self.camera.render(cwT, align)

    def get_tcp_to_hand(self):
        htT = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        return htT

    def get_hand_to_grasp(self):
        ghT = np.eye(4)
        finger_height = 0.069
        ghT[0, 3] = -finger_height
        return ghT

    def setup(self):
        control_joints_name = self.arm_joints_name + self.gripper_joints_name

        joint_type_list = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        JointInfo = namedtuple('JointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity',
                                'controllable'])

        uid = self.uid
        num_joints = p.getNumJoints(uid)
        joints = dict()
        for i in range(num_joints):
            info = p.getJointInfo(uid, i)
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            joint_type = joint_type_list[info[2]]
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]
            joint_max_force = info[10]
            joint_max_vel = info[11]
            controllable = True if joint_name in control_joints_name else False

            info = JointInfo(joint_id, joint_name, joint_type, joint_lower_limit, joint_upper_limit,
                             joint_max_force, joint_max_vel, controllable)
            print(info)
            if info.type == 'REVOLUTE':
                p.setJointMotorControl2(uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            joints[info.name] = info

        return joints, control_joints_name

    def ikines(self, pos, orn):
        pos = np.clip(pos, a_min=[-0.25, -0.4, 0.14], a_max=[0.3, 0.4, 0.7])
        joints_pos = list(p.calculateInverseKinematics(self.uid, self.end_effector_id, pos, orn,
                                                       residualThreshold=10 ^ (-10), maxNumIterations=500))[:6]  # len=8
        return joints_pos

    def move_to(self, pos, orn):
        """Move arm to provided pose.

        Arguments:
        - pos: len=3, position (x, y, z) in world coordinate system
        - orn: len=4, quaternion (x, y, z, w) in world coordinate system
        - finger_angle: numeric, gripper's openess

        Returns:
        - joints_pose: len=8, angles for 8 controllable joints
        """
        pos = np.clip(pos, a_min=[-0.25, -0.4, 0.14], a_max=[0.3, 0.4, 0.7])
        joints_pos = self.ikines(pos, orn)
        self.arm_action(joints_pos)
        return joints_pos

    def reset_joints_pose(self):
        """Move to an ideal init point."""
        self.arm_action(self.arm_zero_joints_states)
        self.gripper_action(self.gripper_zero_joints_states)

    def set_base_pose(self, pos, orn):
        """
        - pos: len=3, (x, y, z) in world coordinate system
        - orn: len=4, (x, y, z, w) orientation in quaternion representation
        """
        self.base_pos = pos
        self.base_orn = orn
        p.resetBasePositionAndOrientation(self.uid, pos, orn)

    def get_joints_state(self):
        """Get all joints' angles and velocities.

        Returns:
        - joints_pos: len=n_joints, angles for all joints
        - joints_vel: len=n_joints, velocities for all joints
        """
        joints_state = p.getJointStates(self.uid, self.arm_control_joints_id)
        joints_pos = [s[0] for s in joints_state]
        joints_vel = [s[1] for s in joints_state]

        return joints_pos, joints_vel

    def get_end_state(self):
        """Get the position and orientation of the end effector.

        Returns:
        - end_pos: len=3, (x, y, z) in world coordinate system
        - end_orn: len=4, orientation in quaternion representation (x, y, z, w)
        """
        end_state = p.getLinkState(self.uid, self.end_effector_id)
        end_pos = end_state[0]
        end_orn = end_state[1]

        return end_pos, end_orn

    """
    gripper
    """

    def grasp(self, q_conf=0.0):
        """
        closes the gripper uniformly + attempts to find a grasp
        this is based on time + not contact points because contact points could just be a finger poking the object
        relies on grip_joints - specified by user/config file which joints should close
        """
        active_grasp_joints = self.gripper_control_joints_id[2:]
        q_conf_grasp_joints = self.gripper_control_joints_id[:2]
        q_conf_pos = [q_conf, -q_conf]
        finish_time = time.time() + 1.0
        while time.time() < finish_time:
            p.stepSimulation()
            for i, joint in enumerate(active_grasp_joints):
                p.setJointMotorControl2(bodyUniqueId=self.uid, jointIndex=joint, controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0.5, force=100)
            for i, joint in enumerate(q_conf_grasp_joints):
                p.setJointMotorControl2(bodyUniqueId=self.uid, jointIndex=joint, controlMode=p.POSITION_CONTROL,
                                        targetPosition=q_conf_pos[i], force=100)

    def relax(self):
        self.gripper_action(self.gripper_zero_joints_states)

    # def control_gripper(self, openness):
    #     """
    #     width: 0~1, 0->close, 1->open
    #     """
    #     openness = np.clip(openness, 0, 1)
    #     motor_pos = (1. - openness) / 25.
    #     gripper_id = [self.joints['left_gripper_motor'].id, self.joints['right_gripper_motor'].id]
    #     p.setJointMotorControlArray(self.uid, gripper_id, p.POSITION_CONTROL,
    #                                 targetPositions=[motor_pos, -motor_pos], targetVelocities=[0, 0],
    #                                 positionGains=[0.03, 0.03], forces=self.control_joints_maxF[-2:])

    # def control_gripper(self, width):
    #     motor_pos = (0.1-width)*0.5
    #     motor_pos = np.clip(motor_pos, 0, 0.05)
    #     gripper_id = [self.joints['left_gripper_motor'].id, self.joints['right_gripper_motor'].id]
    #     p.setJointMotorControlArray(self.uid, gripper_id, p.POSITION_CONTROL,
    #                                 targetPositions=[motor_pos, -motor_pos], targetVelocities=[0, 0],
    #                                 forces=self.control_joints_maxF[-2:])


