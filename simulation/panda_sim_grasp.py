import time
import numpy as np
import math
from collections import namedtuple
import pybullet as p
from camera import Camera, CameraIntrinsic, Frame
from scipy.spatial.transform import Rotation as R
import json


class PandaSim(object):
    def __init__(self, urdf_file, camera_config):
        # p.setPhysicsEngineParameter(solverResidualThreshold=0)

        self.base_pos = [-0.5, 0.0, -0.1]
        self.base_orn = [0, 0, 0, 1]  # quaternion (x, y, z, w)
        self.end_effector_id = 8
        
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.uid = p.loadURDF(urdf_file, self.base_pos, self.base_orn, useFixedBase=True, flags=flags)
        self.joints, self.control_joints_name = self.setup()
        self.control_joints_id = [self.joints[name].id for name in self.control_joints_name]
        self.control_joints_maxF = [self.joints[name].maxForce for name in self.control_joints_name]
        self.reset_joints_pose()

        with open(camera_config, "r") as j:
            config = json.load(j)
        camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        # camera_intrinsic = CameraIntrinsic.from_param(60, 1280, 720)
        self.camera = Camera(camera_intrinsic)

    def get_camera_states(self):
        """设置相机坐标系与末端坐标系的相对位置

        Arguments:
        - end_pos: len=3, end effector position
        - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

        Returns:
        - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
        """
        end_pos, end_orn = self.get_end_state()
        relative_offset = [-0.08, 0, 0.0]  # 相机原点相对于末端执行器局部坐标系的偏移量
        end_orn = R.from_quat(end_orn).as_matrix()
        # end_x_axis, end_y_axis, end_z_axis = end_orn.T

        wcT = np.eye(4)  # w: world, c: camera, ^w_c T
        wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
        return wcT

    def render(self, align):
        wcT = self.get_camera_states()
        cwT = np.linalg.inv(wcT)
        return self.camera.render(cwT, align)

    def get_tcp_to_hand(self):
        htT = np.array([[0, 0, 1, -0.05], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        return htT

    def setup(self):
        control_joints_name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6',
            'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

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

    def action(self, cmds):
        """
        - cmds: len=8, target angles for 8 controllable joints
        """
        n_joints = len(cmds)
        p.setJointMotorControlArray(self.uid, self.control_joints_id[:n_joints], p.POSITION_CONTROL,
                                    targetPositions=cmds, targetVelocities=[0] * n_joints,
                                    positionGains=[0.03] * n_joints, forces=self.control_joints_maxF[:n_joints])

    def ikines(self, pos, orn):
        # pos = np.clip(pos, a_min=[-0.25, -0.4, 0.14], a_max=[0.3, 0.4, 0.7])
        joints_pos = list(p.calculateInverseKinematics(self.uid, self.end_effector_id, pos, orn,
                                                       residualThreshold=10 ^ (-10), maxNumIterations=500))
        return joints_pos

    def move_to(self, pos, orn):
        joints_pos = self.ikines(pos, orn)
        self.action(joints_pos)
        return joints_pos

    def control_gripper(self, width):
        """
        width:
        """
        motor_pos = (0.1-width)*0.5
        motor_pos = np.clip(motor_pos, 0, 0.05)
        gripper_id = [self.joints['panda_finger_joint1'].id, self.joints['panda_finger_joint2'].id]
        p.setJointMotorControlArray(self.uid, gripper_id, p.POSITION_CONTROL,
                                    targetPositions=[motor_pos, motor_pos], targetVelocities=[0, 0],
                                    forces=self.control_joints_maxF[-2:])

    def reset_joints_pose(self):
        """Move to an ideal init point."""
        self.action([0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587,
                     0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0.01999436579245478,
                     0.019977024051412193])
            
    def set_base_pose(self, pos, orn):
        """
        - pos: len=3, (x, y, z) in world coordinate system
        - orn: len=4, (x, y, z, w) orientation in quaternion representation
        """
        self.base_pos = pos
        self.base_orn = orn
        p.resetBasePositionAndOrientation(self.uid, pos, orn)

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

    def get_joints_state(self):
        """Get all joints' angles and velocities.

        Returns:
        - joints_pos: len=n_joints, angles for all joints
        - joints_vel: len=n_joints, velocities for all joints
        """
        joints_state = p.getJointStates(self.uid, self.control_joints_id)
        joints_pos = [s[0] for s in joints_state]
        joints_vel = [s[1] for s in joints_state]

        return joints_pos, joints_vel
