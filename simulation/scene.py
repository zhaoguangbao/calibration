import pybullet as p
import pybullet_data
import math
import random
import numpy as np
from pyquaternion import Quaternion  # Matrix must be orthogonal, i.e. its transpose should be its inverse
from scipy.spatial.transform import Rotation


class Scene(object):
    def __init__(self):
        """
        urdfs_list: urdf filepath list
        """
        self.urdfs_list = []

        self.num_urdf = 0
        self.urdfs_id = []
        self.other_id = []
        self.setup()

    def setup(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF('plane.urdf', [0, 0, -0.63], [0, 0, 0, 1])
        self.table0 = p.loadURDF('table/table.urdf', [0, 0.5, -0.63], [0, 0, 0, 1])
        self.table1 = p.loadURDF('table/table.urdf', [0, -0.5, -0.63], [0, 0, 0, 1])
        # self.bucket = p.loadURDF('tray/tray.urdf', [-0.4, 0.5, 0], [0, 0, 0, 1])
        self.other_id.append(self.plane)
        self.other_id.append(self.table0)
        self.other_id.append(self.table1)

    def rand_distribute(self, file_name, x_min=-0.2, x_max=0.1, y_min=-0.1, y_max=0.1, z_min=0.2, z_max=0.5, scale=1.0):
        xyz = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max], size=3)
        rpy = np.random.uniform(-np.pi, np.pi, size=3)
        orn = p.getQuaternionFromEuler(rpy)
        object_id = p.loadURDF(file_name, xyz, orn, globalScaling=scale)
        return object_id

    def set_distribute(self, file_name, pose, scale=1.0):
        """
        load urdf in specified poses
        """
        xyz = pose[:3, 3]
        # print(np.matmul(pose[:3, :3], pose[:3, :3].T))
        try:
            r = Rotation.from_matrix(pose[:3, :3])
            orn = r.as_quat()
            # orn_ = Quaternion(matrix=pose[:3, :3])
            # orn = (orn_.x, orn_.y, orn_.z, orn_.w)
            object_id = p.loadURDF(file_name, xyz, orn, globalScaling=scale, useFixedBase=True)
            return object_id
        except Exception as e:
            print(e)
            print('set_distribute error:', file_name)
            return -1
    
    def urdf_nums(self):
        return len(self.urdfs_list)

    def loadObjsInURDF(self, urdfs_list, poses_list=None):
        """
        **Inputs:**

        - urdfs_list: list of models path
        - poses_list: list of models poses
        """
        self.urdfs_list = urdfs_list
        print('self.urdfs_filename = ', self.urdfs_list)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        self.num_urdf = len(self.urdfs_list)
        for i, urdf_filename in enumerate(self.urdfs_list):
            if poses_list is None:
                urdf_id = self.rand_distribute(urdf_filename)
            else:
                urdf_id = self.set_distribute(urdf_filename, poses_list[i])
            if urdf_id != -1:
                inf = p.getVisualShapeData(urdf_id)[0]

                self.urdfs_id.append(urdf_id)
                self.urdfs_xyz.append(inf[5])
                self.urdfs_scale.append(inf[3][0])
            else:
                raise NotImplementedError('unable to import urdf models')

    def removeObjsInURDF(self):
        for id in self.urdfs_id:
            p.removeBody(id)
        self.urdfs_id = []
    
