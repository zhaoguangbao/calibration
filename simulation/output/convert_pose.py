import os
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
pose_fn = os.path.join(ROOT_DIR, 'camera_calibration', 'pose_ori.txt')


pose = np.loadtxt(pose_fn)  # pose[0:3] -- tran (mm), pose[3:6] -- xyz(rpy)
pose_process = np.zeros((pose.shape[0], 7))  # pose_process[0:3] -- tran (m), pose_process[3:7] -- quat
r = R.from_euler('xyz', pose[:, 3:], degrees=True)
pose_process[:, 3:] = r.as_quat()
pose_process[:, :3] = pose[:, :3] / 1000.0
np.savetxt(os.path.join(ROOT_DIR, 'camera_calibration', 'pose.txt'), pose_process, fmt='%.6f')

