"""
common util tools for python
"""
import math
import os
import torch
from PIL import Image
import cvxopt as cvx
import cv2
import pickle
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import KDTree
import shutil  # copy file


def query_nn(ref, query, upper_dist=0.03, ret_index=False):
    """
    **Inputs:**

    - ref: np.ndarray (N1, 3)
    - query: : np.ndarray (N2, 3)
    - upper_dist: float scalar, threshold

    **Outputs:**

    nn points locate on query relative ref.
    if ret_index is True, return index (np.ndarray)
    """
    # query
    tree = KDTree(query)
    view_inds_ = tree.query(ref, k=1, distance_upper_bound=upper_dist)
    view_inds = view_inds_[-1]
    view_inds_ = view_inds[view_inds != query.shape[0]]
    query_ = torch.index_select(torch.FloatTensor(query), 0, torch.LongTensor(view_inds_))
    if not ret_index:
        return query_.numpy()
    else:
        return query_.numpy(), view_inds


"""
for grasp (same with models.model_utils.loss_utils)
"""


def generate_grasp_views(N=300, phi=(np.sqrt(5)-1)/2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch

        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y, 1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


"""
for cloud_util
"""


def snr_noise(data: np.ndarray, snr):
    """
    add noise to data, larger snr -- small noise, general in (0, 100)
    data: n x 3
    """
    noise = np.random.randn(data.shape[0], data.shape[1])
    # noise = noise - np.mean(noise, axis=0)
    data_power = np.linalg.norm(data - data.mean(axis=0)) ** 2 / data.shape[0]
    noise_var = data_power / np.power(10., snr / 10.)
    noise = (np.sqrt(noise_var) / np.std(noise)) * noise
    return noise + data


def rot_matrix_to_quaternion(rot_matrix):
    q = Quaternion(rot_matrix)  # x, y, z, w
    return q


def quaternion_to_rot_matrix(q):
    return q.rotation_matrix


def normalize_array(np_array):
    """
    **Inputs:**

    - np_array: shape (n,)
    """
    eps = 0.00001
    max_e = np.max(np_array)
    min_e = np.min(np_array)
    norm_array = (np_array-min_e)/(max_e-min_e+eps)
    return norm_array


def isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def rad2deg(rads):
    return 180. * rads / math.pi


def deg2rad(degs):
    return math.pi * degs / 180.


def show_rgb(rgb, title='show image'):
    cv2.imshow(title, rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_depth(depth, title='show depth'):
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=15), cv2.COLORMAP_JET)
    im = Image.fromarray(im_color)
    im.show(title)


def show_heatmap(heatmap, title='show heatmap'):
    heatmapshow = None
    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    cv2.imshow(title, heatmapshow)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_path(filename):
    save_dir = os.path.dirname(filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def save_image(image_numpy, image_path):
    """
    RGB
    """
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_depth(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    cv2.imwrite(image_path, image_numpy)  # uint [0, 255]


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """
    **Notes:**

    - for [0, 255], camera.scale: 255
    - for [0, 1000], camera.scale: 1000
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth.astype(np.float32) / camera.scale  # camera.scale to normalize depth data
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def rot_z(angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    trans = np.eye(4)
    trans[:3, :3] = rotation
    return trans


def rot_y(angle):
    """
    return the rotation matrix where rotation axis is y
    """
    rotation = np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [np.sin(angle), 0.0, np.cos(angle)]])
    trans = np.eye(4)
    trans[:3, :3] = rotation
    return trans


def rot_x(angle):
    """
    return the rotation matrix where rotation axis is x
    """
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(angle), -np.sin(angle)],
                         [0.0, np.sin(angle), np.cos(angle)]])
    trans = np.eye(4)
    trans[:3, :3] = rotation
    return trans


def copy_urdf_to_models(root, src_file):
    """
    copy model.urdf to models directory
    """
    filename = os.path.basename(src_file)
    for i in range(88):
        dir = os.path.join(root, 'models', '%03d' % i, filename)
        shutil.copyfile(src_file, dir)


def ply_to_simple_obj(root):
    import open3d as o3d
    filename = 'textured.obj'
    for i in range(1):
        path = os.path.join(root, 'models', '%03d' % i)
        filepath = os.path.join(path, filename)
        mesh = o3d.io.read_triangle_mesh(filepath)

        # color_path = os.path.join(path, 'texture_map.png')
        # color_raw = o3d.io.read_image(color_path)

        # mesh.compute_vertex_normals()
        print(f'Input mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 64
        print(f'voxel_size = {voxel_size:e}')
        mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
        # mesh_smp.textures = [o3d.geometry.Image(color_raw)]
        print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
        o3d.io.write_triangle_mesh(os.path.join(path, 'textured_simplified.obj'), mesh_smp)




