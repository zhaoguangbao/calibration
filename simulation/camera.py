import numpy as np
import open3d as o3d
import pybullet as p


# some codes are copied from https://github.com/ethz-asl/vgn.git


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy, scale=255.0):
        self.width = width
        self.height = height
        self.K = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]]
        )
        self.scale_ = scale

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def intrinsic_matrix(self):
        return self.K

    @property
    def scale(self):
        return self.scale_

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
            "scale": self.scale_
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
            scale=data["scale"]
        )
        return intrinsic

    @classmethod
    def from_param(cls, fov, width, height):
        """
        create CameraIntrinsic instance from param
        """
        fov = fov / 180 * np.pi
        focal_length = (height / 2) / np.tan(fov / 2)
        intrinsic = cls(
            width=width,
            height=height,
            fx=focal_length,
            fy=focal_length,
            cx=width / 2,
            cy=height / 2,
        )
        return intrinsic


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, intrinsic, near=0.01, far=4):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")

    def render(self, extrinsic, align=False, noise=False):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
            align: align world frame
            noise: if to add noise
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb, z_buffer, seg = np.ascontiguousarray(result[2][:, :, :3]), result[3], result[4]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        if noise:
            depth = self.add_noise(depth).astype(np.float32)
        if align:
            return Frame(rgb, depth, self.intrinsic, seg, extrinsic)
        else:
            return Frame(rgb, depth, self.intrinsic, seg)

    def add_noise(self,
                  depth,
                  lateral=True,
                  axial=True,
                  missing_value=True,
                  default_angle=85.0):
        """
        Add noise according to kinect noise model.
        Please refer to the paper "Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking".
        """
        h, w = depth.shape
        intrinsic = self.intrinsic.intrinsic_matrix
        point_cloud = self.compute_point_cloud(depth, intrinsic)
        surface_normal = self.compute_surface_normal_central_difference(point_cloud)
        # surface_normal = self.compute_surface_normal_least_square(point_cloud)
        cos = np.squeeze(np.dot(surface_normal, np.array([[0.0, 0.0, 1.0]], dtype=surface_normal.dtype).T))
        angles = np.arccos(cos)
        # adjust angles that don't satisfy the domain of noise model ([0, pi/2) for kinect noise model).
        cos[angles >= np.pi / 2] = np.cos(np.deg2rad(default_angle))
        angles[angles >= np.pi / 2] = np.deg2rad(default_angle)
        # add lateral noise
        if lateral:
            sigma_lateral = 0.8 + 0.035 * angles / (np.pi / 2 - angles)
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            # add noise offset to x axis
            new_x = x + np.round(np.random.normal(scale=sigma_lateral)).astype(np.int)
            # remove points that are out of range
            invalid_ids = np.logical_or(new_x < 0, new_x >= w)
            new_x[invalid_ids] = x[invalid_ids]
            # add noise offset to y axis
            new_y = y + np.round(np.random.normal(scale=sigma_lateral)).astype(np.int)
            # remove points that are out of range
            invalid_ids = np.logical_or(new_y < 0, new_y >= h)
            new_y[invalid_ids] = y[invalid_ids]
            depth = depth[new_y, new_x]
        # add axial noise
        if axial:
            # axial noise
            sigma_axial = 0.0012 + 0.0019 * (depth - 0.4) ** 2
            depth = np.random.normal(depth, sigma_axial)
        # remove some value according to the angle
        # the larger the angle, the higher probability the depth value is set to zero
        if missing_value:
            missing_mask = np.random.uniform(size=cos.shape) > cos
            depth[missing_mask] = 0.0
        return depth

    @staticmethod
    def compute_point_cloud(depth, intrinsic):
        """
        Compute point cloud by depth image and camera intrinsic matrix.
        :param depth: A float numpy array representing the depth image.
        :param intrinsic: A 3x3 numpy array representing the camera intrinsic matrix
        :return: Point cloud in camera space.
        """
        h, w = depth.shape
        w_map, h_map = np.meshgrid(np.arange(w), np.arange(h))
        image_coordinates = np.stack([w_map, h_map, np.ones_like(h_map, dtype=np.float32)], axis=2).astype(np.float32)
        inv_intrinsic = np.linalg.inv(intrinsic)
        camera_coordinates = np.expand_dims(depth, axis=2) * np.dot(image_coordinates, inv_intrinsic.T)
        return camera_coordinates

    @staticmethod
    def compute_surface_normal_central_difference(point_cloud):
        """
        Compute surface normal from point cloud.
        Notice: it only applies to point cloud map represented in camera space.
        The x axis directs in width direction, and y axis is in height direction.
        :param point_cloud: An HxWx3-d numpy array representing the point cloud map.The point cloud map
                            is restricted to the map in camera space without any other transformations.
        :return: An HxWx3-d numpy array representing the corresponding normal map.
        """
        h, w, _ = point_cloud.shape
        gradient_y, gradient_x, _ = np.gradient(point_cloud)
        normal = np.cross(gradient_x, gradient_y, axis=2)
        normal[normal == np.nan] = 0
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        flag = norm[..., 0] != 0
        normal[flag] = normal[flag] / norm[flag]
        return normal


class Frame(object):
    def __init__(self, rgb, depth, intrinsic: CameraIntrinsic, seg=None, extrinsic=None):
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )
        self.intrinsic_ = intrinsic

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        self.extrinsic = extrinsic if extrinsic is not None \
            else np.eye(4)

        self.seg = seg
    
    def color_image(self):
        return np.asarray(self.rgbd.color)
    
    def depth_image(self):
        return np.asarray(self.rgbd.depth)  # float [0, 1]

    def seg_imgae(self):
        # return np.asarray(self.seg, dtype=np.int8)
        return np.asarray(self.seg)

    def point_cloud(self):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=self.rgbd,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic
        )

        return pc


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho
