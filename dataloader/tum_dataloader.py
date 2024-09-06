from .base_dataloader import BaseDataLoader
from typing import Optional, Tuple, Dict
from scipy.spatial.transform import Rotation
import os, json, sys
import numpy as np
from natsort import natsorted
import open3d as o3d
from tqdm import tqdm
import imageio

sys.path.insert(0, "..")
from utils import depth_utils

class TUMDataloader(BaseDataLoader):
    def __init__(
            self, 
            data_path: str, 
            evaluation_indices: Optional[Tuple[int]], 
            focal_length_x: Optional[float] = None,
            focal_length_y: Optional[float] = None,
            map_pointcloud_cache_path: Optional[str] = None,
            start_file_index: Optional[int] = 0,
            last_file_index: Optional[int] = None,
            sampling_period: Optional[int] = 10,
        ):
        """
        Keep the evaluation_indices parameter as an empty list if you have separate directories
        for eval and env datasets.
        """
        super().__init__(data_path, evaluation_indices)

        # Setup paths
        self._depth_images_dir_path=os.path.join(self.data_path, "depth")
        self._rgb_images_dir_path=os.path.join(self.data_path, "rgb")
        self._pose_information_file_path=os.path.join(self.data_path, "groundtruth.txt")

        # setup file path
        self._depth_images_paths = [
            os.path.join(self._depth_images_dir_path, filename) 
            for filename in 
            natsorted(os.listdir(self._depth_images_dir_path))
        ]
        self._rgb_images_paths = [
            os.path.join(self._rgb_images_dir_path, filename) 
            for filename in 
            natsorted(os.listdir(self._rgb_images_dir_path))
        ]
        assert len(self._depth_images_paths) == len(self._rgb_images_paths), "No. of depth and RGB images are not the same!"

        # subsample file names 
        if last_file_index == None:
            last_file_index = len(self._depth_images_paths)
        self._depth_images_paths = self._depth_images_paths[start_file_index:last_file_index:sampling_period]
        self._rgb_images_paths = self._rgb_images_paths[start_file_index:last_file_index:sampling_period]

        # addn rot matrix
        R2 = Rotation.from_euler('xyz', [0,  np.pi, 0]).as_matrix()

        # Set poses
        self._poses = []
        with open(self._pose_information_file_path, 'r') as f:
            pose_file_data = f.readlines()
            for pose in pose_file_data:
                split_pose = pose.split()

                # account for kinect world frame being different
                R1 = Rotation.from_quat([float(i) for i in split_pose[3:]]).as_matrix()
                q = Rotation.from_matrix(R1 @ R2).as_quat()

                split_pose[:3] = [-float(i) for i in split_pose[:3]]
                split_pose[3:] = q

                # x y z qx qy qz qw
                p = np.array([float(i) for i in split_pose])
                self._poses.append(p)

        # subsample poses
        self._poses = self._poses[start_file_index:last_file_index:sampling_period]

        if map_pointcloud_cache_path is not None and os.path.exists(map_pointcloud_cache_path):
            print("Retrieving map's pointcloud from cache")
            self.map_pointcloud = o3d.io.read_point_cloud(map_pointcloud_cache_path)
        else:
            print("Creating the map's pointcloud")

            self.focal_length_x = focal_length_x
            self.focal_length_y = focal_length_y
            self.setup_map_pointcloud()

            # Save the pointcloud if the user has given a path
            if map_pointcloud_cache_path is not None:
                print("Saving the map's pointcloud")
                o3d.io.write_point_cloud(map_pointcloud_cache_path, self.get_pointcloud())

    def setup_map_pointcloud(self) -> None:
        """
            Function that creates the pointcloud of the environment
        """
        self.map_pointcloud = o3d.geometry.PointCloud()
        for env_idx in tqdm([i for i in self.environment_indices][::50], desc="Forming pointcloud map from env. images"):
            rgb_image = np.asarray(imageio.imread(self._rgb_images_paths[env_idx]))
            depth_img = np.asarray(imageio.imread(self._depth_images_paths[env_idx]), dtype=np.float32)
            depth_img /= 5000.0 # for kinect camera
            cur_pointcloud = \
                depth_utils.get_coloured_pointcloud_from_depth(
                    depth_img, rgb_image, self.focal_length_x, self.focal_length_y
                )
            # print(cur_pointcloud)
            transformed_cur_pointcloud = depth_utils.transform_pointcloud_kinect(cur_pointcloud, self._poses[env_idx])
            self.map_pointcloud += transformed_cur_pointcloud
        
        self.map_pointcloud = self.map_pointcloud.voxel_down_sample(0.025)

    def _get_environment_indices(self) -> Tuple[int, ...]:
        return [i for i in range(len(self._depth_images_paths)) if i not in self.evaluation_indices]

    def get_image_data(self, index: int) -> Tuple[str, Optional[str], np.ndarray]:
        cur_pose = self._poses[index]

        return self._rgb_images_paths[index], self._depth_images_paths[index], cur_pose

    def get_pointcloud(self, bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        if bounding_box is not None:
            raise NotImplementedError
        
        return self.map_pointcloud
    
    def get_visible_pointcloud(self, pose, fov, near_clip, far_clip) -> o3d.geometry.PointCloud:
        t = pose[:3]
        q = pose[3:]

        q /= np.linalg.norm(q)
        R = Rotation.from_quat(q).as_matrix()
        R_inv = R.T
        
        # Get points and colors from the point cloud
        pointcloud_points = np.asarray(self.get_pointcloud().points)
        pointcloud_colors = np.asarray(self.get_pointcloud().colors)
        
        # Transform points to the camera's coordinate system
        transformed_pointcloud = np.dot(pointcloud_points - t, R_inv.T)

        # Define camera parameters
        fov_rad = np.deg2rad(fov)
        tan_half_fov = np.tan(fov_rad / 2)

        # Filter points and colors based on the camera's FOV and clipping planes
        visible_points = []
        visible_colors = []
        for point, color in zip(transformed_pointcloud, pointcloud_colors):
            x, y, z = point
            # print(z)
            if z < near_clip or z > far_clip:
                continue
            if abs(x / z) > tan_half_fov or abs(y / z) > tan_half_fov:
                continue
            visible_points.append(point)
            visible_colors.append(color)

        # Create and return a new point cloud with visible points and colors
        visible_pointcloud = o3d.geometry.PointCloud()
        visible_pointcloud.points = o3d.utility.Vector3dVector(np.array(visible_points))
        visible_pointcloud.colors = o3d.utility.Vector3dVector(np.array(visible_colors))
        
        return visible_pointcloud