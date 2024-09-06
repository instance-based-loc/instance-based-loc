from .base_dataloader import BaseDataLoader
from typing import Optional, Tuple, Dict
from scipy.spatial.transform import Rotation
import os, json, sys
import numpy as np
from natsort import natsorted
import open3d as o3d
from tqdm import tqdm
import imageio
import ast

from utils import depth_utils

class HM3DDataloader(BaseDataLoader):
    def __init__(
            self, 
            data_path: str, 
            evaluation_indices: Optional[Tuple[int]], 
            focal_length_x: Optional[float] = None,
            focal_length_y: Optional[float] = None,
            map_pointcloud_cache_path: Optional[str] = None
        ):
        """
        Keep the evaluton_indices parameter as an empty list if you have separate directories
        for eval and env datasets.
        """
        super().__init__(data_path, evaluation_indices)

        # Setup paths
        self._depth_images_dir_path = os.path.join(self.data_path, "depth")
        self._rgb_images_dir_path = os.path.join(self.data_path, "rgb")
        self._pose_information_file_path = os.path.join(self.data_path, "poses.npy")

        # Setup file paths
        self._depth_images_original_paths = [
            os.path.join(self._depth_images_dir_path, filename) 
            for filename in natsorted(os.listdir(self._depth_images_dir_path))
        ]
        self._depth_images_paths = []
        self._rgb_images_paths = [
            os.path.join(self._rgb_images_dir_path, filename) 
            for filename in natsorted(os.listdir(self._rgb_images_dir_path))
        ]  
        
        assert len(self._depth_images_original_paths) == len(self._rgb_images_paths), "No. of depth and RGB images are not the same!"

        # Create a new directory for squeezed depth images if it doesn't exist
        self._squeezed_depth_images_dir_path = os.path.join(self.data_path, "depth_squeezed")
        os.makedirs(self._squeezed_depth_images_dir_path, exist_ok=True)

        # Process and save squeezed depth images
        for depth_image_original_path in self._depth_images_original_paths:
            depth_image = np.load(depth_image_original_path)
            squeezed_depth_image = np.squeeze(depth_image)

            # Create a new filename with "_squeezed" added before the .npy extension
            original_filename = os.path.basename(depth_image_original_path)
            new_filename = f"{os.path.splitext(original_filename)[0]}_squeezed.npy"
            new_filepath = os.path.join(self._squeezed_depth_images_dir_path, new_filename)

            # Save the squeezed depth image to the new directory
            np.save(new_filepath, squeezed_depth_image)

            # Append the new file path to the list
            self._depth_images_paths.append(new_filepath)

        print("Squeezed all depth images")

        # Set poses
        unadjusted_pose_data = np.load(self._pose_information_file_path)
        self._poses = []
        for pose in unadjusted_pose_data:
            adjusted_pose = pose
            adjusted_pose[-2] *= -1
            self._poses.append(adjusted_pose)

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
        for env_idx in tqdm(self.environment_indices, desc="Forming pointcloud map from env. images"):
            cur_pointcloud = \
                depth_utils.get_coloured_pointcloud_from_depth(
                    np.load(self._depth_images_paths[env_idx]), np.asarray(imageio.imread(self._rgb_images_paths[env_idx])), self.focal_length_x, self.focal_length_y, None
                )
            transformed_cur_pointcloud = depth_utils.transform_pointcloud(cur_pointcloud, self._poses[env_idx])
            self.map_pointcloud += transformed_cur_pointcloud
        print("Formed map pointcloud")
        print(self.map_pointcloud)

    def _get_environment_indices(self) -> Tuple[int, ...]:
        return [i for i in range(len(self._depth_images_paths)) if i not in self.evaluation_indices]

    def get_image_data(self, index: int):
        cur_pose = self._poses[index]
        return self._rgb_images_paths[index], self._depth_images_paths[index], cur_pose

    def get_pointcloud(self, bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> o3d.geometry.PointCloud:
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