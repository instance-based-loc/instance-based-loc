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

class EightRoomDataLoader(BaseDataLoader):
    def __init__(
            self, 
            data_path: str, 
            evaluation_indices: Optional[Tuple[int]], 
            focal_length_x: Optional[float] = None,
            focal_length_y: Optional[float] = None,
            map_pointcloud_cache_path: Optional[str] = None,
            rot_correction: Optional[float] = 0.0,
            start_file_index: Optional[int] = 0,
            last_file_index: Optional[int] = None,
            sampling_period: Optional[int] = 10,
        ):
        """
        Keep the evaluton_indices parameter as an empty list if you have separate directories
        for eval and env datasets.

        Start, last and sampling period define the subsampling from all available files
        """
        super().__init__(data_path, evaluation_indices)

        # Setup paths
        self._depth_images_dir_path=os.path.join(self.data_path, "depth")
        self._rgb_images_dir_path=os.path.join(self.data_path, "rgb")
        self._pose_information_file_path=os.path.join(self.data_path, "pose")

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
        self._pose_file_paths = [
            os.path.join(self._pose_information_file_path, filename) 
            for filename in 
            natsorted(os.listdir(self._pose_information_file_path))
        ]  
        assert len(self._depth_images_paths) == len(self._rgb_images_paths), "No. of depth and RGB images are not the same!"
        assert len(self._depth_images_paths) == len(self._pose_file_paths), "No. of depth and pose images are not the same!"
        assert len(self._pose_file_paths) == len(self._rgb_images_paths), "No. of pose and RGB images are not the same!"

        # subsample file names 
        if last_file_index == None:
            last_file_index = len(self._depth_images_paths)
        self._depth_images_paths = self._depth_images_paths[start_file_index:last_file_index:sampling_period]
        self._rgb_images_paths = self._rgb_images_paths[start_file_index:last_file_index:sampling_period]
        self._pose_file_paths = self._pose_file_paths[start_file_index:last_file_index:sampling_period]

        # Set poses
        self._poses = []
        for pose_file_path in self._pose_file_paths:
            with open(pose_file_path, 'r') as file:
                pose_dict = file.read()
            pose_dict = ast.literal_eval(pose_dict)
            pose_dict = {
                "position": {
                    "x": pose_dict[0]['x'],
                    "y": pose_dict[0]['y'],
                    "z": pose_dict[0]['z']
                },
                "rotation": {
                    "x": pose_dict[1]['x'] + rot_correction,
                    "y": pose_dict[1]['y'],
                    "z": pose_dict[1]['z']
                }
            }

            q = Rotation.from_euler('xyz', [r for _, r in pose_dict["rotation"].items()], degrees=True).as_quat()
            t = np.array([x for _, x in pose_dict["position"].items()])

            pose = np.concatenate([t, q])
            self._poses.append(pose)

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
                    np.load(self._depth_images_paths[env_idx]), np.asarray(imageio.imread(self._rgb_images_paths[env_idx])), self.focal_length_x, self.focal_length_y
                )
            transformed_cur_pointcloud = depth_utils.transform_pointcloud(cur_pointcloud, self._poses[env_idx])
            self.map_pointcloud += transformed_cur_pointcloud

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

# NOTE - can be used to form depth map, but it will only **look like** the depth map in the dataset
#        called "sense" of depthmap as we negate the y axis while forming it
def get_sense_of_depthmap_from_pointcloud(
        pointcloud: o3d.geometry.PointCloud,
        image_width: int,
        image_height: int,
        focal_length_x: float,
        focal_length_y: float
) -> np.ndarray:
    points = np.asarray(pointcloud.points)
    
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    x_pixel = (X * focal_length_x / Z) + (image_width / 2)
    y_pixel = (Y * focal_length_y / Z) + (image_height / 2)

    x_pixel = np.clip(np.round(x_pixel).astype(int), 0, image_width - 1)
    y_pixel = np.clip(np.round(y_pixel).astype(int), 0, image_height - 1)

    depth_map = np.zeros((image_height, image_width), dtype=np.float32)

    depth_map[-y_pixel, x_pixel] = Z

    return depth_map