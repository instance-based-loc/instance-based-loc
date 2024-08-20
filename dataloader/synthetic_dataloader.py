from .base_dataloader import BaseDataLoader
from typing import Optional, Tuple, Dict
from scipy.spatial.transform import Rotation
import os, json, sys
import numpy as np
from natsort import natsorted
import open3d as o3d
from tqdm import tqdm
import imageio

from utils import depth_utils

class SynthDataloader(BaseDataLoader):
    def __init__(
            self, 
            data_path: str, 
            evaluation_indices: Optional[Tuple[int]], 
            camera_focal_lenth: Optional[float] = None,
            map_pointcloud_cache_path: Optional[str] = None
        ):
        """
        Keep the evaluton_indices parameter as an empty list if you have separate directories
        for eval and env datasets.
        """
        super().__init__(data_path, evaluation_indices)

        # Setup paths
        self._depth_images_dir_path=os.path.join(self.data_path, "depth")
        self._rgb_images_dir_path=os.path.join(self.data_path, "rgb")
        self._pose_information_file_path=os.path.join(self.data_path, "poses.json")

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

        # Set poses
        self._poses = []
        with open(self._pose_information_file_path, 'r') as f:
            pose_file_data = json.load(f)
            for view in pose_file_data["views"]:
                t = np.array([view["position"]["x"], view["position"]["y"], view["position"]["z"]])

                rotation_euler = [view["rotation"]["x"], view["rotation"]["y"], view["rotation"]["z"]]

                q = Rotation.from_euler('xyz', rotation_euler, degrees=True).as_quat()

                pose = np.concatenate([t, q])
                self._poses.append(pose)

        if map_pointcloud_cache_path is not None and os.path.exists(map_pointcloud_cache_path):
            print("Retrieving map's pointcloud from cache")
            self.map_pointcloud = o3d.io.read_point_cloud(map_pointcloud_cache_path)
        else:
            print("Creating the map's pointcloud")

            self.focal_length = camera_focal_lenth
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
                    np.load(self._depth_images_paths[env_idx]), np.asarray(imageio.imread(self._rgb_images_paths[env_idx])), self.focal_length, self.focal_length
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