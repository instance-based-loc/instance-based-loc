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

class SynthDataloader(BaseDataLoader):
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
                q = Rotation.from_euler('zyx', [r for _, r in view["rotation"].items()], degrees=True).as_quat()
                t = np.array([x for _, x in view["position"].items()])
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
                depth_utils.get_pointcloud_from_depth(
                    np.load(self._depth_images_paths[env_idx]), self.focal_length_x, self.focal_length_y
                )
            transformed_cur_pointcloud = depth_utils.transform_pointcloud(cur_pointcloud, self._poses[env_idx])
            self.map_pointcloud += transformed_cur_pointcloud

    def _get_environment_indices(self) -> Tuple[int, ...]:
        return [i for i in range(len(self._depth_images_paths)) if i not in self.evaluation_indices]

    def get_image_data(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        cur_rgb_image = np.asarray(imageio.imread(self._rgb_images_paths[index]))
        cur_depth_image = np.load(self._depth_images_paths[index])
        cur_pose = self._poses[index]

        print(cur_rgb_image.shape, cur_depth_image.shape, cur_pose.shape)
        return cur_rgb_image, cur_depth_image, cur_pose

    def get_pointcloud(self, bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        if bounding_box is not None:
            raise NotImplementedError
        
        return self.map_pointcloud