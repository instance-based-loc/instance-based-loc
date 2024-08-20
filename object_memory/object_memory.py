from typing import Type
import torch
import open3d as o3d
import numpy as np
from typing import Callable
import imageio

from .object_finder import ObjectFinder
from .object_info import ObjectInfo
from utils.logging import conditional_log
from utils.depth_utils import get_mask_coloured_pointclouds_from_depth, transform_pointcloud, DEFAULT_OUTLIER_REMOVAL_CONFIG
from utils.datatypes import TypedList

print("\033[34mLoaded modules for object_memory.object_memory\033[0m")

def default_load_rgb(path: str) -> np.ndarray:
    return np.asarray(imageio.imread(path))

def default_load_depth(path: str) -> np.ndarray:
    return np.load(path)

class ObjectMemory():
    def _load_rgb_image(self, path: str) -> np.ndarray:
        return self.load_rgb_image_func(path)

    def _load_depth_image(self, path: str) -> np.ndarray:
        return self.load_depth_image_func(path)

    def _get_embeddings(self, *args) -> torch.Tensor:
        return self.get_embeddings_func(*args)

    def _log(self, statement: any) -> None:
        """
        Conditionally log a statement if logging is enabled.

        Args:
            statement (any): The statement to log.
        """
        conditional_log(statement, self.log_enabled)

    def __init__(
        self,
        device: Type[str],
        ram_pretrained_path: Type[str],
        sam_checkpoint_path: Type[str],
        camera_focal_lenth_x: Type[float],
        camera_focal_lenth_y: Type[float],
        get_embeddings_func,
        log_enabled: Type[bool] = True,
        mem_formation_bounding_box_threshold = 0.3,
        mem_formation_occlusion_overlap_threshold = 0.9,
        object_info_max_embeddings_num = 5,
        load_rgb_image_func = default_load_rgb,
        load_depth_image_func = default_load_depth,
    ):
        # ***************************************************************************
        # Setup encoder in the init of the concrete class and then do init for super
        # ***************************************************************************

        self.device = device
        self.ram_pretrained_path = ram_pretrained_path
        self.sam_checkpoint_path = sam_checkpoint_path
        self.camera_focal_lenth_x = camera_focal_lenth_x
        self.camera_focal_lenth_y = camera_focal_lenth_y
        self.log_enabled = log_enabled
        self.mem_formation_bounding_box_threshold = mem_formation_bounding_box_threshold
        self.mem_formation_occlusion_overlap_threshold = mem_formation_occlusion_overlap_threshold
        self.object_info_max_embeddings_num = object_info_max_embeddings_num
        self.get_embeddings_func = get_embeddings_func
        self.load_rgb_image_func = load_rgb_image_func
        self.load_depth_image_func = load_depth_image_func

        ObjectFinder.setup(
            device = self.device,
            ram_pretrained_path = self.ram_pretrained_path,
            sam_checkpoint_path = self.sam_checkpoint_path,
            log_enabled = self.log_enabled
        )

        self.memory = TypedList[ObjectInfo]()

    def _get_object_info(self, rgb_image_path, depth_image_path, consider_floor, outlier_removal_config):
        obj_grounded_imgs, obj_bounding_boxes, obj_masks, obj_phrases = ObjectFinder.find(rgb_image_path, consider_floor)

        if obj_grounded_imgs is None:
            return None, None, None
        
        embs = np.array(
            self._get_embeddings(
                obj_grounded_imgs, 
                obj_bounding_boxes, 
                obj_masks, 
                obj_phrases,
                rgb_image_path,
                depth_image_path,
                consider_floor
            ).cpu()
        )

        obj_pointclouds = get_mask_coloured_pointclouds_from_depth(
            depth_image = self._load_depth_image(depth_image_path),
            rgb_image = self._load_rgb_image(rgb_image_path),
            masks =  obj_masks,
            focal_length_x = self.camera_focal_lenth_x,
            focal_length_y = self.camera_focal_lenth_y,
            outlier_removal_config = outlier_removal_config
        )

        assert(len(obj_grounded_imgs) == len(obj_bounding_boxes) \
                and len(obj_bounding_boxes) == len(obj_masks) \
                and len(obj_masks) == len(obj_phrases) \
                and len(embs) == len(obj_phrases))
        
        return obj_phrases, embs, obj_pointclouds

    def process_image(
        self,
        rgb_image_path,
        depth_image_path,
        pose,
        outlier_removal_config = DEFAULT_OUTLIER_REMOVAL_CONFIG,
        add_noise = False,
        pose_noise = {'trans': 0.0005, 'rot': 0.0005},
        depth_noise = 0.003,
        consider_floor= True,
        min_points = 500,
        will_cluster_later = True
    ):
        def num_points_in_pointcloud(pcd):
            return len(np.asarray(pcd.points))

        obj_phrases, embs, obj_pointclouds = self._get_object_info(rgb_image_path, depth_image_path, consider_floor, outlier_removal_config)
        
        if obj_phrases is None:
            self._log("BaseObjectMemory.process_image did NOT find any objects")
            return
        else:
            self._log(f"BaseObjectMemory.process_image found: {obj_phrases}")

        if pose is None: 
            raise NotImplementedError("Although we can implement ICP for poses between subsequent image, we require it now.")
        
        if add_noise:
            def add_noise_to_array(array, noise_level):
                noise = np.random.normal(0, noise_level, array.shape)
                noisy_array = array + noise
                return noisy_array
            def normalize_quaternion(quaternion):
                norm = np.linalg.norm(quaternion)
                if norm == 0:
                    return quaternion
                return quaternion / norm
            
            # Add noise to pose
            pose[:3] = add_noise_to_array(pose[:3], pose_noise['trans'])
            pose[3:] = normalize_quaternion(add_noise_to_array(pose[3:], pose_noise['rot']))
        
            def add_noise_to_pointcloud(pcd, noise_level):
                points = np.asarray(pcd.points)
                noisy_points = add_noise_to_array(points, noise_level)
                
                # Create a new point cloud with noisy points
                noisy_pcd = o3d.geometry.PointCloud()
                noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)
                
                # Preserve colors
                colors = np.asarray(pcd.colors)
                noisy_pcd.colors = o3d.utility.Vector3dVector(colors)
                
                return noisy_pcd

            # Add noise to pointclouds
            obj_pointclouds = [add_noise_to_pointcloud(obj_pointcloud, depth_noise) for obj_pointcloud in obj_pointclouds]

        transformed_pointclouds = [transform_pointcloud(obj_pointcloud, pose) for obj_pointcloud in obj_pointclouds]

        for i, (obj_phrase, obj_emb, obj_pointcloud) in enumerate(zip(obj_phrases, embs, transformed_pointclouds)):
            self._log(f"\tCurrent Object Phrase under consideration for BaseObjectMemory.process_image: {obj_phrase}")

            if num_points_in_pointcloud(obj_pointcloud) < min_points:
                self._log(f"\t\tSkipping as number of points {num_points_in_pointcloud(obj_pointcloud)} < min_points = {min_points}.")
                continue

            obj_is_unique = True
            if not will_cluster_later: 
                raise NotImplementedError("Only final clustering available currently")
                for obj_info in self.memory:
                    break 
                    # We do clustering later now

            if obj_is_unique:
                new_obj_info = ObjectInfo(
                    len(self.memory),
                    obj_phrase,
                    obj_emb,
                    obj_pointcloud,
                    self.object_info_max_embeddings_num
                )

                self.memory.append(new_obj_info)
                self._log(f"\tObject Added: {new_obj_info}")

