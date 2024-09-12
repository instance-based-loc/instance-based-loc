import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import os, pickle

from .object_info import ObjectInfo
from .object_memory import ObjectMemory

from typing import Type
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
import imageio
import os
import pickle
import copy
from scipy.spatial.transform import Rotation
from sklearn.cluster import AgglomerativeClustering

from .object_finder import ObjectFinder
from utils.logging import conditional_log
from utils.depth_utils import get_mask_coloured_pointclouds_from_depth, \
    transform_pointcloud, \
    DEFAULT_OUTLIER_REMOVAL_CONFIG, \
    combine_point_clouds
from utils.similarity_volume import SimVolume
from utils.fpfh_register import register_point_clouds, evaluate_transform, downsample_and_compute_fpfh
from .object_finder_phrases import check_if_floor
from .lora_module import LoraRevolver, LoraConfig


class ObjectDatasetInfo(ObjectInfo):
        def __init__(self, id: int, name: str, emb: np.ndarray, pointcloud: o3d.geometry.PointCloud, max_embeddings_num: int, rgb: torch.Tensor, depth: torch.Tensor):
            super().__init__(id, name, emb, pointcloud, max_embeddings_num)

            self.rgb_imgs = [rgb]
            self.depth_imgs = [depth]

        def _add_images(self, rgb_images, depth_images):
            self.rgb_imgs += rgb_images
            self.depth_imgs += depth_images

            assert len(self.rgb_imgs) == len(self.depth_imgs)

        def __add__(self, new_obj_info):
            super()._add_names(new_names = new_obj_info.names)
            super()._add_embeddings(new_embs = new_obj_info.embeddings)
            super()._add_pointcloud(new_pointcloud = new_obj_info.pointcloud)
            self._add_images(rgb_images = new_obj_info.rgb_imgs, depth_images = new_obj_info.depth_imgs)
            return self

        def __repr__(self):
            return (
            f"TRAINING INFO OBJ == Names: {self.names}, Mean_Emb: {self.mean_emb.shape}, Num. Points: {self.pcd.shape}, Num images: {len(self.rgb_imgs)},{len(self.depth_imgs)}, "
        )

def default_load_rgb(path: str) -> np.ndarray:
    return np.asarray(imageio.imread(path))

def default_load_depth(path: str) -> np.ndarray:
    if path.split('.')[-1] == 'npy':
        depth_img = np.load(path)
    else:
        depth_img = np.asarray(imageio.imread(path))

    return depth_img

class ObjectDatasetMemory(ObjectMemory):
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
        object_info_max_embeddings_num = 1000000,
        load_rgb_image_func = default_load_rgb,
        load_depth_image_func = default_load_depth,
        dataset_floor_thickness = 0.1,
        lora_path=None
    ):
        super().__init__(
        device,
        ram_pretrained_path,
        sam_checkpoint_path,
        camera_focal_lenth_x,
        camera_focal_lenth_y,
        get_embeddings_func,
        log_enabled,
        mem_formation_bounding_box_threshold,
        mem_formation_occlusion_overlap_threshold,
        object_info_max_embeddings_num,
        load_rgb_image_func,
        load_depth_image_func,
        dataset_floor_thickness,
        lora_path)

    # modified to return rgb & depth images as well
    def _get_object_info(self, rgb_image_path, depth_image_path, consider_floor, outlier_removal_config, depth_factor=1.):
        obj_grounded_imgs, obj_grounded_depths, obj_bounding_boxes, obj_masks, obj_phrases = ObjectFinder.find_for_training(rgb_image_path, depth_image_path)

        if obj_grounded_imgs is None:
            return None, None, None
        
        embs = np.stack(
            [
                np.array(self._get_embeddings(
                    current_obj_grounded_img = obj_grounded_imgs[i], 
                    current_obj_bounding_box = obj_bounding_boxes[i], 
                    current_obj_mask = obj_masks[i], 
                    current_obj_phrase = obj_phrases[i],
                    full_rgb_image = self._load_rgb_image(rgb_image_path),
                    full_depth_image = self._load_depth_image(depth_image_path),
                    consider_floor = consider_floor,
                    device = self.device
                ).cpu())
                    for i in range(len(obj_grounded_imgs))
            ]
        )

        obj_pointclouds = get_mask_coloured_pointclouds_from_depth(
            depth_image = self._load_depth_image(depth_image_path) / depth_factor,
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
        
        return obj_grounded_imgs, obj_grounded_depths, obj_phrases, embs, obj_pointclouds

    def process_image(
        self,
        rgb_image_path: str,
        depth_image_path: str,
        pose: np.ndarray,
        consider_floor: bool,
        outlier_removal_config = DEFAULT_OUTLIER_REMOVAL_CONFIG,
        add_noise = False,
        pose_noise = {'trans': 0.0005, 'rot': 0.0005},
        depth_noise = 0.003,
        min_points = 500,
        will_cluster_later = True,
        depth_factor=1.
    ):
        def num_points_in_pointcloud(pcd):
            return len(np.asarray(pcd.points))

        obj_grounded_imgs, obj_grounded_depths, obj_phrases, embs, obj_pointclouds = self._get_object_info(rgb_image_path, depth_image_path, consider_floor, outlier_removal_config,
                                                                   depth_factor=depth_factor)
        
        if obj_phrases is None:
            self._log("ObjectMemory.process_image did NOT find any objects")
            return
        else:
            self._log(f"ObjectMemory.process_image found: {obj_phrases}")
        
    
        transformed_pointclouds = [transform_pointcloud(obj_pointcloud, pose) for obj_pointcloud in obj_pointclouds]

        for i, (obj_rgb, obj_depth, obj_phrase, obj_emb, obj_pointcloud) in enumerate(zip(obj_grounded_imgs, obj_grounded_depths, obj_phrases, embs, transformed_pointclouds)):
            self._log(f"\tCurrent Object Phrase under consideration for ObjectMemory.process_image: {obj_phrase}")

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
                new_obj_info = ObjectDatasetInfo(
                    len(self.memory),
                    obj_phrase,
                    obj_emb,
                    obj_pointcloud,
                    self.object_info_max_embeddings_num,
                    obj_rgb,
                    obj_depth
                )

                # add to floor object or to list of emm objects
                if check_if_floor(new_obj_info.names):
                    if self.floors == None:
                        self.floors = new_obj_info
                    else:
                        self.floors = self.floors + new_obj_info
                    self._log(f"\tFloor Added: {new_obj_info}")
                else:
                    self.memory.append(new_obj_info)
                    self._log(f"\tObject Added: {new_obj_info}")

    def dump_dataset(self, dataset_root):
        # make root dir
        os.system(f"mkdir -p {dataset_root}")

        # create directories per object, dump all rgb/depth pairs
        for obj in self.memory:
            obj_name = f"{obj.names[0]}_{obj.id}"
            os.system(f"mkdir {os.path.join(dataset_root, obj_name)}")

            for i, (rgb, depth) in enumerate(zip(obj.rgb_imgs, obj.depth_imgs)):
                rgb_name = obj_name + "_" + str(i) + "_rgb.png"
                depth_name = obj_name + "_" + str(i) + "_depth.npy"

                rgb_path = os.path.join(dataset_root, obj_name, rgb_name)
                depth_path = os.path.join(dataset_root, obj_name, depth_name)

                imageio.imwrite(rgb_path, rgb)
                np.save(depth_path, depth)