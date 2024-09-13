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

from clip_loc.ellipsoid_utils import fit_ellipsoid_to_point_cloud
from utils.logging import conditional_log
from object_memory.object_info import ObjectInfo
from .clip_loc_object_info import ClipLocObjectInfo
from utils import depth_utils
from .loc_utils import p3p_pose_estimation, project_pointcloud_to_image
from .clip_utils import *
from .yolo_utils import *

def default_load_rgb(path: str) -> np.ndarray:
    return np.asarray(imageio.imread(path))

def default_load_depth(path: str) -> np.ndarray:
    if path.split('.')[-1] == 'npy':
        depth_img = np.load(path)
    else:
        depth_img = np.asarray(imageio.imread(path))

    return depth_img

class ClipLocObjectMemory:
    def _log(self, statement: any) -> None:
        """
        Conditionally log a statement if logging is enabled.

        Args:
            statement (any): The statement to log.
        """
        conditional_log(statement, self.log_enabled)

    def _load_rgb_image(self, path: str) -> np.ndarray:
        return self.load_rgb_image_func(path)

    def _load_depth_image(self, path: str) -> np.ndarray:
        return self.load_depth_image_func(path)

    def _process_memory(self):
        """
        Optimization to create an embedding->objectID mapping
        """
        self.emb_to_index = []
        for i, obj in enumerate(self.memory):
            self.emb_to_index.append([obj.text_embedding, i])

    def __init__(self, 
        base_memory: list[ObjectInfo], 
        load_rgb_image_func = default_load_rgb,
        load_depth_image_func = default_load_depth,
        log_enabled = True
    ):
        self.memory: list[ClipLocObjectInfo] = []
        self.pcd = o3d.geometry.PointCloud()
        self.log_enabled = log_enabled

        self.load_rgb_image_func = load_rgb_image_func
        self.load_depth_image_func = load_depth_image_func

        for obj in base_memory:
            cur_object_ellipsoid = fit_ellipsoid_to_point_cloud(obj.pointcloud)
            cur_object_ellipsoid.paint_uniform_color([0, 1, 0]) 
            text = " ".join(obj.names)

            emb = embed_text(text)

            self.memory.append(ClipLocObjectInfo(
                obj.id, text, emb, obj.pointcloud, cur_object_ellipsoid
            ))

            self.pcd += obj.pointcloud
            self.pcd += cur_object_ellipsoid

        self._process_memory()

    def __len__(self) -> int:
        return len(self.memory)

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for obj in self.memory:
            obj.save(os.path.join(save_dir, str(obj.id)))

        o3d.io.write_point_cloud(os.path.join(
            save_dir, "pcd.ply"
        ), self.pcd)

    @classmethod
    def load(cls, 
        load_dir,
        load_rgb_image_func = default_load_rgb,
        load_depth_image_func = default_load_depth,
        log_enabled = True
    ):
        clip_loc_mem_obj = cls([], load_rgb_image_func, load_depth_image_func, log_enabled)

        object_dirs = [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))]

        for object_dir in object_dirs:
            full_object_dir = os.path.join(load_dir, object_dir)

            obj_info = ClipLocObjectInfo.load(full_object_dir)

            clip_loc_mem_obj.memory.append(obj_info)

            clip_loc_mem_obj.pcd += obj_info.pointcloud
            clip_loc_mem_obj.pcd += obj_info.ellipsoid

        clip_loc_mem_obj._process_memory()

        clip_loc_mem_obj._log(f"Loaded ClipLocObjectMemory has {len(clip_loc_mem_obj)} objects")
        for obj in clip_loc_mem_obj.memory:
            clip_loc_mem_obj._log(obj)

        return clip_loc_mem_obj
    

    """
    **********************************************************************************************************************
    Localization code (ik, ik, bad coding practice)
    """
    
    def localize(self, img_path, intrinsic_matrix, k = 3, method = 'ransac'):
        for i in self.memory:
            print(i)

        img = self._load_rgb_image(img_path)
        detections = detect_objects(img)

        self._log(f"\tDetected {len(detections)} objects in the current image")
        for i in detections:
            self._log(f"\t\tDetected {i['class_name']}")

        if len(detections) == 0:
            self._log(f"No objects were detected in {img_path}")
            return None
        
        embeddings = encode_object_images(img, detections)

        top_k_indices = []

        for embedding in embeddings:
            similarities = []
            for stored_embedding, index in self.emb_to_index:
                sim = clip_similarity(embedding, stored_embedding)
                similarities.append((sim, index))

            similarities.sort(key=lambda x: x[0], reverse=True)
            top_k_indices.append([index for _, index in similarities[:k]])

        prosac_sampling_list = []
        for i_th_closest in range(k):
            for det_idx in range(len(detections)):
                prosac_sampling_list.append([det_idx, top_k_indices[det_idx][i_th_closest]])

        self._log(prosac_sampling_list)

        def compute_object_center_in_image(detection):
            x1, y1, x2, y2 = map(int, detection['bbox'])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            return [center_x, center_y]

        best_pose = None

        if method == "ransac":
            best_score = -1
            for _ in range(100):
                sample = np.random.choice(len(prosac_sampling_list), size=3, replace=False)
                sample_indices = [prosac_sampling_list[i] for i in sample]

                pts3d = np.array([depth_utils.compute_center(self.memory[idx].pointcloud) for _, idx in sample_indices], dtype=np.float32)
                pts2d = np.array([compute_object_center_in_image(detections[idx]) for idx, _ in sample_indices], dtype=np.float32)

                camera_poses = p3p_pose_estimation(pts3d, pts2d, intrinsic_matrix)

                mask = np.zeros_like(img)
                for _, idx in sample_indices:
                    x1, y1, x2, y2 = [int(coord) for coord in detections[idx]['bbox']]
                    mask[y1:y2, x1:x2] = 1

                total_corres_ellipse_pcd = o3d.geometry.PointCloud()
                for _, idx in sample_indices:
                    total_corres_ellipse_pcd += self.memory[idx].pointcloud 

                if len(camera_poses) > 0:
                    for pose in camera_poses:
                        # this has 1s in the positions where the ellipsoid is
                        corres_ellipsoids_image = project_pointcloud_to_image(
                            total_corres_ellipse_pcd, intrinsic_matrix, pose, img.shape
                        )
                        
                        score = np.sum(np.logical_and(corres_ellipsoids_image, mask))

                        if score > best_score:
                            best_score = score
                            best_pose = pose

        if best_pose is not None:
            return depth_utils.decompose_pose_matrix(best_pose)

        return None

