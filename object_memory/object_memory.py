from typing import Type
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
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
        dataset_floor_thickness = 0.1
    ):
        # *******************************************************************************************
        # Setup encoder in the pipeline and pass a wrapper function to `get_embeddings_func`
        # *******************************************************************************************

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
        self.dataset_floor_thickness = dataset_floor_thickness

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
                self._load_rgb_image(rgb_image_path),
                self._load_depth_image(depth_image_path),
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

    def remove_points_below_floor(self):
        """
        Remove points from objects that are below a specified floor height 
        plus a given thickness. This helps in filtering out points 
        that belong to the floor of the objects.

        Uses self.dataset_floor_thickness
        """
        
        # Initialize floor_height to infinity to find the minimum height
        floor_height = float('inf')

        # First pass: Determine the lowest floor height from non-floor objects
        for info in self.memory:
            if "floor" in info.names:  # Skip objects that are classified as floors
                continue
            
            # Find the minimum height of the current point cloud
            low = np.min(info.pcd[1, :])
            floor_height = min(low, floor_height)  # Update the lowest floor height

        # Second pass: Remove points below the calculated floor height plus thickness
        for info in self.memory:
            if "floor" in info.names:  # Skip objects that are classified as floors
                continue
            
            # Filter out points that are below the floor height plus thickness
            mask = info.pcd[1, :] > floor_height + self.dataset_floor_thickness
            info.update_pointcloud_with_mask(mask)  # Update the point cloud using the mask

            # Remove the object from memory if it has no points left
            if len(info.pointcloud.points) == 0:
                self.memory.remove(info)

    def recluster_objects_with_dbscan(self, eps=0.2, min_points_per_cluster=300, visualize=False):
        """
        Recluster objects in memory using the DBSCAN algorithm.

        :param eps: The maximum distance between two samples for them to be considered 
                    as in the same neighborhood.
        :param min_points_per_cluster: The minimum number of points required to form a dense region.
        :param visualize: If True, display progress during clustering.
        """
        self._log("Clustering using DBSCAN")
        
        # Load all points into a single point cloud (PCD) and prepare for clustering
        all_points = np.concatenate([obj.pcd for obj in self.memory], axis=-1).T  # Shape: 3xN
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(all_points)

        self._log(f"\tCombined points shape: {all_points.shape}, Example object shape: {self.memory[0].pcd.shape}")

        # Perform clustering using DBSCAN
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points_per_cluster, print_progress=visualize))
        self._log(f"\tUnique labels: {np.unique(labels)}, Labels shape: {labels.shape}")

        # Function to check if a point is present in an array of points
        def is_point_in_array(points_array, query_point):
            return np.any(np.all(points_array == query_point, axis=1))

        # Initialize assignments for new object clusters
        new_object_assignments = np.full(len(self.memory), -1)

        # Iterate through each object and assign cluster labels
        for index, obj in tqdm(enumerate(self.memory), total=len(self.memory)):
            query_point = obj.pcd[:, 0]  # Select a random point from the object's point cloud
            for label in np.unique(labels):
                if label == -1:
                    continue

                cluster_points = all_points[labels == label]

                if is_point_in_array(cluster_points, query_point):
                    if new_object_assignments[index] != -1:
                        self._log("\t\tWarning: Multiple labels detected for the same object.")
                    
                    new_object_assignments[index] = label

        self._log(f"\tNew assignments: {new_object_assignments}, Unique clusters: {len(np.unique(new_object_assignments))}")

        # Create new clustered objects based on the assignments
        clustered_objects = []
        for label in np.unique(labels):
            if label == -1:
                continue

            self._log(f"\tProcessing label: {label}")
            objects_to_cluster = [self.memory[i] for i in range(len(self.memory)) if new_object_assignments[i] == label]

            self._log(f"\tObjects to cluster: {len(objects_to_cluster)}")
            if not objects_to_cluster or len(objects_to_cluster) == 0:
                continue

            # Merge objects into a single clustered object
            clustered_object_info = objects_to_cluster[0]
            for obj_info in objects_to_cluster[1:]:
                self._log(f"\tMerging {obj} into {clustered_object_info}")
                clustered_object_info = clustered_object_info + obj_info

            clustered_objects.append(clustered_object_info)

        self._log(f"\tTotal clustered objects: {len(clustered_objects)}")
        
        # Update memory with new clustered objects
        self.memory = clustered_objects
        print(f"Updated memory size: {len(self.memory)}")

        # Update object IDs
        for i, obj_info in enumerate(self.memory):
            obj_info.id = i
