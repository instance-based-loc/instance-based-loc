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
from .object_info import ObjectInfo
from utils.logging import conditional_log
from utils.depth_utils import get_mask_coloured_pointclouds_from_depth, \
    transform_pointcloud, \
    DEFAULT_OUTLIER_REMOVAL_CONFIG, \
    combine_point_clouds
from utils.similarity_volume import SimVolume
from utils.fpfh_register import register_point_clouds, evaluate_transform, downsample_and_compute_fpfh
from .object_finder_phrases import check_if_floor
from .lora_module import LoraRevolver, LoraConfig

from utils.IoU_ops import calculate_obj_aligned_3d_IoU

print("\033[34mLoaded modules for object_memory.object_memory\033[0m")

def default_load_rgb(path: str) -> np.ndarray:
    return np.asarray(imageio.imread(path))

def default_load_depth(path: str) -> np.ndarray:
    if path.split('.')[-1] == 'npy':
        depth_img = np.load(path)
    else:
        depth_img = np.asarray(imageio.imread(path))

    return depth_img

class ObjectMemory():
    def _load_rgb_image(self, path: str) -> np.ndarray:
        return self.load_rgb_image_func(path)

    def _load_depth_image(self, path: str) -> np.ndarray:
        return self.load_depth_image_func(path)

    def _get_embeddings(self, **kwargs) -> torch.Tensor:
        return self.get_embeddings_func(**kwargs)

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
        object_info_max_embeddings_num = 1000000,
        load_rgb_image_func = default_load_rgb,
        load_depth_image_func = default_load_depth,
        dataset_floor_thickness = 0.1,
        lora_path=None
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

        self.loraModule = LoraRevolver(self.device)
        if lora_path != None:
            self.loraModule.load_lora_ckpt_from_file(lora_path, "5x40")
        else:
            raise NotImplementedError

        ##  HARDCODED TO LORA
        self.get_embeddings_func = self.loraModule.encode_image

        self.memory: list[ObjectInfo] = []
        self.floors = None # stoors all floors or ground in one pcd

    def __repr__(self):
        repr = ""
        for obj_info in self.memory:
            repr += f"\t{obj_info}\n"
        if repr == "":
            repr = "\tNo objects in memory yet."
        return repr

    def _get_object_info(self, rgb_image_path, depth_image_path, consider_floor, outlier_removal_config, depth_factor=1.):
        obj_grounded_imgs, obj_bounding_boxes, obj_masks, obj_phrases = ObjectFinder.find(rgb_image_path, consider_floor)

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
        
        return obj_phrases, embs, obj_pointclouds

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

        obj_phrases, embs, obj_pointclouds = self._get_object_info(rgb_image_path, depth_image_path, consider_floor, outlier_removal_config,
                                                                   depth_factor=depth_factor)
        
        if obj_phrases is None:
            self._log("ObjectMemory.process_image did NOT find any objects")
            return
        else:
            self._log(f"ObjectMemory.process_image found: {obj_phrases}")
        
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
                new_obj_info = ObjectInfo(
                    len(self.memory),
                    obj_phrase,
                    obj_emb,
                    obj_pointcloud,
                    self.object_info_max_embeddings_num
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

    def downsample_all_objects(self, voxel_size):
        self._log("Downsampling all objects")
        for obj in self.memory:
            obj.downsample(voxel_size)
        if self.floors != None:
            self.floors.downsample(voxel_size)

    def remove_points_below_floor(self):
        """
        Remove points from objects that are below a specified floor height 
        plus a given thickness. This helps in filtering out points 
        that belong to the floor of the objects.

        Uses self.dataset_floor_thickness
        """
        self._log("Removing points below floor")

        floor_height = float('inf')

        # First pass: Determine the lowest floor height from non-floor objects
        for info in self.memory:
            # Find the minimum height of the current point cloud
            low = np.min(info.pcd[1, :])
            floor_height = min(low, floor_height)  # Update the lowest floor height

        # Second pass: Remove points below the calculated floor height plus thickness
        for info in self.memory:
            # Filter out points that are below the floor height plus thickness
            mask = info.pcd[1, :] > floor_height + self.dataset_floor_thickness
            info.update_pointcloud_with_mask(mask)  # Update the point cloud using the mask

            # Remove the object from memory if it has no points left
            if len(info.pointcloud.points) == 0:
                self.memory.remove(info)

    # def recluster_via_IoU(self):


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
                self._log(f"\t\tMerging {obj_info} into {clustered_object_info}")
                clustered_object_info = clustered_object_info + obj_info

            self._log(f"\tClustered Object is finally {clustered_object_info}")
            clustered_objects.append(clustered_object_info)

        self._log(f"\tTotal clustered objects: {len(clustered_objects)}")
        
        # Update memory with new clustered objects
        self.memory = clustered_objects
        self._log(f"Updated memory size: {len(self.memory)}")

        # Update object IDs
        for i, obj_info in enumerate(self.memory):
            obj_info.id = i

    def recluster_via_agglomerative_clustering(self, distance_func=None, embedding_distance_threshold=0.4, distance_threshold=0.1):
        if distance_func == None:
            def df(all_obj_embs, all_obj_centroids, distance_threshold=0.1):       # expects N x D 
                norms = np.linalg.norm(all_obj_embs, axis=1, keepdims=True)
                normalized_embeddings = all_obj_embs / norms

                emb_distance_matrix = 1 - np.dot(normalized_embeddings, normalized_embeddings.T)
                
                # compute pairwise distances, NxNx3
                centroid_distances = np.linalg.norm(all_obj_centroids[np.newaxis, :, :] - all_obj_centroids[:, np.newaxis, :])
                
                # mask out all pairs where centroid distance is greater than the allowed threshold
                emb_distance_matrix = np.where(centroid_distances < distance_threshold, emb_distance_matrix, 1)
                
                return emb_distance_matrix
            
            distance_func = df

        all_mean_embs = np.array([obj.mean_emb for obj in self.memory])
        all_centroids = np.array([obj.centroid for obj in self.memory])
        distance_matrix = df(all_mean_embs, all_centroids, distance_threshold=distance_threshold)

        # sklearn agglomerative clustering
        self._log("Clustering agglomeratively")
        agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=embedding_distance_threshold, metric='precomputed', linkage='average')
        agg_clustering.fit(distance_matrix)

        # Get the cluster labels
        labels = agg_clustering.labels_

        unique_labels = set(labels)

        self._log(f"{len(unique_labels)} objects clustered")

        # reassign memory
        new_memory = [None for _ in unique_labels]
        for new_label, old_obj_info in zip(labels, self.memory):
            if new_memory[new_label] == None:
                new_memory[new_label] = old_obj_info
            else:
                new_memory[new_label] = new_memory[new_label] + old_obj_info
        
        # save new memory, fix IDs
        self.memory = new_memory
        for i, _ in enumerate(self.memory):
            self.memory[i].id = i

    def recluster_via_combined(self, distance_func=None, embedding_distance_threshold=0.4, eps=0.4, min_points_per_cluster=150):
        if distance_func == None:
            def df(all_obj_embs, all_obj_centroids):       # expects N x D 
                norms = np.linalg.norm(all_obj_embs, axis=1, keepdims=True)
                normalized_embeddings = all_obj_embs / norms

                emb_distance_matrix = 1 - np.dot(normalized_embeddings, normalized_embeddings.T)
                
                # compute pairwise distances, NxNx3
                centroid_distances = np.linalg.norm(all_obj_centroids[np.newaxis, :, :] - all_obj_centroids[:, np.newaxis, :])
                
                return emb_distance_matrix
            
            distance_func = df

        all_mean_embs = np.array([obj.mean_emb for obj in self.memory])
        all_centroids = np.array([obj.centroid for obj in self.memory])
        distance_matrix = df(all_mean_embs, all_centroids)

        import matplotlib.pyplot as plt
        cax = plt.imshow(distance_matrix)
        cbar = plt.colorbar(cax)
        plt.savefig('/home2/aneesh.chavan/instance-based-loc/lora_sims.png')

        # import pdb;
        # pdb.set_trace()

        # sklearn agglomerative clustering
        self._log("Clustering agglomeratively")
        agg_clustering = AgglomerativeClustering(
                            n_clusters=None, 
                            distance_threshold=embedding_distance_threshold, 
                            metric='precomputed', 
                            linkage='average')
        agg_clustering.fit(distance_matrix)

        # Get the cluster labels
        labels = agg_clustering.labels_

        unique_labels = set(labels)

        self._log(f"{len(unique_labels)} clusters initially")

        def is_point_in_array(points_array, query_point):
            return np.any(np.all(points_array == query_point, axis=1))

        # DBScan within clusters ONLY
        objects_post_dbscan = []
        for u in unique_labels:
            # print("debugs: ", len(objects_post_dbscan), " label ", u)

            # get objects assigned label `u`
            objs_to_consider = []
            for i, obj in enumerate(self.memory):
                if labels[i] == u:
                    objs_to_consider.append(obj)

            # print(len(objs_to_consider), " objs to consider")

            # dbscan these points
            all_points = np.concatenate([obj.pcd for obj in objs_to_consider], axis=-1).T  # Shape: 3xN
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(all_points)
            dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points_per_cluster, print_progress=False))

            # print(dbscan_labels, np.unique(dbscan_labels))

            # assign each object to 
            new_object_assignments = np.full(len(objs_to_consider), -1)
            for index, obj in enumerate(objs_to_consider):
                query_point = obj.pcd[:,0]
                for label in np.unique(dbscan_labels):
                    if label == -1:
                        continue

                    cluster_points = all_points[dbscan_labels == label]
                    if is_point_in_array(cluster_points, query_point):
                        if new_object_assignments[index] != -1:
                            self._log("\t\tMULTIPLE LABELS IN COMBINED RECLUSTERING")

                        new_object_assignments[index] = label

            # create new objects for each dbscan cluster 
            dbscanned_objects = []
            for label in np.unique(dbscan_labels):
                if label == -1:
                    continue

                to_combine = [objs_to_consider[i] for i in range(len(objs_to_consider)) if new_object_assignments[i] == label] 
                if len(to_combine) == 0:
                    continue

                combined_obj_info = to_combine[0]
                for obj_info in to_combine[1:]:
                    combined_obj_info = combined_obj_info + obj_info
                
                dbscanned_objects.append(combined_obj_info)

            # add the new objects into objects_post_dbscan, update count
            objects_post_dbscan = objects_post_dbscan + dbscanned_objects

        # reassign memory
        new_memory = objects_post_dbscan
        
        # save new memory, fix IDs
        print("Clustering done")
        self.memory = new_memory
        for i, _ in enumerate(self.memory):
            self.memory[i].id = i

    def recluster_via_clustering_and_IoU(self, distance_func=None, embedding_distance_threshold=0.4, eps=0.4, min_points_per_cluster=150, IoU_threshold=0.4):
        if distance_func == None:
            def df(all_obj_embs, all_obj_centroids):       # expects N x D 
                norms = np.linalg.norm(all_obj_embs, axis=1, keepdims=True)
                normalized_embeddings = all_obj_embs / norms

                emb_distance_matrix = 1 - np.dot(normalized_embeddings, normalized_embeddings.T)
                
                # compute pairwise distances, NxNx3
                centroid_distances = np.linalg.norm(all_obj_centroids[np.newaxis, :, :] - all_obj_centroids[:, np.newaxis, :])
                
                return emb_distance_matrix
            
            distance_func = df

        all_mean_embs = np.array([obj.mean_emb for obj in self.memory])
        all_centroids = np.array([obj.centroid for obj in self.memory])
        distance_matrix = df(all_mean_embs, all_centroids)

        import matplotlib.pyplot as plt
        cax = plt.imshow(distance_matrix)
        cbar = plt.colorbar(cax)
        plt.savefig('/home2/aneesh.chavan/instance-based-loc/lora_sims.png')

        # import pdb;
        # pdb.set_trace()

        # sklearn agglomerative clustering
        self._log("Clustering agglomeratively")
        agg_clustering = AgglomerativeClustering(
                            n_clusters=None, 
                            distance_threshold=embedding_distance_threshold, 
                            metric='precomputed', 
                            linkage='average')
        agg_clustering.fit(distance_matrix)

        # Get the cluster labels
        labels = agg_clustering.labels_

        unique_labels = set(labels)

        self._log(f"{len(unique_labels)} clusters initially")

        def is_point_in_array(points_array, query_point):
            return np.any(np.all(points_array == query_point, axis=1))

        # DBScan within clusters ONLY
        objects_post_dbscan = []
        for u in unique_labels:
            # print("debugs: ", len(objects_post_dbscan), " label ", u)

            # get objects assigned label `u`
            objs_to_consider = []
            for i, obj in enumerate(self.memory):
                if labels[i] == u:
                    objs_to_consider.append(obj)

            # print(len(objs_to_consider), " objs to consider")

            # dbscan these points
            all_points = np.concatenate([obj.pcd for obj in objs_to_consider], axis=-1).T  # Shape: 3xN
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(all_points)
            dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points_per_cluster, print_progress=False))

            # print(dbscan_labels, np.unique(dbscan_labels))

            # assign each object to 
            new_object_assignments = np.full(len(objs_to_consider), -1)
            for index, obj in enumerate(objs_to_consider):
                query_point = obj.pcd[:,0]
                for label in np.unique(dbscan_labels):
                    if label == -1:
                        continue

                    cluster_points = all_points[dbscan_labels == label]
                    if is_point_in_array(cluster_points, query_point):
                        if new_object_assignments[index] != -1:
                            self._log("\t\tMULTIPLE LABELS IN COMBINED RECLUSTERING")

                        new_object_assignments[index] = label

            # create new objects for each dbscan cluster 
            dbscanned_objects = []
            for label in np.unique(dbscan_labels):
                if label == -1:
                    continue

                to_combine = [objs_to_consider[i] for i in range(len(objs_to_consider)) if new_object_assignments[i] == label] 
                if len(to_combine) == 0:
                    continue

                combined_obj_info = to_combine[0]
                for obj_info in to_combine[1:]:
                    combined_obj_info = combined_obj_info + obj_info
                
                dbscanned_objects.append(combined_obj_info)

            # add the new objects into objects_post_dbscan, update count
            objects_post_dbscan = objects_post_dbscan + dbscanned_objects

        # reassign memory
        pre_IoU_memory = objects_post_dbscan

        # check IoU between all new memory objects
        # distance_matrix = df(all_mean_embs, all_centroids, distance_threshold=distance_threshold)
        IoUs = np.zeros((len(pre_IoU_memory), len(pre_IoU_memory)))
        IoU_threshold = 1 - IoU_threshold       # agg clustering discards high values, we want the opposite
        for i in range(len(pre_IoU_memory)):
            for j in range(i, len(pre_IoU_memory)):
                IoUs[i][j] = 1 - calculate_obj_aligned_3d_IoU(np.asarray(pre_IoU_memory[i].pointcloud.points),
                                                          np.asarray(pre_IoU_memory[j].pointcloud.points))
                IoUs[j][i] = IoUs[i][j]                                                          

        # sklearn agglomerative clustering
        self._log("Clustering with IoUs")
        agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=IoU_threshold, metric='precomputed', linkage='complete')
        agg_clustering.fit(IoUs)

        # Get the cluster labels
        labels = agg_clustering.labels_

        unique_labels = set(labels)

        self._log(f"{len(unique_labels)} objects clustered")

        # reassign memory
        new_memory = [None for _ in unique_labels]
        for new_label, old_obj_info in zip(labels, pre_IoU_memory):
            if new_memory[new_label] == None:
                new_memory[new_label] = old_obj_info
            else:
                new_memory[new_label] = new_memory[new_label] + old_obj_info

        
        # save new memory, fix IDs
        print("Clustering done")
        self.memory = new_memory
        for i, _ in enumerate(self.memory):
            self.memory[i].id = i

    def save(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        individual_obj_save_dir = os.path.join(save_directory, "objects")
        os.makedirs(individual_obj_save_dir, exist_ok=True)
        individual_floor_save_dir = os.path.join(save_directory, "floors")
        os.makedirs(individual_floor_save_dir, exist_ok=True)

        textual_info_path = os.path.join(save_directory, "memory.txt")
        combined_pointcloud_save_path = os.path.join(save_directory, "combined_pointcloud.ply")
        combined_pointcloud_with_floor_save_path = os.path.join(save_directory, "combined_pointcloud_with_floor.ply")

        with open(textual_info_path, "w") as f:
            f.write(self.__repr__())

        combined_pointcloud = combine_point_clouds(obj.pointcloud for obj in self.memory)
        o3d.io.write_point_cloud(combined_pointcloud_save_path, combined_pointcloud)
        combined_pointcloud_with_floor = combine_point_clouds([obj.pointcloud for obj in self.memory] + [self.floors.pointcloud])
        o3d.io.write_point_cloud(combined_pointcloud_with_floor_save_path, combined_pointcloud_with_floor)

        for obj in self.memory:
            current_obj_save_dir = os.path.join(individual_obj_save_dir, f"{obj.id}")
            obj.save(current_obj_save_dir)

        current_floor_save_dir = os.path.join(individual_floor_save_dir, f"all_floors")
        self.floors.save(current_floor_save_dir)

        self._log(f"Saved memory to {save_directory}")

    def save_to_pkl(self, save_directory: str):
        pklable_memory = []
        for objinfo in self.memory:
            blank_info = ObjectInfo(
                id=0,
                name="",
                emb=objinfo.embeddings[0],
                pointcloud=o3d.geometry.PointCloud(),
                max_embeddings_num=1e5
            )

            blank_info.id = id
            blank_info.names = objinfo.names
            blank_info.embeddings = objinfo.embeddings
            blank_info.max_embeddings_num = objinfo.max_embeddings_num
            blank_info.mean_emb = objinfo.mean_emb
            blank_info.centroid = objinfo.centroid
            blank_info.pointcloud = None

            pcd_points = np.asarray(objinfo.pointcloud.points)
            pcd_colors = np.asarray(objinfo.pointcloud.colors)

            info_tuple = (blank_info, pcd_points, pcd_colors)
            pklable_memory.append(info_tuple)
        
        blank_info = ObjectInfo(
            id=0,
            name="",
            emb=objinfo.embeddings[0],
            pointcloud=o3d.geometry.PointCloud(),
            max_embeddings_num=1e5
        )

        blank_info.id = id
        blank_info.names = objinfo.names
        blank_info.embeddings = objinfo.embeddings
        blank_info.max_embeddings_num = objinfo.max_embeddings_num
        blank_info.mean_emb = objinfo.mean_emb
        blank_info.centroid = objinfo.centroid
        blank_info.pointcloud = None

        pcd_points = np.asarray(objinfo.pointcloud.points)
        pcd_colors = np.asarray(objinfo.pointcloud.colors)

        info_tuple = (blank_info, pcd_points, pcd_colors)
        pklable_floors = info_tuple

        pickle.dump((pklable_memory, pklable_floors), open(save_directory, 'wb'))

    """
    Designed to load objInfos into memory from a pickleable object created by save, above
    """
    def load(self, load_directory: str):
        pkl = pickle.load(open(load_directory, 'rb'))

        pklable_memory, pklable_floors = pkl

        def conv_info_tuple(info_tuple):
            blank_info, pcd_points, pcd_colors = info_tuple
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

            blank_info.pointcloud = pcd
            return blank_info
        
        self.memory = [conv_info_tuple(it) for it in pklable_memory]
        self.floors = conv_info_tuple(pklable_floors)


    """
    The sauce.
    """
    def localise(self, image_path, depth_image_path, testname="", 
                 subtest_name="",
                 save_point_clouds=False,
                 outlier_removal_config=None, 
                 fpfh_global_dist_factor = 2, fpfh_local_dist_factor = 0.4, 
                 fpfh_voxel_size = 0.05, topK=5, useLora=True,
                 save_localised_pcd_path=None,
                 consider_floor=False,
                 perform_semantic_icp=True,
                 depth_factor = 1.,
                 max_detected_object_num=7):
        """
        Given an image and a corresponding depth image in an unknown frame, consult the stored memory
        and output a pose in the world frame of the point clouds stored in memory.

        Args:
        - image_path (str): Path to the RGB image file.
        - depth_image_path (str): Path to the depth image file in .npy format.
        - icp_threshold (float): Threshold for ICP (Iterative Closest Point) algorithm.
        - testname (str): Name for test-specific files.
        - perform_semantic_icp(bool): Initialise transformation by performing a simple p2p semantic ICP first
        
        Returns:
        - np.ndarray: Localized pose in the world frame as [x, y, z, qw, qx, qy, qz].
        """
        # Default outlier removal config
        if outlier_removal_config == None:
            outlier_removal_config = {
                "radius_nb_points": 8,
                "radius": 0.05,
            }

        consider_floor=False
        # Extract all objects currently seen, get embeddings, point clouds in the local unknown frame
        detected_phrases, detected_embs, detected_pointclouds = self._get_object_info(image_path, depth_image_path, consider_floor=consider_floor, 
                                                                                      outlier_removal_config=outlier_removal_config,
                                                                                      depth_factor=depth_factor)


        # TODO deal with no objects detected
        if detected_embs is None:
            return np.array([0.,0.,0.,0.,0.,0.,1.]), [[],[]]
 
        # take top 7 largest pointclouds, phrases, embs
        if len(detected_pointclouds) > max_detected_object_num:
            print(f"Taking top {max_detected_object_num} objects")

            pairs = [[p,e,pcd] for p,e,pcd in zip(detected_phrases, detected_embs, detected_pointclouds)]
            pairs = sorted(pairs, key=lambda x: np.asarray(x[-1].points).shape[0], reverse=True)

            detected_phrases = [p for p, _, _ in pairs[:max_detected_object_num]]
            detected_embs = [e for _, e, _ in pairs[:max_detected_object_num]]
            detected_pointclouds = [pcd for _, _, pcd in pairs[:max_detected_object_num]]

            print(len(pairs))

        # Correlate embeddings with objects in memory for all seen objects
        # TODO maybe a KNN search will do better?
        for m in self.memory:
            m._compute_means()  # Update object info means

        memory_embs = np.array([m.mean_emb for m in self.memory])

        if len(detected_embs) > len(self.memory):
            print("Not enough memory objects")
            detected_embs = detected_embs[:len(memory_embs)]

        all_memory_embs = [np.array([e/np.linalg.norm(e) for e in m.embeddings]) for m in self.memory]      # all embeddings per object

        detected_embs /= np.linalg.norm(detected_embs, axis=-1, keepdims=True)
        memory_embs /= np.linalg.norm(memory_embs, axis=-1, keepdims=True)      # mean embedding

        # Detected x Mem x Emb sized
        cosine_similarities = np.inner(detected_embs, memory_embs)
        print(cosine_similarities.shape, f" {len(detected_embs)} objects detected")

        # get the closest similarity from each object
        closest_similarities = np.zeros_like(cosine_similarities)
        for i, d in enumerate(detected_embs):
            for j, m in enumerate(all_memory_embs):
                closest_similarities[i][j] = np.max(np.dot(m, d))

        # closest L2-norm similarities
        L2_closest_similarities = np.zeros_like(cosine_similarities)
        for i, d in enumerate(detected_embs):
            row = np.zeros_like(cosine_similarities[0])
            for j, m in enumerate(all_memory_embs):
                row[i] = np.max(np.linalg.norm(m - d))
            L2_closest_similarities[i] = row

        # Save point clouds
        save_root = f"pcds/{testname}/"
        subsave_root = os.path.join(save_root, str(subtest_name))
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        if save_point_clouds:
            if not os.path.exists(subsave_root):
                os.makedirs(subsave_root)

            init_pcd = o3d.geometry.PointCloud()
            temp_pcd = o3d.geometry.PointCloud()
            for i, d in enumerate(detected_pointclouds):
                init_pcd += d            

            for j, m in enumerate(self.memory):
                init_pcd += m

            o3d.io.write_point_cloud(os.path.join(subsave_root , "_init_pcd_" + str(subtest_name) + ".ply"), init_pcd)
            print("Initial ICP point clouds saved")

        # TODO unseen objects in detected objects are not being dealt with, 
        # assuming that all detected objects can be assigned to mem objs

        print("Getting assignments")
        print(closest_similarities.shape)
        # sv = SimVolume(closest_similarities)
        sv = SimVolume(closest_similarities)
        # rep_vol, _ = sv.construct_volume()
        # print(rep_vol.shape)
        # best_coords = sv.get_top_indices(rep_vol, 10)
        # assns = sv.conv_coords_to_pairs(rep_vol, best_coords)
        
        subvolume_size = min(len(detected_pointclouds), 3)      # 3 represents the dimensionality of the subvolumes constructed
        sv.fast_construct_volume(subvolume_size)
        assns = sv.get_top_indices_from_subvolumes(num_per_length=4)
        assns_to_consider = assns
        del sv

        print("Phrases: ", detected_phrases)
        # print(cosine_similarities)
        # print("                 V?")
        # print(closest_similarities)
        print("Assignments being considered: ", assns_to_consider)

        assn_data = [ assn for assn in assns_to_consider ]

        # clean up outliers from detected pcds before registration
        cleaned_detected_pcds = []
        for obj in detected_pointclouds:
            print(obj)
            obj_filtered, _ = obj.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                            radius=outlier_removal_config["radius"])
            cleaned_detected_pcds.append(obj_filtered)
        detected_pointclouds = cleaned_detected_pcds

        # prepare a full memory and detected pcd
        all_memory_points = []
        for obj in self.memory:
            all_memory_points.append(obj)
        
        all_memory_pcd = o3d.geometry.PointCloud()
        for pcd in all_memory_points:
            all_memory_pcd = all_memory_pcd + pcd.pointcloud        # get pcd from obj memory

        all_detected_points = []
        for obj in detected_pointclouds:
            all_detected_points.append(obj)
        
        all_detected_pcd = o3d.geometry.PointCloud()
        for pcd in all_detected_points:
            all_detected_pcd = all_detected_pcd + pcd 

        # go through all top K assingments, record ICP costs
        for assn_num, assn in tqdm(enumerate(assn_data)):
            # use ALL object pointclouds together

            # centering all the pointclouds
            chosen_detected_pcd = o3d.geometry.PointCloud()
            chosen_memory_pcd = o3d.geometry.PointCloud()

            for d_index, m_index in assn:
                chosen_detected_pcd = chosen_detected_pcd + detected_pointclouds[d_index]
                chosen_memory_pcd = chosen_memory_pcd + self.memory[m_index].pointcloud

            detected_mean = np.mean(np.asarray(chosen_detected_pcd.points), axis=0)
            memory_mean = np.mean(np.asarray(chosen_memory_pcd.points), axis=0)
            
            chosen_detected_pcd.translate(-detected_mean)
            chosen_memory_pcd.translate(-memory_mean)

            # chosen_detected_pcd.points = o3d.utility.Vector3dVector(chosen_detected_points.T - detected_mean)
            # chosen_memory_pcd.points = o3d.utility.Vector3dVector(chosen_memory_points.T - memory_mean)

            if perform_semantic_icp:            
                raise NotImplementedError
                # generate labels that match objects
                chosen_detected_labels = np.zeros(len(chosen_detected_pcd.points))
                chosen_memory_labels = np.zeros(len(chosen_memory_pcd.points))

                det_ptr = 0
                mem_ptr = 0
                for d_index, m_index in assn:
                    chosen_detected_labels[det_ptr:] = d_index
                    chosen_memory_labels[mem_ptr:] = d_index

                    # update ptrs
                    det_ptr += len(detected_pointclouds[d_index])
                    mem_ptr += len(self.memory[m_index].pcd)

                # heavy downsample
                ideal_num_points = 25*len(assn)
                det_skip = len(chosen_detected_labels) // ideal_num_points
                mem_skip = len(chosen_memory_labels) // ideal_num_points

                # det_indices = np.random.choice(len(chosen_detected_labels), size=25*len(assn), replace=False)
                # mem_indices = np.random.choice(len(chosen_memory_labels), size=25*len(assn), replace=False)

                ds_detected_points = np.asarray(chosen_detected_pcd.points)[::det_skip]
                ds_memory_points = np.asarray(chosen_memory_pcd.points)[::mem_skip]

                chosen_detected_labels = chosen_detected_labels[::det_skip]
                chosen_memory_labels = chosen_memory_labels[::mem_skip]

                # o3d.io.write_point_cloud(f"./temp/{str(assn)}-{testname}-detmem.ply", chosen_memory_pcd + chosen_detected_pcd)

                # calculate initial coarse alignment based on semantic matching (better init?)
                semantic_icp_transform = semantic_icp(torch.tensor(ds_detected_points), 
                                                      torch.tensor(ds_memory_points),
                                                    chosen_detected_labels, chosen_memory_labels,
                                                    max_iterations=2,
                                                    tolerance=0.001)
                chosen_detected_pcd = chosen_detected_pcd.transform(semantic_icp_transform)    
                
                transform, rmse, fitness = register_point_clouds(chosen_detected_pcd, chosen_memory_pcd, 
                                                voxel_size = fpfh_voxel_size, global_dist_factor = fpfh_global_dist_factor, 
                                                local_dist_factor = fpfh_local_dist_factor)
                
                transform = transform @ semantic_icp_transform

            # no semantics
            else:
                transform, rmse, fitness = register_point_clouds(chosen_detected_pcd, chosen_memory_pcd, 
                                                voxel_size = fpfh_voxel_size, global_dist_factor = fpfh_global_dist_factor, 
                                                local_dist_factor = fpfh_local_dist_factor)

            # save transformation pcds
            if save_point_clouds:              
                o3d.io.write_point_cloud(os.path.join(subsave_root, f"only_chosen_" + str(assn) + ".ply"), chosen_memory_pcd + chosen_detected_pcd.transform(transform))

            # get transforms in the global frame, account for mean centering
            global_frame_transform = copy.deepcopy(transform)
            R = copy.copy(transform[:3, :3])
            tx = copy.copy(transform[:3, 3])

            global_frame_transform[:3, :3] = R
            global_frame_transform[:3,  3] = tx + memory_mean - R@detected_mean

            # apply candidate transform to ALL pcds, check rmse            
            full_memory_rmse, full_memory_fitness = evaluate_transform(all_detected_pcd, all_memory_pcd, trans_init=global_frame_transform)

            assn_data[assn_num] = [assn, transform, rmse, fitness, full_memory_rmse, full_memory_fitness]       # fitness

        # USE THE BEST CHOSEN ASSIGNMENT
        # GET TX/RX error
        best_assn_acc_to_fitness = sorted(assn_data, key=lambda x: x[-1], reverse=True)    # reverse required for fitness
        best_assn_acc_to_rmse = sorted(assn_data, key=lambda x: x[-2])

        best_assn = best_assn_acc_to_fitness

        for a in best_assn:
            print("Assn: ", a[0], " | chosen RMSE: ", a[3] , " | full RMSE: ", a[5] , " | chosen fitness: ", a[4] , " | full memory fitness: ", a[-1])

        best_transform = best_assn[0][1]
        best_assn = best_assn[0][0]

        moved_objs = [n for n in range(len(detected_pointclouds)) if n not in assn]

        R = copy.copy(best_transform[:3,:3])
        t = copy.copy(best_transform[:3, 3])
        
        tAvg = t + memory_mean - R@detected_mean  
        # tAvg = t + detected_mean - R@memory_mean #incorrect smh
        qAvg = Rotation.from_matrix(R).as_quat()

        localised_pose = np.concatenate((tAvg, qAvg))
        print("Best assn: ", best_assn)

        ## DEBUG
        print("R: ", R )
        print("t: ", t)

        # use ALL object pointclouds together, save pcd
        if save_point_clouds:
            # transform full memory
            all_detected_pcd = o3d.geometry.PointCloud()
            all_memory_pcd = o3d.geometry.PointCloud()
            
            for i in detected_pointclouds:
                all_detected_pcd = all_detected_pcd + i
            for i in self.memory:
                all_memory_pcd = all_memory_pcd + i.pointcloud

            # centering all the pointclouds (needed as best_transform is between centered pcds)
            all_detected_pcd.translate(-detected_mean)
            all_memory_pcd.translate(-memory_mean)

            # remove outliers from detected pcds
            all_detected_pcd_filtered, _ = all_detected_pcd.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                            radius=outlier_removal_config["radius"])

            all_memory_pcd.paint_uniform_color([0,1,0])
            all_detected_pcd_filtered.paint_uniform_color([1,0,0])

            o3d.io.write_point_cloud(os.path.join(subsave_root, f"_best_full_pcd" + str(best_assn) + ".ply"), all_memory_pcd + all_detected_pcd_filtered.transform(best_transform))
        # # import pdb; pdb.set_trace()

        # save rgb image
        if save_point_clouds:
            os.system(f"cp {image_path} {os.path.join(subsave_root, 'rgb_image.' + image_path.split('.')[-1])}")

        # return localised_pose, [assn, moved_idx]
        return localised_pose, [assn, None]