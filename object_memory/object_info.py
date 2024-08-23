import numpy as np
import open3d as o3d
from utils.depth_utils import voxel_down_sample_with_colors
from sklearn.neighbors import NearestNeighbors
import os, pickle

class ObjectInfo:
    def __init__(self, id: int, name: str, emb: np.ndarray, pointcloud: o3d.geometry.PointCloud, max_embeddings_num: int):
        self.id = id
        self.names: list[str] = [name]
        self.embeddings: list[np.ndarray] = [emb]
        self.pointcloud: o3d.geometry.PointCloud = pointcloud
        self.max_embeddings_num: int = max_embeddings_num

        self._process_pointcloud()

        self.mean_emb = None
        self.centroid = None

        self._compute_means()

    def __repr__(self):
        """
        Returns a string representation of the object information.
        """
        return (
            f"ObjectInfo == ID: {self.id}, Names: {self.names}, Mean_Emb: {self.mean_emb}, Num. Points: {self.pcd.shape}"
        )
    
    def _add_name(self, new_name: str):
        if new_name not in self.names:
            self.names.append(new_name)

    def _add_names(self, new_names: list[str]):
        for new_name in new_names:
            self._add_name(new_name)

    def _add_embedding(self, new_emb: np.ndarray):
        raise NotImplementedError

        # Case 1: If the number of embeddings is less than max limit, append it
        if len(self.embeddings) < self.max_embeddings_num:
            self.embeddings.append(new_emb)
        else:
            # Case 2: Use KNN to find the least similar embedding
            embeddings_array = np.array(self.embeddings)

            # TODO: is this right?
            knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
            knn.fit(embeddings_array)

            distances, indices = knn.kneighbors(new_emb.reshape(1, -1), n_neighbors=2)
            least_similar_index = indices[0][1]  # The second closest (least similar)

            least_similar_distance = distances[0][1]
            new_emb_distance = knn.kneighbors(embeddings_array[least_similar_index].reshape(1, -1), n_neighbors=1)[0][0][0]

            if new_emb_distance < least_similar_distance:
                self.embeddings[least_similar_index] = new_emb

    def _add_embeddings(self, new_embs: list[np.ndarray]):
        self.embeddings.extend(new_embs)
        # TODO: limit the number of embeddings

    def _add_pointcloud(self, new_pointcloud: o3d.geometry.PointCloud):
        combined_points = np.vstack((np.asarray(self.pointcloud.points), np.asarray(new_pointcloud.points)))
        combined_colors = np.vstack((np.asarray(self.pointcloud.colors), np.asarray(new_pointcloud.colors)))

        self.pointcloud.points = o3d.utility.Vector3dVector(combined_points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(combined_colors) 

        self._process_pointcloud()

    def _process_pointcloud(self):
        self.pcd = np.asarray(self.pointcloud.points).T
        self.pcd_colors = np.asarray(self.pointcloud.colors).T

    def _compute_means(self):
        self.mean_emb = np.mean(np.array(self.embeddings), axis=0)
        self.centroid = np.mean(self.pcd, axis=-1)

    
    def __add__(self, new_obj_info):
        self._add_names(new_names = new_obj_info.names)
        self._add_embeddings(new_embs = new_obj_info.embeddings)
        self._add_pointcloud(new_pointcloud = new_obj_info.pointcloud)
        return self

    def downsample(self, voxel_size):
        self.pointcloud = voxel_down_sample_with_colors(self.pointcloud, voxel_size)
        self._process_pointcloud()

    def add_info(self, new_name: str, new_emb: np.ndarray, new_pointcloud: o3d.geometry.PointCloud, align: bool = False, max_iteration=30, max_correspondence_distance=0.01):
        if align:
            raise NotImplementedError("Aligning is a To-Do")
        
        self._add_name(new_name)
        self._add_embedding(new_emb)
        self._add_pointcloud(new_pointcloud)

        self._compute_means()

    def update_pointcloud_with_mask(self, mask: np.ndarray):
        mask = np.asarray(mask)
        
        self.pointcloud.points = o3d.utility.Vector3dVector(np.asarray(self.pointcloud.points)[mask, :])
        self.pointcloud.colors = o3d.utility.Vector3dVector(np.asarray(self.pointcloud.colors)[mask, :])

        self._process_pointcloud()

    def save(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        o3d.io.write_point_cloud(os.path.join(save_directory, "pointcloud.ply"), self.pointcloud)
        with open(os.path.join(save_directory, "info.pkl"), "wb") as f:
            pickle.dump({
                "names": self.names,
                "embeddings": self.embeddings,
                "max_embeddings_num": self.max_embeddings_num
            }, f)
