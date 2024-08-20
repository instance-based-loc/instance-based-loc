import numpy as np
import open3d as o3d
from utils.depth_utils import voxel_down_sample_with_colors
from utils.datatypes import TypedList
from sklearn.neighbors import NearestNeighbors

class ObjectInfo:
    def _add_name(self, new_name):
        if new_name not in self.names:
            self.names.append(new_name)

    def _add_embedding(self, new_emb: np.ndarray):
        # Case 1: If the number of embeddings is less than max limit, append it
        if len(self.embeddings) < self.max_embeddings_num:
            self.embeddings.append(new_emb)
        else:
            # Case 2: Use KNN to find the least similar embedding
            embeddings_array = np.array(self.embeddings)

            knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
            knn.fit(embeddings_array)

            distances, indices = knn.kneighbors(new_emb.reshape(1, -1), n_neighbors=2)
            least_similar_index = indices[0][1]  # The second closest (least similar)

            least_similar_distance = distances[0][1]
            new_emb_distance = knn.kneighbors(embeddings_array[least_similar_index].reshape(1, -1), n_neighbors=1)[0][0][0]

            if new_emb_distance < least_similar_distance:
                self.embeddings[least_similar_index] = new_emb


    def _add_pointcloud(self, new_pointcloud: o3d.geometry.PointCloud):
        combined_points = np.vstack((np.asarray(self.pointcloud.points), np.asarray(new_pointcloud.points)))
        combined_colors = np.vstack((np.asarray(self.pointcloud.colors), np.asarray(new_pointcloud.colors)))

        self.pointcloud.points = o3d.utility.Vector3dVector(combined_points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(combined_colors) 

        self._process_pointcloud()

    def _process_pointcloud(self):
        self.pcd = np.asarray(self.pointcloud.points).T
        self.pcd_colours = np.asarray(self.pointcloud.colors).T

    def _compute_means(self):
        self.mean_emb = np.mean(np.array(self.embeddings), axis=0)
        self.centroid = np.mean(self.pcd, axis=-1)

    def __init__(self, id: int, name: str, emb: np.ndarray, pointcloud: o3d.geometry.PointCloud, max_embeddings_num: int):
        self.id = id
        self.names = TypedList[str](name)
        self.embeddings = TypedList[np.ndarray](emb) # so that we don't accidentally add a torch tensor
        self.pointcloud = pointcloud
        self.max_embeddings_num = max_embeddings_num

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

    def downsample(self, voxel_size):
        self.pointcloud = voxel_down_sample_with_colors(self.pointcloud, voxel_size)
        self._process_pointcloud()

    def add_info(self, new_name: str, new_emb: np.ndarray, new_pointcloud: o3d.geometry.PointCloud, align: bool, max_iteration=30, max_correspondence_distance=0.01):
        if align:
            raise NotImplementedError("Aligning is a To-Do")
        
        self._add_name(new_name)
        self._add_embedding(new_emb)
        self._add_pointcloud(new_pointcloud)

        self._compute_means()

