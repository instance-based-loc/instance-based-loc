import numpy as np
import open3d as o3d
import os, pickle

class ClipLocObjectInfo:
    def __init__(self, id: int, text: str, text_embedding: np.ndarray, pointcloud: o3d.geometry.PointCloud, ellipsoid: o3d.geometry.PointCloud):
        self.id = id
        self.text = text
        self.text_embedding = text_embedding
        self.pointcloud: o3d.geometry.PointCloud = pointcloud
        self.ellipsoid: o3d.geometry.PointCloud = ellipsoid

    def save(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        o3d.io.write_point_cloud(os.path.join(save_directory, "pointcloud.ply"), self.pointcloud)
        o3d.io.write_point_cloud(os.path.join(save_directory, "ellipsoid.ply"), self.pointcloud)
        with open(os.path.join(save_directory, "info.pkl"), "wb") as f:
            pickle.dump({
                "text": self.text,
                "text_embedding": self.text_embedding,
                "id": self.id
            }, f)

    def __repr__(self):
        return (f"ClipLocObjectInfo(id={self.id}, text='{self.text}', "
                f"text_embedding_shape={self.text_embedding.shape}, "
                f"pointcloud_points={len(np.asarray(self.pointcloud.points))}, "
                f"ellipsoid_points={len(np.asarray(self.ellipsoid.points))})")

    @classmethod
    def load(cls, load_directory: str):
        pointcloud = o3d.io.read_point_cloud(os.path.join(load_directory, "pointcloud.ply"))
        ellipsoid = o3d.io.read_point_cloud(os.path.join(load_directory, "ellipsoid.ply"))

        with open(os.path.join(load_directory, "info.pkl"), "rb") as f:
            info = pickle.load(f)

        return cls(
            id=info["id"],
            text=info["text"],
            text_embedding=info["text_embedding"],
            pointcloud=pointcloud,
            ellipsoid=ellipsoid
        )
