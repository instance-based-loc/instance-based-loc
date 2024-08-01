import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def get_pointcloud_from_depth(
        depth_image: np.ndarray, 
        focal_lenth: float,
        outlier_removal_config = \
            {
                "radius_nb_points": 12,
                "radius": 0.05,
            }
) -> o3d.geometry.PointCloud:
    w, h = depth_image.shape

    horizontal_distance = np.linspace(-h / 2, h / 2, h, dtype=np.float32)
    vertical_distance = np.linspace(w / 2, -w / 2, w, dtype=np.float32).reshape(-1, 1)

    horizontal_distance = np.tile(horizontal_distance, (w, 1))
    vertical_distance = np.tile(vertical_distance, (1, h))

    X = horizontal_distance * depth_image / focal_lenth
    Y = vertical_distance * depth_image / focal_lenth
    Z = depth_image

    pointcloud_points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

    # Filtering out [0, 0, 0] points
    pointcloud_points = pointcloud_points[pointcloud_points[:, 2] != 0]

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_points)

    # Outlier removal
    if outlier_removal_config is not None:
        pointcloud, _ = pointcloud.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                        radius=outlier_removal_config["radius"])
        
    return pointcloud

def transform_pointcloud(
        pointcloud: o3d.geometry.PointCloud,
        pose: np.ndarray
) -> o3d.geometry.PointCloud:
    t = pose[:3]
    q = pose[3:]

    q /= np.linalg.norm(q)
    R = Rotation.from_quat(q).as_matrix()

    pointcloud_points = np.asarray(pointcloud.points)

    transformed_pointcloud_points = pointcloud_points @ R.T
    transformed_pointcloud_points += t

    transformed_pointcloud = o3d.geometry.PointCloud()
    transformed_pointcloud.points = o3d.utility.Vector3dVector(transformed_pointcloud_points)

    return transformed_pointcloud

