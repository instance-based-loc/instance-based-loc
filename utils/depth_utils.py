import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

DEFAULT_OUTLIER_REMOVAL_CONFIG = \
    {
        "radius_nb_points": 12,
        "radius": 0.05,
    }

def get_pointcloud_from_depth(
        depth_image: np.ndarray, 
        focal_lenth_x: float,
        focal_lenth_y: float,
        outlier_removal_config = DEFAULT_OUTLIER_REMOVAL_CONFIG
) -> o3d.geometry.PointCloud:
    w, h = depth_image.shape
    # print(depth_image)

    horizontal_distance = np.linspace(-h / 2, h / 2, h, dtype=np.float32)
    vertical_distance = np.linspace(w / 2, -w / 2, w, dtype=np.float32).reshape(-1, 1)

    horizontal_distance = np.tile(horizontal_distance, (w, 1))
    vertical_distance = np.tile(vertical_distance, (1, h))

    X = horizontal_distance * depth_image / focal_lenth_x
    Y = vertical_distance * depth_image / focal_lenth_y
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

def get_coloured_pointcloud_from_depth(
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        focal_lenth_x: float,
        focal_lenth_y: float,
        outlier_removal_config = DEFAULT_OUTLIER_REMOVAL_CONFIG

) -> o3d.geometry.PointCloud:
    # NOTE: ensure this is the calling function
    assert depth_image.shape[:2] == rgb_image.shape[:2], "Depth and RGB image dimensions do not match"

    w, h = depth_image.shape

    horizontal_distance = np.linspace(-h / 2, h / 2, h, dtype=np.float32)
    vertical_distance = np.linspace(w / 2, -w / 2, w, dtype=np.float32).reshape(-1, 1)

    horizontal_distance = np.tile(horizontal_distance, (w, 1))
    vertical_distance = np.tile(vertical_distance, (1, h))

    X = horizontal_distance * depth_image / focal_lenth_x
    Y = vertical_distance * depth_image / focal_lenth_y
    Z = depth_image

    # Flatten the arrays
    pointcloud_points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

    # Filtering out [0, 0, 0] points
    valid_mask = pointcloud_points[:, 2] != 0
    pointcloud_points = pointcloud_points[valid_mask]

    # Map RGB colors to points
    rgb_image = rgb_image.astype(np.float32) / 255.0  # Normalize RGB values
    rgb_values = rgb_image.reshape(-1, 3)[valid_mask]

    # Create the point cloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_points)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_values)

    # Outlier removal
    if outlier_removal_config is not None:
        pointcloud, _ = pointcloud.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                        radius=outlier_removal_config["radius"])
        
    return pointcloud

def transform_pointcloud(
        pointcloud: o3d.geometry.PointCloud,
        pose: np.ndarray
) -> o3d.geometry.PointCloud:
    # Extract translation and quaternion from the pose
    t = pose[:3]
    q = pose[3:]

    # Normalize the quaternion
    q /= np.linalg.norm(q)
    R = Rotation.from_quat(q).as_matrix()

    # Get points and colors from the point cloud
    pointcloud_points = np.asarray(pointcloud.points)
    pointcloud_colors = np.asarray(pointcloud.colors)

    # Apply transformation to the points
    transformed_pointcloud_points = (R @ pointcloud_points.T).T + t

    # Create and return a new point cloud with transformed points and original colors
    transformed_pointcloud = o3d.geometry.PointCloud()
    transformed_pointcloud.points = o3d.utility.Vector3dVector(transformed_pointcloud_points)
    transformed_pointcloud.colors = o3d.utility.Vector3dVector(pointcloud_colors)
    
    return transformed_pointcloud
