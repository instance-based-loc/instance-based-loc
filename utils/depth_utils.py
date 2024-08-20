import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch

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


def get_mask_pointclouds_from_depth(
        depth_image: np.ndarray, 
        masks: torch.Tensor, 
        focal_length_x: float, 
        focal_length_y: float, 
        outlier_removal_config=DEFAULT_OUTLIER_REMOVAL_CONFIG
) -> list[o3d.geometry.PointCloud]:
    """
    Returns a list of 3D point clouds for each segmented object based on depth information.

    Parameters:
    - depth_image (np.ndarray): The depth image as a 2D numpy array.
    - masks (torch.Tensor): Binary segmentation masks for each object.
    - focal_length_x (float): Focal length in the x direction for depth-to-distance conversion.
    - focal_length_y (float): Focal length in the y direction for depth-to-distance conversion.
    - outlier_removal_config (dict, optional): Configuration for outlier removal in the point cloud.

    Returns:
    - pointclouds (list[o3d.geometry.PointCloud]): List of 3D point clouds for each segmented object.
    """
    pointclouds = []
    for i in range(masks.shape[0]):
        mask = masks[i].squeeze().cpu().numpy()
        masked_depth = depth_image * mask
        
        pointcloud = get_pointcloud_from_depth(masked_depth, focal_length_x, focal_length_y, outlier_removal_config)
        pointclouds.append(pointcloud)
    
    return pointclouds

def get_mask_coloured_pointclouds_from_depth(
        depth_image: np.ndarray, 
        rgb_image: np.ndarray,
        masks: torch.Tensor, 
        focal_length_x: float, 
        focal_length_y: float, 
        outlier_removal_config=DEFAULT_OUTLIER_REMOVAL_CONFIG
) -> list[o3d.geometry.PointCloud]:
    """
    Returns a list of 3D point clouds for each segmented object based on depth information.

    Args:
        depth_image (np.ndarray): The depth image as a 2D numpy array.
        rgb_image (np.ndarray): The rgb image as a 2D numpy array.
        masks (torch.Tensor): Binary segmentation masks for each object.
        focal_length_x (float): Focal length in the x direction for depth-to-distance conversion.
        focal_length_y (float): Focal length in the y direction for depth-to-distance conversion.
        outlier_removal_config (dict, optional): Configuration for outlier removal in the point cloud.

    Returns:
        pointclouds (list[o3d.geometry.PointCloud]): List of 3D point clouds for each segmented object.
    """
    pointclouds = []
    for i in range(masks.shape[0]):
        mask = masks[i].squeeze().cpu().numpy()
        masked_depth = depth_image * mask
        
        pointcloud = get_coloured_pointcloud_from_depth(masked_depth, rgb_image, focal_length_x, focal_length_y, outlier_removal_config)
        pointclouds.append(pointcloud)
    
    return pointclouds

def voxel_down_sample_with_colors(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Downsamples an Open3D point cloud using a voxel grid and preserves the averaged colors.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud to be downsampled.
        voxel_size (float): The voxel grid size for downsampling.
    
    Returns:
        o3d.geometry.PointCloud: The downsampled point cloud with preserved colors.
    """
    # Downsample the point cloud
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Compute the voxel grid indices
    voxel_indices = np.floor(np.asarray(pcd.points) / voxel_size).astype(np.int64)
    
    # Create a dictionary to average points and colors within each voxel
    voxel_dict = {}
    
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = {
                "points": [],
                "colors": []
            }
        voxel_dict[key]["points"].append(pcd.points[i])
        if pcd.has_colors():
            voxel_dict[key]["colors"].append(pcd.colors[i])
    
    # Average points and colors in each voxel
    downsampled_points = []
    downsampled_colors = []
    
    for voxel in voxel_dict.values():
        downsampled_points.append(np.mean(voxel["points"], axis=0))
        if pcd.has_colors():
            downsampled_colors.append(np.mean(voxel["colors"], axis=0))
    
    # Assign the averaged points and colors to the downsampled point cloud
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    if pcd.has_colors():
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)
    
    return downsampled_pcd