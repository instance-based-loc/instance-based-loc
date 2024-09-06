import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

from utils import depth_utils

episode_path = "out/hm3d_trial_obj_nav_sq_image/episode_2"
pose_data_path = os.path.join(episode_path, "poses.npy")
rgb_images_save_path = os.path.join(episode_path, "rgb")
depth_images_save_path = os.path.join(episode_path, "depth")

pose_data = np.load(pose_data_path)

full_point_cloud = o3d.geometry.PointCloud()

# Camera intrinsic parameters (based on Habitat config)
image_width = 600  # From your Habitat config
image_height = 600  # From your Habitat config
image_hfov = 90.0
fx = fy = 0.5 * image_width / np.tan(0.5 * np.radians(image_hfov))  # Calculate based on HFOV
cx = image_width / 2
cy = image_height / 2

intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

intrinsic_inverse = np.linalg.inv(intrinsic_matrix)

for i in tqdm(range(len(pose_data))[27:35]):
    rgb_image_path = os.path.join(rgb_images_save_path, f"{i}.png")
    depth_image_path = os.path.join(depth_images_save_path, f"{i}.npy")

    rgb_image = cv2.imread(rgb_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    depth_image = np.load(depth_image_path)
    depth_image = np.squeeze(depth_image)

    depth_image = depth_image.astype(np.float32)

    position = pose_data[i][:3]
    stored_orientation = pose_data[i][3:]

    # Ensure the quaternion is in the format [qx, qy, qz, qw]
    def normalize_quaternion(q):
        norm = np.linalg.norm(q)
        return q / norm if norm > 0 else q

    # Normalize the quaternion
    orientation = normalize_quaternion(np.array([stored_orientation[1], stored_orientation[2], stored_orientation[3], stored_orientation[0]]))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(orientation)

    # Construct the transformation matrix
    transformation_matrix = np.eye(4)

    # Apply rotation and handle the Z-axis inversion
    transformation_matrix[:3, :3] = rotation_matrix @ np.array([[1, 0, 0],
                                                                [0, 1, 0],
                                                                [0, 0, -1]])  # Invert Z-axis

    # Set the translation
    transformation_matrix[:3, 3] = position  # Use position directly without inversion

    depth_image = np.squeeze(depth_image)
    h, w = depth_image.shape
    uv_grid = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).reshape(-1, 2)

    depth_image[depth_image == 0] = np.nan  # Set invalid depth values to NaN
    z_values = depth_image.flatten()
    valid_depth_mask = ~np.isnan(z_values)  # Create a mask for valid depth values
    uv_homogeneous = np.concatenate([uv_grid, np.ones((uv_grid.shape[0], 1))], axis=1)

    points_camera_frame = (intrinsic_inverse @ uv_homogeneous.T).T * z_values[:, np.newaxis]

    # Apply mask to filter out invalid points
    points_camera_frame = points_camera_frame[valid_depth_mask]
    rgb_colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1] for Open3D
    rgb_colors = rgb_colors[valid_depth_mask]  # Filter RGB colors based on valid depth

    # pcd = depth_utils.get_coloured_pointcloud_from_depth(depth_image, rgb_image, fx, fy, None)

    # Transform points to the world frame using the pose matrix
    points_world_frame = (transformation_matrix @ np.hstack((points_camera_frame, np.ones((points_camera_frame.shape[0], 1)))).T).T[:, :3]

    # Add the points and colors to the point cloud
    full_point_cloud.points = o3d.utility.Vector3dVector(np.vstack((full_point_cloud.points, points_world_frame)))
    full_point_cloud.colors = o3d.utility.Vector3dVector(np.vstack((full_point_cloud.colors, rgb_colors)))

    # pose = pose_data[i]

    # new_pose = [
    #     pose[0],
    #     pose[1],
    #     pose[2],

    #     pose[4],
    #     pose[5],
    #     pose[6],
    #     pose[3],
    # ]

    # pcd_transformed = depth_utils.transform_pointcloud(pcd, pose)

    # full_point_cloud += pcd_transformed

# Visualize the final colored point cloud
o3d.io.write_point_cloud(f"./out/hehe.ply", full_point_cloud)

print(full_point_cloud)

# import os
# import numpy as np
# import open3d as o3d
# import cv2
# from tqdm import tqdm
# from utils.fpfh_register import register_point_clouds

# def normalize_quaternion(q):
#     norm = np.linalg.norm(q)
#     return q / norm if norm > 0 else q

# episode_path = "out/hm3d_trial_obj_nav_sq_image/episode_2"
# pose_data_path = os.path.join(episode_path, "poses.npy")
# rgb_images_save_path = os.path.join(episode_path, "rgb")
# depth_images_save_path = os.path.join(episode_path, "depth")

# pose_data = np.load(pose_data_path)

# full_point_cloud = o3d.geometry.PointCloud()

# # Camera intrinsic parameters (based on Habitat config)
# image_width = 600  # From your Habitat config
# image_height = 600  # From your Habitat config
# image_hfov = 90.0
# fx = fy = 0.5 * image_width / np.tan(0.5 * np.radians(image_hfov))  # Calculate based on HFOV
# cx = image_width / 2
# cy = image_height / 2

# intrinsic_matrix = np.array([[fx, 0, cx],
#                              [0, fy, cy],
#                              [0, 0, 1]])

# intrinsic_inverse = np.linalg.inv(intrinsic_matrix)

# # Initialize the previous point cloud for ICP
# previous_point_cloud = None

# for i in tqdm(range(len(pose_data))[41:43]):
#     rgb_image_path = os.path.join(rgb_images_save_path, f"{i}.png")
#     depth_image_path = os.path.join(depth_images_save_path, f"{i}.npy")

#     rgb_image = cv2.imread(rgb_image_path)
#     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

#     depth_image = np.load(depth_image_path)
#     depth_image = np.squeeze(depth_image)

#     depth_image = depth_image.astype(np.float32)

#     position = pose_data[i][:3]
#     stored_orientation = pose_data[i][3:]

#     # Normalize the quaternion before using it
#     orientation = normalize_quaternion(np.array([stored_orientation[1], stored_orientation[2], stored_orientation[3], stored_orientation[0]]))
#     rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(orientation)

#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = rotation_matrix
#     transformation_matrix[:3, 3] = position

#     depth_image[depth_image == 0] = np.nan  # Set invalid depth values to NaN
#     z_values = depth_image.flatten()
#     valid_depth_mask = ~np.isnan(z_values)  # Create a mask for valid depth values
    
#     h, w = depth_image.shape
#     uv_grid = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).reshape(-1, 2)
#     uv_homogeneous = np.concatenate([uv_grid, np.ones((uv_grid.shape[0], 1))], axis=1)

#     points_camera_frame = (intrinsic_inverse @ uv_homogeneous.T).T * z_values[:, np.newaxis]
#     points_camera_frame = points_camera_frame[valid_depth_mask]
#     rgb_colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1] for Open3D
#     rgb_colors = rgb_colors[valid_depth_mask]  # Filter RGB colors based on valid depth

#     # Create a new point cloud for the current frame
#     current_point_cloud = o3d.geometry.PointCloud()
#     current_point_cloud.points = o3d.utility.Vector3dVector(points_camera_frame)
#     current_point_cloud.colors = o3d.utility.Vector3dVector(rgb_colors)

#     # If there is a previous point cloud, apply ICP
#     if previous_point_cloud is not None:
#         # Define stricter parameters
#         max_correspondence_distance = 0.001  # Stricter distance
#         voxel_size = 0.01  # Voxel size for downsampling
#         nb_neighbors = 20  # Number of neighbors for outlier removal
#         std_ratio = 2.0  # Standard deviation ratio for outlier removal

#         # Preprocess point clouds
#         current_point_cloud = current_point_cloud.voxel_down_sample(voxel_size)
#         previous_point_cloud = previous_point_cloud.voxel_down_sample(voxel_size)

#         # Remove outliers from the current point cloud
#         cl, ind = current_point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
#         current_point_cloud = current_point_cloud.select_by_index(ind)

#         # Run ICP with stricter parameters
#         estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
#         criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000, relative_fitness=1e-6, relative_rmse=1e-6)

#         reg_icp = o3d.pipelines.registration.registration_icp(
#             current_point_cloud, previous_point_cloud,
#             max_correspondence_distance=max_correspondence_distance,
#             estimation_method=estimation_method,
#             criteria=criteria
#         )

#         # trans, _, _ = register_point_clouds(current_point_cloud, previous_point_cloud, 0.5, 1.5,)
        
#         # Transform the current point cloud using the ICP result
#         current_point_cloud.transform(reg_icp.transformation)

#     # Add the aligned points to the full point cloud
#     full_point_cloud.points.extend(current_point_cloud.points)
#     full_point_cloud.colors.extend(current_point_cloud.colors)

#     # Update the previous point cloud
#     previous_point_cloud = current_point_cloud

# # Visualize the final colored point cloud
# o3d.io.write_point_cloud(f"./out/hehe.ply", full_point_cloud)

# print(full_point_cloud)
