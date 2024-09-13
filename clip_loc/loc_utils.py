import cv2
import numpy as np
import open3d as o3d

def get_camera_pose(rvec, tvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Create a 4x4 transformation matrix
    pose = np.eye(4)  # Start with an identity matrix
    pose[:3, :3] = R  # Set the rotation part
    pose[:3, 3] = tvec.flatten()  # Set the translation part

    return pose

def p3p_pose_estimation(objectPoints, imagePoints, cameraMatrix, distCoeffs = np.zeros((1, 4), dtype=np.float32)):
    """
    Estimate all possible camera poses using the P3P method.
    
    Args:
        objectPoints (np.ndarray): 3x3 array of 3D object points.
        imagePoints (np.ndarray): 3x2 array of 2D image points.
        cameraMatrix (np.ndarray): 3x3 camera intrinsic matrix.
        distCoeffs (np.ndarray): 1x4, 1x5, 1x8 or 1x12 vector of distortion coefficients.
    
    Returns:
        retval (int): Number of valid solutions.
        rvecs (list of np.ndarray): Rotation vectors for each solution.
        tvecs (list of np.ndarray): Translation vectors for each solution.
    """
    flags = cv2.SOLVEPNP_P3P
    
    rvecs = []
    tvecs = []
    
    retval, rv, tv = cv2.solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=flags)
    
    poses = []

    for i in range(retval):
        poses.append(get_camera_pose(rv[i], tv[i]))
    
    return poses

def project_pointcloud_to_image(pcd: o3d.geometry.PointCloud, 
                                camera_intrinsics: np.ndarray, 
                                camera_pose: np.ndarray, 
                                image_shape: tuple) -> np.ndarray:
    """
    Projects a 3D point cloud to a 2D image plane using the camera's intrinsic matrix and pose.

    Args:
        pcd (open3d.geometry.PointCloud): The 3D point cloud to be projected.
        camera_intrinsics (np.ndarray): The camera intrinsic matrix (3x3).
        camera_pose (np.ndarray): The camera pose as a 4x4 matrix.
        image_shape (tuple): The shape of the image (height, width).
        
    Returns:
        np.ndarray: A 2D image with the projected points marked.
    """
    points = np.asarray(pcd.points)  # Shape: (N, 3)

    extrinsics = np.linalg.inv(camera_pose)

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    camera_coords = (extrinsics @ points_homogeneous.T).T  # Shape: (N, 4)

    camera_coords = camera_coords[:, :3]

    camera_coords = camera_coords[camera_coords[:, 2] > 0]

    pixel_coords = (camera_intrinsics @ camera_coords.T).T  # Shape: (N, 3)

    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:3]

    pixel_coords = np.round(pixel_coords).astype(int)

    image = np.zeros(image_shape, dtype=np.uint8)

    height, width = image_shape[:2]
    valid_indices = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) & \
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
    pixel_coords = pixel_coords[valid_indices]

    image[pixel_coords[:, 1], pixel_coords[:, 0]] = 1

    return image

# # Example usage
# objectPoints = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
# imagePoints = np.array([[100, 200], [150, 200], [100, 250]], dtype=np.float32)
# cameraMatrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

# retval, rvecs, tvecs = p3p_pose_estimation(objectPoints, imagePoints, cameraMatrix)
# print(f"Number of valid solutions: {retval}")

# for i in range(retval):
#     print(f"Solution {i+1}:")
#     print("Rotation vector:", rvecs[i])
#     print("Translation vector:", tvecs[i])