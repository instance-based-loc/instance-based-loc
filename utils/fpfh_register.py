import os
import numpy as np
import copy
import open3d as o3d
import cv2
import pickle
from tqdm import tqdm
from PIL import Image
import shutil
import time

from scipy.spatial.transform import Rotation as R
from scipy.spatial import procrustes

"""
expects two Nx3 arrays

gets the optimal transform assuming correct rowwise correspondences via SVD

SLOW for many points

returns R and t
"""
def get_transformation(source, target):
    """
    Perform Procrustes analysis to align 'source' points with 'target' points.
    
    Args:
        source (np.ndarray): Nx3 array of source points.
        target (np.ndarray): Nx3 array of target points.
    
    Returns:
        R_matrix (np.ndarray): 3x3 rotation matrix.
        t_vector (np.ndarray): 3x1 translation vector.
    """
    # Ensure input arrays are numpy arrays
    source = np.array(source)
    target = np.array(target)
    
    # Compute centroids of the two sets of points
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    print(centroid_source.shape, "AAAAAAAAAAAAAAA")
    
    # Center the points around their centroids
    source_centered = source - centroid_source
    target_centered = target - centroid_target
    
    # Compute the covariance matrix
    H = np.dot(source_centered.T, target_centered)
    
    # Perform Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)
    
    # Ensure the rotation matrix is a proper rotation matrix
    if np.linalg.det(R_matrix) < 0:
        Vt[2, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    
    # Compute the translation vector
    t_vector = centroid_target - np.dot(centroid_source, R_matrix)
    
    return R_matrix, t_vector

def get_SVD_transform(p, q):
    u_p = np.mean(p, axis=0)
    u_q = np.mean(q, axis=0)    
    p_dash = p - u_p
    q_dash = q - u_q
    W = np.zeros((3,3))
    for i in range(len(q_dash)):
        W += np.matmul(q_dash[i, np.newaxis].T, p_dash[i, np.newaxis])
    u, s, vh = np.linalg.svd(W, full_matrices=True)
    M = np.diag([1,1, np.linalg.det(u) * np.linalg.det(vh)])
    R_recovered = u @ M @ vh
    t_recovered = u_q - R_recovered @ u_p
    T_recovered = np.hstack((R_recovered, np.array([[t_recovered[0]], [t_recovered[1]], [t_recovered[2]]])))
    T_recovered = np.vstack((T_recovered, np.array([[0,0,0,1]])))

    return T_recovered



def downsample_and_compute_fpfh(pcd, voxel_size):
    # Downsample the point cloud using Voxel Grid
    pcd_down = copy.deepcopy(pcd)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def register_point_clouds(source, target, voxel_size, global_dist_factor = 1.5, local_dist_factor = 0.4):
    try:    # catch cases where normals cant be computed
        source_down, source_fpfh = downsample_and_compute_fpfh(source, voxel_size)
        target_down, target_fpfh = downsample_and_compute_fpfh(target, voxel_size)


        distance_threshold = voxel_size * global_dist_factor
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.99))

        # fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        # maximum_correspondence_distance=0.5,  # Adjust based on your data
        # )
        
        # Perform fast global registration
        # result_ransac = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        #     source_down, target_down, source_fpfh, target_fpfh, fgr_option
        # )


        # Refine the registration using ICP
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, voxel_size*local_dist_factor, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP()
        )

    except:
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source, target, voxel_size*local_dist_factor, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForColoredICP()
        )

    return result_icp.transformation, result_icp.inlier_rmse, result_icp.fitness

def evaluate_transform(source, target, trans_init, threshold=0.02):
    res = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )

    return res.inlier_rmse, res.fitness
