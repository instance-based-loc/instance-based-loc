import os
import numpy as np
import pandas as pd
import copy
import open3d as o3d
import cv2
import pickle
from tqdm import tqdm
from PIL import Image
import shutil
import time

from scipy.spatial.transform import Rotation as R

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

        # Refine the registration using ICP
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_size*local_dist_factor, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
    except:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size*local_dist_factor, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    return result_icp.transformation, result_icp.inlier_rmse, result_icp.fitness

def evaluate_transform(source, target, trans_init, threshold=0.02):
    res = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )

    return res.inlier_rmse, res.fitness
