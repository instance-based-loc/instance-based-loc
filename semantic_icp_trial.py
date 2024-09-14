# import open3d as o3d
# import numpy as np

# def create_cylinder(radius, height, resolution, color):
#     mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
#     mesh_cylinder.compute_vertex_normals()
    
#     pcd_cylinder = mesh_cylinder.sample_points_uniformly(number_of_points=5000)
    
#     colors = np.full((np.asarray(pcd_cylinder.points).shape[0], 3), color, dtype=np.float32)
#     pcd_cylinder.colors = o3d.utility.Vector3dVector(colors)
    
#     return pcd_cylinder

# def main():
#     blue = [0.0, 0.0, 1.0]
#     red = [1.0, 0.0, 0.0]
    
#     pcd_cylinder1 = create_cylinder(radius=1.0, height=2.0, resolution=30, color=red)
#     pcd_cylinder2 = create_cylinder(radius=1.0, height=2.0, resolution=30, color=blue)
    
#     pcd_cylinder2.translate((4, 0, 0))
    
#     pcd_combined = pcd_cylinder1 + pcd_cylinder2
    
#     o3d.io.write_point_cloud("./out/target_cylinders.ply", pcd_combined)
    
# if __name__ == "__main__":
#     main()
import open3d as o3d
import numpy as np
from utils.depth_utils import voxel_down_sample_with_colors

def assign_labels_from_color(pcd):
    colors = np.asarray(pcd.colors)
    labels = np.zeros(colors.shape[0])
    
    blue_threshold = np.array([0, 0, 1])
    red_threshold = np.array([1, 0, 0])
    
    labels[np.all(np.isclose(colors, blue_threshold, atol=0.1), axis=1)] = 1
    labels[np.all(np.isclose(colors, red_threshold, atol=0.1), axis=1)] = 0
    
    return labels

def create_positional_encoding(labels, num_features):
    pos_enc = np.zeros((len(labels), num_features))
    pos_enc[np.arange(len(labels)), labels.astype(int)] = 1
    return pos_enc

def downsample_and_compute_fpfh_with_labels(pcd, voxel_size):
    pcd_down = voxel_down_sample_with_colors(pcd, voxel_size)
    print(np.asarray(pcd_down.colors)[0], np.asarray(pcd_down.colors)[-1])

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100))
    labels = assign_labels_from_color(pcd_down)
    max_objects_in_corres = 10
    pos_enc = create_positional_encoding(labels, max_objects_in_corres)
    
    combined_fpfh = np.hstack((np.asarray(pcd_fpfh.data).T, pos_enc))
    pcd_fpfh_combined = o3d.pipelines.registration.Feature()
    pcd_fpfh_combined.data = np.asarray(combined_fpfh).T
    
    return pcd_down, pcd_fpfh_combined

def register_point_clouds(source, target, voxel_size, global_dist_factor=1.5, local_dist_factor=0.4):
    try:
        source_down, source_fpfh_combined = downsample_and_compute_fpfh_with_labels(source, voxel_size)
        target_down, target_fpfh_combined = downsample_and_compute_fpfh_with_labels(target, voxel_size)

        fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=0.5,
        )

        result_ransac = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh_combined, target_fpfh_combined, fgr_option
        )

        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_size * local_dist_factor, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    except Exception as e:
        print(f"Exception during registration: {e}")
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * local_dist_factor, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    return result_icp.transformation, result_icp.inlier_rmse, result_icp.fitness

def main():
    source = o3d.io.read_point_cloud("./out/source_cylinders.ply")
    target = o3d.io.read_point_cloud("./out/target_cylinders.ply")

    print(np.asarray(source.colors)[0], np.asarray(source.colors)[-1])
    print(np.asarray(target.colors)[0], np.asarray(target.colors)[-1])
    
    voxel_size = 0.05
    
    transformation, inlier_rmse, fitness = register_point_clouds(source, target, voxel_size)
    
    print("Transformation Matrix:")
    print(transformation)
    print("Inlier RMSE:", inlier_rmse)
    print("Fitness:", fitness)
    
    source.transform(transformation)
    
    combined_pcd = source + target
    
    o3d.io.write_point_cloud("./out/combined_point_cloud.ply", combined_pcd)

if __name__ == "__main__":
    main()


