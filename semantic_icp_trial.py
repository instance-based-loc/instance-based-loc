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

def assign_labels_from_num_points(object_point_lengths):
    labels = np.zeros(np.sum(object_point_lengths))
    for i in range(1, len(object_point_lengths) - 1):
        start = np.sum(object_point_lengths[:i])
        end = np.sum(object_point_lengths[:i+1])

        labels[start:end] = i

    return labels

def create_positional_encoding(labels, num_features):
    pos_enc = np.zeros((len(labels), num_features))
    pos_enc[np.arange(len(labels)), labels.astype(int)] = 1
    return pos_enc

# supply point lengths, returns onehot encodings
def generate_onehot_array(arr):
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError("Input must be a 1-dimensional numpy array of integers.")
    
    # Determine the size of the 2D array (max index + 1)
    max_index = len(arr)
    num_rows = np.sum(arr)
    
    # Initialize the 2D array with zeros
    result = np.zeros((num_rows, max_index), dtype=float)
    
    row_start = 0
    for index, count in enumerate(arr):
        if count > 0:
            # Create count rows of one-hot encoding with 1 in position `index`
            result[row_start:row_start + count, index] = 1
            row_start += count
    
    return result


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

# supply label
def downsample_and_compute_fpfh_with_one_hot(pcd, voxel_size, one_hot):
    pcd_down = voxel_down_sample_with_colors(pcd, voxel_size)
    print(np.asarray(pcd_down.colors)[0], np.asarray(pcd_down.colors)[-1])

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100))
    pos_enc = one_hot

    combined_fpfh = np.hstack((np.asarray(pcd_fpfh.data).T, pos_enc))
    pcd_fpfh_combined = o3d.pipelines.registration.Feature()
    pcd_fpfh_combined.data = np.asarray(combined_fpfh).T

    return pcd_down, pcd_fpfh_combined

def register_point_clouds(source, target, voxel_size, source_num_points, target_num_points, global_dist_factor=1.5, local_dist_factor=0.4):
    try:
        source_one_hot = generate_onehot_array(source_num_points)
        target_one_hot = generate_onehot_array(target_num_points)

        source_down, source_fpfh_combined = downsample_and_compute_fpfh_with_one_hot(source, voxel_size, source_one_hot)
        target_down, target_fpfh_combined = downsample_and_compute_fpfh_with_one_hot(target, voxel_size, target_one_hot)

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