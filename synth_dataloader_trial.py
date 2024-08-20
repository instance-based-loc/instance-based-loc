from dataloader.synthetic_dataloader import SynthDataloader, get_sense_of_depthmap_from_pointcloud
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from utils import depth_utils
import imageio

def main(args):
    dataloader = SynthDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length_x,
        focal_length_y=args.focal_length_y,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    rgb_path, depth_path, pose = dataloader.get_image_data(0)
    rgb = np.asarray(imageio.imread(rgb_path))
    depth = np.load(depth_path)

    pcd = dataloader.get_visible_pointcloud(pose, 100, 0.05, 20)

    proj_depth = get_sense_of_depthmap_from_pointcloud(pcd, depth.shape[0], depth.shape[1], args.focal_length, args.focal_length)

    reformed_pcd = depth_utils.get_pointcloud_from_depth(proj_depth, args.focal_length, args.focal_length)
    # o3d.visualization.draw_geometries([pcd, reformed_pcd])
    # o3d.visualization.draw_geometries([pcd, dataloader.get_pointcloud()])

    # in actual position (not in camera frame)
    o3d.visualization.draw_geometries([depth_utils.transform_pointcloud(pcd, pose), dataloader.get_pointcloud(), depth_utils.transform_pointcloud(reformed_pcd, pose)])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to synthetic data",
        default="./data/our-synthetic/360_basic_test"
    )
    parser.add_argument(
        "-e",
        "--eval-img-inds",
        type=int,
        nargs='+',
        help="Indices to be evaluated",
        default=[4]
    )
    parser.add_argument(
        "--focal-length-x",
        type=float,
        help="Focal length of camera's x-axis",
        default=300
    )
    parser.add_argument(
        "--focal-length-y",
        type=float,
        help="Focal length of camera's y-axis",
        default=300
    )
    parser.add_argument(
        "--map-pcd-cache-path",
        type=str,
        help="Location where the map's pointcloud is cached for future use",
        default="./cache/360_zip_cache_map_coloured.pcd"
    )
    args = parser.parse_args()

    main(args)