from dataloader.real_dataloader import RealDataloader
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from utils import depth_utils
import imageio

def main(args):
    dataloader = RealDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length_x,
        focal_length_y=args.focal_length_y,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    rgb_path, depth_path, pose = dataloader.get_image_data(1)
    rgb = np.asarray(imageio.imread(rgb_path))
    depth = np.asarray(imageio.imread(depth_path), dtype=np.float32)
    depth /= 1000.0
    
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

    pcd = dataloader.get_visible_pointcloud(pose, 100, 0.05, 20)

    # proj_depth = get_sense_of_depthmap_from_pointcloud(pcd, depth.shape[0], depth.shape[1], args.focal_length, args.focal_length)

    reformed_pcd = depth_utils.get_pointcloud_from_depth(depth, args.focal_length_x, args.focal_length_y)
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
        default="./data/new_lab"
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
        help="x-Focal length of camera",
        default=385.28887939453125
    )
    parser.add_argument(
        "--focal-length-y",
        type=float,
        help="y-Focal length of camera",
        default=384.3631591796875
    )
    parser.add_argument(
        "--map-pcd-cache-path",
        type=str,
        help="Location where the map's pointcloud is cached for future use",
        default="./.cache/gf_lab4_cache.pcd"
    )
    args = parser.parse_args()

    main(args)