from dataloader.hm3d_dataloder import HM3DDataloader
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from utils import depth_utils
import imageio

image_width = 600  # From your Habitat config
image_height = 600  # From your Habitat config
image_hfov = 90.0
fx = fy = 0.5 * image_width / np.tan(0.5 * np.radians(image_hfov))  # Calculate based on HFOV

def main():
    dataloader = HM3DDataloader(
        evaluation_indices=[],
        data_path="/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/object_nav_trajectories/episode_2",
        focal_length_x=fx,
        focal_length_y=fy,
        map_pointcloud_cache_path="/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/object_nav_trajectories/episode_2/full_env_pcd.ply"
    )


if __name__ == "__main__":
    main()