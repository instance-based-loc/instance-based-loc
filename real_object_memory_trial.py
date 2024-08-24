from dataloader.real_dataloader import RealDataloader
from object_memory.object_finder import ObjectFinder
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from utils.os_env import get_user

def main(args):
    dataloader = RealDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length_x,
        focal_length_y=args.focal_length_y,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    ObjectFinder.setup(
        device = args.device,
        ram_pretrained_path = args.ram_pretrained_path,
        sam_checkpoint_path = args.sam_checkpoint_path
    )


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
    parser.add_argument(
        "--device",
        type=str,
        help="Device that the things is being run on",
        default="cuda"
    )
    parser.add_argument(
        "--sam-checkpoint-path",
        type=str,
        help="Path to checkpoint being used for SAM",
        default=f'/scratch/{get_user()}/sam_vit_h_4b8939.pth'
    )
    parser.add_argument(
        "--ram-pretrained-path",
        type=str,
        help="Path to pretained model being used for RAM",
        default=f'/scratch/{get_user()}/ram_swin_large_14m.pth'
    )
    args = parser.parse_args()

    main(args)
