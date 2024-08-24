from dataloader.synthetic_dataloader import SynthDataloader
from object_memory.object_memory import ObjectMemory
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
from utils.os_env import get_user

from utils.logging import get_mem_stats

def main(args):
    dataloader = SynthDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length,
        focal_length_y=args.focal_length,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    def dummy_get_embs(
        **kwargs
    ):
        return torch.tensor([1, 2, 3], device=torch.device(kwargs["device"]))
    memory = ObjectMemory(
        device = args.device,
        ram_pretrained_path = args.ram_pretrained_path,
        sam_checkpoint_path = args.sam_checkpoint_path,
        camera_focal_lenth_x = args.focal_length,
        camera_focal_lenth_y = args.focal_length,
        get_embeddings_func = dummy_get_embs
    )

    for idx in dataloader.environment_indices:
        print(f"Making env from index {idx} currently.")
        rgb_image_path, depth_image_path, pose = dataloader.get_image_data(idx)

        memory.process_image(
            rgb_image_path,
            depth_image_path,
            pose,
            consider_floor = True
        )

        mem_usage, gpu_usage = get_mem_stats()
        print(f"Using {mem_usage} GB of memory and {gpu_usage} GB of GPU")

    # Downsample
    memory.downsample_all_objects(voxel_size=0.01)

    # Remove below floors
    memory.remove_points_below_floor()
    
    # Recluster
    memory.recluster_objects_with_dbscan(visualize=True)
    

    print("\nMemory is")
    print(memory)

    memory.save(save_directory = "./out/360_trial_with_floor")


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
        "--focal-length",
        type=float,
        help="Focal length of camera",
        default=300
    )
    parser.add_argument(
        "--map-pcd-cache-path",
        type=str,
        help="Location where the map's pointcloud is cached for future use",
        default="./cache/360_zip_cache_map_coloured.pcd"
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
