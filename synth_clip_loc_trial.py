from dataloader.synthetic_dataloader import SynthDataloader
from object_memory.object_memory import ObjectMemory
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
from utils.os_env import get_user

from utils.logging import get_mem_stats

from clip_loc.clip_loc_object_memory import ClipLocObjectMemory

def get_intrinsic_matrix(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

def main(args):
    dataloader = SynthDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length,
        focal_length_y=args.focal_length,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    # def dummy_get_embs(
    #     **kwargs
    # ):
    #     return torch.tensor([1, 2, 3], device=torch.device(kwargs["device"]))

    # memory = ObjectMemory(
    #     device = args.device,
    #     ram_pretrained_path = args.ram_pretrained_path,
    #     sam_checkpoint_path = args.sam_checkpoint_path,
    #     camera_focal_lenth_x = args.focal_length,
    #     camera_focal_lenth_y = args.focal_length,
    #     get_embeddings_func = dummy_get_embs
    # )

    # for idx in dataloader.environment_indices:
    #     print(f"Making env from index {idx} currently.")
    #     rgb_image_path, depth_image_path, pose = dataloader.get_image_data(idx)

    #     memory.process_image(
    #         rgb_image_path,
    #         depth_image_path,
    #         pose,
    #         consider_floor = True
    #     )

    #     mem_usage, gpu_usage = get_mem_stats()
    #     print(f"Using {mem_usage} GB of memory and {gpu_usage} GB of GPU")

    # # Downsample
    # memory.downsample_all_objects(voxel_size=0.01)

    # # Remove below floors
    # memory.remove_points_below_floor()
    
    # # Recluster
    # memory.recluster_objects_with_dbscan(visualize=True)
    

    # print("\nMemory is")
    # print(memory)

    # memory.save(save_directory = "./out/360_trial_with_floor_dummy")

    # # TODO - change this above code to use the changes made to use the new clustering method

    # # Now, convert this memory into clip-loc based memory
    # clip_loc_memory = ClipLocObjectMemory(memory.memory)

    # clip_loc_memory.save("./out/360_clip_loc_mem")

    # Now, we've written loading functionality
    clip_loc_memory = ClipLocObjectMemory.load("./out/360_clip_loc_mem")

    print(f"Loaded clip-loc memory with {len(clip_loc_memory)} objects")

    intrinsic_matrix = get_intrinsic_matrix(300, 300, 300, 300) # Or, use the intrinsic matrix I hardcoded for HM3D

    actual_poses = []
    calc_poses = []

    for idx in dataloader.evaluation_indices:
        print(f"Processing index {idx} now")

        img_path, _, actual_pose = dataloader.get_image_data(idx)
        calc_pose = clip_loc_memory.localize(img_path, intrinsic_matrix)

        calc_poses.append(calc_pose)
        actual_poses.append(actual_pose)

    for i in range(len(actual_poses)):
        print("i:")
        print("\t actual    :", actual_poses[i])
        print("\t calculated:", calc_poses[i])

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
        default=range(8)
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
