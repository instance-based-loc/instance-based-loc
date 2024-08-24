from dataloader.tum_dataloader import TUMDataloader
from object_memory.object_memory import ObjectMemory
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
import pickle
from copy import deepcopy
from utils.os_env import get_user
from tqdm import tqdm

from utils.quaternion_ops import QuaternionOps
from utils.logging import get_mem_stats

def dummy_get_embs(
    **kwargs
):
    return torch.tensor([1, 2, 3], device=torch.device(kwargs["device"]))

def main(args):
    dataloader = TUMDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length,
        focal_length_y=args.focal_length,
        map_pointcloud_cache_path=args.map_pcd_cache_path,
        # rot_correction=args.rot_correction,
        start_file_index=args.start_file_index,
        last_file_index=args.last_file_index,
        sampling_period=args.sampling_period
    )

    
    # define and create memory
    memory = ObjectMemory(
        device = args.device,
        ram_pretrained_path = args.ram_pretrained_path,
        sam_checkpoint_path = args.sam_checkpoint_path,
        camera_focal_lenth_x = args.focal_length,
        camera_focal_lenth_y = args.focal_length,
        get_embeddings_func = dummy_get_embs,
        lora_path=args.lora_path
    )
    if args.load_memory == False:

        for idx in dataloader.environment_indices:
            print(f"Making env from index {idx}/{len(dataloader.environment_indices)} currently.")
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
        # memory.remove_points_below_floor()

        # Recluster
        memory.recluster_objects_with_dbscan(visualize=True)


        print("\nMemory is")
        print(memory)

        memory.save_to_pkl(args.memory_load_path)
        print("Memory dumped")
    else:
        memory.load(args.memory_load_path)
        print("Memory loaded")

    exit(0)
    ########### begin localisation ############

    eval_dataloader = TUMDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length,
        focal_length_y=args.focal_length,
        map_pointcloud_cache_path=args.map_pcd_cache_path,
        rot_correction=args.rot_correction,
        start_file_index=args.loc_start_file_index,
        last_file_index=args.loc_last_file_index,
        sampling_period=args.loc_sampling_period
    )

    tgt = []
    pred = []
    trans_errors = []
    rot_errors = []
    chosen_assignments = []

    import matplotlib.pyplot as plt
    import imageio
    import os
    print("Begin localisation")
    for idx in tqdm(eval_dataloader.environment_indices, total=len(eval_dataloader.environment_indices)):
        print(f"Localising {idx}/{len(eval_dataloader.environment_indices)} currently.")
        rgb_image_path, depth_image_path, target_pose = eval_dataloader.get_image_data(idx)
        print(rgb_image_path)
        os.system(f"cp {rgb_image_path} {os.path.join('./out/imgs/', str(idx) + '.png')}")

    exit(0)
    for idx in tqdm(eval_dataloader.environment_indices, total=len(eval_dataloader.environment_indices)):
        print(f"Localistion {idx}/{len(eval_dataloader.environment_indices)} currently.")
        rgb_image_path, depth_image_path, target_pose = eval_dataloader.get_image_data(idx)

        estimated_pose, chosen_assignment = memory.localise(image_path=rgb_image_path, 
                                            depth_image_path=depth_image_path,
                                            testname=args.testname,
                                            subtest_name=f"{idx}" ,
                                            save_point_clouds=args.save_point_clouds,
                                            fpfh_global_dist_factor = args.fpfh_global_dist_factor, 
                                            fpfh_local_dist_factor = args.fpfh_local_dist_factor, 
                                            fpfh_voxel_size = args.fpfh_voxel_size, useLora = True,
                                            consider_floor = False,
                                            perform_semantic_icp=False)

        print("Target pose: ", target_pose)
        print("Estimated pose: ", estimated_pose)

        translation_error = np.linalg.norm(target_pose[:3] - estimated_pose[:3]) 
        rotation_error = QuaternionOps.quaternion_error(target_pose[3:], estimated_pose[3:])

        print("Translation error: ", translation_error)
        print("Rotation_error: ", rotation_error)

        tgt.append(target_pose)
        pred.append(estimated_pose.tolist())
        trans_errors.append(translation_error)
        rot_errors.append(rotation_error)
        chosen_assignments.append(chosen_assignment)

    for idx, _ in enumerate(tqdm(eval_dataloader.environment_indices, total=len(eval_dataloader.environment_indices))):
        print(f"Pose {idx + 1}, image {len(eval_dataloader.environment_indices)}")
        print("Translation error", trans_errors[idx])
        print("Rotation errors", rot_errors[idx])
        print("Assignment: ", chosen_assignments[idx][0])
        print("Moved objects: ", chosen_assignments[idx][1])
        if trans_errors[idx] < 0.6 and rot_errors[idx] < 0.3:
            print("SUCCESS")
        else:
            print("MISALIGNED")
        print()

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    parser.add_argument(
        "-t",
        "--testname",
        type=str,
        help="Experiment name",
        default="8room"
    )
    # dataset params
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the 8room sequence",
        default="/scratch/sarthak/synced_data2"
    )
    parser.add_argument(
        "-e",
        "--eval-img-inds",
        type=int,
        nargs='+',
        help="Indices to be evaluated",
        default=[0]
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
        default="./cache/tum_zip_cache_map_coloured.pcd"
    )
    #device
    parser.add_argument(
        "--device",
        type=str,
        help="Device that the things is being run on",
        default="cuda"
    )
    # checkpoint paths
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
    parser.add_argument(
        "--rot-correction",
        type=float,
        help="correction to roll",
        default=0.0
    )
    # sampling params
    parser.add_argument(
        "--start-file-index",
        type=int,
        help="beginning of file sampling",
        default=0
    )
    parser.add_argument(
        "--last-file-index",
        type=int,
        help="last file to sample",
        default=2000
    )
    parser.add_argument(
        "--sampling-period",
        type=int,
        help="sampling period",
        default=20
    )

    # eval sampling params
    parser.add_argument(
        "--loc-start-file-index",
        type=int,
        help="eval beginning of file sampling",
        default=280
    )
    parser.add_argument(
        "--loc-last-file-index",
        type=int,
        help="eval last file to sample",
        default=1400
    )
    parser.add_argument(
        "--loc-sampling-period",
        type=int,
        help="eval sampling period",
        default=13
    )
    # Memory dump/load args
    parser.add_argument(
        "--load-memory",
        type=bool,
        help="should memory be loaded from a file",
        default=False
    )
    parser.add_argument(
        "--memory-load-path",
        type=str,
        help="file to load memory from, or save it to",
        default='./out/8room_with_floor/tum_desk_memory.pt'
    )

    # lora path
    parser.add_argument(
        "--lora-path",
        type=str,
        help="finetuned lora path",
        default='/home2/aneesh.chavan/instance-based-loc/models/vit_finegrained_5x40_procthor.pt'
    )
    # lora path
    parser.add_argument(
        "--save-point-clouds",
        type=bool,
        default=False
    )
    # icp/fpfh config
    parser.add_argument(
        "--fpfh-global-dist-factor",
        type=float,
        default=1.5
    )
    parser.add_argument(
        "--fpfh-local-dist-factor",
        type=float,
        default=1.5
    )
    parser.add_argument(
        "--fpfh-voxel-size",
        type=float,
        default=0.05
    )

    args = parser.parse_args()
    main(args)
