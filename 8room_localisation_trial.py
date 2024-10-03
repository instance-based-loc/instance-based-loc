from dataloader.eightroom_dataloader import EightRoomDataLoader
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
from utils.embeddings import get_all_clip_embeddings, get_all_dino_embeddings, get_dator_embeddings, get_all_vit_embeddings

def dummy_get_embs(
    **kwargs
):
    return torch.tensor([1, 2, 3], device=torch.device(kwargs["device"]))

def main(args):
    if args.embeddings == "clip":
        embeddings_func = get_all_clip_embeddings
    elif args.embeddings == "dino":
        embeddings_func = get_all_dino_embeddings
    elif args.embeddings == "dator":
        embeddings_func = get_dator_embeddings
    elif args.embeddings == "vit":
        embeddings_func = get_all_vit_embeddings
    else:
        raise ValueError("Invalid embeddings function")

    dataloader = EightRoomDataLoader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length,
        focal_length_y=args.focal_length,
        map_pointcloud_cache_path=args.map_pcd_cache_path,
        rot_correction=args.rot_correction,
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
        get_embeddings_func = embeddings_func,
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



        # save memory point cloud
        pcd_list = []
        
        for info in memory.memory:
            object_pcd = info.pcd
            pcd_list.append(object_pcd)

        combined_pcd = o3d.geometry.PointCloud()

        for bhencho in range(len(pcd_list)):
            pcd_np = pcd_list[bhencho]
            pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
            pcd = o3d.geometry.PointCloud()
            pcd.points = pcd_vec
            pcd.paint_uniform_color(np.random.rand(3))
            combined_pcd += pcd
    
        save_path = f"/home2/aneesh.chavan/instance-based-loc/pcds/cached_{args.testname}_before_cons.ply"
        o3d.io.write_point_cloud(save_path, combined_pcd)



        # Downsample
        memory.downsample_all_objects(voxel_size=0.01)

        memory._recluster_IoU(0.3)
        # memory.recluster_via_combined(eps=0.05, embedding_distance_threshold=0.5, min_points_per_cluster=1)

        memory.recluster_via_clustering_and_IoU(eps=0.05, embedding_distance_threshold=0.5, IoU_threshold=0.25, min_points_per_cluster=50)



        # save
        pcd_list = []

        for info in memory.memory:
            object_pcd = info.pcd
            pcd_list.append(object_pcd)

        combined_pcd = o3d.geometry.PointCloud()

        for bhencho in range(len(pcd_list)):
            pcd_np = pcd_list[bhencho]
            pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
            pcd = o3d.geometry.PointCloud()
            pcd.points = pcd_vec
            pcd.paint_uniform_color(np.random.rand(3))
            combined_pcd += pcd

        save_path = f"/home2/aneesh.chavan/instance-based-loc/pcds/cached_{args.testname}_after_cons.ply"
        o3d.io.write_point_cloud(save_path, combined_pcd)


        print("\nMemory is")
        print(memory)

        memory.save_to_pkl(args.memory_load_path)
        print("Memory dumped")
    else:
        memory.load(args.memory_load_path)
        print("Memory loaded")


    ########### begin localisation ############

    eval_dataloader = EightRoomDataLoader(
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

    f = open(f"./out/{args.testname}_results.txt", "w")

    # bin the results by storing them in a dictionary
    d_tr = {'0.1': 0, '0.3': 0, '0.6': 0, '1.0': 0, '1.5': 0, '3.0': 0, 'other': 0}
    r_tr = {'0.1': 0, '0.3': 0, '0.6': 0, '1.0': 0, '1.5': 0, 'other': 0}

    # Output results
    total = 0
    successes = 0
    avg_trans_error = 0
    avg_rot_error = 0

    # Output results
    for idx, _ in enumerate(tqdm(eval_dataloader.environment_indices, total=len(eval_dataloader.environment_indices))):
        print(f"Pose {idx + 1}, image {len(eval_dataloader.environment_indices)}")
        print("Translation error", trans_errors[idx])
        print("Rotation errors", rot_errors[idx])
        print("Assignment: ", chosen_assignments[idx][0])
        print("Moved objects: ", chosen_assignments[idx][1])
        if trans_errors[idx] < 0.6 and rot_errors[idx] < 0.3:
            print("SUCCESS")
            successes += 1
        else:
            print("MISALIGNED")
        total += 1
        print()
        f.write(f"Pose {idx + 1}, image {len(eval_dataloader.environment_indices)}\n")
        f.write(f"Translation error: {trans_errors[idx]}\n")
        f.write(f"Rotation errors: {rot_errors[idx]}\n")
        f.write(f"Assignment: {chosen_assignments[idx][0]}\n")
        f.write(f"Moved objects: {chosen_assignments[idx][1]}\n")
        if trans_errors[idx] < 0.6 and rot_errors[idx] < 0.3:
            f.write("SUCCESS\n")
        else:
            f.write("MISALIGNED\n")

        avg_trans_error += trans_errors[idx]
        avg_rot_error += rot_errors[idx] 
        
        if trans_errors[idx] < 0.1:
            d_tr['0.1'] += 1
        if trans_errors[idx] < 0.3:
            d_tr['0.3'] += 1
        if trans_errors[idx] < 0.6:
            d_tr['0.6'] += 1
        if trans_errors[idx] < 1.0:
            d_tr['1.0'] += 1
        if trans_errors[idx] < 1.5:
            d_tr['1.5'] += 1
        if trans_errors[idx] < 3.0:
            d_tr['3.0'] += 1
        else:
            d_tr['other'] += 1

        if rot_errors[idx] < 0.1:
            r_tr['0.1'] += 1
        if rot_errors[idx] < 0.3:
            r_tr['0.3'] += 1
        if rot_errors[idx] < 0.6:
            r_tr['0.6'] += 1
        if rot_errors[idx] < 1.0:
            r_tr['1.0'] += 1
        if rot_errors[idx] < 1.5:
            r_tr['1.5'] += 1
        else:
            r_tr['other'] += 1

        f.write("\n")
    f.write(f"Bagged results for {len(eval_dataloader.environment_indices)} eval indices\n")
    f.write(f"Translation error less than 0.1: {d_tr['0.1']}\n")
    f.write(f"Translation error less than 0.3: {d_tr['0.3']}\n")
    f.write(f"Translation error less than 0.6: {d_tr['0.6']}\n")
    f.write(f"Translation error less than 1.0: {d_tr['1.0']}\n")
    f.write(f"Translation error less than 1.5: {d_tr['1.5']}\n")
    f.write(f"Translation error less than 3.0: {d_tr['3.0']}\n")
    f.write(f"Translation error greater than 3.0: {d_tr['other']}\n")
    f.write("\n")
    f.write(f"Rotation error less than 0.1: {r_tr['0.1']}\n")
    f.write(f"Rotation error less than 0.3: {r_tr['0.3']}\n")
    f.write(f"Rotation error less than 0.6: {r_tr['0.6']}\n")
    f.write(f"Rotation error less than 1.0: {r_tr['1.0']}\n")
    f.write(f"Rotation error less than 1.5: {r_tr['1.5']}\n")
    f.write(f"Rotation error greater than 1.5: {r_tr['other']}\n")
    f.write("\n")  
    f.write(f"Average Translation Error: {avg_trans_error/total}\n")
    f.write(f"Average Rotation Error: {avg_rot_error/total}\n")
    f.write(f"Median Translation Error: {np.median(trans_errors)}\n")
    f.write(f"Median Rotation Error: {np.median(rot_errors)}\n")
    f.write(f"Total Success Rate: {successes/total*100}\n")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    parser.add_argument(
        "-t",
        "--testname",
        type=str,
        help="Experiment name",

        default="8room_agg_clustering"

    )
    # dataset params
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the 8room sequence",
        default="/scratch/aneesh.chavan/8room/8-room-v1/1/"
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
        default="./cache/360_zip_cache_map_coloured.pcd"
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
        default=200
    )
    parser.add_argument(
        "--last-file-index",
        type=int,
        help="last file to sample",
        default=1500
    )
    parser.add_argument(
        "--sampling-period",
        type=int,
        help="sampling period",
        default=15
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
        default=26
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
        default='./out/8room_with_floor/8room_memory.pt'
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
