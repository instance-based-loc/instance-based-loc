#!/bin/bash

datasets=(
# "rgbd_dataset_freiburg1_360"
# "rgbd_dataset_freiburg1_desk"
# "rgbd_dataset_freiburg1_desk2"
# "rgbd_dataset_freiburg1_floor"
# "rgbd_dataset_freiburg1_room"
# "rgbd_dataset_freiburg2_360_hemisphere"
# "rgbd_dataset_freiburg2_360_kidnap"
# "rgbd_dataset_freiburg2_large_no_loop"
# "rgbd_dataset_freiburg2_large_with_loop"
# "/scratch/sarthak/ground_floor_dataset"
# "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/instance_nav_trajectories/episode_0"
# "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/instance_nav_trajectories/episode_7"
# "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/instance_nav_trajectories/episode_10"
# "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/instance_nav_trajectories/episode_14"
# "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/instance_nav_trajectories/episode_15"

# rgbd_dataset_freiburg1_desk
# rgbd_dataset_freiburg1_desk2
# rgbd_dataset_freiburg1_360    #TODO
# rgbd_dataset_freiburg1_room
rgbd_dataset_freiburg2_desk
rgbd_dataset_freiburg3_long_office_household

)

GENDATA='/scratch/aneesh.chavan/gen_data/'

for i in ${datasets[@]};
do
echo "Processing " $i
python tum_gen_dataset_trial.py --dump-dir="/scratch/aneesh.chavan/gen_data/$i" \
                                --data-path="/scratch/instance-loc/tum_datasets/$i/synced_data" \
                                -t=$i
done;