#!/bin/bash

# Find all folders starting with 'rgbd_dataset_freiburg' in the specified directory
datasets=($(ls -d /scratch/instance-loc/tum_datasets/rgbd_dataset_freiburg*/))

# Process all datasets for cuda:1
for (( i=${#datasets[@]}-1; i>=4; i-- ))
do
    data_path="${datasets[$i]}"
    dataset=$(basename ${data_path})

    # Add the 'dator' suffix to the dataset name
    dataset_dator="${dataset}_dino"

    map_pcd_cache_path="./cache/${dataset_dator}.pcd"
    memory_load_path="./out/${dataset_dator}.pt"
    log_file="logs/log_${dataset_dator}.txt"

    echo "Processing ${dataset_dator} on cuda:1..."
    
    # Running the python script for DATOR embedding extraction
    CUDA_VISIBLE_DEVICES=1 python tum_localisation_trial.py -t ${dataset_dator} --data-path ${data_path}synced_data/ --map-pcd-cache-path ${map_pcd_cache_path} --memory-load-path ${memory_load_path} --embeddings dino | tee ${log_file}
done

echo "All datasets processed on cuda:1."
