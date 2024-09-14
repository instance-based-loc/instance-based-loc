#!/bin/bash

# Find all folders starting with 'rgbd_dataset_freiburg' in the specified directory
datasets=($(ls -d /scratch/sarthak/rgbd_dataset_freiburg*/))

# Process all datasets for cuda:0
for (( i=0; i<${#datasets[@]}; i++ ))
do
    data_path="${datasets[$i]}"
    dataset=$(basename ${data_path})
    
    # Add the 'clip' suffix to the dataset name
    dataset_clip="${dataset}_clip"

    map_pcd_cache_path="./cache/${dataset_clip}.pcd"
    memory_load_path="./out/${dataset_clip}.pt"
    log_file="./logs/log_${dataset_clip}.txt"

    echo "Processing ${dataset_clip} on cuda:0..."
    CUDA_VISIBLE_DEVICES=0 python tum_localisation_trial.py -t ${dataset_clip} --data-path ${data_path}synced_data/ --map-pcd-cache-path ${map_pcd_cache_path} --memory-load-path ${memory_load_path} --embeddings clip > ${log_file}
done

echo "All datasets processed on cuda:0."
