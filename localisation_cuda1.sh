#!/bin/bash

# Find all folders starting with 'rgbd_dataset_freiburg' in the specified directory
datasets=($(ls -d /scratch/sarthak/rgbd_dataset_freiburg*/))

# Process all datasets for cuda:1
for (( i=2; i<${#datasets[@]}; i++ ))
do
    data_path="${datasets[$i]}"
    dataset=$(basename ${data_path})
    
    # Add the 'dino' suffix to the dataset name
    dataset_dino="${dataset}_dino"

    map_pcd_cache_path="./cache/${dataset_dino}.pcd"
    memory_load_path="./out/${dataset_dino}.pt"
    log_file="logs/log_${dataset_dino}.txt"

    echo "Processing ${dataset_dino} on cuda:1..."
    CUDA_VISIBLE_DEVICES=1 python tum_localisation_trial.py -t ${dataset_dino} --data-path ${data_path}synced_data/ --map-pcd-cache-path ${map_pcd_cache_path} --memory-load-path ${memory_load_path} --embeddings dino > ${log_file}
done

echo "All datasets processed on cuda:1."
