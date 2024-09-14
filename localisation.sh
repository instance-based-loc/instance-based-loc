#!/bin/bash

# Find all folders starting with 'rgbd_dataset_freiburg2' in the specified directory
datasets=($(ls -d /scratch/sarthak/rgbd_dataset_freiburg2*/))

# Loop over each dataset and run the Python script
for data_path in "${datasets[@]}"
do
    # Extract the dataset name from the path
    dataset=$(basename ${data_path})
    
    map_pcd_cache_path="./cache/${dataset}.pcd"
    memory_load_path="./out/${dataset}.pt"
    log_file="log_${dataset}.txt"

    echo "Processing ${dataset}..."
    python tum_localisation_trial.py -t ${dataset} --data-path ${data_path}synced_data/ --map-pcd-cache-path ${map_pcd_cache_path} --memory-load-path ${memory_load_path} > ${log_file}
    echo "Finished processing ${dataset}. Output saved to ${log_file}"
done

echo "All datasets processed."
