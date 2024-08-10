#!/bin/bash

# Get the current user's username
current_user=$USER

# Create the directory with the username
mkdir -p /scratch/"$current_user"/

# Download the models using the username in the path
wget -O /scratch/"$current_user"/ram_swin_large_14m.pth https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
wget -O /scratch/"$current_user"/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
