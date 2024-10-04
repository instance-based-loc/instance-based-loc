#!/bin/bash

# the config file is currently set to procthor_depth
python3 train.py --config_file configs/Market/vit_jpm.yml MODEL.DEVICE_ID "('0')"  
