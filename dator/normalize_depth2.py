import os 
import shutil 
import numpy as np 
import cv2 

ROOT_DIR = "data/procthor_depth/train"
NUM_INST = 8
MAX_DEPTH = 8.0
if not os.path.exists(f"vis_depth"):
    os.mkdir(f"vis_depth")

all_data = np.empty((len(os.listdir(ROOT_DIR)) * NUM_INST, 256, 128))
count = 0
for ctg in os.listdir(ROOT_DIR):
    for idx in range(NUM_INST):
        depth_img = cv2.imread(os.path.join(ROOT_DIR, ctg, f"{idx}_d.jpg"), cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.resize(depth_img, (128, 256))
        depth_img = np.clip(depth_img, 0.0, MAX_DEPTH)
        depth_img = depth_img / MAX_DEPTH 
        # img = cv2.imread(os.path.join(ROOT_DIR, ctg, f"{idx}.jpg")) 
        all_data[count] = depth_img 
        count += 1

print(f"mean = {np.mean(all_data)}") 
print(f"std = {np.std(all_data)}")

mean = np.mean(all_data)
std = np.std(all_data)
all_data = all_data - mean 
# all_data = all_data / std
print(f"max = {np.max(all_data)}")
print(f"min = {np.min(all_data)}")