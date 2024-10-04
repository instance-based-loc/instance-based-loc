import os 
import matplotlib.pyplot as plt 
import shutil 
import cv2

ROOT_DIR = "data/procthor_depth/test"
NUM_INST = 8
if not os.path.exists(f"vis_depth"):
    os.mkdir(f"vis_depth")

for ctg in os.listdir(ROOT_DIR):
    for idx in range(NUM_INST):
        depth_img = cv2.imread(os.path.join(ROOT_DIR, ctg, f"{idx}_d.jpg"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(ROOT_DIR, ctg, f"{idx}.jpg")) 
        print(f"depth_image.dtype = {depth_img.dtype}")
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(depth_img)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=depth_img.min(), vmax=depth_img.max()))
        sm._A = []  
        cbar = plt.colorbar(sm, ax=axs[1], label='Color Scale')  
        plt.savefig(f"vis_depth/{ctg}_{idx}.jpg")