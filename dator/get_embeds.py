import os
import torch
from config import cfg
import argparse
from datasets.make_dataloader_depth import make_dataloader_depth 
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np
import cv2
import os
import shutil
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import sys 

import copy 

GPU0 = 0 
GPU1 = 0 
TARGET_GPU = 0 

MAX_DEPTH = 50 
MIN_DEPTH = 0 

#
#
if __name__ == "__main__":
        # parser = argparse.ArgumentParser(description="ReID Baseline Training")
        # parser.add_argument(
        #     "--config_file", default="", help="path to config file", type=str
        # )
        # parser.add_argument("opts", help="Modify config options using the command-line", default=None,
        #                     nargs=argparse.REMAINDER)
        #
        # args = parser.parse_args()
    
        # if args.config_file != "":
        #     cfg.merge_from_file(args.config_file)
         #cfg.merge_from_list(args.opts)
    cfg.merge_from_file("config.yml")
    cfg.MODEL.DEVICE_ID = "0"
    # cfg.TEST.WEIGHT = "/ssd_scratch/cvit/vaibhav/ckpts/DATOR_new_attempt2_tum/240.pth" 
    cfg.TEST.WEIGHT = "/ssd_scratch/cvit/vaibhav/ckpts/DATOR_new_attempt1_rrc/240.pth" 
    # cfg.TEST.WEIGHT = "dator_best_tum.pth" 
    cfg.freeze()
    print(cfg)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
        # logger = setup_logger("transreid", output_dir, if_train=False)
        # logger.info(args)
    
        # if args.config_file != "":
        #     logger.info("Loaded configuration file {}".format(args.config_file))
        #     with open(args.config_file, 'r') as cf:
        #         config_str = "\n" + cf.read()
        #         logger.info(config_str)
        # logger.info("Running with config:\n{}".format(cfg))
    
        # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    
        # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader_depth(cfg)
    
    # model = make_model(cfg, num_class=241, camera_num=1, view_num=1, )
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, gpu0 = GPU0, gpu1 = GPU1, target_gpu = TARGET_GPU) 
    model.load_param(trained_path=cfg.TEST.WEIGHT, load_classifier=False)  

    val_transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    
        # img = cv2.imread(f"sample.png")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_t = val_transforms(img)
        # model.eval()
        # with torch.no_grad():
        #     img_t = img_t.unsqueeze(0)
        #     output = model(img_t)
        #     print(output.shape)
    
    test_path = "/ssd_scratch/cvit/vaibhav/rrc/train/"
    test_classes = []
    # for coarse in os.listdir(test_path):
    #     for fine in os.listdir(osp.join(test_path, coarse)):
    #         test_classes.append(osp.join(coarse, fine))
    for coarse in os.listdir(test_path): 
        test_classes.append(osp.join(test_path, coarse)) 

    test_images = [os.listdir(os.path.join(test_path, c)) for c in test_classes]
    test_images = [[img_name for img_name in subgroup if img_name.find("depth") == -1] for subgroup in test_images] 
    # print(f"{test_images = }")
    
    test_images_depth = copy.deepcopy(test_images) 

    for i in range(len(test_classes)):
        for j in range(len(test_images[i])):
            test_images[i][j] = os.path.join(
                test_path, test_classes[i], test_images[i][j]
            )

    assert len(test_classes) == len(test_images) 
    for i in range(len(test_classes)):
        for j in range(len(test_images[i])):
            rgb_path = test_images[i][j] 
            assert osp.exists(rgb_path) 
            depth_path = test_images[i][j].replace("rgb", "depth") 
            test_images[i][j] = cv2.imread(rgb_path)
            test_images[i][j] = cv2.cvtColor(test_images[i][j], cv2.COLOR_BGR2RGB) 
            assert os.path.exists(depth_path) 
            assert os.path.exists(rgb_path)  

            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) 
            depth = cv2.resize(depth, (128, 256)) 
            depth = np.repeat(depth[None, :, :], 3, axis=0) 
            depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH) 
            depth = (depth - MIN_DEPTH)/ (MAX_DEPTH - MIN_DEPTH) 
            depth = depth - 0.5 
            depth = depth / 0.5 
            depth = torch.tensor(depth) 

            test_images_depth[i][j] = depth  

            assert type(test_images[i][j]) != str 
    
    total_count = 0 
    for i in range(len(test_classes)): 
        for j in range(len(test_images[i])): 
            assert type(test_images[i][j]) != str 
            total_count += 1 
    #
    model.eval()
    w = []
    with torch.no_grad():
        with tqdm(total=total_count) as bar:
            for row_idx, row in enumerate(test_images):
                r = []
                for rgb_idx, rgb in enumerate(row): 
                    assert type(rgb) != str 
                    depth = test_images_depth[row_idx][rgb_idx] 
                    im = val_transforms(rgb) 
                    with torch.no_grad():
                        k = model(im.unsqueeze(0), depth.unsqueeze(0)) 
                    r.append(k)
                    bar.update(1)
                    w.append(k)  
                # w.append(torch.stack(r))
                
    w = torch.stack(w)
    w = w.reshape((-1, w.shape[-1]))

    scores = (
        (F.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1))
        .detach()
        .cpu()
        .numpy()
    )
    # with open(f"w.pkl", "wb") as f:
    #     pickle.dump(w, f)

    # with open(f"w.pkl", "rb") as f:
    #     w = pickle.load(f)

    # scores = (
    #     (F.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1))
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )


    # scores = torch.zeros((w.shape[0], w.shape[0])).cpu().numpy()
    # for i in range(scores.shape[0]):
    #     for j in range(scores.shape[1]):
    #         scores[i][j] = w[i] @ w[j] / (torch.norm(w[i]) * torch.norm(w[j])) 

    # scores = scores * scores * scores * scores * scores   


    line_locs = [0] 
    for i in range(len(test_images)): 
        line_locs.append(len(test_images[i]) + line_locs[-1])  

    # print(f"{line_locs = }") 
    # print(f"{scores.shape = }") 

    # for line_loc in line_locs: 
    #     for i in range(scores.shape[0]): 
    #         for j in range(scores.shape[0]): 
    #             if i == line_loc or j == line_loc: 
    #                 scores[i][j] = 0 

    plt.figure(figsize=(15, 15))
    for line_loc in line_locs: 
        plt.axvline(x=line_loc, color="blue", linestyle="-")  
        plt.axhline(y=line_loc, color="blue", linestyle="-")   

    plt.imshow(scores, cmap="hot")
    plt.colorbar()


    # Show the heatmap
    plt.title("DATOR")
    plt.savefig("heatmap.jpg")