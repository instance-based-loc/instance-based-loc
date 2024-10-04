import torch 
import torch.nn as nn 

rgb_pretrained_path = "./logs/Experiment9a_rgbonly/120.pth"
depth_pretrained_path = "./logs/Experiment9a_depthonly/"

def load_param(trained_path):
    param_dict = torch.load(trained_path)
    for i in param_dict:
        print(i)
        # self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
    print('Loading pretrained model from {}'.format(trained_path))

load_param(rgb_pretrained_path)