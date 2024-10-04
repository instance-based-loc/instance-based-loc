import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import time
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np 
import cv2
import shutil 
import os.path as osp


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.features4 = None
        self.features20 = None
        # print(self.model)

        for name, module in self.model.named_modules():
            if name == "features.4" or name == "features.20":
                module.register_forward_hook(
                    # lambda module, input, output : print(f"features.shape = {output.shape}")
                    # lambda module, input, output : output 
                    self.hook_fn(module, name)
                )
        
    def hook_fn(self, module, name):
        def fn(_, __, output):
            if name == "features.4":
                self.features4 = output
            if name == "features.20":
                self.features20 = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self.features4, self.features20


class build_DepthNet3(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        print(f"<===================== building DepthNet3 =========================>")
        super().__init__()
        self.reduced_dim = 128
        self.vgg = VGGFeatures()
        self.device = "cuda:0"
        self.merge_local_global_feat = nn.Linear(128 + 128, self.reduced_dim)
        self.ffn_global = nn.Conv2d(512, 128, 3, 1, 1)
        self.classifier = nn.Linear(self.reduced_dim, num_classes)
        self.vis_count = 0
        self.max_vis = 100


    def forward(self, _, depth, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        B = depth.shape[0]
        depth = depth.float()
        features4, features20 = self.vgg(depth)  
        # features4.shape = (B, 128, 112, 112)
        features20 = F.interpolate(features20, (112, 112))
        global_feat = self.ffn_global(features20)
        local_cat_global_feat = torch.cat((
            global_feat,
            features4
        ), 1).permute(0, 2, 3, 1).reshape(B, 112 * 112, 128 + 128)
        x = self.merge_local_global_feat(local_cat_global_feat)
        x = torch.mean(x, -2)
        cls_score = self.classifier(x)
        if self.training:
            return cls_score, x
        else:
            return x


class build_DepthNet(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        print(f"<===================== building DepthNet =========================>")
        super().__init__()
        self.reduced_dim = 128
        self.vgg = VGGFeatures()
        self.device = "cuda:0"
        self.ffn = nn.Conv2d(512, self.reduced_dim, 7, 1, 0)
        self.classifier = nn.Linear(self.reduced_dim, num_classes)
        self.vis_count = 0
        self.max_vis = 100



class build_DepthNet2(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        print(f"<===================== building DepthNet2 =========================>")
        super().__init__()
        self.reduced_dim = 128
        self.vgg = VGGFeatures()
        self.device = "cuda:0"
        self.merge_local_global_feat = nn.Linear(128 + 512, self.reduced_dim)
        self.classifier = nn.Linear(self.reduced_dim, num_classes)
        self.vis_count = 0
        self.max_vis = 100
        self.ffn = nn.Conv2d(512, self.reduced_dim, 7, 1, 0)
        self.classifier = nn.Linear(self.reduced_dim, num_classes)


    def forward(self, _, depth, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        B = depth.shape[0]
        if self.training:
            x = depth.float()
        else:
            x = depth.float()
        features4, features20 = self.vgg(x)  
        global_feat = torch.mean(features20.reshape(B, 512, 49), -1)
        # print(f"features4.shape = {features4.shape}")
        local_cat_global_feat = torch.cat((
            features4.reshape(B, 128, 112 * 112).permute(0, 2, 1),
            global_feat.unsqueeze(1).repeat(1, 112 * 112, 1)
        ), -1)
        x = self.merge_local_global_feat(local_cat_global_feat)
        x = torch.mean(x, -2)
        # x = features20
        # x = self.ffn(x).squeeze(-1).squeeze(-1)
        cls_score = self.classifier(x)
        if self.training:
            return cls_score, x
        else:
            return x


class build_DepthNet(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        print(f"<===================== building DepthNet =========================>")
        super().__init__()
        self.reduced_dim = 128
        self.vgg = VGGFeatures()
        self.device = "cuda:0"
        self.ffn = nn.Conv2d(512, self.reduced_dim, 7, 1, 0)
        self.classifier = nn.Linear(self.reduced_dim, num_classes)
        self.vis_count = 0
        self.max_vis = 100


    def forward(self, _, depth, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        B = depth.shape[0]
        # x = torch.repeat(depth, 3, axis=1)
        # x = depth.half()
        # if self.vis_count < self.max_vis:
        #     if os.path.exists(f"vis"):
        #         shutil.rmtree(f"vis")
        #     os.mkdir(f"vis")
        #     for batch_idx in range(B):
        #         depth_img = depth[batch_idx][0]
        #         rgb_img = (rgb[batch_idx].permute(1, 2, 0) + 0.5) * 255.0
        #         rgb_img = torch.clamp(rgb_img, 0.0, 255.0).type(torch.int32) 
        #         fig, axs = plt.subplots(1, 2)
        #         axs[0].imshow(depth_img.cpu().numpy())
        #         axs[1].imshow(rgb_img.cpu().numpy())
        #         if not os.path.exists(f"vis/{label[batch_idx]}"):
        #             os.mkdir(f"vis/{label[batch_idx]}")
        #         plt.savefig(f"vis/{label[batch_idx]}/depth{self.vis_count}.jpg")
        #         plt.close()
        #         self.vis_count += 1
        if self.training:
            x = depth.half()
        else:
            x = depth.float()
        # print(f"it is actually this~!")
        # print(f"x.dtype = {x.dtype}") 
        # print(f"x.shape = {x.shape}")
        features4, features20 = self.vgg(x)  
        # print(f"vgg.shape = {x.shape}")
        x = features20
        x = self.ffn(x).squeeze(-1).squeeze(-1)
        cls_score = self.classifier(x)
        # print(f"cls_score.shape = {cls_score.shape}")
        # print(f"cls_score.dtype = {cls_score.dtype}")
        # print(f"final_embedding.shape = {x.shape}")
        if self.training:
            return cls_score, x
        else:
            return x


class build_SimpleDepthNet(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        print(f"<===================== building SimpleDepthNet =========================>")
        super().__init__()
        self.reduced_dim = 128
        self.project_depth = nn.Sequential(
            nn.Conv2d(1, self.reduced_dim // 2, 3, 2, 1),
            nn.Conv2d(self.reduced_dim // 2, self.reduced_dim, 3, 2, 1),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, 2, 1),
        )

        self.device = "cuda:0"
        self.classifier = nn.Linear(self.reduced_dim, num_classes)


    def forward(self, rgb, depth, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        B = depth.shape[0]
        x = depth.half()
        # x = torch.cat((rgb, depth), 1) 
        x = self.project_depth(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.reduced_dim)
        x = torch.mean(x, 1)
        cls_score = self.classifier(x)
        # print(f"cls_score.shape = {cls_score.shape}")
        # print(f"cls_score.dtype = {cls_score.dtype}")
        # print(f"final_embedding.shape = {x.shape}")
        return cls_score, x


class build_FourDNet(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, gpu0, gpu1, target_gpu, rearrange):
        print(f"<===================== building FourDNet =========================>")
        super().__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.gpu0 = int(gpu0)
        self.gpu1 = int(gpu1)
        self.target_gpu = int(target_gpu)


        # the pretrained modalities 
        self.rgb_pretrained_path = "rgb.pth"
        self.depth_pretrained_path = "depth.pth"
        

        print(f"using model parallel with GPU0 = {self.gpu0}, GPU1 = {self.gpu1} and TARGET_GPU = {self.target_gpu}")

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # configuring the SIE info
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0


        # defining the RGB backbone
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH).to(self.gpu0)

        # defining the depth backbone
        self.base2 = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH).to(self.gpu1)


        # if pretrain_choice == 'imagenet':
        #     self.base.load_param(model_path)
        #     self.base2.load_param(model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))
        

        # model's channel dimension
        self.reduced_dim = 128


        # project the RGB features to smaller dimension
        self.project_local_rgb = nn.Linear(self.in_planes, self.reduced_dim).to(self.gpu0)
        self.project_global_rgb = nn.Linear(self.in_planes, self.reduced_dim).to(self.gpu0)
        self.merge_local_global_rgb = nn.Linear(2 * self.reduced_dim, self.reduced_dim).to(self.gpu0)


        # project the depth features to smaller dimension
        self.project_local_depth = nn.Linear(self.in_planes, self.reduced_dim).to(self.gpu1)
        self.project_global_depth = nn.Linear(self.in_planes, self.reduced_dim).to(self.gpu1)
        self.merge_local_global_depth = nn.Linear(2 * self.reduced_dim, self.reduced_dim).to(self.gpu1)


        # defining the GPUs
        self.r2r_gpu = self.gpu0
        self.r2d_gpu = self.gpu0
        self.d2d_gpu = self.gpu1
        self.d2r_gpu = self.gpu1


        # query and value transformations
        self.Q_r = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.r2r_gpu)
        self.V_r = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.r2r_gpu)
        self.Q_d = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.d2d_gpu)
        self.V_d = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.d2d_gpu)


        # R2D cross attention
        self.r2d_k = 3
        self.r2d_m = 8
        self.r2d_selector = nn.Sequential(
            nn.Linear(self.reduced_dim, 2 * self.r2d_m * self.r2d_k),
            nn.Sigmoid(),
        ).to(self.r2d_gpu) 
        self.r2d_attn_weights = nn.Sequential(
            nn.Linear(self.reduced_dim, self.r2d_m * self.r2d_k),
            nn.Softmax(dim=-1)
        ).to(self.r2d_gpu)
        self.r2d_norm = nn.LayerNorm(self.reduced_dim).to(self.gpu0)
        self.r2d_ffn = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.r2d_gpu)


        # R2R self attention
        self.r2r_k = 3
        self.r2r_m = 8
        self.r2r_selector = nn.Sequential(
            nn.Linear(self.reduced_dim, 2 * self.r2r_m * self.r2r_k),
            nn.Sigmoid(),
        ).to(self.r2r_gpu) 
        self.r2r_attn_weights = nn.Sequential(
            nn.Linear(self.reduced_dim, self.r2r_m * self.r2r_k),
            nn.Softmax(dim=-1)
        ).to(self.r2r_gpu)
        self.r2r_norm = nn.LayerNorm(self.reduced_dim).to(self.gpu0)
        self.r2r_ffn = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.r2r_gpu)


        # D2R cross attention
        self.d2r_k = 3
        self.d2r_m = 8
        self.d2r_selector = nn.Sequential(
            nn.Linear(self.reduced_dim, 2 * self.d2r_m * self.d2r_k),
            nn.Sigmoid(),
        ).to(self.d2r_gpu) 
        self.d2r_attn_weights = nn.Sequential(
            nn.Linear(self.reduced_dim, self.d2r_m * self.d2r_k),
            nn.Softmax(dim=-1)
        ).to(self.d2r_gpu)
        self.d2r_norm = nn.LayerNorm(self.reduced_dim).to(self.gpu1)
        self.d2r_ffn = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.d2r_gpu)



        # D2D self attention
        self.d2d_k = 3
        self.d2d_m = 8
        self.d2d_selector = nn.Sequential(
            nn.Linear(self.reduced_dim, 2 * self.d2d_m * self.d2d_k),
            nn.Sigmoid(),
        ).to(self.d2d_gpu) 
        self.d2d_attn_weights = nn.Sequential(
            nn.Linear(self.reduced_dim, self.d2d_m * self.d2d_k),
            nn.Softmax(dim=-1)
        ).to(self.d2d_gpu)
        self.d2d_norm = nn.LayerNorm(self.reduced_dim).to(self.gpu1)
        self.d2d_ffn = nn.Linear(self.reduced_dim, self.reduced_dim).to(self.d2d_gpu)


        # the final classifier
        self.classifier = nn.Linear(self.reduced_dim, num_classes).to(self.target_gpu)


        # development stage
        self.visualize = True
        self.vis_count = 0
        self.max_vis = 25
        if self.visualize and osp.exists(f"vis"):
            shutil.rmtree(f"vis")
        self.dropout = False


        # hypernet
        self.hypernet_gpu = self.gpu0
        self.hypernet = nn.Sequential(
            nn.Conv2d(2 * self.reduced_dim, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 2, 3, 1, 1),
            nn.Softmax(dim=1)
        ).to(self.hypernet_gpu)


        # loading the pretrained modalities
        self.load_rgb_pretrained()
        self.load_depth_pretrained()


    def load_rgb_pretrained(self):
        param_dict = torch.load(self.rgb_pretrained_path)
        for i in param_dict:
            if i.find("base2") != -1:
                continue
            elif i.find("base") != -1:
                self.state_dict()[i].copy_(param_dict[i])
            elif i.find("rgb") != -1 or i.find("r2r") != -1 or i.find("Q_r") != -1 or i.find("V_r") != -1:
                self.state_dict()[i].copy_(param_dict[i])


    def load_depth_pretrained(self):
        param_dict = torch.load(self.depth_pretrained_path)
        for i in param_dict:
            if i.find("base2") != -1:
                self.state_dict()[i].copy_(param_dict[i])
            elif i.find("depth") != -1 or i.find("d2d") != -1 or i.find("Q_d") != -1 or i.find("V_d") != -1:
                self.state_dict()[i].copy_(param_dict[i])



    def forward(self, rgb, depth, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        rgb = rgb.float().to(self.gpu0)
        depth = depth.float().to(self.gpu1)


        B = rgb.shape[0]
        if self.dropout and self.training:
            with torch.no_grad():
                # random modality dropout
                p = torch.randint(0, 5, size=(B, ))
                rgb_dropout_ids = ((p == 0) | (p == 2)).to(self.gpu0)
                depth_dropout_ids = ((p == 1) | (p == 3)).to(self.gpu1)
                rgb[rgb_dropout_ids] = torch.zeros(rgb[0].shape).to(self.gpu0)
                depth[depth_dropout_ids] = torch.zeros(depth[0].shape).to(self.gpu1)


        # visualizing the inputs
        if self.visualize and self.vis_count < self.max_vis:
            for batch_idx in range(B):
                depth_img = depth[batch_idx][0]
                rgb_img = (rgb[batch_idx].permute(1, 2, 0) + 0.5) * 255.0
                rgb_img = torch.clamp(rgb_img, 0.0, 255.0).type(torch.int32) 
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(depth_img.cpu().numpy())
                axs[1].imshow(rgb_img.cpu().numpy())
                if not os.path.exists(f"vis/{label[batch_idx]}"):
                    os.mkdir(f"vis/{label[batch_idx]}")
                plt.savefig(f"vis/{label[batch_idx]}/depth{self.vis_count}.jpg")
                plt.close()
                self.vis_count += 1



        # extracting the RGB features
        rgb_features = self.base(rgb, cam_label=cam_label, view_label=view_label)
        N = rgb_features.shape[1] - 1


        # global rgb features
        global_rgb = rgb_features[:, 0]
        global_rgb = self.project_global_rgb(global_rgb)


        # local rgb features
        local_rgb = rgb_features[:, 1:]
        local_rgb = self.project_local_rgb(local_rgb)


        # concatenating local and global rgb features 
        local_cat_global_rgb = torch.cat((global_rgb.unsqueeze(1).repeat(1, N, 1), local_rgb), -1)
        local_cat_global_rgb = self.merge_local_global_rgb(local_cat_global_rgb)


        # extracting depth features
        depth_features = self.base2(depth.to(self.gpu1), cam_label=cam_label, view_label=view_label)
        N = depth_features.shape[1] - 1


        # global depth features
        global_depth = depth_features[:, 0]
        global_depth = self.project_global_depth(global_depth)


        # local depth features
        local_depth = depth_features[:, 1:]
        local_depth = self.project_local_depth(local_depth)


        # concatenating local and global rgb features 
        local_cat_global_depth = torch.cat((global_depth.unsqueeze(1).repeat(1, N, 1), local_depth), -1)
        local_cat_global_depth = self.merge_local_global_depth(local_cat_global_depth)


        # the hypernet features
        depth_feat_spatial = local_cat_global_depth.reshape(B, 16, 8, self.reduced_dim).permute(0, 3, 1, 2)
        rgb_feat_spatial = local_cat_global_rgb.reshape(B, 16, 8, self.reduced_dim).permute(0, 3, 1, 2)
        filters = self.hypernet(torch.cat((depth_feat_spatial.to(self.hypernet_gpu), rgb_feat_spatial.to(self.hypernet_gpu)), dim=1))
        assert filters.shape == (B, 2, 16, 8)
        rgb_filter = filters[:, 0, ...].to(self.gpu0)
        depth_filter = filters[:, 1, ...].to(self.gpu1)
        assert rgb_filter.shape == (B, 16, 8)
        assert depth_filter.shape == (B, 16, 8)


        # defining the queries and values for both RGB and depth
        q_r = self.Q_r(local_cat_global_rgb.to(self.r2r_gpu))
        v_r = self.V_r(local_cat_global_rgb.to(self.r2r_gpu))
        q_d = self.Q_d(local_cat_global_depth.to(self.d2d_gpu))
        v_d = self.V_d(local_cat_global_depth.to(self.d2d_gpu))


        """R2R Self Attention"""
        # selecting key positions and their attention weights
        selector_outputs = self.r2r_selector(q_r)
        attention_scores = self.r2r_attn_weights(q_r)
        locations_x = selector_outputs[:, :, 0 : self.r2r_m * self.r2r_k]
        locations_y = selector_outputs[:, :, self.r2r_m * self.r2r_k :] 


        # performing sampling of the value feature map at the given locations
        v = v_r.permute(0, 2, 1).reshape(B, self.reduced_dim, 16, 8)
        grid = torch.stack((locations_x, locations_y), -1)
        grid = grid * 2 - 1
        interpolated_feat = F.grid_sample(v, grid, align_corners=True).permute(0, 2, 3, 1)


        # performing weighted sum of values
        r2r_feat = torch.sum(interpolated_feat * attention_scores.unsqueeze(-1), dim=-2) 
        r2r_feat = self.r2r_ffn(r2r_feat)


        # adding back to the RGB path
        local_cat_global_rgb = local_cat_global_rgb + r2r_feat.to(self.gpu0)
        local_cat_global_rgb = self.r2r_norm(local_cat_global_rgb)


        # """D2D Self Attention"""
        # # selecting key positions and their attention weights
        # selector_outputs = self.d2d_selector(q_d)
        # attention_scores = self.d2d_attn_weights(q_d)
        # locations_x = selector_outputs[:, :, : self.d2d_m * self.d2d_k]
        # locations_y = selector_outputs[:, :, self.d2d_m * self.d2d_k :] 


        # # performing sampling of the value feature map at the given locations
        # v = v_d.permute(0, 2, 1).reshape(B, self.reduced_dim, 16, 8)
        # grid = torch.stack((locations_x, locations_y), -1)
        # grid = grid * 2 - 1
        # interpolated_feat = F.grid_sample(v, grid, align_corners=True).permute(0, 2, 3, 1)


        # # performing weighted sum of values
        # d2d_feat = torch.sum(interpolated_feat * attention_scores.unsqueeze(-1), dim=-2) 
        # d2d_feat = self.d2d_ffn(d2d_feat)


        # # adding back to the depth path
        # local_cat_global_depth = local_cat_global_depth + d2d_feat.to(self.gpu1)
        # local_cat_global_depth = self.d2d_norm(local_cat_global_depth)


        # """D2R Cross Attention"""
        # # selecting key positions and their attention weights
        # selector_outputs = self.d2r_selector(q_d.to(self.d2r_gpu))
        # attention_scores = self.d2r_attn_weights(q_d.to(self.d2r_gpu))
        # locations_x = selector_outputs[:, :, 0 : self.d2r_m * self.d2r_k]
        # locations_y = selector_outputs[:, :, self.d2r_m * self.d2r_k :] 


        # # performing sampling of the value feature map at the given locations
        # v = v_r.permute(0, 2, 1).reshape(B, self.reduced_dim, 16, 8)
        # grid = torch.stack((locations_x, locations_y), -1)
        # grid = grid * 2 - 1
        # interpolated_feat = F.grid_sample(v.to(self.d2r_gpu), grid, align_corners=True).permute(0, 2, 3, 1)


        # # performing weighted sum of values
        # d2r_feat = torch.sum(interpolated_feat * attention_scores.unsqueeze(-1), dim=-2) 
        # d2r_feat = self.d2r_ffn(d2r_feat)


        # # adding back to the depth path
        # local_cat_global_depth = local_cat_global_depth + d2r_feat.to(self.gpu1) * rgb_filter.reshape(B, 128).unsqueeze(-1).to(self.gpu1)
        # local_cat_global_depth = self.d2r_norm(local_cat_global_depth)


        # """R2D Cross Attention"""
        # # selecting key positions and their attention weights
        # selector_outputs = self.r2d_selector(q_r.to(self.r2d_gpu))
        # attention_scores = self.r2d_attn_weights(q_r.to(self.r2d_gpu))
        # locations_x = selector_outputs[:, :, 0 : self.r2d_m * self.r2d_k]
        # locations_y = selector_outputs[:, :, self.r2d_m * self.r2d_k :] 


        # # performing sampling of the value feature map at the given locations
        # v = v_d.permute(0, 2, 1).reshape(B, self.reduced_dim, 16, 8)
        # grid = torch.stack((locations_x, locations_y), -1)
        # grid = grid * 2 - 1
        # interpolated_feat = F.grid_sample(v.to(self.r2d_gpu), grid, align_corners=True).permute(0, 2, 3, 1)


        # # performing weighted sum of values
        # r2d_feat = torch.sum(interpolated_feat * attention_scores.unsqueeze(-1), dim=-2) 
        # r2d_feat = self.r2d_ffn(r2d_feat)


        # # adding back to the RGB path
        # local_cat_global_rgb = local_cat_global_rgb + r2d_feat.to(self.gpu0) * depth_filter.reshape(B, 128).unsqueeze(-1).to(self.gpu0)
        # local_cat_global_rgb = self.r2d_norm(local_cat_global_rgb)


        """Preparing final features to use for classification"""
        # # performing global average pooling
        # local_cat_global_rgb = torch.mean(local_cat_global_rgb, -2)
        # local_cat_global_depth = torch.mean(local_cat_global_depth, -2)


        # final_embedding = local_cat_global_depth.to(self.target_gpu) * depth_filter.reshape(B, 128).unsqueeze(-1).to(self.target_gpu) + local_cat_global_rgb.to(self.target_gpu) * rgb_filter.reshape(B, 128).unsqueeze(-1).to(self.target_gpu)
        final_embedding = local_cat_global_rgb.to(self.target_gpu) 
        # final_embedding = local_cat_global_depth.to(self.target_gpu) 
        final_embedding = torch.mean(final_embedding, dim=-2)

        # compute the cls scores and return
        cls_score = self.classifier(final_embedding)
        if self.training:
            return cls_score, final_embedding
        else:
            return final_embedding 



class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, rgb, depth, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        x = rgb

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num, gpu0, gpu1, target_gpu):
    # if cfg.MODEL.NAME == 'transformer':
    #     if cfg.MODEL.JPM:
    #         model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    #         print('===========building transformer with JPM module ===========')
    #     else:
    #         model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    #         print('===========building transformer===========')
    # else:
    #     model = Backbone(num_class, cfg)
    #     print('===========building ResNet===========')

    model = build_FourDNet(num_class, camera_num, view_num, cfg, __factory_T_type, gpu0, gpu1, target_gpu, rearrange=cfg.MODEL.RE_ARRANGE)
    # model = build_SimpleDepthNet(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    # model = build_DepthNet(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    # model = build_DepthNet2(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    # model = build_DepthNet3(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    # model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    return model
