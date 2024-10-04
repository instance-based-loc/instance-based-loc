import torch
import torch.nn as nn
import torch.nn.functional as F

import model
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import (
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID,
    deit_small_patch16_224_TransReID,
)
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


def bilinear_interpolation(f, x1, x2, y1, y2, x, y):
    x1 = x1.type(torch.float32) 
    x2 = x2.type(torch.float32) 
    y1 = y1.type(torch.float32) 
    y2 = y2.type(torch.float32) 
    x = x.type(torch.float32) 
    y = y.type(torch.float32) 
    h = y2 - y1
    w = x2 - x1
    print(f"torch.min(h) = {torch.min(h)}")
    print(f"torch.min(w) = {torch.min(w)}")
    assert torch.all(h >= 0) and torch.all(w >= 0)
    coeff = 1 / (h * w)
    coeff[(h == 0) | (w == 0)] = 2
    print(f"torch.max(coeff) = {torch.max(coeff)}")
    print(f"torch.min(coeff) = {torch.min(coeff)}")
    x = torch.cat(((x2 - x).unsqueeze(-1), (x - x1).unsqueeze(-1)), -1)
    y = torch.cat(((y2 - y).unsqueeze(-1), (y - y1).unsqueeze(-1)), -1)
    x[w == 0] = torch.tensor([1.0, 1.0]).to(x.device)
    y[h == 0] = torch.tensor([1.0, 1.0]).to(y.device)
    x = x.unsqueeze(-2).repeat(1, 1, 1, f.size(-1), 1).unsqueeze(-1)
    M = f.transpose(-1, -2).reshape(*f.shape[:-2], f.shape[-1], 2, 2)
    y = y.unsqueeze(-2).repeat(1, 1, 1, f.size(-1), 1).unsqueeze(-1)
    assert not torch.any(coeff.isnan())
    assert not torch.any(M.isnan())
    assert not torch.any(x.isnan())
    assert not torch.any(y.isnan())
    res = coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x.transpose(-1, -2) @ M @ y
    assert not torch.any(res.isnan())
    return res


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat(
        [features[:, begin - 1 + shift :], features[:, begin : begin - 1 + shift]],
        dim=1,
    )
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
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
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

        if model_name == "resnet50":
            self.in_planes = 2048
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3]
            )
            print("using resnet50 as a backbone")
        else:
            print("unsupported backbone! but got {}".format(model_name))

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......from {}".format(model_path))

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
        global_feat = global_feat.view(
            global_feat.shape[0], -1
        )  # flatten to (bs, 2048)

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == "after":
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


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

        print(
            "using Transformer_type: {} as a backbone".format(
                cfg.MODEL.TRANSFORMER_TYPE
            )
        )

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
        )
        if cfg.MODEL.TRANSFORMER_TYPE == "deit_small_patch16_224_TransReID":
            self.in_planes = 384
        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......from {}".format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == "arcface":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = Arcface(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "cosface":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = Cosface(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "amsoftmax":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = AMSoftmax(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "circle":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = CircleLoss(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ("arcface", "cosface", "amsoftmax", "circle"):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == "after":
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


class FourDNet(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super().__init__()
        print(
            f"<=========================== building FourDNet ============================>"
        )
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print(
            "using Transformer_type: {} as a backbone".format(
                cfg.MODEL.TRANSFORMER_TYPE
            )
        )

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
        )

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......from {}".format(model_path))

        # reduce dimensionality of ViT features
        self.reduced_dim = 128
        self.reduce_dims = nn.Linear(self.in_planes, self.reduced_dim)
        self.reduce_dims_global = nn.Linear(self.in_planes, self.reduced_dim)

        # project depth input to higher dimension
        self.project_depth = nn.Sequential(
            nn.Conv2d(1, self.reduced_dim // 2, 3, 2, 1),
            nn.Conv2d(self.reduced_dim // 2, self.reduced_dim, 3, 2, 1),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, 2, 1),
        )
        self.global_depth_feat = nn.Parameter(torch.randn(self.reduced_dim))

        # R2D cross attention
        self.r2d_k = 3
        self.r2d_m = 8
        self.r2d_Q = nn.Linear(2 * self.reduced_dim, self.reduced_dim)
        self.merge_local_global_depth = nn.Linear(2 * self.reduced_dim, self.reduced_dim)
        self.r2d_V = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.r2d_selector = nn.Sequential(
            # + 1 because I the base location is compulsorily selected
            nn.Linear(self.reduced_dim, 3 * self.r2d_m * self.r2d_k),
            nn.Sigmoid(),
        )

        """D2D self attention"""
        self.d2d_k = 3
        self.d2d_m = 8
        self.d2d_Q = nn.Linear(self.reduced_dim, self.reduced_dim)
        # self.d2d_Q = nn.Sequential(
        #     nn.Conv2d(2 * self.reduced_dim, self.reduced_dim, 3, 2, 1),
        #     nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, 2, 1)
        # )
        self.d2d_V = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.d2d_selector = nn.Sequential(
            nn.Linear(self.reduced_dim, 3 * self.r2d_m * self.r2d_k), nn.Sigmoid()
        )
        self.d2d_ffn = nn.Linear(self.reduced_dim, self.reduced_dim)
        
        self.device = "cuda:0"

        # classification head
        self.classifier = nn.Linear(self.reduced_dim, num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)



    def forward(self, rgb, d, label=None, cam_label=None, view_label=None):
        B = rgb.size(0)
        features = self.base(rgb, cam_label=0, view_label=0)

        N = features.size(1) - 1

        # global rgb features
        global_feat = features[:, 0]
        global_feat = self.reduce_dims_global(global_feat)

        # local rgb features
        local_feat = features[:, 1:]
        local_feat = self.reduce_dims(local_feat)

        # concatenating local and global rgb features 
        local_cat_global = torch.cat(
            (global_feat.unsqueeze(1).repeat(1, N, 1), local_feat), -1
        )
        assert local_cat_global.shape == (B, N, 2 * self.reduced_dim)

        # local depth features
        local_depth_feat = (
            self.project_depth(d).permute(0, 2, 3, 1)
        )
        Hd, Wd = local_depth_feat.shape[-3], local_depth_feat.shape[-2]
        local_depth_feat = local_depth_feat.reshape(B, -1, self.reduced_dim)
        assert local_depth_feat.shape == (B, Hd * Wd, self.reduced_dim)
        
        # concatenating local and global depth features
        local_cat_global_depth = torch.cat(
            (
                self.global_depth_feat.unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, local_depth_feat.size(-2), 1),
                local_depth_feat,
            ),
            -1,
        )
        assert local_cat_global_depth.shape == (
            B,
            Hd * Wd,
            # local_depth_feat.size(-2),
            2 * self.reduced_dim,
        )
        local_cat_global_depth = self.merge_local_global_depth(local_cat_global_depth)

        # # D2D self attention
        # # q, v = (
        # #     self.d2d_Q(local_cat_global_depth.reshape(B, Hd, Wd, 2 * self.reduced_dim).permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
        # #     self.d2d_V(local_cat_global_depth),
        # # )

        # print(f"local_cat_global_depth.shape = {local_cat_global_depth.shape}")
        # # q = self.d2d_Q(local_cat_global_depth)
        # # v = self.d2d_V(local_cat_global_depth)
        # # q = torch.rand(local_cat_global_depth.shape).to(0)
        # # v = torch.rand(local_cat_global_depth.shape).to(0)
        # q = local_cat_global_depth
        # v = local_cat_global_depth
        # # Hdq, Wdq = q.shape[-3], q.shape[-2]
        # Hdq, Wdq = Hd, Wd
        # # print(f"Hd = {Hd}, Wd = {Wd}")
        # # print(f"Hdq = {Hdq}, Wdq = {Wdq}")
        # assert q.shape == (B, Hdq * Wdq, self.reduced_dim)
        # q = q.reshape(B, Hdq * Wdq, self.reduced_dim)
        # selector_outputs = self.d2d_selector(q)
        # locations_x = selector_outputs[:, :, 0 : 2 * self.d2d_m * self.r2d_k : 2]
        # locations_y = selector_outputs[:, :, 1 : 2 * self.d2d_m * self.r2d_k : 2]
        # attention_scores = selector_outputs[:, :, 2 * self.d2d_m * self.r2d_k :]
        # assert locations_x.shape == (
        #     B,
        #     Hdq * Wdq,
        #     self.d2d_m * self.d2d_k,
        # )
        # assert locations_x.shape == locations_y.shape
        # assert locations_x.shape == attention_scores.shape
        # attention_scores = F.softmax(attention_scores, -1)

        # # find out the nearest positions for each selected position 
        # # x1, x2, y1, y2
        # nearest_pos = torch.zeros((B, Hdq * Wdq, self.d2d_m * self.d2d_k, 4))
        # stride_x = 1.0 / (Wd - 1)
        # stride_y = 1.0 / (Hd - 1)
        # nearest_pos[..., 0] = locations_x // stride_x
        # nearest_pos[..., 1] = torch.minimum(nearest_pos[..., 0] + 1, torch.tensor(Wd - 1))
        # nearest_pos[..., 2] = locations_y // stride_y
        # nearest_pos[..., 3] = torch.minimum(nearest_pos[..., 2] + 1, torch.tensor(Hd - 1))
        
        # nearest_pos = nearest_pos.type(torch.int32)
        # # print(f"locations_x[0, 0, 0] = {locations_x[0, 0, 0]}")
        # # print(f"locations_y[0, 0, 0] = {locations_y[0, 0, 0]}")
        # # print(f"nearest_pos[0, 0, 0, 0] = {nearest_pos[0, 0, 0, 0]}")
        # # print(f"nearest_pos[0, 0, 0, 1] = {nearest_pos[0, 0, 0, 1]}")
        # # print(f"nearest_pos[0, 0, 0, 2] = {nearest_pos[0, 0, 0, 2]}")
        # # print(f"nearest_pos[0, 0, 0, 3] = {nearest_pos[0, 0, 0, 3]}")

        # # finding features corresponding to these nearest locations
        # # print(f"v.shape = {v.shape}")
        # nearest_feat = torch.zeros(
        #     (B, Hdq * Wdq, self.d2d_m * self.d2d_k, 4, v.shape[-1])
        # )
        # # print(f"nearest_pos.shape = {nearest_pos.shape}")
        # # print(f"nearest_feat.shape = {nearest_feat.shape}")
        # for batch_idx in range(B):
        #     nearest_feat[batch_idx, :, :, 0] = v[batch_idx, nearest_pos[batch_idx, :, :, 0] * Wd + nearest_pos[batch_idx, :, :, 3]]
        #     nearest_feat[batch_idx, :, :, 1] = v[batch_idx, nearest_pos[batch_idx, :, :, 1] * Wd + nearest_pos[batch_idx, :, :, 3]]
        #     nearest_feat[batch_idx, :, :, 2] = v[batch_idx, nearest_pos[batch_idx, :, :, 1] * Wd + nearest_pos[batch_idx, :, :, 2]]
        #     nearest_feat[batch_idx, :, :, 3] = v[batch_idx, nearest_pos[batch_idx, :, :, 0] * Wd + nearest_pos[batch_idx, :, :, 2]]
        
        # nearest_pos = nearest_pos.to(self.device)
        # locations_x = locations_x.to(self.device)
        # locations_y = locations_y.to(self.device)
        # nearest_feat = nearest_feat.to(self.device)
        # # print(f"nearest_feat.shape = {nearest_feat.shape}")
        # assert torch.all(nearest_pos[..., 1] >= nearest_pos[..., 0])
        # assert torch.all(nearest_pos[..., 3] >= nearest_pos[..., 2])
        # interpolated_feat = bilinear_interpolation(nearest_feat, nearest_pos[..., 0], nearest_pos[..., 1], nearest_pos[..., 2], nearest_pos[..., 3], locations_x * Wd, locations_y * Hd).squeeze(-1).squeeze(-1)
        # # print(f"interpolated_feat.shape = {interpolated_feat.shape}")
        # # print(f"interpolated_feat.shape = {interpolated_feat.shape}")
        # # print(f"attention_scores.shape = {attention_scores.shape}")
        
        # d2d_feat = torch.mean(interpolated_feat * attention_scores.unsqueeze(-1), dim=-2)
        # # print(f"d2d_feat.shape = {d2d_feat.shape}")
        # local_cat_global_depth += d2d_feat

        """R2D cross attention"""
        q, v = (
            self.r2d_Q(local_cat_global),
            self.r2d_V(local_cat_global_depth),
        )

        selector_outputs = self.r2d_selector(q)
        locations_x = selector_outputs[:, :, 0 : 2 * self.r2d_m * self.r2d_k : 2]
        locations_y = selector_outputs[:, :, 1 : 2 * self.r2d_m * self.r2d_k : 2]
        attention_scores = selector_outputs[:, :, 2 * self.r2d_m * self.r2d_k :]
        assert locations_x.shape == (B, N, self.r2d_m * self.r2d_k)
        assert locations_x.shape == locations_y.shape
        assert locations_x.shape[-1] == attention_scores.shape[-1]
        attention_scores = F.softmax(attention_scores, -1)

        # find out nearest positions for each selected position
        # x1, x2, y1, y2
        nearest_pos = torch.zeros((B, N, self.r2d_m * self.r2d_k, 4))
        stride_x = 1.0 / (Wd - 1)
        stride_y = 1.0 / (Hd - 1)
        nearest_pos[..., 0] = locations_x // stride_x
        assert torch.all(nearest_pos[..., 0] <= Wd - 1)
        
        nearest_pos[..., 1] = torch.minimum(nearest_pos[..., 0] + 1, torch.tensor(Wd - 1))
        assert torch.all(nearest_pos[..., 1] >= nearest_pos[..., 0])
        nearest_pos[..., 2] = locations_y // stride_y
        nearest_pos[..., 3] = torch.minimum(nearest_pos[..., 2] + 1, torch.tensor(Hd - 1))
        nearest_pos = nearest_pos.type(torch.int32)
        print(f"locations_x[0, 0, 0] = {locations_x[0, 0, 0]}")
        print(f"locations_y[0, 0, 0] = {locations_y[0, 0, 0]}")
        print(f"nearest_pos[0, 0, 0, 0] = {nearest_pos[0, 0, 0, 0]}")
        print(f"nearest_pos[0, 0, 0, 1] = {nearest_pos[0, 0, 0, 1]}")
        print(f"nearest_pos[0, 0, 0, 2] = {nearest_pos[0, 0, 0, 2]}")
        print(f"nearest_pos[0, 0, 0, 3] = {nearest_pos[0, 0, 0, 3]}")

        # finding features corresponding to these nearest locations
        # print(f"v.shape = {v.shape}")
        nearest_feat = torch.zeros(
            (B, N, self.r2d_m * self.r2d_k, 4, v.shape[-1])
        )
        # print(f"nearest_pos.shape = {nearest_pos.shape}")
        # print(f"nearest_feat.shape = {nearest_feat.shape}")
        for batch_idx in range(B):
            nearest_feat[batch_idx, :, :, 0] = v[batch_idx, nearest_pos[batch_idx, :, :, 0] * Wd + nearest_pos[batch_idx, :, :, 3]]
            nearest_feat[batch_idx, :, :, 1] = v[batch_idx, nearest_pos[batch_idx, :, :, 1] * Wd + nearest_pos[batch_idx, :, :, 3]]
            nearest_feat[batch_idx, :, :, 2] = v[batch_idx, nearest_pos[batch_idx, :, :, 1] * Wd + nearest_pos[batch_idx, :, :, 2]]
            nearest_feat[batch_idx, :, :, 3] = v[batch_idx, nearest_pos[batch_idx, :, :, 0] * Wd + nearest_pos[batch_idx, :, :, 2]]
        
        nearest_pos = nearest_pos.to(self.device)
        locations_x = locations_x.to(self.device)
        locations_y = locations_y.to(self.device)
        nearest_feat = nearest_feat.to(self.device)
        assert torch.all(nearest_pos[..., 1] >= nearest_pos[..., 0])
        assert torch.all(nearest_pos[..., 3] >= nearest_pos[..., 2])
        interpolated_feat = bilinear_interpolation(nearest_feat, nearest_pos[..., 0], nearest_pos[..., 1], nearest_pos[..., 2], nearest_pos[..., 3], locations_x * Wd, locations_y * Hd).squeeze(-1).squeeze(-1)
        # print(f"interpolated_feat.shape = {interpolated_feat.shape}")
        # print(f"attention_scores.shape = {attention_scores.shape}")

        assert not torch.any(attention_scores.isnan())
        assert not torch.any(interpolated_feat.isnan())
        r2d_feat = torch.mean(interpolated_feat * attention_scores.unsqueeze(-1), dim=-2)
        print(f"r2d_feat.shape = {r2d_feat.shape}")
        """global average pooling to obtain final embedding"""
        final_embedding = torch.mean(r2d_feat, 1)
        cls_score = self.classifier(final_embedding)
        # return [cls_score], [final_embedding]
        return cls_score, final_embedding

        
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print(
            "using Transformer_type: {} as a backbone".format(
                cfg.MODEL.TRANSFORMER_TYPE
            )
        )

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
        )

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......from {}".format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == "arcface":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = Arcface(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "cosface":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = Cosface(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "amsoftmax":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = AMSoftmax(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
        elif self.ID_LOSS_TYPE == "circle":
            print(
                "using {} with s:{}, m: {}".format(
                    self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN
                )
            )
            self.classifier = CircleLoss(
                self.in_planes,
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE,
                m=cfg.SOLVER.COSINE_MARGIN,
            )
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
        print("using shuffle_groups size:{}".format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print("using shift_num size:{}".format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print("using divide_length size:{}".format(self.divide_length))
        self.rearrange = rearrange

    def forward(
        self, x, label=None, cam_label=None, view_label=None
    ):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
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
        b2_local_feat = x[:, patch_length : patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2 : patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3 : patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ("arcface", "cosface", "amsoftmax", "circle"):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], [
                global_feat,
                local_feat_1,
                local_feat_2,
                local_feat_3,
                local_feat_4,
            ]  # global feature for triplet loss
        else:
            if self.neck_feat == "after":
                return torch.cat(
                    [
                        feat,
                        local_feat_1_bn / 4,
                        local_feat_2_bn / 4,
                        local_feat_3_bn / 4,
                        local_feat_4_bn / 4,
                    ],
                    dim=1,
                )
            else:
                return torch.cat(
                    [
                        global_feat,
                        local_feat_1 / 4,
                        local_feat_2 / 4,
                        local_feat_3 / 4,
                        local_feat_4 / 4,
                    ],
                    dim=1,
                )

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


__factory_T_type = {
    "vit_base_patch16_224_TransReID": vit_base_patch16_224_TransReID,
    "deit_base_patch16_224_TransReID": vit_base_patch16_224_TransReID,
    "vit_small_patch16_224_TransReID": vit_small_patch16_224_TransReID,
    "deit_small_patch16_224_TransReID": deit_small_patch16_224_TransReID,
}


def make_model(cfg, num_class, camera_num, view_num):
    # if cfg.MODEL.NAME == "transformer":
    #     if cfg.MODEL.JPM:
    #         model = build_transformer_local(
    #             num_class,
    #             camera_num,
    #             view_num,
    #             cfg,
    #             __factory_T_type,
    #             rearrange=cfg.MODEL.RE_ARRANGE,
    #         )
    #         print("===========building transformer with JPM module ===========")
    #     else:
    #         model = build_transformer(
    #             num_class, camera_num, view_num, cfg, __factory_T_type
    #         )
    #         print("===========building transformer===========")
    # else:
    #     model = Backbone(num_class, cfg)
    #     print("===========building ResNet===========")
    # return model
    model = FourDNet(
        num_class,
        camera_num,
        view_num,
        cfg,
        __factory_T_type,
        rearrange=cfg.MODEL.RE_ARRANGE,
    )

    # model = build_transformer_local(
    #     num_class,
    #     camera_num,
    #     view_num,
    #     cfg,
    #     __factory_T_type,
    #     rearrange=cfg.MODEL.RE_ARRANGE,
    # )

    return model
