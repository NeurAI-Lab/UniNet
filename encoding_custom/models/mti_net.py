#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    MTI-Net implementation based on HRNet backbone
    https://arxiv.org/pdf/2001.06902.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from encoding_custom.backbones.base import BaseNet
from encoding_custom.backbones.resnet import BasicBlock
from encoding_custom.nn.attention_modules import SEBlock
from encoding_custom.models.padnet import MultiTaskDistillationModule
from encoding_custom.models import *
from encoding_custom.necks.fpn import FPN
from encoding_custom.utils.model_store import ModelLoader


class InitialTaskPredictionModule(nn.Module):
    """ Module to make the initial task predictions """

    def __init__(self, num_outs, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict({task: nn.Sequential(
                BasicBlock(channels, channels), BasicBlock(channels, channels))
                for task in self.auxilary_tasks})

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(
                    input_channels, task_channels, 1, bias=False),
                    nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(
                    input_channels, task_channels, downsample=downsample),
                    BasicBlock(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict({task: nn.Conv2d(
            task_channels, num_outs[task], 1)
            for task in self.auxilary_tasks})

    def forward(self, features_curr_scale, features_prev_scale=None):
        # Concat features that were propagated from previous scale
        if features_prev_scale is not None:
            x = {t: torch.cat((features_curr_scale, F.interpolate(
                features_prev_scale[t], scale_factor=2, mode='bilinear')), 1)
                 for t in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' % t] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' % t])

        return out


class FPM(nn.Module):
    """ Feature Propagation Module """

    def __init__(self, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        # General
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(
            self.shared_channels, self.shared_channels // 4, 1, bias=False),
            nn.BatchNorm2d(self.shared_channels // 4))
        self.non_linear = nn.Sequential(
            BasicBlock(self.shared_channels, self.shared_channels // 4,
                       downsample=downsample),
            BasicBlock(self.shared_channels // 4, self.shared_channels // 4),
            nn.Conv2d(self.shared_channels // 4, self.shared_channels, 1))

        # Dimensionality reduction
        downsample = nn.Sequential(nn.Conv2d(
            self.shared_channels, self.per_task_channels, 1, bias=False),
            nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(
            self.shared_channels, self.per_task_channels,
            downsample=downsample)

        # SEBlock
        self.se = nn.ModuleDict({task: SEBlock(
            self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        # Get shared representation
        concat = torch.cat([x['features_%s' % task] for task in
                            self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        # Per task attention mask
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)
                           ).view(B, -1, H, W)

        # Perform dimensionality reduction
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' % task]

        return out


class MTINet(nn.Module):
    """
        MTI-Net implementation based on HRNet backbone
        https://arxiv.org/pdf/2001.06902.pdf
    """

    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(MTINet, self).__init__()

        self.tasks = tasks
        instance_tasks = []
        for task in tasks:
            if task in ['detect', 'inst_seg', 'inst_depth']:
                instance_tasks.append(task)
        for task in list(set(tasks) & set(instance_tasks)):
            self.tasks.remove(task)
        # aux tasks + segment + depth...
        self.auxilary_tasks = copy.deepcopy(tasks)
        for task in list(set(tasks) & set(cfg.MISC.AUX_TASKS)):
            self.tasks.remove(task)

        # Backbone
        self.base_net = BaseNet(
            backbone, pretrained=pretrained and not cfg.MODEL.LOAD_BACKBONE,
            norm_layer=norm_layer, dcn=cfg.MODEL.DCN_TYPE, **kwargs)
        # only the last stage channel number is required...
        backbone_channels = self.base_net.backbone.feat_channels
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels

        # FPN..
        if cfg.MODEL.NECK.NECK_TYPE == 'fpn':
            self.fpn = FPN(cfg, backbone_channels[1:], norm_layer,
                           cfg.MODEL.DET.FEATURE_CHANNELS)
            self.channels[1:] = [cfg.MODEL.DET.FEATURE_CHANNELS] * 3

        # Feature Propagation Module
        self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])

        num_outs = {'segment': cfg.NUM_CLASSES.SEGMENT, 'depth': 1,
                    'sem_cont': 1, 'sur_nor': 3}
        # Initial task predictions at multiple scales
        self.scale_0 = InitialTaskPredictionModule(
            num_outs, self.auxilary_tasks, self.channels[0] + self.channels[1],
            self.channels[0])
        self.scale_1 = InitialTaskPredictionModule(
            num_outs, self.auxilary_tasks, self.channels[1] + self.channels[2],
            self.channels[1])
        self.scale_2 = InitialTaskPredictionModule(
            num_outs, self.auxilary_tasks, self.channels[2] + self.channels[3],
            self.channels[2])
        self.scale_3 = InitialTaskPredictionModule(
            num_outs, self.auxilary_tasks, self.channels[3], self.channels[3])

        # Distillation at multiple scales
        self.distillation_scale_0 = MultiTaskDistillationModule(
            self.tasks, self.auxilary_tasks, self.channels[0])
        self.distillation_scale_1 = MultiTaskDistillationModule(
            self.tasks, self.auxilary_tasks, self.channels[1])
        self.distillation_scale_2 = MultiTaskDistillationModule(
            self.tasks, self.auxilary_tasks, self.channels[2])
        self.distillation_scale_3 = MultiTaskDistillationModule(
            self.tasks, self.auxilary_tasks, self.channels[3])

        # Feature aggregation through HRNet heads
        self.heads = torch.nn.ModuleDict()
        if 'segment' in tasks:
            self.heads.update({
                'segment': get_segment_head(cfg, self.channels,
                                            name='hr_head')})
        if 'depth' in tasks:
            self.heads.update({
                'depth': get_depth_head(cfg, self.channels,
                                        name='hr_head')})

        if 'detect' in instance_tasks:
            if not hasattr(self, 'fpn'):
                self.neck = get_neck(cfg.MODEL.NECK.NECK_TYPE, norm_layer,
                                     self.channels[1:], cfg)
            self.heads.update({
                'detect': get_detect_head(cfg, tasks + instance_tasks)})
            self.tasks = self.tasks.copy()
            tasks += instance_tasks

    def forward(self, x, targets=None):
        img_size = x.size()[-2:]
        out = {}

        # Backbone
        backbone_out = self.base_net(x)

        if hasattr(self, 'fpn'):
            backbone_out = list(backbone_out)
            backbone_out[1:4] = self.fpn(backbone_out[1:])

        # Predictions at multiple scales
        # Scale 3
        x_3 = self.scale_3(backbone_out[3])
        x_3_fpm = self.fpm_scale_3(x_3)
        # Scale 2
        x_2 = self.scale_2(backbone_out[2], x_3_fpm)
        x_2_fpm = self.fpm_scale_2(x_2)
        # Scale 1
        x_1 = self.scale_1(backbone_out[1], x_2_fpm)
        x_1_fpm = self.fpm_scale_1(x_1)
        # Scale 0
        x_0 = self.scale_0(backbone_out[0], x_1_fpm)

        out['deep_supervision'] = {
            'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}

        # Distillation + Output
        features_0 = self.distillation_scale_0(x_0)
        features_1 = self.distillation_scale_1(x_1)
        features_2 = self.distillation_scale_2(x_2)
        features_3 = self.distillation_scale_3(x_3)
        multi_scale_features = {t: [features_0[t], features_1[t], features_2[t],
                                    features_3[t]] for t in self.tasks}

        # Feature aggregation
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](multi_scale_features[t]),
                                   img_size, mode='bilinear')

        if 'detect' in self.heads.keys():
            if hasattr(self, 'neck'):
                neck_features = self.neck(backbone_out[1:4])
            else:
                neck_features = backbone_out[1:4]
            out.update({'detect': self.heads['detect'](
                neck_features, targets=targets, image_size=img_size)})

        return out


def get_mtinet(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = MTINet(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks, model_name='mtinet')

    return model
