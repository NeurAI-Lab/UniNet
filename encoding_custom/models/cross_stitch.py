#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license
# (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of cross-stitch networks
    https://arxiv.org/abs/1604.03539
"""

# Taken from
# https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.backbones.base import BaseNet
from encoding_custom.models import *
from encoding_custom.utils.model_store import ModelLoader


class ChannelWiseMultiply(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels),
                                  requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1, -1, 1, 1), x)


class CrossStitchUnit(nn.Module):
    def __init__(self, tasks, num_channels, alpha, beta):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_unit = nn.ModuleDict(
            {t: nn.ModuleDict({t: ChannelWiseMultiply(num_channels)
                               for t in tasks}) for t in tasks})

        for t_i in tasks:
            for t_j in tasks:
                if t_i == t_j:
                    self.cross_stitch_unit[t_i][t_j].init_value(alpha)
                else:
                    self.cross_stitch_unit[t_i][t_j].init_value(beta)

    def forward(self, task_features):
        out = {}
        for t_i in task_features.keys():
            prod = torch.stack([self.cross_stitch_unit[t_i][t_j](
                task_features[t_j]) for t_j in task_features.keys()])
            out[t_i] = torch.sum(prod, dim=0)
        return out


class CrossStitchNetwork(nn.Module):
    """
        Implementation of cross-stitch networks.
        We insert a cross-stitch unit, to combine features from the
         task-specific backbones after every stage.
    """

    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(CrossStitchNetwork, self).__init__()

        tasks = tasks.copy()
        inst_tasks = []
        for task in ['inst_seg', 'inst_depth']:
            if task in tasks:
                inst_tasks.append(task)
                tasks.remove(task)
        self.tasks = tasks
        self.backbone = {}

        for task in tasks:
            bb = BaseNet(
                backbone, pretrained=pretrained and not cfg.MODEL.LOAD_BACKBONE,
                norm_layer=norm_layer, dcn=cfg.MODEL.DCN_TYPE, **kwargs)
            self.backbone[task] = bb
            self.add_module(f'bb_{task}', bb)
        backbone_channels = self.backbone[tasks[0]].backbone.feat_channels
        self.heads = torch.nn.ModuleDict()
        if 'segment' in tasks:
            self.heads.update({
                'segment': get_segment_head(cfg, backbone_channels,
                                            name='aspp')})
        if 'depth' in tasks:
            self.heads.update({
                'depth': get_depth_head(cfg, backbone_channels,
                                        name='aspp')})

        if 'detect' in tasks:
            self.neck = get_neck(cfg.MODEL.NECK.NECK_TYPE, norm_layer,
                                 backbone_channels[1:], cfg)
            self.detect_feats = cfg.MODEL.FPN.FPN_STRIDES
            self.heads.update({
                'detect': get_detect_head(cfg, tasks + inst_tasks)})

        # Cross-stitch units
        alpha = 0.9
        beta = 0.1
        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        self.cross_stitch = nn.ModuleDict(
            {stage: CrossStitchUnit(self.tasks, backbone_channels[idx],
                                    alpha, beta)
             for idx, stage in enumerate(self.stages)})

        if 'detect' in tasks:
            self.neck = get_neck(
                cfg.MODEL.NECK.NECK_TYPE, norm_layer, backbone_channels[1:], cfg)

    def forward(self, x, targets=None):
        img_size = x.size()[-2:]
        # Feed as input to every single-task network
        x = {task: x for task in self.tasks}

        det_level_outs = []
        # Backbone
        for stage in self.stages:

            # Forward through next stage of task-specific network
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)

            # Cross-stitch the task-specific features
            level_out = self.cross_stitch[stage](x)
            if 'detect' in self.heads.keys():
                det_level_outs.append(level_out['detect'])
            x = level_out

        # Task-specific heads
        out = {task: self.heads[task](x[task]) for task in self.tasks
               if task in ['segment', 'depth']}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear')
               for task in self.tasks if task in ['segment', 'depth']}

        if 'detect' in self.heads.keys():
            neck_features = self.neck(det_level_outs[1:])
            out.update({'detect': self.heads['detect'](
                neck_features, targets=targets, image_size=img_size)})

        return out


def get_cross_stitch(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = CrossStitchNetwork(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks, model_name='cross_stitch')

    return model
