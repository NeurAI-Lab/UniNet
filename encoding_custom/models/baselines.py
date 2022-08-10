#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license
# (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.backbones.base import BaseNet
from encoding_custom.models import *
from encoding_custom.utils.model_store import ModelLoader


class SingleTaskModel(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(SingleTaskModel, self).__init__()
        assert len(tasks) == 1
        self.task = tasks[0]
        self.base_net = BaseNet(
            backbone, pretrained=pretrained and not cfg.MODEL.LOAD_BACKBONE,
            norm_layer=norm_layer, dcn=cfg.MODEL.DCN_TYPE, **kwargs)
        backbone_channels = self.base_net.backbone.feat_channels
        if 'segment' in tasks:
            self.head = get_segment_head(cfg, backbone_channels, name='aspp')
        elif 'depth' in tasks:
            self.head = get_depth_head(cfg, backbone_channels, name='aspp')

    def forward(self, x, targets=None):
        out_size = x.size()[2:]
        out = self.head(self.base_net(x)[-1])
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(MultiTaskModel, self).__init__()
        self.tasks = tasks
        self.base_net = BaseNet(
            backbone, pretrained=pretrained and not cfg.MODEL.LOAD_BACKBONE,
            norm_layer=norm_layer, dcn=cfg.MODEL.DCN_TYPE, **kwargs)
        backbone_channels = self.base_net.backbone.feat_channels
        self.heads = torch.nn.ModuleDict()
        if 'segment' in tasks:
            self.heads.update({
                'segment': get_segment_head(cfg, backbone_channels,
                                            name='aspp')})
        if 'depth' in tasks:
            self.heads.update({
                'depth': get_depth_head(cfg, backbone_channels,
                                        name='aspp')})
        self.tasks = tasks

    def forward(self, x, targets=None):
        out_size = x.size()[2:]
        shared_representation = self.base_net(x)
        return {task: F.interpolate(
            self.heads[task](shared_representation[-1]),
            out_size, mode='bilinear') for task in self.tasks}


def get_mtl_baseline(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = MultiTaskModel(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks)

    return model


def get_single_task_baseline(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = SingleTaskModel(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks)

    return model
