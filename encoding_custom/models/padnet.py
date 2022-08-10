#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of PAD-Net.
    https://arxiv.org/abs/1805.04409
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from encoding_custom.backbones.base import BaseNet
from encoding_custom.backbones.resnet import Bottleneck
from encoding_custom.nn.attention_modules import SABlock
from encoding_custom.models import *
from encoding_custom.utils.model_store import ModelLoader


class BackboneFuse(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(BackboneFuse, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_outputs,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(num_outputs, momentum=0.1),
            nn.ReLU(inplace=False))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x


class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """

    def __init__(self, num_aux_outs, tasks, input_channels,
                 intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = tasks
        layers = {}
        conv_out = {}

        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(
                    input_channels, intermediate_channels, kernel_size=1,
                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels // 4,
                                     downsample=downsample)
            bottleneck2 = Bottleneck(
                intermediate_channels, intermediate_channels // 4,
                downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, num_aux_outs[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}

        for task in self.tasks:
            out['features_%s' % task] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' % task])

        return out


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks.copy()
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(
                channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' % a])
                        for a in self.auxilary_tasks if a != t}
                    for t in self.tasks}
        out = {t: x['features_%s' % t] + torch.sum(torch.stack(
            [v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out


class PADNet(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(PADNet, self).__init__()

        tasks = tasks.copy()
        instance_tasks = []
        for task in tasks:
            if task in ['detect', 'inst_seg', 'inst_depth']:
                instance_tasks.append(task)
        for task in list(set(tasks) & set(instance_tasks)):
            tasks.remove(task)

        self.tasks = tasks
        # aux tasks + segment + depth...
        self.auxilary_tasks = copy.deepcopy(tasks)
        for task in list(set(tasks) & set(cfg.MISC.AUX_TASKS)):
            self.tasks.remove(task)

        # Backbone
        self.base_net = BaseNet(
            backbone, pretrained=pretrained and not cfg.MODEL.LOAD_BACKBONE,
            norm_layer=norm_layer, dcn=cfg.MODEL.DCN_TYPE, **kwargs)
        self.channels = self.base_net.backbone.feat_channels

        # FPN..
        if cfg.MODEL.NECK.NECK_TYPE == 'fpn':
            self.fpn = FPN(cfg, self.channels[1:], norm_layer,
                           cfg.MODEL.DET.FEATURE_CHANNELS)
            self.channels[1:] = [cfg.MODEL.DET.FEATURE_CHANNELS] * 3

        self.fuse = BackboneFuse(self.channels, 256)

        num_outs = {'segment': cfg.NUM_CLASSES.SEGMENT, 'depth': 1,
                    'sem_cont': 1, 'sur_nor': 3}
        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(
            num_outs, self.auxilary_tasks, 256)

        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(
            self.tasks, self.auxilary_tasks, 256)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(256, 256 // 4, downsample=None)
            bottleneck2 = Bottleneck(256, 256 // 4, downsample=None)
            conv_out_ = nn.Conv2d(256, num_outs[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)

        if 'detect' in instance_tasks:
            if not hasattr(self, 'fpn'):
                self.neck = get_neck(cfg.MODEL.NECK.NECK_TYPE, norm_layer,
                                     self.channels[1:], cfg)
            self.detect_feats = cfg.MODEL.FPN.FPN_STRIDES
            self.heads.update({
                'detect': get_detect_head(cfg, tasks + instance_tasks)})

    def forward(self, x, targets=None):
        img_size = x.size()[-2:]
        out = {}

        # Backbone
        backbone_out = self.base_net(x)

        if hasattr(self, 'fpn'):
            backbone_out = list(backbone_out)
            backbone_out[1:4] = self.fpn(backbone_out[1:])

        x = self.fuse(backbone_out)

        # Initial predictions for every task including auxiliary tasks
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' % task] = x[task]

        # Refine features through multi-modal distillation
        x = self.multi_modal_distillation(x)

        # Make final prediction with task-specific heads
        for task in self.auxilary_tasks:
            if task in self.heads.keys():
                pred = self.heads[task](x[task])
            else:
                pred = out[f'initial_{task}']
            out[task] = F.interpolate(pred, img_size, mode='bilinear')

        if 'detect' in self.heads.keys():
            if not hasattr(self, 'fpn'):
                neck_features = self.neck(backbone_out[1:4])
            else:
                neck_features = backbone_out[1:4]
            out.update({'detect': self.heads['detect'](
                neck_features, targets=targets, image_size=img_size)})

        return out


def get_padnet(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = PADNet(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks, model_name='padnet')

    return model
