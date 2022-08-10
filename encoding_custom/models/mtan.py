#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license
# (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of MTAN
    https://arxiv.org/abs/1803.10704
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.nn.common import conv1x1
from encoding_custom.backbones.resnet import Bottleneck
from encoding_custom.backbones.base import BaseNet
from encoding_custom.models import *
from encoding_custom.utils.model_store import ModelLoader


class AttentionLayer(nn.Sequential):
    """
        Attention layer: Takes a feature representation as input and
        generates an attention mask
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(AttentionLayer, self).__init__(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid())


class RefinementBlock(nn.Sequential):
    """
        Refinement block uses a single Bottleneck layer to refine the features
         after applying task-specific attention.
    """

    def __init__(self, in_channels, out_channels):
        downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1),
                                   nn.BatchNorm2d(out_channels))
        super(RefinementBlock, self).__init__(Bottleneck(
            in_channels, out_channels // 4, downsample=downsample))


class MTAN(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(MTAN, self).__init__()

        self.tasks = tasks.copy()
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

        if 'detect' in tasks:
            self.neck = get_neck(cfg.MODEL.NECK.NECK_TYPE, norm_layer,
                                 backbone_channels[1:], cfg)
            self.detect_feats = cfg.MODEL.FPN.FPN_STRIDES
            self.heads.update({
                'detect': get_detect_head(cfg, tasks)})
            for task in ['inst_seg', 'inst_depth']:
                self.tasks.remove(task)

        # Task-specific attention modules
        self.attention_1 = nn.ModuleDict({task: AttentionLayer(
            backbone_channels[0], backbone_channels[0] // 4,
            backbone_channels[0]) for task in self.tasks})
        self.attention_2 = nn.ModuleDict({task: AttentionLayer(
            2 * backbone_channels[1], backbone_channels[1] // 4,
            backbone_channels[1]) for task in self.tasks})
        self.attention_3 = nn.ModuleDict({task: AttentionLayer(
            2 * backbone_channels[2], backbone_channels[2] // 4,
            backbone_channels[2]) for task in self.tasks})
        self.attention_4 = nn.ModuleDict({task: AttentionLayer(
            2 * backbone_channels[3], backbone_channels[3] // 4,
            backbone_channels[3]) for task in self.tasks
            if task != 'detect'})

        # Shared refinement
        self.refine_1 = RefinementBlock(
            backbone_channels[0], backbone_channels[1])
        self.refine_2 = RefinementBlock(
            backbone_channels[1], backbone_channels[2])
        self.refine_3 = RefinementBlock(
            backbone_channels[2], backbone_channels[3])

        # Downsample
        downsample = {'layer1': True, 'layer2': True, 'layer3': True}
        if backbone == 'dilated_resnet50':
            downsample.update({'layer2': False, 'layer3': False})
        self.downsample = {stage: nn.MaxPool2d(
            kernel_size=2, stride=2) if downsample else nn.Identity() for
                           stage, downsample in downsample.items()}

    def forward(self, x, targets=None):
        img_size = x.size()[-2:]

        # Shared backbone
        # In case of ResNet we apply attention over
        # the last bottleneck in each block.
        u_1_b, u_1_t = self.base_net.forward_stage_with_last_block(
            x, 'layer1')
        u_2_b, u_2_t = self.base_net.forward_stage_with_last_block(
            u_1_t, 'layer2')
        u_3_b, u_3_t = self.base_net.forward_stage_with_last_block(
            u_2_t, 'layer3')
        u_4_b, u_4_t = self.base_net.forward_stage_with_last_block(
            u_3_t, 'layer4')

        # Apply attention over the first Resnet Block -> Over last bottleneck
        a_1_mask = {task: self.attention_1[task](u_1_b) for task in self.tasks}
        a_1 = {task: a_1_mask[task] * u_1_t for task in self.tasks}
        a_1 = {task: self.downsample['layer1'](self.refine_1(a_1[task]))
               for task in self.tasks}

        # Apply attention over the second Resnet Block -> Over last bottleneck
        a_2_mask = {task: self.attention_2[task](
            torch.cat((u_2_b, a_1[task]), 1)) for task in self.tasks}
        a_2 = {task: a_2_mask[task] * u_2_t for task in self.tasks}
        a_2 = {task: self.downsample['layer2'](
            self.refine_2(a_2[task])) for task in self.tasks}

        # Apply attention over the third Resnet Block -> Over last bottleneck
        a_3_mask = {task: self.attention_3[task](
            torch.cat((u_3_b, a_2[task]), 1)) for task in self.tasks}
        a_3 = {task: a_3_mask[task] * u_3_t for task in self.tasks}
        a_3 = {task: self.downsample['layer3'](
            self.refine_3(a_3[task])) for task in self.tasks}

        # Apply attention over the last Resnet Block ->
        # No more refinement since we have task-specific
        # heads anyway. Testing with extra self.refin_4 did
        # not result in any improvements btw.
        a_4_mask = {task: self.attention_4[task](
            torch.cat((u_4_b, a_3[task]), 1)) for task in self.tasks
            if task != 'detect'}
        a_4 = {task: a_4_mask[task] * u_4_t for task in self.tasks
               if task != 'detect'}

        # Task-specific heads
        out = {task: self.heads[task](a_4[task]) for task in self.tasks
               if task in ['segment', 'depth']}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear')
               for task in self.tasks if task in ['segment', 'depth']}

        if 'detect' in self.heads.keys():
            neck_features = self.neck([a_1['detect'], a_2['detect'],
                                       a_3['detect']])
            out.update({'detect': self.heads['detect'](
                neck_features, targets=targets, image_size=img_size)})

        return out


def get_mtan(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = MTAN(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks, model_name='mtan')

    return model
