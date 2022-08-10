# this file is from backbones dla in
# https://github.com/aim-uofa/AdelaiDet/
# adet/modeling/backbone/dla.py
# DLA: Deep Layer Aggregation

import math
import torch
from torch import nn

from encoding_custom.backbones.resnet import BasicBlock
from encoding_custom.utils.model_store import get_model_file

WEB_ROOT = 'http://dl.yf.io/dla/models'


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual,
                 norm_layer=nn.BatchNorm2d):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, norm_layer=nn.BatchNorm2d):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(out_channels))

    def forward(self, x, residual=None, children=None):
        if self.training and residual is not None:
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

    def forward_stage_with_last_block(self, x, residual=None, children=None):
        if self.training and residual is not None:
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x1, x


class DLA(nn.Module):
    def __init__(self, levels, channels, block=BasicBlock,
                 residual_root=False, norm_layer=nn.BatchNorm2d):
        super(DLA, self).__init__()
        self.channels = channels
        self.norm_layer = norm_layer

        self._out_features = ["level{}".format(i) for i in range(6)]
        self._out_feature_channels = {k: channels[i] for i, k
                                      in enumerate(self._out_features)}
        self._out_feature_strides = {k: 2 ** i for i, k
                                     in enumerate(self._out_features)}

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            norm_layer(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                self.norm_layer(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            name = 'level{}'.format(i)
            x = getattr(self, name)(x)
            y.append(x)
        return y[2:]

    def get_level(self, stage):
        stage = stage.replace('layer', 'level')
        num = int(stage[-1]) + 1
        stage = stage[:-1] + str(num)
        return getattr(self, stage)

    def forward_stage(self, x, stage):
        if stage == 'layer1':
            x = self.base_layer(x)
            for i in range(3):
                name = 'level{}'.format(i)
                x = getattr(self, name)(x)
            return x

        else:  # Stage 2, 3 or 4
            layer = self.get_level(stage)
            return layer(x)

    def forward_stage_with_last_block(self, x, stage):
        if stage == 'layer1':
            x = self.base_layer(x)
            x = self.level0(x)
            x = self.level1(x)
            return self.level2.forward_stage_with_last_block(x)

        else:  # Stage 2, 3 or 4
            layer = self.get_level(stage)
            return layer.forward_stage_with_last_block(x)


def dla34(pretrained=False, root="~/.encoding/models", **kwargs):
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    model.feat_channels = [64, 128, 256, 512]
    if pretrained:
        model.load_state_dict(
            torch.load(get_model_file("dla34", root=root)), strict=False)
    return model
