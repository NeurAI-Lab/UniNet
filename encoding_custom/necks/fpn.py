import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Adapted from:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py

    Only includes 3 output levels for real time performance..
    """

    def __init__(self, cfg, en_feat_channels, norm_layer, neck_out_channels):
        super(FPN, self).__init__()
        backbone_levels = ['c3', 'c4', 'c5']
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(backbone_levels)):
            l_conv = nn.Conv2d(
                en_feat_channels[i], neck_out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(
                neck_out_channels, neck_out_channels, kernel_size=3,
                padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.interpolation_method = cfg.MODEL.FPN.INTERPOLATION_METHOD
        self.activation = None
        if cfg.MODEL.FPN.ACTIVATION == 'relu':
            self.activation = nn.ReLU(inplace=True)

        self.add_levels = nn.ModuleList()
        for idx, stride in enumerate(cfg.MODEL.FPN.ADDITIONAL_STRIDES):
            conv = nn.Conv2d(
                neck_out_channels, neck_out_channels, kernel_size=3,
                padding=1, stride=2)
            self.add_module(f'conv_p{6 + idx}', conv)
            self.add_levels.append(conv)

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            align_corners = False if self.interpolation_method is not 'nearest'\
                else None
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode=self.interpolation_method,
                align_corners=align_corners)

        outs = [
            conv(laterals[idx]) for idx, conv in enumerate(self.fpn_convs)]
        if self.activation is not None:
            outs = [self.activation(o) for o in outs]

        for levels in self.add_levels:
            outs.append(levels(outs[-1]))
        return outs


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
