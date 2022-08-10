import torch.nn as nn

from encoding_custom import backbones
from functools import partial


__all__ = ["BaseNet"]


class BaseNet(nn.Module):
    def __init__(self, name, dilated=False, norm_layer=None,
                 root="~/.encoding/models", dilate_only_last_layer=False,
                 pretrained=True, multi_grid=True, multi_dilation=None,
                 dcn=None, **kwargs):
        super(BaseNet, self).__init__()
        self.name = name
        if multi_dilation is None:
            multi_dilation = [4, 8, 16]

        self.backbone = None
        # copying modules from pretrained models
        if name.startswith("resnet"):
            self.backbone = {
                "resnet18": backbones.resnet18,
                "resnet34": backbones.resnet34,
                "resnet50": backbones.resnet50,
                "resnet101": backbones.resnet101,
                "resnet152": backbones.resnet152,
            }[name](pretrained=pretrained, dilated=dilated, norm_layer=norm_layer,
                    root=root, dilate_only_last_layer=dilate_only_last_layer,
                    dcn=dcn)

        elif name == 'dilated_resnet50':
            self.backbone = backbones.ResnetDilated(
                backbones.resnet50(pretrained=pretrained, dilated=dilated, norm_layer=norm_layer,
                                   root=root, dilate_only_last_layer=dilate_only_last_layer))
            self.backbone.feat_channels = [256, 512, 1024, 2048]

        elif name == 'dla34':
            self.backbone = backbones.dla34(pretrained=pretrained)
        else:
            raise RuntimeError("unknown backbone: {}".format(name))

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        return c1, c2, c3, c4

    def forward_stage(self, x, stage):
        return self.backbone.forward_stage(x, stage)

    def forward_stage_with_last_block(self, x, stage):
        return self.backbone.forward_stage_with_last_block(x, stage)
