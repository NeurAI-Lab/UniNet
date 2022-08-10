import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float),
                                  requires_grad=True)

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):

    def __init__(self, cfg, in_channels, inst_depth=False, inst_seg=False):
        super(FCOSHead, self).__init__()
        num_classes = cfg.NUM_CLASSES.DETECT - 1
        self.fpn_strides = cfg.MODEL.FPN.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.depth_on_reg = cfg.MODEL.INST_DEPTH.DEPTH_ON_REG
        self.share_cls_inst_heads = cfg.MODEL.INST_SEG.SHARE_CLS_INST_HEADS
        self.share_bbox_inst_heads = cfg.MODEL.INST_SEG.SHARE_BBOX_INST_HEADS
        self.inst_depth = inst_depth
        self.inst_seg = inst_seg

        cls_tower, bbox_tower, mask_tower = self.build_fcos_towers(cfg, in_channels)

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1)

        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1)
        heads = [self.cls_tower, self.bbox_tower, self.cls_logits,
                 self.bbox_pred, self.centerness]
        if self.inst_depth:
            self.inst_depth_logits = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1, padding=1)
            heads.append(self.inst_depth_logits)

        if self.inst_seg:
            if not (self.share_cls_inst_heads or self.share_bbox_inst_heads):
                self.add_module('mask_tower', nn.Sequential(*mask_tower))

            # the output channels is num components replacing num classes in
            # classification head..
            self.inst_seg_logits = nn.Conv2d(
                in_channels, cfg.MODEL.MEINST.NUM_COMPONENTS, kernel_size=3,
                stride=1, padding=1)
            heads.append(self.inst_seg_logits)

        # initialization
        for modules in heads:
            for mod in modules.modules():
                if isinstance(mod, nn.Conv2d):
                    torch.nn.init.normal_(mod.weight, std=0.01)
                    torch.nn.init.constant_(mod.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(scale=1.0) for _ in self.fpn_strides])

    def build_fcos_towers(self, cfg, in_channels):
        conv_args = [in_channels, in_channels]
        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1,
                       'bias': True}
        convs_tower = [partial(nn.Conv2d, *conv_args, **conv_kwargs)]
        convs_tower *= cfg.MODEL.FCOS.NUM_CONVS - 1
        convs_tower.append(partial(nn.Conv2d, *conv_args,
                                   **conv_kwargs))

        return self.build_towers(in_channels, convs_tower)

    def build_towers(self, in_channels, convs_tower):
        cls_tower = []
        bbox_tower = []
        mask_tower = []

        for conv in convs_tower:
            cls_tower.append(conv())
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(conv())
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

            if self.inst_seg and not (self.share_cls_inst_heads
                                      or self.share_bbox_inst_heads):
                mask_tower.append(conv())
                mask_tower.append(nn.GroupNorm(32, in_channels))
                mask_tower.append(nn.ReLU())

        return cls_tower, bbox_tower, mask_tower

    def forward(self, x, locations=None, targets=None):
        box_cls = []
        bbox_reg = []
        inst_depth_preds = []
        centerness = []
        inst_seg_preds = []
        for level, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            box_cls.append(self.cls_logits(cls_tower))

            cent_in = box_tower if self.centerness_on_reg else cls_tower
            centerness.append(self.centerness(cent_in))

            bbox_pred = self.scales[level](self.bbox_pred(box_tower))

            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred))
            else:
                bbox_reg.append(torch.exp(bbox_pred))

            if self.inst_depth:
                ins_depth_in = box_tower if self.depth_on_reg else cls_tower
                inst_depth_preds.append(
                    F.relu(self.inst_depth_logits(ins_depth_in)))

            if self.inst_seg:
                if not (self.share_cls_inst_heads or self.share_bbox_inst_heads):
                    mask_tower = self.mask_tower(feature)
                elif self.share_bbox_inst_heads:
                    mask_tower = box_tower
                else:
                    mask_tower = cls_tower
                inst_seg_preds.append(self.inst_seg_logits(mask_tower))

        predictions = [box_cls, bbox_reg, centerness]

        if self.inst_depth:
            predictions += [inst_depth_preds]
        if self.inst_seg:
            predictions += [inst_seg_preds]
        return predictions


class FCOSModule(nn.Module):

    def __init__(self, cfg, inst_depth, inst_seg, head_fn=FCOSHead):
        super(FCOSModule, self).__init__()

        in_channels = cfg.MODEL.DET.FEATURE_CHANNELS
        self.head = head_fn(cfg, in_channels, inst_depth, inst_seg)
        self.inst_depth = inst_depth
        self.inst_seg = inst_seg
        self.cfg = cfg
        self.fpn_strides = cfg.MODEL.FPN.FPN_STRIDES

    def forward(self, features, targets=None):
        with torch.no_grad():
            locations = self.compute_locations(features)
        predictions = self.head(features, locations=locations, targets=targets)
        predictions = [locations] + predictions

        return predictions

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    @staticmethod
    def compute_locations_per_level(h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device)
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
