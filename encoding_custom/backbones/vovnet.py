# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.utils.model_store import get_model_file

__all__ = ["VoVNet", "build_vovnet"]

_NORM = False

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64], 'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512], "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1], "eSE": True, "dw": True}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64], "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024], "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1], "eSE": True, "dw": True}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128], 'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512], 'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1], 'eSE': True, "dw": False}

VoVNet19_eSE = {
    'stem': [64, 64, 128], "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024], "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1], "eSE": True, "dw": False}

VoVNet39_eSE = {
    'stem': [64, 64, 128], "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024], "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2], "eSE": True, "dw": False}

VoVNet57_eSE = {
    'stem': [64, 64, 128], "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024], "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3], "eSE": True, "dw": False}

VoVNet99_eSE = {
    'stem': [64, 64, 128], "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024], "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3], "eSE": True, "dw": False}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE, "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE, "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE, "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE}


def dw_conv3x3(in_channels, out_channels, module_name, postfix, norm_layer,
               stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [('{}_{}/dw_conv3x3'.format(module_name, postfix),
             nn.Conv2d(
                 in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, groups=out_channels,
                 bias=False)),
            ('{}_{}/pw_conv1x1'.format(module_name, postfix),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1,
                padding=0, groups=1, bias=False)),
            ('{}_{}/pw_norm'.format(module_name, postfix),
             norm_layer(out_channels)),
            ('{}_{}/pw_relu'.format(module_name, postfix),
             nn.ReLU(inplace=True))]


def conv3x3(in_channels, out_channels, module_name, postfix, norm_layer,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv',
             nn.Conv2d(
                 in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, groups=groups,
                 bias=False)),
            (f'{module_name}_{postfix}/norm', norm_layer(out_channels)),
            (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def conv1x1(in_channels, out_channels, module_name, postfix, norm_layer,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv',
             nn.Conv2d(
                 in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, groups=groups,
                 bias=False)),
            (f'{module_name}_{postfix}/norm', norm_layer(out_channels)),
            (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1,
                            padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        inp = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return inp * x


class _OSA_module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block,
                 module_name, norm_layer, SE=False, identity=False,
                 depthwise=False):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(OrderedDict(
                conv1x1(in_channel, stage_ch, f"{module_name}_reduction",
                        "0", norm_layer)))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(
                    dw_conv3x3(stage_ch, stage_ch, module_name,
                               i, norm_layer))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(
                    conv3x3(in_channel, stage_ch, module_name,
                            i, norm_layer))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(
            in_channel, concat_ch, module_name, "concat", norm_layer)))

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x
        output = [x]

        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)

        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage,
                 layer_per_block, stage_num, norm_layer, SE=False,
                 depthwise=False, stage2_down=False):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2 or stage2_down:
            self.add_module("Pooling", nn.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(module_name, _OSA_module(
            in_ch, stage_ch, concat_ch, layer_per_block, module_name,
            norm_layer, SE, depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(module_name, _OSA_module(
                concat_ch, stage_ch, concat_ch, layer_per_block,
                module_name, norm_layer, SE, identity=True,
                depthwise=depthwise))


class VoVNet(nn.Module):

    def __init__(self, stage_specs, input_ch, norm_layer,
                 out_features=None, add_stem=True, stage2_down=False):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()

        if type(stage_specs) is not dict:
            stage_specs = _STAGE_SPECS[stage_specs]

        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features

        in_ch_list = config_concat_ch[:-1]
        current_stirde = 4
        self._out_feature_strides = {"stage2": current_stirde}
        self._out_feature_channels = {}

        # Stem module
        if add_stem:
            stem_ch = stage_specs["stem"]
            conv_type = dw_conv3x3 if depthwise else conv3x3
            stem = conv3x3(input_ch, stem_ch[0], "stem", "1", norm_layer, 2)
            stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", norm_layer, 1)
            stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", norm_layer, 2)
            self.add_module("stem", nn.Sequential((OrderedDict(stem))))
            stem_out_ch = [stem_ch[2]]
            in_ch_list = stem_out_ch + in_ch_list

            self._out_feature_strides.update({"stem": current_stirde})
            self._out_feature_channels.update({"stem": stem_ch[2]})

        # OSA stages
        self.stage_names = []
        for i in range(len(self._out_features)):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(name, _OSA_stage(
                in_ch_list[i], config_stage_ch[i], config_concat_ch[i],
                block_per_stage[i], layer_per_block, i + 2, norm_layer,
                SE, depthwise, stage2_down=stage2_down))

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(
                    current_stirde * 2)

        # initialize weights
        # self._initialize_weights()

        self.backbone_feat_channels = []

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs[name] = x

        out = []
        for name in self._out_features:
            out.append(outputs[name])

        return out


def build_vovnet(stage_spec_name, input_channels, out_features,
                 norm_layer=nn.BatchNorm2d, pretrained=False,
                 root="~/.encoding/models"):
    vovnet = VoVNet(stage_spec_name, input_channels, norm_layer,
                    out_features=out_features)
    stage_spec = _STAGE_SPECS[stage_spec_name]
    vovnet.feat_channels = \
        stage_spec["stage_out_ch"][:len(out_features)]
    if pretrained:
        weights = OrderedDict()
        loaded = torch.load(get_model_file("vovnet39", root=root))
        for i, v in loaded.items():
            k = i.replace("backbone.bottom_up.", "")
            weights[k] = v

        vovnet.load_state_dict(weights)

    return vovnet
