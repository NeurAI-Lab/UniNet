import torch.nn as nn

from encoding_custom.backbones.vovnet import VoVNet


class UninetEncoder(VoVNet):

    STAGE_SPECS = {'stem': [64, 64, 128], 'stage_conv_ch': [64, 80, 96, 112],
                   'layer_per_block': 3, 'block_per_stage': [1, 1, 1, 1],
                   'eSE': True, 'dw': False}

    def __init__(self, input_ch, norm_layer, out_features=None, **kwargs):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        cfg = kwargs.get('cfg', None)
        if hasattr(cfg, 'DATASET') and cfg.DATASET == 'uninet_cs':
            stage_out_ch = 1024
            add_stem = True
            stage2_down = False
        else:
            stage_out_ch = input_ch
            add_stem = False
            stage2_down = True
        UninetEncoder.STAGE_SPECS["stage_out_ch"] = [stage_out_ch] * 4
        super(UninetEncoder, self).__init__(
            UninetEncoder.STAGE_SPECS, input_ch, norm_layer,
            out_features=out_features, add_stem=add_stem,
            stage2_down=stage2_down)
        self.add_stem = add_stem

    def forward(self, x):
        outputs = {}
        if self.add_stem:
            x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs[name] = x

        out = []
        for name in self._out_features:
            out.append(outputs[name])

        return out


def vovnet_encoder(input_channels, out_features, norm_layer=nn.BatchNorm2d,
                   **kwargs):
    vovnet = UninetEncoder(input_channels, norm_layer,
                           out_features=out_features, **kwargs)
    vovnet.feat_channels = \
        UninetEncoder.STAGE_SPECS["stage_out_ch"][:len(out_features)]

    return vovnet
