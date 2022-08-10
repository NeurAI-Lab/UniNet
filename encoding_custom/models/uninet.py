import torch.nn as nn
from collections import OrderedDict

from encoding_custom.backbones.base import BaseNet
from encoding_custom.models import *
from encoding_custom.utils.model_store import ModelLoader


class Uninet(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(Uninet, self).__init__()

        encoder_type = cfg.MODEL.ENCODER.ENCODER_TYPE
        early_detect = cfg.MODEL.DET.EARLY_DETECT
        neck_type = cfg.MODEL.NECK.NECK_TYPE
        decoder_type = cfg.MODEL.DECODER.DECODER_TYPE

        self.base_net = BaseNet(
            backbone, pretrained=pretrained and not cfg.MODEL.LOAD_BACKBONE,
            norm_layer=norm_layer, dcn=cfg.MODEL.DCN_TYPE, **kwargs)
        backbone_feat_channels = self.base_net.backbone.feat_channels

        self.encoder_decoder = UninetEncoderDecoder(
            backbone_feat_channels, tasks, norm_layer, cfg=cfg,
            encoder_type=encoder_type, early_detect=early_detect,
            neck_type=neck_type, decoder_type=decoder_type, **kwargs)

        self.head = UninetHead(tasks, cfg, early_detect=early_detect, **kwargs)

    def forward(self, x, targets=None):
        image_size = list(x.shape[2:])
        backbone_features = self.base_net(x)
        encoder_features, decoder_features, compressed_features = self.encoder_decoder(
            backbone_features)

        out = self.head(decoder_features, compressed_features,
                        image_size, targets=targets)
        return out


class UninetEncoderDecoder(nn.Module):
    def __init__(self, backbone_feat_channels, tasks, norm_layer, cfg=None,
                 encoder_type='resnet', early_detect=False, neck_type='v1',
                 decoder_type='resnetv1', **kwargs):
        super(UninetEncoderDecoder, self).__init__()

        self.num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES
        segment_or_depth = 'segment' in tasks or 'depth' in tasks
        self.only_detect = not segment_or_depth
        self.uninet_encoder, uni_en_feat_channels = get_encoder(
            encoder_type, norm_layer, backbone_feat_channels,
            segment_or_depth, cfg)
        en_feat_channels = backbone_feat_channels + uni_en_feat_channels
        en_feat_channels = en_feat_channels[:self.num_en_features]

        self.detect_feats = cfg.MODEL.FPN.FPN_STRIDES
        stride_to_index = {4: 0, 8: 1, 16: 2, 32: 3, 64: 4, 128: 5}
        self.feature_indices = [stride_to_index[stride] for stride
                                in self.detect_feats]
        self.early_detect = early_detect
        self.neck_type = neck_type
        self.use_neck = 'detect' in tasks and early_detect
        if self.use_neck:
            self.neck = get_neck(
                neck_type, norm_layer,
                en_feat_channels[min(self.feature_indices):
                                 max(self.feature_indices) + 1], cfg)

        self.use_decoder = segment_or_depth or not self.use_neck
        if self.use_decoder:
            self.use_neck_features = cfg.MODEL.DECODER.USE_NECK_FEATURES
            self.decoder = get_decoder(
                decoder_type, norm_layer, en_feat_channels, self.only_detect,
                self.use_neck, len(self.detect_feats), cfg)

    def forward(self, backbone_features):
        l1, l2, l3, l4 = backbone_features
        uni_en_outs = []
        if self.uninet_encoder is not None:
            uni_en_outs = self.uninet_encoder(l4)
        encoder_features = [l1, l2, l3, l4] + uni_en_outs

        decoder_input = encoder_features
        neck_features = None
        if self.early_detect and self.use_neck:
            features_at_stride = [encoder_features[idx] for idx
                                  in self.feature_indices]
            neck_features = self.neck(features_at_stride)

        decoder_features = None
        if self.use_decoder:
            if self.use_neck_features:
                decoder_input[min(self.feature_indices):
                              max(self.feature_indices) + 1] = neck_features
            if self.only_detect:
                decoder_input = decoder_input[min(self.feature_indices):]
            decoder_features = self.decoder(decoder_input)

        return encoder_features, decoder_features, neck_features


class UninetHead(nn.Module):
    def __init__(self, tasks, cfg, early_detect=False, **kwargs):

        super(UninetHead, self).__init__()
        self.detect = 'detect' in tasks
        self.segment = 'segment' in tasks
        self.depth = 'depth' in tasks
        self.sem_cont = 'sem_cont' in tasks
        self.sur_nor = 'sur_nor' in tasks

        self.early_detect = early_detect
        num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES - 1

        if self.segment:
            self.seg_head = get_segment_head(cfg, num_en_features)
        if self.detect:
            self.detect_feats = cfg.MODEL.FPN.FPN_STRIDES
            self.det_head = get_detect_head(cfg, tasks)
        if self.depth:
            self.depth_head = get_depth_head(cfg, num_en_features)
        if self.sem_cont:
            self.sem_cont_head = get_sem_cont_head(cfg, num_en_features)
        if self.sur_nor:
            self.sur_nor_head = get_sur_nor_head(cfg, num_en_features)

    def forward(self, decoder_features, compressed_features, image_size,
                targets=None):
        results = OrderedDict()

        if self.segment:
            results["segment"] = self.seg_head(decoder_features, image_size)
        if self.depth:
            results["depth"] = self.depth_head(decoder_features, image_size)
        if self.sem_cont:
            results["sem_cont"] = self.sem_cont_head(decoder_features, image_size)
        if self.sur_nor:
            results["sur_nor"] = self.sur_nor_head(decoder_features, image_size)

        if self.detect:
            if self.early_detect:
                det_input = compressed_features
            else:
                det_input = decoder_features[1:len(self.detect_feats) + 1]
                det_input = list(reversed(det_input))

            results["detect"] = self.det_head(det_input, targets=targets)

        return results


def get_uninet(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = Uninet(backbone, tasks, norm_layer, cfg, pretrained, **kwargs)
    if pretrained:
        loader = ModelLoader()
        model = loader(model, cfg, tasks)

    return model
