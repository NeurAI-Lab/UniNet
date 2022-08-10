from ..encoders_decoders.decoder import Decoder, DecoderBlock, DecoderBlockV2
from ..encoders_decoders.vovnet_encoder import vovnet_encoder
from ..necks.fpn import FPN
from ..heads.detect.fcos_head import FCOSModule
from encoding_custom.heads.segment.seg_head import SegHead
from encoding_custom.heads.dense_heads import DeepLabHead, HighResolutionHead
from ..heads.depth_head import DepthHead
from ..heads.aux_heads import SemanticContourHead, SurfaceNormalHead


def get_encoder(encoder_type, norm_layer, backbone_feat_channels,
                segment_or_depth, cfg):
    models = {'vovnet': vovnet_encoder}
    model_args = {'vovnet': [backbone_feat_channels[-1],
                             ["stage2", "stage3"]]}
    model_kwargs = {'vovnet': dict(norm_layer=norm_layer, cfg=cfg)}
    if encoder_type not in models:
        raise ValueError('Unknown encoder...')
    model = models[encoder_type.lower()]
    args = model_args[encoder_type.lower()]
    kwargs = model_kwargs[encoder_type.lower()]

    create_uni_en = False
    if not segment_or_depth and max(cfg.MODEL.FPN.FPN_STRIDES) > 32:
        create_uni_en = True
    elif segment_or_depth and cfg.MODEL.ENCODER.NUM_EN_FEATURES > 4:
        create_uni_en = True

    if create_uni_en:
        encoder = model(*args, **kwargs)
        uni_en_feat_channels = encoder.feat_channels
    else:
        encoder = None
        uni_en_feat_channels = []
    return encoder, uni_en_feat_channels


def get_decoder(decoder_type, norm_layer, en_feat_channels, only_detect,
                use_neck, num_detect_feats, cfg):
    if decoder_type == "resnetv1":
        expansion = DecoderBlock.expansion
    else:
        expansion = DecoderBlockV2.expansion
    de_out_channels = cfg.MODEL.DECODER.OUTPLANES * expansion
    use_neck_features = cfg.MODEL.DECODER.USE_NECK_FEATURES
    if use_neck_features and use_neck:
        en_feat_channels[1:4] = [de_out_channels] * 3
    de_in_channels = [channels + de_out_channels for channels
                      in en_feat_channels[:-2]]
    de_in_channels += [en_feat_channels[-2:]]
    de_in_channels = list(reversed(de_in_channels))

    if only_detect:
        de_in_channels = de_in_channels[:num_detect_feats + 1]
    uni_decoder = Decoder(cfg, de_in_channels, norm_layer, only_detect,
                          decoder_type=decoder_type)
    return uni_decoder


def get_neck(neck_type, norm_layer, en_feat_channels, cfg):
    en_out_channels = cfg.MODEL.DET.FEATURE_CHANNELS
    compressors_dict = {'fpn': FPN}

    neck = compressors_dict.get(neck_type, None)
    assert neck is not None, 'Unknown compressor type...'

    return neck(cfg, en_feat_channels, norm_layer, en_out_channels)


def get_detect_head(cfg, tasks):

    name_to_head = {'FCOS': FCOSModule}
    head = name_to_head[cfg.MODEL.DET.HEAD_NAME]
    return head(
        cfg, inst_depth='inst_depth' in tasks, inst_seg='inst_seg' in tasks)


def get_segment_head(cfg, num_en_features, name='seg_head'):
    heads = {'seg_head': SegHead, 'aspp': DeepLabHead,
             'hr_head': HighResolutionHead}
    head = heads.get(name, None)
    if head is not None:
        return head(cfg, cfg.NUM_CLASSES.SEGMENT, num_en_features)
    else:
        raise ValueError('Unknown head...')


def get_depth_head(cfg, num_en_features, name='depth_head'):
    heads = {'depth_head': DepthHead, 'aspp': DeepLabHead,
             'hr_head': HighResolutionHead}
    head = heads.get(name, None)
    if head is not None:
        return head(cfg, 1, num_en_features)
    else:
        raise ValueError('Unknown head...')


def get_inst_depth_head():
    pass


def get_inst_seg_head():
    pass


def get_sem_cont_head(cfg, num_en_features):
    return SemanticContourHead(cfg, cfg.NUM_CLASSES.SEGMENT,
                               num_en_features)


def get_sur_nor_head(cfg, num_en_features):
    return SurfaceNormalHead(cfg, num_en_features)
