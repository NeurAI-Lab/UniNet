# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL options
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'uninet'
_C.MODEL.BACKBONE_NAME = 'dla34'
_C.MODEL.NORM_LAYER = 'bn'
_C.MODEL.DCN_TYPE = 'None'
# model loading args..
_C.MODEL.PRETRAINED_PATH = ""
_C.MODEL.IS_FULL_MODEL = False
_C.MODEL.LOAD_BACKBONE = True
_C.MODEL.BACKBONE_LOAD_NAME = 'backbone'
_C.MODEL.NECK_LOAD_NAMES = ['fpn', 'neck']
_C.MODEL.HEAD_LOAD_NAME = 'head'

# -----------------------------------------------------------------------------
# INPUT options
# -----------------------------------------------------------------------------
_C.INPUT = CN()


# --------------------------------------------------------------------------- #
# Dataloader Options
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.YEAR = 2014
_C.DATALOADER.ANNOTATION_FOLDER = 'gtFine/annotations_coco_format_v1'
_C.DATALOADER.IMG_FOLDER = 'leftImg8bit'
_C.DATALOADER.ANN_FILE_FORMAT = 'instances_%s.json'
_C.DATALOADER.MEAN = [0.28689554, 0.32513303, 0.28389177]
_C.DATALOADER.STD = [0.18696375, 0.19017339, 0.18720214]
_C.DATALOADER.TRAIN_TRANSFORMS = ['Expand', 'RandomSampleCrop', 'PreProcessBoxes',
                                  'ResizeMultiScale', 'HorizontalFlip', 'ColorJitter', 'PostProcessBoxes',
                                  'ConvertFromInts', 'ToTensor', 'Normalize']
_C.DATALOADER.VAL_TRANSFORMS = ['PreProcessBoxes', 'Resize', 'PostProcessBoxes',
                                'ToTensor', 'Normalize']
_C.DATALOADER.TRAIN_BATCH_TRANSFORMS = []
# Multi scale augmentation defaults..
_C.DATALOADER.MS_MULTISCALE_MODE = 'value'
_C.DATALOADER.MS_RATIO_RANGE = [0.75, 1]
_C.DATALOADER.PHOTOMETRIC_DISTORT_KWARGS = '{}'
_C.DATALOADER.INST_SEG_ENCODING = 'MEINST'
_C.DATALOADER.MIN_DEPTH = 1e-3
_C.DATALOADER.MAX_DEPTH = 80.
_C.DATALOADER.SIZE_DIVISIBILITY = 128

# ---------------------------------------------------------------------------- #
# Task options
# ---------------------------------------------------------------------------- #
_C.TASKS = CN()
_C.TASKS.BALANCING_METHOD = 'geometric'


# --------------------------------------------------------------------------- #
# Neck Options
# ---------------------------------------------------------------------------- #
_C.MODEL.NECK = CN()
_C.MODEL.NECK.NECK_TYPE = 'fpn'


# --------------------------------------------------------------------------- #
# Backbone and encoder Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.ENCODER_TYPE = 'vovnet'
_C.MODEL.ENCODER.NUM_EN_FEATURES = 6
_C.MODEL.ENCODER.OUT_CHANNELS_BEFORE_EXPANSION = 128
_C.MODEL.ENCODER.FEAT_CHANNELS = [512, 512, 512, 512]


# --------------------------------------------------------------------------- #
# Decoder Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.DECODER_TYPE = 'resnetv1'
_C.MODEL.DECODER.OUTPLANES = 64
_C.MODEL.DECODER.INIT_WEIGHTS = False
_C.MODEL.DECODER.USE_NECK_FEATURES = False


# --------------------------------------------------------------------------- #
# FPN Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.FPN_STRIDES = [8, 16, 32]
_C.MODEL.FPN.INTERPOLATION_METHOD = 'nearest'
_C.MODEL.FPN.ACTIVATION = None
_C.MODEL.FPN.ADDITIONAL_STRIDES = []


# --------------------------------------------------------------------------- #
# Object Detection Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DET = CN()
_C.MODEL.DET.HEAD_NAME = "FCOS"
_C.MODEL.DET.EARLY_DETECT = True
_C.MODEL.DET.FEATURE_CHANNELS = 256
_C.MODEL.DET.WEIGHTS_PER_CLASS = [1] * 8
_C.MODEL.DET.CLS_LOSS_TYPE = 'vari_focal_loss'
# Focal loss parameter: alpha
_C.MODEL.DET.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.DET.LOSS_GAMMA = 2.0
_C.MODEL.DET.LOSS_BETA = 0.9999
_C.MODEL.DET.PRIOR_PROB = 0.01


# --------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N_TRAIN = 1000
_C.MODEL.FCOS.POST_NMS_TOP_N_TRAIN = 100
_C.MODEL.FCOS.PRE_NMS_TOP_N_INFER = 1000
_C.MODEL.FCOS.POST_NMS_TOP_N_INFER = 100
# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4
# if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
_C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 1.5
# IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
_C.MODEL.FCOS.IOU_LOSS_TYPE = "giou"
_C.MODEL.FCOS.NORM_REG_TARGETS = True
_C.MODEL.FCOS.CENTERNESS_ON_REG = True


# --------------------------------------------------------------------------- #
# Segmentation Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SEG = CN()
_C.MODEL.SEG.INPLANES = 64
_C.MODEL.SEG.OUTPLANES = 64


# --------------------------------------------------------------------------- #
# Depth Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DEPTH = CN()
_C.MODEL.DEPTH.INPLANES = 64
_C.MODEL.DEPTH.OUTPLANES = 64
_C.MODEL.DEPTH.ACTIVATION_FN = 'sigmoid'

# --------------------------------------------------------------------------- #
# Instance depth Options
# ---------------------------------------------------------------------------- #
_C.MODEL.INST_DEPTH = CN()
_C.MODEL.INST_DEPTH.DEPTH_ON_REG = True


# --------------------------------------------------------------------------- #
# Instance segmentation Options
# ---------------------------------------------------------------------------- #
_C.MODEL.INST_SEG = CN()
_C.MODEL.INST_SEG.SHARE_CLS_INST_HEADS = False
_C.MODEL.INST_SEG.SHARE_BBOX_INST_HEADS = True

# --------------------------------------------------------------------------- #
# MEINST Options
# ---------------------------------------------------------------------------- #

_C.MODEL.MEINST = CN()
# mask encoding type
_C.MODEL.MEINST.ENCODING_TYPE = 'explicit'
# is inverse sigmoid and sigmoid used for finding pca components
_C.MODEL.MEINST.SIGMOID = True
# is whiten used for finding pca components
_C.MODEL.MEINST.WHITEN = True
# path to pca params file
_C.MODEL.MEINST.PCA_PATH = ''
# number of components in the encoded mask
_C.MODEL.MEINST.NUM_COMPONENTS = 60
# dimension to which all instance masks are reshaped to
_C.MODEL.MEINST.ENCODING_DIM = 28


# --------------------------------------------------------------------------- #
# Miscellaneous Options
# ---------------------------------------------------------------------------- #

_C.MISC = CN()
_C.MISC.CITYS_INST_SEG_EVAL = False
_C.MISC.AUX_TASKS = ['sem_cont', 'sur_nor']
_C.MISC.SEM_CONT_MULTICLASS = False
_C.MISC.SEM_CONT_POS_WEIGHT = 0.95


# --------------------------------------------------------------------------- #
# Dict config
# ---------------------------------------------------------------------------- #

TASKS_DICT = dict(detect=True, segment=True, depth=True, inst_depth=True,
                  inst_seg=True, sem_cont=False, sur_nor=False)
TASK_TO_LOSS_NAME = dict(detect='default', segment='default', depth='default',
                         inst_depth='default', inst_seg='default',
                         sem_cont='default', sur_nor='default')
TASK_TO_LOSS_ARGS = dict()
TASK_TO_LOSS_KWARGS = dict()
TASK_TO_LOSS_CALL_KWARGS = dict(segment=dict(ignore_index=-1))
TASK_TO_MIN_OR_MAX = dict(detect=1, segment=1, depth=-1, inst_depth=-1, inst_seg=1,
                          sem_cont=1, sur_nor=-1)
LOSS_INIT_WEIGHTS = dict(detect_cls_loss=1., detect_reg_loss=1.,
                         detect_centerness_loss=1., segment_loss=1.,
                         depth_loss=8., inst_depth_l1_loss=0.05)
LOSS_START_EPOCH = dict(detect_cls_loss=1, detect_reg_loss=1,
                        detect_centerness_loss=1, segment_loss=1,
                        depth_loss=1, inst_depth_l1_loss=1, inst_seg_loss=1)

LR_SCHEDULER_ARGS = dict()
OPTIMIZER_ARGS = dict()
# For Lookahead..
OPTIMIZER_WRAP_ARGS = dict(k=5, alpha=0.5)
