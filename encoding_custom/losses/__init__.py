from .detect_losses.fcos_loss import FCOSLossComputation
from .segment_losses.seg_losses_all import cross_entropy_loss, \
    ClassBalancedSegmentationLosses
from .segment_losses.focal_loss import FocalLoss
from .depth_losses import RMSE,DepthL1Loss
from .aux_losses import SemanticContLoss, NormalsCosineLoss, \
    BalancedBinaryCrossEntropyLoss, NormalsL1Loss
