import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit


__all__ = ["build_classification_loss"]


def build_classification_loss(cfg):
    if cfg.MODEL.DET.CLS_LOSS_TYPE == 'focal_loss':
        return FocalLoss(cfg)
    elif cfg.MODEL.DET.CLS_LOSS_TYPE == 'vari_focal_loss':
        return VariFocalLoss(cfg)
    else:
        raise ValueError('Unknown classification loss..')


class FocalLoss(nn.Module):
    def __init__(self, cfg):
        super(FocalLoss, self).__init__()
        self.focal_loss_alpha = cfg.MODEL.DET.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.DET.LOSS_GAMMA

    def forward(self, logits, targets, pos_inds, quality_scores,
                reduction="sum", avg_factor=1.0):
        # prepare one_hot
        # latest adelai fcos has num_cls as background instead of 0..
        targets = targets - 1
        class_target = torch.zeros_like(logits)
        # targets now ignores -1 background labels as its not
        # included in pos_inds
        class_target[pos_inds, targets[pos_inds]] = 1

        cls_loss = sigmoid_focal_loss_jit(
            logits, class_target, alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma, reduction=reduction)

        return cls_loss / avg_factor


class VariFocalLoss(nn.Module):
    def __init__(self, cfg):
        super(VariFocalLoss, self).__init__()
        self.alpha = 1 - cfg.MODEL.DET.LOSS_ALPHA
        self.gamma = cfg.MODEL.DET.LOSS_GAMMA

    def forward(self, logits, targets, pos_inds, quality_scores,
                reduction="sum", avg_factor=1.0):
        targets = targets - 1
        class_target = torch.zeros_like(logits)
        class_target[pos_inds, targets[pos_inds]] = quality_scores

        pred_sigmoid = logits.sigmoid()
        class_target = class_target.type_as(class_target)
        focal_weight = class_target * (
                class_target > 0.0).float() + self.alpha * (
                pred_sigmoid - class_target).abs().pow(
            self.gamma) * (class_target <= 0.0).float()
        loss = F.binary_cross_entropy_with_logits(
            logits, class_target, reduction='none') * focal_weight

        if reduction == "sum":
            loss = loss.sum()
        if reduction == "mean":
            loss = loss.mean()

        if avg_factor is not None:
            loss /= torch.sum(quality_scores)

        return loss
