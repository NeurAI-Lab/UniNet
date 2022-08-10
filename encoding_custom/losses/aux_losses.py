import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from kornia.geometry import depth_to_normals


def extract_semantic_contours(seg_mask, num_seg_classes, multi_class=True):
    if not multi_class:
        seg_mask = seg_mask.float()
        if len(list(seg_mask.shape)) < 3:
            seg_mask = seg_mask[None, :, :]
        dilated = dilate(seg_mask)
        eroded = -dilate(-seg_mask)

        cont = dilated != eroded
        if len(list(cont.shape)) < 3:
            cont = cont.unsqueeze(0)
        return cont.float().unsqueeze(1)

    ch_mask = torch.zeros(tuple(seg_mask.size()) + (num_seg_classes,)).to(
        seg_mask.device)
    for cls in range(num_seg_classes):
        current_mask = (seg_mask == cls).float()
        if len(list(current_mask.shape)) < 3:
            current_mask = current_mask[None, :, :]
        dilated = dilate(current_mask)
        eroded = -dilate(-current_mask)

        cont = dilated != eroded
        ch_mask[..., cls] = cont
    return ch_mask.permute(0, 3, 1, 2)


def dilate(seg_mask):
    dilated = F.max_pool2d(seg_mask, [3, 3], stride=[1, 1], padding=[1, 1])
    # dilated = F.adaptive_avg_pool2d(seg_mask, seg_mask.shape[1:])
    dilated = dilated.squeeze()
    return dilated


class SemanticContLoss(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(SemanticContLoss, self).__init__()
        self.num_seg_classes = cfg.NUM_CLASSES.SEGMENT
        self.multiclass = cfg.MISC.SEM_CONT_MULTICLASS

    def weigh_classes(self, sem_cont_target):
        pass

    @staticmethod
    def balanced_loss(pred, target):
        pred = torch.clamp_max(torch.sigmoid(pred) + 1e-12, 1)
        beta = 1 - (torch.sum(target, dim=(2, 3)) / (
                target.size(2) * target.size(3)))
        beta = beta[:, :, None, None]
        loss = beta * target * torch.log(pred) + (
                1 - beta) * (1 - target) * torch.log(1 - pred)
        return torch.neg(torch.mean(loss))

    def forward(self, pred, target):
        sem_cont_target = extract_semantic_contours(
            target['segment'], self.num_seg_classes,
            multi_class=self.multiclass)

        if pred.shape != sem_cont_target.shape:
            sem_cont_target = F.interpolate(
                sem_cont_target.float(), pred.shape[2:],
                mode='nearest').squeeze(1).long()

        sem_cont_loss = self.balanced_loss(pred, sem_cont_target)
        return sem_cont_loss


class BalancedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, cfg, size_average=True, batch_average=True,
                 **kwargs):
        super(BalancedBinaryCrossEntropyLoss, self).__init__()
        self.num_seg_classes = cfg.NUM_CLASSES.SEGMENT
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = cfg.MISC.SEM_CONT_POS_WEIGHT

    def forward(self, output, target):
        sem_cont_target = extract_semantic_contours(
            target['segment'], self.num_seg_classes, multi_class=False)

        if output.shape != sem_cont_target.shape:
            sem_cont_target = F.interpolate(
                sem_cont_target.float(), output.shape[2:],
                mode='nearest').squeeze(1).long()

        labels = torch.ge(sem_cont_target, 0.5).float()
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)
        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        # Weighting of the loss, default is HED-style
        final_loss = self.pos_weight * loss_pos + (
                1 - self.pos_weight) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(sem_cont_target.size()))
        elif self.batch_average:
            final_loss /= sem_cont_target.size()[0]

        return final_loss


def extract_surface_normals(depth_map, intrinsics):
    # doesn't seem to exactly resemble derivation in:
    # Taskology: Utilizing task relations at scale
    return depth_to_normals(depth_map, intrinsics, normalize_points=True)


class NormalsCosineLoss(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(NormalsCosineLoss, self).__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.cfg = cfg

    def forward(self, pred, target, ignore_label=255):
        tar = target
        if type(target) is dict:
            tar = extract_surface_normals(
                target['depth'][:, None, :, :], target['intrinsics'])
        pred_flatten = pred.permute(0, 2, 3, 1).reshape(-1, 3)
        tar_flatten = tar.permute(0, 2, 3, 1).reshape(-1, 3)
        mask = (tar_flatten != ignore_label)
        tar_flatten = tar_flatten[mask]
        pred_flatten = pred_flatten[mask]
        y = torch.ones(tar_flatten.shape[0]).to(tar.device)
        sur_nor_loss = self.cosine_loss(pred_flatten, tar_flatten, y)
        return sur_nor_loss


class NormalsL1Loss(nn.Module):
    def __init__(self, cfg, size_average=True, **kwargs):
        super(NormalsL1Loss, self).__init__()
        self.size_average = size_average
        self.loss_func = F.l1_loss

    def forward(self, out, target, ignore_label=255):
        tar = target
        if type(target) is dict:
            tar = extract_surface_normals(
                target['depth'][:, None, :, :], target['intrinsics'])

        if out.shape[2:] != tar.shape[2:]:
            tar = F.interpolate(
                tar.float(), out.shape[2:], mode='nearest').long()

        mask = (tar != ignore_label)
        n_valid = torch.sum(mask).item()

        qn = torch.norm(out, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        out_norm = out.div(qn)
        loss = self.loss_func(
            torch.masked_select(out_norm, mask),
            torch.masked_select(tar, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(tar.size())))
                return ret_loss

        return loss


class InstSegDenseL1Loss(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(InstSegDenseL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.cfg = cfg

    def forward(self, pred, target):
        instance_target = target[:, :2, :, :]
        instance_mask = target[:, 2:, :, :]
        instance_mask = instance_mask.float()

        target = instance_target.float() * instance_mask
        mult_loss = self.l1_loss(pred * instance_mask, target)
        num_nonzero = torch.nonzero(target).size(0)
        if num_nonzero > 0:
            mult_loss /= num_nonzero
        else:
            mult_loss = torch.zeros_like(mult_loss)

        return mult_loss
