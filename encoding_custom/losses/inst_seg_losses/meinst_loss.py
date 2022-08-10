import torch
from torch import nn

from encoding_custom.utils.inst_seg.meinst_utils import\
    ReadMEInstParams

__all__ = ['MEInstLossComputation']


class MEInstLossComputation(ReadMEInstParams):

    def __init__(self, cfg):
        super(MEInstLossComputation, self).__init__(cfg)
        self.detertor_name = cfg.MODEL.DET.HEAD_NAME
        if self.encoding_type == 'explicit':
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, inst_seg_preds, targets, detect_loss_args):
        pos_inds = detect_loss_args['pos_inds']
        if pos_inds.numel() == 0:
            return {'losses/inst_seg_loss': torch.tensor(0).float().to(
                pos_inds.device)}

        if self.detertor_name == 'FCOS':
            return self.fcos_based_loss(inst_seg_preds, targets,
                                        detect_loss_args)
        else:
            raise ValueError('Unknown detector...')

    def compute_loss(self, predictions, targets, centerness_targets,
                     ctrness_norm):
        # compute regression loss..
        if self.encoding_type == 'implicit':
            predictions = torch.transpose(predictions, 0, 1)
            predictions = torch.matmul(self.w, predictions)
            predictions = torch.transpose(predictions, 0, 1)

        mask_loss = self.loss_fn(targets, predictions)
        if centerness_targets is not None:
            mask_loss = mask_loss.sum(1) * centerness_targets
        if self.encoding_type == 'explicit':
            inst_seg_loss = mask_loss.sum() / max(
                ctrness_norm * self.num_components, 1.0)
        else:
            inst_seg_loss = mask_loss.sum() / max(
                ctrness_norm * self.encoding_dim[0] ** 2, 1.0)
        return inst_seg_loss

    def fcos_based_loss(self, inst_seg_preds, inst_seg_tars, detect_loss_args):
        pos_inds = detect_loss_args['pos_inds']
        num_points_per_level = detect_loss_args['num_points_per_level']
        locations = detect_loss_args['locations']
        centerness_targets = detect_loss_args['centerness_targets']
        ctrness_norm = detect_loss_args['sum_centerness_targets_avg_per_gpu']
        exist_targets = detect_loss_args['exist_targets']

        for i in range(len(inst_seg_tars)):
            inst_seg_tars[i] = torch.split(inst_seg_tars[i],
                                           num_points_per_level, dim=0)

        inst_seg_tars_level_first = []
        for level in range(len(locations)):
            inst_seg_tars_level_first.append(
                torch.cat([inst_segs_per_im[level]
                           for inst_segs_per_im in inst_seg_tars], dim=0))
        inst_seg_tars = inst_seg_tars_level_first

        targets = []
        for l in range(len(inst_seg_tars)):
            if self.encoding_type == 'explicit':
                reshape_to = self.num_components
            else:
                reshape_to = self.encoding_dim[0] * self.encoding_dim[0]
            targets.append(inst_seg_tars[l].reshape(-1, reshape_to))
        targets = torch.cat(targets, dim=0)

        for feat_level in range(len(inst_seg_preds)):
            inst_seg_preds[feat_level] = inst_seg_preds[feat_level][exist_targets]

        predictions = torch.empty((0, self.num_components)).to(pos_inds.device)
        for ins in inst_seg_preds:
            predictions = torch.cat(
                (predictions, ins.permute(0, 2, 3, 1).reshape(-1, self.num_components)))

        # only select positives...
        predictions = predictions[pos_inds]
        targets = targets[pos_inds]

        inst_seg_loss = self.compute_loss(predictions, targets,
                                          centerness_targets, ctrness_norm)

        return {'losses/inst_seg_loss': inst_seg_loss}
