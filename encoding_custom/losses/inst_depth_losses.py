import torch
from torch import nn

__all__ = ['InstDepthLoss']


class InstDepthLoss:

    def __init__(self, cfg):
        self.cfg = cfg
        self.detertor_name = cfg.MODEL.DET.HEAD_NAME
        self.loss_fn = nn.SmoothL1Loss(reduction="sum")

    def __call__(self, inst_depth_preds, inst_depth_tars, detect_loss_args):
        pos_inds = detect_loss_args['pos_inds']
        if pos_inds.numel() == 0:
            inst_depth_l1_loss = torch.tensor(0).float().to(pos_inds.device)
            losses = {'key_metrics/inst_depth_l1_loss': inst_depth_l1_loss,
                      'losses/inst_depth_l1_loss': inst_depth_l1_loss}
            return losses

        if self.detertor_name == 'FCOS':
            predictions, targets, labels, avg_factor = self.fcos_based_loss(
                inst_depth_preds, inst_depth_tars, detect_loss_args)
        else:
            raise ValueError('Unknown detector...')

        inst_depth_l1_loss = self.loss_fn(
            predictions, targets) / avg_factor

        all_losses = {}
        if self.cfg.MISC.LOG_PER_CLASS_METRICS:
            instance_names = self.cfg.INSTANCE_NAMES
            for i in range(1, len(instance_names)):
                ind = labels[pos_inds] == i
                # TODO: pos ids size mismatch again an issue..
                preds_flatten_selected = predictions[ind]
                tars_flatten_selected = targets[ind]
                if tars_flatten_selected.numel() != 0:
                    class_error = (abs(preds_flatten_selected - tars_flatten_selected)
                                   / tars_flatten_selected).mean()
                else:
                    class_error = torch.tensor(0).float().to(
                        tars_flatten_selected.device)
                all_losses.update({
                    f'classwise/inst_depth_mean_error_{instance_names[i]}':
                        class_error})

        all_losses.update({'key_metrics/inst_depth_l1_loss': inst_depth_l1_loss,
                           'losses/inst_depth_l1_loss': inst_depth_l1_loss})
        return all_losses

    @staticmethod
    def fcos_based_loss(inst_depth_preds, inst_depth_tars, detect_loss_args):
        pos_inds = detect_loss_args['pos_inds']
        num_points_per_level = detect_loss_args['num_points_per_level']
        locations = detect_loss_args['locations']
        labels_flatten = detect_loss_args['labels_flatten']
        num_pos_avg_per_gpu = detect_loss_args['num_pos_avg_per_gpu']
        exist_targets = detect_loss_args['exist_targets']

        for i in range(len(inst_depth_tars)):
            inst_depth_tars[i] = torch.split(inst_depth_tars[i],
                                             num_points_per_level, dim=0)

        inst_depth_tars_level_first = []
        for level in range(len(locations)):
            inst_depth_tars_level_first.append(
                torch.cat([depths_per_im[level]
                           for depths_per_im in inst_depth_tars], dim=0))
        inst_depth_tars = inst_depth_tars_level_first

        for feat_level in range(len(inst_depth_preds)):
            inst_depth_preds[feat_level] = \
                inst_depth_preds[feat_level][exist_targets]

        tars_flatten = []
        preds_flatten = []

        for l in range(len(inst_depth_tars)):
            tars_flatten.append(inst_depth_tars[l].reshape(-1))
            preds_flatten.append(
                inst_depth_preds[l].permute(0, 2, 3, 1).reshape(-1))

        preds_flatten = torch.cat(preds_flatten, dim=0)
        tars_flatten = torch.cat(tars_flatten, dim=0)

        preds_flatten = preds_flatten[pos_inds]
        tars_flatten = tars_flatten[pos_inds]

        return preds_flatten, tars_flatten, labels_flatten, num_pos_avg_per_gpu
