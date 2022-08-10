import torch
from torch import nn
import os

from encoding_custom.losses.detect_losses.iou_loss import IOULoss
from encoding_custom.losses.inst_depth_losses import InstDepthLoss
from encoding_custom.losses.inst_seg_losses import build_inst_seg_loss
from encoding_custom.losses.classification_losses import build_classification_loss

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


class FCOSLossComputation(object):
    def __init__(self, cfg, inst_depth=False, inst_seg=False, **kwargs):
        self.fpn_strides = cfg.MODEL.FPN.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.num_points_per_level = None

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.cls_loss_func = build_classification_loss(cfg)

        self.inst_seg = inst_seg
        self.inst_depth = inst_depth
        if inst_depth:
            self.inst_depth_loss_fn = InstDepthLoss(cfg)
        if inst_seg:
            self.inst_seg_loss_fn = build_inst_seg_loss(cfg)

    @staticmethod
    def get_sample_region(boxes, strides, num_loc_list, loc_xs, loc_ys, radius=1):
        # This function is from adelai repository
        # https://github.com/aim-uofa/AdelaiDet/
        # fcos outputs file...
        center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
        center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(
                ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(
                xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(
                ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets, box_cls):
        object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512],
                                    [512, INF]]
        expanded_object_sizes_of_interest = []
        for level, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[level])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(
                    len(points_per_level), -1))

        expanded_object_sizes_of_interest = torch.cat(
            expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)

        targets_out = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, box_cls)
        labels, reg_targets = targets_out[:2]

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            if len(labels) == 0:
                break
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0))

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        level_firsts = [labels_level_first, reg_targets_level_first] + targets_out[2:]
        return level_firsts

    def compute_targets_for_locations(self, locations, targets,
                                      object_sizes_of_interest, box_cls):
        # box_cls is used by ATSS to get grid shape..
        labels = []
        inst_depth_targets = []
        inst_seg_targets = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        idx_target = []
        pos = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes_per_im = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_labels
            inst_depth_per_im, inst_seg_per_im = None, None
            if self.inst_depth:
                inst_depth_per_im = targets_per_im.gt_inst_depth
            if self.inst_seg:
                inst_seg_per_im = targets_per_im.gt_inst_seg
            area = targets_per_im.gt_boxes.area()

            left = xs[:, None] - bboxes_per_im[:, 0][None]
            top = ys[:, None] - bboxes_per_im[:, 1][None]
            right = bboxes_per_im[:, 2][None] - xs[:, None]
            bottom = bboxes_per_im[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([left, top, right, bottom], dim=2)

            if reg_targets_per_im.shape[1] == 0:
                num_locs = locations.size(0)
                idx_target.append(torch.ones(1, num_locs).long().to(
                    reg_targets_per_im.device) * -1)
                pos.append(torch.zeros(1, num_locs).bool().to(
                    reg_targets_per_im.device))
                continue

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes_per_im, self.fpn_strides, self.num_points_per_level,
                    xs, ys, radius=self.center_sampling_radius)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)),
                                                    locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

            if inst_depth_per_im is not None:
                inst_depth_per_im = inst_depth_per_im[locations_to_gt_inds]
                inst_depth_targets.append(inst_depth_per_im)

            if inst_seg_per_im is not None:
                inst_seg_per_im = inst_seg_per_im[locations_to_gt_inds]
                inst_seg_targets.append(inst_seg_per_im)

            idx_target.append(locations_to_gt_inds[None])
            positives = labels_per_im > 0
            pos.append(positives[None])

        tars = [labels, reg_targets, pos, idx_target]
        if self.inst_depth:
            tars += [inst_depth_targets]
        if self.inst_seg:
            tars += [inst_seg_targets]

        return tars

    def __call__(self, predictions, targets):
        """
        Arguments:
            predictions: locations (list[Tensor]), box_cls (list[Tensor]),
                box_regression (list[Tensor]), centerness (list[Tensor]),
                Optionally inst_seg and inst_depth targets
            targets (list[BoxList])

        Returns:
            Dict of loss tensors
        """
        exist_targets = []
        for idx, bl in enumerate(targets):
            if len(bl) == 0:
                continue
            exist_targets.append(idx)

        locations, box_cls, box_regression, centerness = predictions[:4]
        for feat_level in range(len(box_cls)):
            box_cls[feat_level] = box_cls[feat_level][exist_targets]
            box_regression[feat_level] = box_regression[feat_level][exist_targets]
            centerness[feat_level] = centerness[feat_level][exist_targets]

        predictions = predictions[4:]
        inst_depth_preds, inst_seg_preds = None, None
        if self.inst_depth:
            inst_depth_preds = predictions.pop(0)
        if self.inst_seg:
            inst_seg_preds = predictions.pop(0)
        num_classes = box_cls[0].size(1)

        prepared_targets = self.prepare_targets(locations, targets, box_cls)
        labels, reg_targets = prepared_targets[:2]
        pos, idx_target = prepared_targets[2:4]
        prepared_targets = prepared_targets[4:]
        inst_depth_tars, inst_seg_tars = None, None
        if self.inst_depth:
            inst_depth_tars = prepared_targets.pop(0)
        if self.inst_seg:
            inst_seg_tars = prepared_targets.pop(0)

        box_cls_flatten = []
        box_reg_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for level in range(len(labels)):
            box_cls_flatten.append(
                box_cls[level].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_reg_flatten.append(
                box_regression[level].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[level].reshape(-1))
            reg_targets_flatten.append(reg_targets[level].reshape(-1, 4))
            centerness_flatten.append(centerness[level].reshape(-1))

        if len(box_cls_flatten) == 0:
            return None
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_reg_flatten = torch.cat(box_reg_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_reg_flatten = box_reg_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        sum_centerness_targets_avg_per_gpu = 1.
        centerness_targets = None
        if pos_inds.numel() > 0:
            centerness_targets = compute_centerness_targets(
                reg_targets_flatten)
            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss, ious = self.box_reg_loss_func(
                box_reg_flatten, reg_targets_flatten,
                centerness_targets, get_iou=True)
            reg_loss = reg_loss / sum_centerness_targets_avg_per_gpu

            centerness_loss = self.centerness_loss_func(
                centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_reg_flatten.sum() * 0.
            ious = torch.zeros_like(box_reg_flatten)
            centerness_loss = centerness_flatten.sum() * 0.

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten,
                                      pos_inds, ious.clone().detach(),
                                      avg_factor=num_pos_avg_per_gpu)

        losses = {'detect': {
            'losses/detect_cls_loss': cls_loss,
            'losses/detect_reg_loss': reg_loss,
            'losses/detect_centerness_loss': centerness_loss}}

        fcos_args = {'pos_inds': pos_inds,
                     'num_points_per_level': self.num_points_per_level,
                     'locations': locations, 'exist_targets': exist_targets,
                     'labels_flatten': labels_flatten, 'targets': targets,
                     'pos': pos, 'idx_target': idx_target}
        if self.inst_seg:
            if pos_inds.numel() > 0:
                fcos_args.update({'centerness_targets': centerness_targets,
                                  'sum_centerness_targets_avg_per_gpu':
                                      sum_centerness_targets_avg_per_gpu,
                                  'num_pos_avg_per_gpu': num_pos_avg_per_gpu,
                                  'reg_targets_flatten': reg_targets_flatten})
            losses['inst_seg'] = self.inst_seg_loss_fn(
                inst_seg_preds, inst_seg_tars, fcos_args)

        if self.inst_depth:
            fcos_args.update({'num_pos_avg_per_gpu': num_pos_avg_per_gpu})
            losses['inst_depth'] = self.inst_depth_loss_fn(
                inst_depth_preds, inst_depth_tars, fcos_args)

        return losses
