from torch import nn
import torch
from detectron2.structures import Instances
from detectron2.structures import Boxes
from detectron2.layers import batched_nms


class FCOSInference(nn.Module):
    def __init__(self, cfg, inst_depth=False):
        super(FCOSInference, self).__init__()
        self.pre_nms_thresh = cfg.MODEL.FCOS.INFERENCE_TH
        self.nms_thresh = cfg.MODEL.FCOS.NMS_TH
        self.fpn_strides = cfg.MODEL.FPN.FPN_STRIDES
        self.image_sizes = cfg.INPUT.IMAGE_SIZE
        self.inst_depth = inst_depth
        self.locs_so_far = [0]

        self.pre_nms_top_n = cfg.MODEL.FCOS.PRE_NMS_TOP_N_TRAIN
        self.post_nms_top_n = cfg.MODEL.FCOS.POST_NMS_TOP_N_TRAIN
        self.cfg = cfg

    def forward_for_single_feature_map(
            self, level, locations, box_cls, box_regression, centerness,
            inst_depth_preds=None):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        if self.inst_depth:
            inst_depth_preds = inst_depth_preds.view(N, 1, H, W).permute(0, 2, 3, 1)
            inst_depth_preds = inst_depth_preds.reshape(N, -1)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.contiguous().view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            # +1 likely because foreground classes start from 1...
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i] * self.fpn_strides[level]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            per_inst_depth = None
            if self.inst_depth:
                per_inst_depth = inst_depth_preds[i]
                per_inst_depth = per_inst_depth[per_box_loc]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if self.inst_depth:
                    per_inst_depth = per_inst_depth[top_k_indices]
                per_box_loc = per_box_loc[top_k_indices]

            h, w = self.image_sizes
            instances = Instances((int(h), int(w)))
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3]], dim=1)
            instances.pred_boxes = Boxes(detections)
            instances.pred_classes = per_class
            instances.scores = torch.sqrt(per_box_cls)
            instances.box_loc_idx = per_box_loc + sum(self.locs_so_far)
            if self.inst_depth:
                instances.inst_depth_tars = per_inst_depth
            instances.pred_boxes.clip(instances.image_size)
            results.append(instances)

        self.locs_so_far.append(H * W)
        return results

    def forward(self, locations, box_cls, box_regression, centerness,
                inst_depth_preds=None):
        """
        Arguments:
            locations:
            box_cls: list[tensor]
            box_regression: list[tensor]
            centerness:
            inst_depth_preds:
        Returns:
            boxes (list[Boxes]): the post-processed anchors, after
                applying box decoding and NMS
        """
        if self.training:
            self.pre_nms_top_n = self.cfg.MODEL.FCOS.PRE_NMS_TOP_N_TRAIN
            self.post_nms_top_n = self.cfg.MODEL.FCOS.POST_NMS_TOP_N_TRAIN
        else:
            self.pre_nms_top_n = self.cfg.MODEL.FCOS.PRE_NMS_TOP_N_INFER
            self.post_nms_top_n = self.cfg.MODEL.FCOS.POST_NMS_TOP_N_INFER

        sample_instances = []
        self.locs_so_far = [0]
        to_zip = [locations, box_cls, box_regression, centerness]
        if self.inst_depth:
            to_zip.append(inst_depth_preds)
        for level, preds in enumerate(zip(*to_zip)):
            args = [level] + list(preds)
            sample_instances.append(self.forward_for_single_feature_map(*args))

        instances = list(zip(*sample_instances))
        instances = [Instances.cat(inst) for inst in instances]
        instances = self.select_over_all_levels(instances)

        return instances

    def select_over_all_levels(self, instances):
        num_images = len(instances)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = nms(instances[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def nms(instances, nms_thresh, max_proposals=-1):
    if nms_thresh <= 0:
        return instances
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    labels = instances.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    instances = instances[keep]
    return instances


def get_inferer(cfg, tasks):
    name_to_inferer = {'FCOS': FCOSInference}
    inferer = name_to_inferer[cfg.MODEL.DET.HEAD_NAME]
    return inferer(cfg, inst_depth='inst_depth' in tasks)
