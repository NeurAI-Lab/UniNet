import numpy as np
import cv2
import torch
from torch.nn import functional as F

from detectron2.structures import Instances, Boxes, PolygonMasks
from encoding_custom.utils.inst_seg import meinst_utils


class FCOSTargetTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, bboxes=None, labels=None, inst_depths=None,
                 inst_masks=None):
        target = Instances(self.img_size)
        target.gt_boxes = Boxes(bboxes)
        labels = torch.tensor(labels).long()
        target.gt_labels = labels
        if inst_depths is not None:
            inst_depths = torch.tensor(inst_depths)
            target.gt_inst_depth = inst_depths
        if inst_masks is not None:
            if type(inst_masks) is list and len(inst_masks) != 0:
                inst_masks = torch.stack(inst_masks)
            elif isinstance(inst_masks, PolygonMasks):
                pass
            target.gt_inst_seg = inst_masks
        return target, []


class MEInstTargetTransform:
    def __init__(self, cfg, image_size, device='cpu', **kwargs):
        self.cfg = cfg
        self.image_size = image_size
        self.params = meinst_utils.ReadMEInstParams(cfg)
        self.params.load_pca()
        self.params_ts = meinst_utils.ReadMEInstParams(cfg)
        self.params_ts.load_pca(is_ts=True, device=device)

    def __call__(self, boxes, inst_masks):
        if inst_masks.numel() == 0:
            return inst_masks
        inst_seg_tars = []
        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = [int(b) for b in box]
            if x1 - x0 > 0 and y1 - y0 > 0 and all(
                    [i >= 0 for i in [x0, y0, x1, y1]]):
                crop_mask = inst_masks[y0: y1, x0: x1, idx]
                crop_mask = crop_mask.reshape((y1 - y0, x1 - x0))
                crop_mask = cv2.resize(
                    crop_mask.numpy(), self.params.encoding_dim,
                    interpolation=cv2.INTER_NEAREST)
            else:
                crop_mask = np.zeros(self.params.encoding_dim,
                                     dtype=np.int64)
            crop_mask = crop_mask.flatten()
            if self.params.encoding_type == 'explicit':
                crop_mask = meinst_utils.mask_encode(
                    crop_mask, self.params.pca_params,
                    self.params.sigmoid, self.params.whiten)
            crop_mask = torch.from_numpy(crop_mask).float()
            inst_seg_tars.append(crop_mask)

        return inst_seg_tars

    def decode_inst_mask(self, box, en_mask_ts, image_size=None):
        en_mask_ts = meinst_utils.mask_decode(
            en_mask_ts, self.params_ts.pca_params,
            self.params_ts.sigmoid, self.params_ts.whiten)
        en_mask_ts = en_mask_ts.reshape(
            (1, 1) + self.params_ts.encoding_dim).float()
        x0, y0, x1, y1 = box
        en_mask_ts = F.interpolate(
            en_mask_ts, size=(y1 - y0, x1 - x0), mode='bilinear')
        en_mask_ts[en_mask_ts >= 0.5] = 1
        en_mask_ts = en_mask_ts.int()
        image_size = self.image_size if image_size is None else image_size
        inst_mask = torch.zeros(image_size).int()
        # first two dimension are 1 and therefore removing them...
        inst_mask[y0: y1, x0: x1] = en_mask_ts[0][0]

        return inst_mask


class InstSegTargetTransform:
    def __init__(self, cfg, image_size, **kwargs):
        self.cfg = cfg
        self.image_size = image_size

    def __call__(self, boxes, inst_masks):
        if inst_masks.ndim == 3:
            return [inst_masks[:, :, i] for i in
                    range(list(inst_masks.shape)[2])]
        else:
            return []

    def decode_inst_mask(self, box, en_mask_ts, image_size=None):
        # TODO: complete this...
        raise NotImplementedError
