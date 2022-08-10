import numpy as np
import pycocotools.mask as coco_mask_util
import torch
import torch.nn.functional as F

from encoding_custom.utils.inst_seg.meinst_utils import \
    ReadMEInstParams, mask_decode


def meinst_fcos_inference(inst_seg, instances, meinst_params):
    inst_preds = []
    for img_num in range(len(instances)):
        per_img_preds = []
        for ins in inst_seg:
            per_img_preds.append(ins[img_num].permute(1, 2, 0).reshape(
                -1, meinst_params.num_components))
        per_img_preds = torch.cat(per_img_preds, dim=0)
        inst_preds.append(per_img_preds)
    for img_num, (inst, ip) in enumerate(zip(instances, inst_preds)):
        if len(inst) == 0:
            continue
        selected_segs = ip[inst.box_loc_idx]

        decoded_masks = mask_decode(
            selected_segs, meinst_params.pca_params,
            meinst_params.sigmoid, meinst_params.whiten)
        decoded_masks = decoded_masks.reshape(
            (-1,) + meinst_params.encoding_dim)

        inst.remove("box_loc_idx")
        inst.pred_masks = decoded_masks

    return instances


def meinst_inference(instances, inst_seg, cfg):
    meinst_params = ReadMEInstParams(cfg)
    meinst_params.load_pca(is_ts=True)
    if cfg.MODEL.DET.HEAD_NAME == 'FCOS':
        instances = meinst_fcos_inference(inst_seg, instances, meinst_params)
    else:
        raise ValueError('Unknow detector')
    return instances


def get_inst_mask(bbox, pred_masks, img_h, img_w, cfg):
    x0, y0 = int(bbox[0]), int(bbox[1])
    x1, y1 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    inst_mask = np.zeros((img_h, img_w))

    if x1 - x0 > 0 and y1 - y0 > 0 and all(
                [i >= 0 for i in [x0, y0, x1, y1]]):
        bbox_mask = F.interpolate(
            pred_masks[None, None, :, :], (y1 - y0, x1 - x0),
            mode='bilinear', align_corners=False)[0, 0]
        bbox_mask.gt_(0.5)
        inst_mask[y0: y1, x0: x1] = bbox_mask.cpu().numpy()

    return np.asarray(inst_mask, dtype=np.uint8)


def encode_bin_mask(bin_mask):
    bin_mask = coco_mask_util.encode(np.array(bin_mask[:, :, None], order="F",
                                              dtype="uint8"))[0]
    bin_mask["counts"] = bin_mask["counts"].decode("utf-8")

    return bin_mask
