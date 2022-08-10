import json
import cv2
from cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
from cityscapesscripts.helpers.labels import name2label
from pycocotools.cocoeval import COCOeval
import tempfile
import sys
import os
import copy
from detectron2.evaluation.fast_eval_api import COCOeval_opt

from encoding_custom.utils.inst_seg.inference import get_inst_mask, encode_bin_mask
from utilities.train_utils import mkdir

detect_keys = ["key_metrics/detect_AP_box", "metrics/AP50_box", "metrics/AP75_box",
               "metrics/APs_box", "metrics/APm_box", "metrics/APl_box"]
inst_seg_keys = ["key_metrics/inst_seg_AP_mask", "metrics/AP50_mask", "metrics/AP75_mask",
                 "metrics/APs_mask", "metrics/APm_mask", "metrics/APl_mask"]


def process_predictions(predictions, image_idxs, cfg, dataset, inst_seg=False,
                        encode_inst_mask=True, mask_dir=None, txt_dir=None):
    all_results = []
    for idx, prediction in zip(image_idxs, predictions):
        boxes = prediction.pred_boxes.tensor.numpy().tolist()
        pred_classes = prediction.pred_classes.numpy().tolist()
        scores = prediction.scores.numpy().tolist()

        class_mapper = dataset.contiguous_id_to_coco_id
        image_id, img_info = dataset.get_img_info(idx)
        height = img_info['height'] / prediction.image_size[0]
        width = img_info['width'] / prediction.image_size[1]
        if len(pred_classes) == 0:
            continue

        instance_dir, img_name, txt_path, texts = None, None, None, None
        if mask_dir is not None:
            img_name = '_'.join(img_info['file_name'].split('_')[:-1])
            txt_path = os.path.join(txt_dir, f'{img_name}.txt')
            texts = []
            instance_dir = os.path.join(mask_dir, img_name)
            mkdir(instance_dir)

        per_img_results = []
        for k, box in enumerate(boxes):
            bbox = [box[0] * width, box[1] * height, (box[2] - box[0]) * width,
                    (box[3] - box[1]) * height]
            res = {"image_id": image_id,
                   "category_id": class_mapper[pred_classes[k]],
                   "bbox": bbox, "score": scores[k]}
            if inst_seg:
                pred_masks = prediction.pred_masks
                inst_mask = get_inst_mask(
                    bbox, pred_masks[k], img_info['height'], img_info['width'],
                    cfg)

                if mask_dir is not None:
                    instance_path = os.path.join(
                        instance_dir, 'instance' + str(k) + '.png')
                    score = res['score']
                    inst_name = dataset.INSTANCE_NAMES[res['category_id']]
                    cv2.imwrite(instance_path, inst_mask)
                    instance_path = os.path.join(
                        '../mask', img_name, 'instance' + str(k) + '.png')
                    texts.append(
                        f'{instance_path} {name2label[inst_name].id} {score}\n')

                if encode_inst_mask:
                    inst_mask = encode_bin_mask(inst_mask)
                res.update({"segmentation": inst_mask})
            per_img_results.append(res)
        all_results.append(per_img_results)

        if mask_dir is not None:
            with open(txt_path, 'w') as f:
                for txt in texts:
                    f.write(txt)

    return all_results


def calc_metrics(keys, instance_ids, coco_eval):
    metrics = {}
    coco_eval.params.catIds = instance_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    for i, key in enumerate(keys):
        metrics.update({key: coco_eval.stats[i]})

    return metrics


def coco_evaluation(dataset, results, inst_seg=False, use_fast_eval=True,
                    per_class=False, instance_names=None):
    def update_per_cls(name_prefix, task_eval):
        sys.stdout = open(os.devnull, 'w')
        for inst_id, inst_name in zip(instance_ids, instance_names):
            metrics.update(
                calc_metrics([f'{name_prefix}{inst_name}'],
                             [inst_id], task_eval))
        sys.stdout = sys.__stdout__

    instance_ids = dataset.get_detect_ids()
    # Ignore background.. only use the instance ids for evaluation..
    instance_ids = instance_ids[1:]
    instance_names = instance_names[1:]
    evaluator = COCOeval_opt if use_fast_eval else COCOeval
    temp_f = tempfile.NamedTemporaryFile(suffix='.json')
    with open(temp_f.name, "w") as f:
        json.dump(results, f)

    metrics = {}
    if len(results) != 0:
        coco_gt = copy.deepcopy(dataset.coco)
        coco_dt = coco_gt.loadRes(temp_f.name)
        detect_eval = evaluator(coco_gt, coco_dt, 'bbox')
        metrics.update(calc_metrics(detect_keys, instance_ids, detect_eval))
        if per_class:
            update_per_cls('classwise/AP_box_', detect_eval)

        if inst_seg:
            inst_seg_eval = evaluator(coco_gt, coco_dt, 'segm')
            metrics.update(calc_metrics(inst_seg_keys, instance_ids,
                                        inst_seg_eval))
            if per_class:
                update_per_cls('classwise/AP_mask_', inst_seg_eval)
    else:
        # ensure detect and inst seg model saving happens..
        for key in detect_keys + inst_seg_keys:
            metrics.update({key: 0})

    return metrics


def cityscapes_inst_eval(dataset, result_dir, txt_dir, split='val'):
    prediction = [os.path.join(txt_dir, txt) for txt in os.listdir(txt_dir)]
    gt = []
    for pred in prediction:
        txt = pred.split('/')[-1]
        img_name = txt.split('.')[0]
        city = img_name.split('_')[0]
        gt.append(os.path.join(
            dataset.root, 'gtFine', split, city,
            img_name + '_gtFine_instanceIds.png'))

    args = evalInstanceLevelSemanticLabeling.args
    args.predictionPath = os.path.abspath(result_dir)
    res_dict = evalInstanceLevelSemanticLabeling.evaluateImgLists(
        prediction, gt, args)
    return {"key_metrics/inst_seg_AP_mask": res_dict['averages']['allAp'],
            "metrics/AP50_mask": res_dict['averages']['allAp50%']}
