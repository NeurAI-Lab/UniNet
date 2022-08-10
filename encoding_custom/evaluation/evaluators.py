from abc import abstractmethod
import logging
import torch
from torch import nn
import torch.distributed
from detectron2.utils import comm
import os
import math

from utilities import dist_utils
from encoding_custom.evaluation import *
from encoding_custom.losses.depth_losses import RMSE
from encoding_custom.losses.aux_losses import extract_semantic_contours,\
    extract_surface_normals
from utilities.metric_utils import SmoothedValue
from utilities.train_utils import mkdir


def get_mtl_evaluator(cfg, tasks, dataset_val, root_dir):
    mtl_evaluator = MultiTaskEvaluator(cfg)
    evaluators = {}
    if 'detect' in tasks:
        evaluators.update({'detect': DetectEvaluator(
            cfg, dataset_val, tasks, root_dir)})
    if 'segment' in tasks:
        evaluators.update({'segment': SegmentEvaluator(cfg)})
    if 'depth' in tasks:
        evaluators.update({'depth': DepthEvaluator(cfg)})
    if 'sem_cont' in tasks:
        evaluators.update({'sem_cont': SemanticContourEvaluator(cfg)})
    if 'sur_nor' in tasks:
        evaluators.update({'sur_nor': SurfaceNormalsEvaluator(cfg, dataset_val)})

    mtl_evaluator.evaluators = evaluators

    return mtl_evaluator


class BaseEvaluator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.log_per_cls_metrics = cfg.MISC.LOG_PER_CLASS_METRICS
        self.instance_names = cfg.INSTANCE_NAMES
        self.semantic_names = cfg.SEMANTIC_NAMES

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def process(self, targets, predictions, image_idxs):
        pass

    @abstractmethod
    def reduce_from_all_processes(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class MultiTaskEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        # based on detectron2 DatasetEvaluators
        super(MultiTaskEvaluator, self).__init__(cfg)
        self.aux_tasks = cfg.MISC.AUX_TASKS
        self._evaluators = {}

    @property
    def evaluators(self):
        return self._evaluators

    @evaluators.setter
    def evaluators(self, evaluators_dict):
        self._evaluators = evaluators_dict

    def reset(self):
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def process(self, targets, predictions, image_idxs):
        for task, evaluator in self.evaluators.items():
            if task in self.aux_tasks:
                evaluator.process(targets, predictions, image_idxs)
            else:
                evaluator.process(targets[task], predictions[task], image_idxs)

    def evaluate(self):
        self.reduce_from_all_processes()
        results = {}
        for evaluator in self.evaluators.values():
            result = evaluator.evaluate()
            if dist_utils.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (k not in results), \
                        "Different evaluators produce results with " \
                        "the same key {}".format(k)
                    results[k] = v
        return results

    def reduce_from_all_processes(self):
        for evaluator in self.evaluators.values():
            evaluator.reduce_from_all_processes()


class DetectEvaluator(BaseEvaluator):

    def __init__(self, cfg, dataset_val, tasks, root_dir):
        super(DetectEvaluator, self).__init__(cfg)
        self.dataset_val = dataset_val
        self.inst_seg = 'inst_seg' in tasks
        self.detect_preds = {}
        self.citys_inst_seg_eval = cfg.MISC.CITYS_INST_SEG_EVAL
        self.result_dir = None
        self.mask_dir = None
        self.txt_dir = None
        if self.citys_inst_seg_eval:
            self.result_dir = os.path.join(root_dir, 'cityscapes_eval')
            self.mask_dir = os.path.join(self.result_dir, 'mask')
            self.txt_dir = os.path.join(self.result_dir, 'text')
            mkdir(self.mask_dir)
            mkdir(self.txt_dir)

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        boxlists = [bl.to(torch.device("cpu")) for bl in predictions]
        processed = process_predictions(
            boxlists, image_idxs, self.cfg, self.dataset_val,
            inst_seg=self.inst_seg, mask_dir=self.mask_dir, txt_dir=self.txt_dir)
        self.detect_preds.update({i: pred for i, pred
                                  in zip(image_idxs, processed)})

    def reduce_from_all_processes(self):
        all_predictions = comm.all_gather(self.detect_preds)
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        image_idxs = list(sorted(predictions.keys()))
        if len(image_idxs) == 0:
            logging.warning("No detections predictions after nms...")
        elif len(image_idxs) != image_idxs[-1] + 1:
            logging.warning("Number of images that were gathered from multiple"
                            " processes is not a contiguous set. Some images "
                            "might be missing from the evaluation")
        self.detect_preds = [predictions[i] for i in image_idxs]
        self.detect_preds = sum(self.detect_preds, [])

    def evaluate(self):
        result = coco_evaluation(
            self.dataset_val, self.detect_preds,
            inst_seg=self.inst_seg and not self.citys_inst_seg_eval,
            per_class=self.log_per_cls_metrics,
            instance_names=self.instance_names)
        if self.citys_inst_seg_eval:
            inst_result = cityscapes_inst_eval(
                self.dataset_val, self.result_dir, self.txt_dir)
            result.update(inst_result)
        return result


class SegmentEvaluator(BaseEvaluator):

    def __init__(self, cfg, num_classes=None):
        super(SegmentEvaluator, self).__init__(cfg)
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = cfg.NUM_CLASSES.SEGMENT
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix.zero_()

    def process(self, targets, predictions, image_idxs):
        tars_flat = targets.flatten()
        preds_flat = predictions.argmax(1).flatten()
        n = self.num_classes
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(
                (n, n), dtype=torch.int64, device=tars_flat.device)
        with torch.no_grad():
            k = (tars_flat >= 0) & (tars_flat < n)
            inds = n * tars_flat[k].to(torch.int64) + preds_flat[k]
            self.confusion_matrix += torch.bincount(
                inds, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.confusion_matrix.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.confusion_matrix)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\n average row correct: {}\n"
                "IoU: {}\n mean IoU: {:.1f}").format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

    def evaluate(self):
        acc_global, acc, iu = self.compute()
        acc_global = acc_global.item()
        mean_iou = iu.mean().item() * 100
        class_iu = (iu * 100).tolist()
        result = {'metrics/segment_pixel_acc': acc_global,
                  'key_metrics/segment_MIoU': mean_iou}
        if self.log_per_cls_metrics:
            for sem, c_iu in zip(self.semantic_names, class_iu):
                result.update({f'classwise/iou_{sem}': c_iu})
        return result


class DepthEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super(DepthEvaluator, self).__init__(cfg)
        self.min_depth = cfg.DATALOADER.MIN_DEPTH
        self.max_depth = cfg.DATALOADER.MAX_DEPTH
        self.rmse_fn = RMSE(cfg)
        self.rmse = SmoothedValue()
        self.abs_rel = SmoothedValue()
        self.abs_err = SmoothedValue()
        self.sq_rel = SmoothedValue()
        self.rmse_log = SmoothedValue()
        self.a1 = SmoothedValue()
        self.a2 = SmoothedValue()
        self.a3 = SmoothedValue()

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        # scale back to range 0 to max_depth from 0 to 1..
        targets = self.max_depth * targets
        predictions = self.max_depth * predictions
        rmse_value = self.rmse_fn(predictions, targets)
        self.rmse.update(rmse_value)

        mask = torch.where(targets >= 0)
        predictions = torch.squeeze(predictions, dim=1)
        (abs_rel, abs_err, sq_rel, rmse_log, a1, a2, a3) = compute_depth_errors(
            targets[mask], predictions[mask])
        self.abs_rel.update(abs_rel)
        self.abs_err.update(abs_err)
        self.sq_rel.update(sq_rel)
        self.rmse_log.update(rmse_log)
        self.a1.update(a1)
        self.a2.update(a2)
        self.a3.update(a3)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        return {'key_metrics/depth_rmse': self.rmse.global_avg,
                'metrics/depth_abs_rel': self.abs_rel.global_avg,
                'metrics/depth_abs_err': self.abs_err.global_avg,
                'metrics/depth_sq_rel': self.sq_rel.global_avg,
                'metrics/depth_rmse_log': self.rmse_log.global_avg,
                'metrics/depth_a1': self.a1.global_avg,
                'metrics/depth_a2': self.a2.global_avg,
                'metrics/depth_a3': self.a3.global_avg}


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.where((gt / pred) > (pred / gt), (gt / pred), (pred / gt))
    a1 = thresh[(thresh < 1.25)].mean()
    a2 = thresh[(thresh < 1.25 ** 2)].mean()
    a3 = thresh[(thresh < 1.25 ** 3)].mean()

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    abs_err = torch.mean(torch.abs(gt - pred))

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, abs_err, sq_rel, rmse_log, a1, a2, a3


class SemanticContourEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super(SemanticContourEvaluator, self).__init__(cfg)
        self.sem_consist = SmoothedValue()
        self.consist_loss = nn.BCEWithLogitsLoss()
        self.num_seg_classes = cfg.NUM_CLASSES.SEGMENT
        self.multiclass = cfg.MISC.SEM_CONT_MULTICLASS

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        convert_cont = extract_semantic_contours(
            predictions['segment'].argmax(1).clone(),
            self.num_seg_classes, multi_class=self.multiclass)
        consistency_loss = self.consist_loss(
            predictions['sem_cont'], convert_cont)
        self.sem_consist.update(consistency_loss)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        # TODO: add semantic contours metrics...
        return {'metrics/semantic_consistency': self.sem_consist.global_avg}


class SurfaceNormalsEvaluator(BaseEvaluator):

    def __init__(self, cfg, dataset_val):
        super(SurfaceNormalsEvaluator, self).__init__(cfg)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.geo_consist = SmoothedValue()
        self.cos_sim = SmoothedValue()
        self.nor_mean = SmoothedValue()
        self.nor_rmse = SmoothedValue()
        self.nor_11_25 = SmoothedValue()
        self.nor_22_5 = SmoothedValue()
        self.nor_30 = SmoothedValue()
        self.dataset_val = dataset_val

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        intrinsics = targets.get('intrinsics', None)
        if intrinsics is None:
            intrinsics = self.dataset_val.K.copy()
            intrinsics = torch.from_numpy(intrinsics)[None]
            intrinsics = intrinsics.repeat(predictions['sur_nor'].size(0), 1, 1)
            intrinsics = intrinsics.to(predictions['sur_nor'].device)

        if targets.get('sur_nor', None) is not None:
            target_nor = targets['sur_nor']
        else:
            target_nor = extract_surface_normals(
                targets['depth'][:, None, :, :], intrinsics)

        l2_norm = torch.sqrt(
                target_nor[:, 0, ...] ** 2 + target_nor[:, 1, ...] ** 2 +
                target_nor[:, 2, ...] ** 2)
        valid_mask = l2_norm != 0
        invalid_mask = 1 - valid_mask.long()
        pred = predictions['sur_nor'].permute(0, 2, 3, 1)
        pred[invalid_mask == 1, :] = 0
        pred = pred.permute(0, 3, 1, 2)

        convert_nor = extract_surface_normals(
            predictions['depth'], intrinsics)
        consistency_loss = self.cosine_similarity(pred, convert_nor)
        self.geo_consist.update(consistency_loss.mean())
        cos_sim = self.cosine_similarity(pred, target_nor)
        self.cos_sim.update(cos_sim.mean())

        deg_diff = (180 / math.pi) * (torch.acos(torch.clamp(torch.sum(
            pred * target_nor, 1), min=-1, max=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        nor_mean, nor_rmse, nor_11_25, nor_22_5, nor_30 = compute_normal_errors(
            deg_diff)
        self.nor_mean.update(nor_mean)
        self.nor_rmse.update(nor_rmse)
        self.nor_11_25.update(nor_11_25)
        self.nor_22_5.update(nor_22_5)
        self.nor_30.update(nor_30)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        return {'metrics/geometric_consistency': self.geo_consist.global_avg,
                'metrics/mean': self.nor_mean.global_avg,
                'metrics/rmse': self.nor_rmse.global_avg,
                'metrics/11.25': self.nor_11_25.global_avg,
                'metrics/22.5': self.nor_22_5.global_avg,
                'metrics/30': self.nor_30.global_avg,
                'key_metrics/sur_nor_cosine_similarity': self.cos_sim.global_avg}


def compute_normal_errors(deg_diff):
    n = deg_diff.numel()
    nor_mean = torch.sum(deg_diff).item() / n
    nor_rmse = torch.sum(torch.sqrt(torch.pow(deg_diff, 2))).item() / n
    nor_11_25 = torch.sum((deg_diff < 11.25).float()).item() * 100 / n
    nor_22_5 = torch.sum((deg_diff < 22.5).float()).item() * 100 / n
    nor_30 = torch.sum((deg_diff < 30).float()).item() * 100 / n

    return nor_mean, nor_rmse, nor_11_25, nor_22_5, nor_30
