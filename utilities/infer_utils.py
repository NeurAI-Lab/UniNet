import random
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
from detectron2.structures import ImageList

from encoding_custom.utils.detect.inference import get_inferer
from encoding_custom.utils.inst_seg.inference import meinst_inference
from utilities import train_utils


class BatchCollator:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.size_divisibility = cfg.DATALOADER.SIZE_DIVISIBILITY

    def __call__(self, batch):
        images = ImageList.from_tensors(
            batch, size_divisibility=self.size_divisibility)
        return images.tensor


def run_inference(predictions, cfg, tasks):
    # TODO: clean up this method...
    detect_inference = None
    if 'detect' in tasks:
        detect_inference = get_inferer(cfg, tasks)
    boxes = None
    if 'detect' in predictions.keys():
        inst_seg_preds = None
        detect_pred_idx = 0
        if 'inst_seg' in tasks:
            inst_seg_preds = predictions['detect'][-1]
            detect_pred_idx -= 1
        inst_depth_preds = None
        if detect_inference.inst_depth:
            detect_pred_idx -= 1
            index = -1 if inst_seg_preds is None else -2
            inst_depth_preds = predictions['detect'][index]
        detect_preds = predictions['detect']
        if detect_pred_idx < 0:
            detect_preds = detect_preds[:detect_pred_idx]
        boxes = detect_inference(
            *detect_preds, inst_depth_preds=inst_depth_preds)
        if 'inst_seg' in tasks:
            boxes = meinst_inference(boxes, inst_seg_preds, cfg)

    return boxes, predictions


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Multitask Inference')
    parser.add_argument(
        "--config-file", default="", metavar="FILE",
        help="path to config file", type=str)

    # dataset...
    parser.add_argument(
        '--dataset', default='uninet_cs', help='dataset')
    parser.add_argument(
        '--split', default='test',
        help='dataset split; use infer for inferring from folder')
    parser.add_argument(
        '--data-folder', type=str, default='data/',
        help='training dataset folder')

    # inference params...
    parser.add_argument(
        '--device', default='cuda', help='device')
    parser.add_argument(
        '--crop-size', type=int, default=[512, 512], nargs="*",
        help='crop image size')
    parser.add_argument(
        '-j', '--workers', default=16, type=int, metavar='N',
        help='number of data loading workers (default: 16)')
    parser.add_argument(
        '-b', '--batch-size', default=1, type=int)

    # misc...
    parser.add_argument(
        '--output-dir', default='./runs', help='path where to save')
    parser.add_argument(
        '--resume', default=None, help='resume from checkpoint')
    parser.add_argument(
        '--function-name', default="measure_fps", type=str,
        help='function to call')
    parser.add_argument(
        "--test-it", default=0, type=int,
        help="number of iterations for turnaround measure")
    parser.add_argument(
        '--save-path', default=None, type=str,
        help='img save path for turnaround measure')

    args = parser.parse_args()

    return args


def setup_infer_model(args, cfg, seed=0, num_workers=0, batch_size=1,
                      collate_fn=None, cfg_hook=None):
    # create attributes to use functions from train.py
    args.distributed = False
    args.backbone = False
    args.resume_after_suspend = False
    args.log_per_class_metrics = False
    args.local_rank = 0
    args.pretrained = False

    cfg = train_utils.update_config_node(cfg, args)
    if cfg_hook is not None:
        cfg_hook()
    cfg.freeze()

    if seed >= 0:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)

    data_kwargs = {'base_size': None, 'crop_size': args.crop_size,
                   'root': args.data_folder, 'cfg': cfg}
    dataset = train_utils.get_multitask_dataset(
        args.dataset, split=args.split, **data_kwargs)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    dl = torch.utils.data.DataLoader(
        dataset, sampler=sampler, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, batch_size=batch_size,
        drop_last=False)

    device = torch.device(args.device)

    norm_layer = train_utils.get_norm_layer(cfg, args)
    tasks = [task for task, status in dict(cfg.TASKS_DICT).items() if status]
    model = train_utils.get_model(cfg, args, norm_layer, tasks)
    model.to(device)
    if args.resume is not None:
        train_utils.load_model(args, model, None)
    model.eval()

    return model, tasks, dl, device


def forward_transform(image, cfg, scale=1.):
    image = image / scale
    image[:, 0, :, :] = (image[:, 0, :, :] -
                         cfg.DATALOADER.MEAN[0]) / cfg.DATALOADER.STD[0]
    image[:, 1, :, :] = (image[:, 1, :, :] -
                         cfg.DATALOADER.MEAN[1]) / cfg.DATALOADER.STD[1]
    image[:, 2, :, :] = (image[:, 2, :, :] -
                         cfg.DATALOADER.MEAN[2]) / cfg.DATALOADER.STD[2]
    return image


def back_transform(image, cfg, scale=1.):
    image[:, 0, :, :] = (image[:, 0, :, :] *
                         cfg.DATALOADER.STD[0]) + cfg.DATALOADER.MEAN[0]
    image[:, 1, :, :] = (image[:, 1, :, :] *
                         cfg.DATALOADER.STD[1]) + cfg.DATALOADER.MEAN[1]
    image[:, 2, :, :] = (image[:, 2, :, :] *
                         cfg.DATALOADER.STD[2]) + cfg.DATALOADER.MEAN[2]
    return image * scale
