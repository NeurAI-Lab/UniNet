import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from encoding_custom.losses import *


class MTLLoss(nn.Module):

    def __init__(self, cfg, tasks_dict, n_epochs=1, batch_size=1):
        super(MTLLoss, self).__init__()
        self.batch_size = batch_size
        self.cfg = cfg

        task_to_loss_name = dict(cfg.TASK_TO_LOSS_NAME)
        task_to_loss_args = dict(cfg.TASK_TO_LOSS_ARGS)
        task_to_loss_kwargs = dict(cfg.TASK_TO_LOSS_KWARGS)
        self.task_to_call_kwargs = dict(cfg.TASK_TO_LOSS_CALL_KWARGS)

        task_to_loss_fn = {'detect': {'default': FCOSLossComputation,
                                      'fcos_loss': FCOSLossComputation},
                           'segment': {'default': ClassBalancedSegmentationLosses,
                                       'balanced': ClassBalancedSegmentationLosses,
                                       'cross_entropy': cross_entropy_loss},
                           'depth': {'default': RMSE, 'rmse': RMSE},
                           'inst_depth': {'default': None},
                           'inst_seg': {'default': None},
                           'sem_cont': {'default': SemanticContLoss,
                                        'binary_ce': BalancedBinaryCrossEntropyLoss},
                           'sur_nor': {'default': NormalsCosineLoss,
                                       'l1_loss': NormalsL1Loss}}

        self.tasks = [task for task, status in tasks_dict.items() if status]
        # all losses have access to what tasks are predicted..
        for task in tasks_dict.keys():
            if task in task_to_loss_kwargs.keys():
                task_to_loss_kwargs[task].update(tasks_dict)
            task_to_loss_kwargs.update({task: tasks_dict})

        self.task_to_fn = {}
        for task, status in tasks_dict.items():
            if not status:
                continue
            loss_name = task_to_loss_name.get(task, 'default')
            loss_fn = task_to_loss_fn[task][loss_name]
            if loss_fn is None:
                continue
            self.task_to_fn.update({task: loss_fn(
                cfg, *task_to_loss_args.get(task, []),
                **task_to_loss_kwargs.get(task, {}))})

        self.loss_to_weights = dict(cfg.LOSS_INIT_WEIGHTS)
        self.loss_to_weights = {'losses/' + key: value for key, value
                                in self.loss_to_weights.items()}
        self.loss_to_start = dict(cfg.LOSS_START_EPOCH)
        self.loss_to_start = {'losses/' + key: value for key, value
                              in self.loss_to_start.items()}

        self.balancing_method = cfg.TASKS.BALANCING_METHOD
        self.dyn_loss_idx = {loss: idx for idx, loss in
                             enumerate(self.loss_to_weights.keys())}

    @staticmethod
    def get_losses(loss_dict):
        losses = []
        loss_names = []
        for loss_name, curr_loss in loss_dict.items():
            if 'losses' not in loss_name:
                continue
            losses.append(curr_loss)
            loss_names.append(loss_name)
        return torch.stack(losses), loss_names

    def run_backprop(self, loss, model, loss_dict):
        loss.mean().backward()

    def handcrafted_balancing(self, losses_to_add):
        loss = 0
        for loss_name, curr_loss in losses_to_add.items():
            curr_loss = curr_loss * self.loss_to_weights.get(loss_name, 1)
            loss = loss + curr_loss
            losses_to_add[loss_name] = curr_loss

        return loss, losses_to_add

    def loss_balancing(self, losses_to_add, epoch):
        if self.balancing_method == 'geometric':
            loss = 1.
            for loss_name, curr_loss in losses_to_add.items():
                curr_loss = curr_loss * self.loss_to_weights.get(loss_name, 1)
                loss = loss * curr_loss
                losses_to_add[loss_name] = curr_loss
            loss = loss ** (1/len(self.loss_to_weights))
        else:
            loss, losses_to_add = self.handcrafted_balancing(losses_to_add)

        return loss, losses_to_add

    @staticmethod
    def check_losses(task, task_losses, loss_dict, losses_to_add):
        if type(task_losses) is not dict:
            task_losses = {task: {f'losses/{task}_loss': task_losses}}

        for losses in task_losses.values():
            for loss_name, loss_ts in losses.items():
                if 'losses/' in loss_name:
                    losses_to_add.update({loss_name: loss_ts})
                else:
                    loss_dict.update({loss_name: loss_ts})

        return loss_dict, losses_to_add

    def forward_padnet(self, predictions, targets, epoch=1):
        loss_dict = {}
        losses_to_add = {}
        for task, pred in predictions.items():
            if 'initial_' in task:
                task = task.replace('initial_', '')
                fn = self.task_to_fn[task]
                fn_args = self.task_to_call_kwargs.get(task, {})
                tar = targets.get(task, None)
                loss_name = f'losses/{task}_initial_loss'
                if tar is not None:
                    tar = tar.clone()
                    tar = tar[:, None, :, :] if tar.ndim == 3 else tar
                    tar = F.interpolate(
                        tar.float(), pred.shape[2:], mode='nearest').squeeze(1)
                    task_loss = fn(pred, tar, **fn_args)
                else:
                    task_loss = fn(pred, targets, **fn_args)
                losses_to_add.update({loss_name: task_loss})
            elif 'sem_cont' in task or 'sur_nor' in task:
                # skip final predictions of sem cont and sur nor...
                continue
            else:
                fn = self.task_to_fn[task]
                task_losses = fn(pred, targets[task],
                                 **self.task_to_call_kwargs.get(task, {}))
                if task_losses is None:
                    continue
                loss_dict, losses_to_add = self.check_losses(
                    task, task_losses, loss_dict, losses_to_add)

        loss, losses_to_add = self.loss_balancing(losses_to_add, epoch)
        loss_dict.update(losses_to_add)
        return loss, loss_dict

    def forward_mtinet(self, predictions, targets, epoch=1):
        loss_dict = {}
        losses_to_add = {}
        img_size = self.cfg.INPUT.IMAGE_SIZE
        dense_tasks = ['segment', 'depth', 'sem_cont', 'sur_nor']
        for task, pred in predictions.items():
            if 'deep_supervision' in task:
                for scale, pred_scale in pred.items():
                    pred_scale = {t: F.interpolate(
                        pred_scale[t], img_size, mode='bilinear')
                        for t in self.tasks if t in dense_tasks}
                    for t, p_scale in pred_scale.items():
                        fn_args = self.task_to_call_kwargs.get(task, {})
                        fn = self.task_to_fn[t]
                        loss_name = f'losses/{t}_{scale}_loss'
                        if t in targets.keys():
                            task_loss = fn(p_scale, targets[t], **fn_args)
                        else:
                            task_loss = fn(p_scale, targets, **fn_args)
                        losses_to_add.update({loss_name: task_loss})
            elif 'sem_cont' in task or 'sur_nor' in task:
                # skip final predictions of sem cont and sur nor...
                continue
            else:
                fn = self.task_to_fn[task]
                task_losses = fn(pred, targets[task],
                                 **self.task_to_call_kwargs.get(task, {}))
                if task_losses is None:
                    continue
                loss_dict, losses_to_add = self.check_losses(
                    task, task_losses, loss_dict, losses_to_add)

        loss, losses_to_add = self.loss_balancing(losses_to_add, epoch)
        loss_dict.update(losses_to_add)
        return loss, loss_dict

    def forward(self, predictions, targets, epoch=1):
        if self.cfg.MODEL.NAME == 'padnet':
            return self.forward_padnet(predictions, targets, epoch)
        if self.cfg.MODEL.NAME == 'mtinet':
            return self.forward_mtinet(predictions, targets, epoch)
        loss_dict = {}
        losses_to_add = {}
        for task, fn in self.task_to_fn.items():
            task_losses = fn(predictions[task], targets.get(task, targets),
                             **self.task_to_call_kwargs.get(task, {}))
            if task_losses is None:
                continue
            loss_dict, losses_to_add = self.check_losses(
                task, task_losses, loss_dict, losses_to_add)

        loss, losses_to_add = self.loss_balancing(losses_to_add, epoch)
        loss_dict.update(losses_to_add)
        return loss, loss_dict
