from torch import nn
import torch
import torch.nn.functional as F

__all__ = ['DepthL1Loss', 'RMSE']


class DepthL1Loss(nn.Module):
    def __init__(self, cfg, loss=True, **kwargs):
        super(DepthL1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target, unsqueeze=True):
        if unsqueeze:
            target = target.unsqueeze(1)

        if not pred.shape == target.shape:
            print(pred.shape)
            print(target.shape)
            _, _, H, W = target.shape
            pred = F.upsample(pred, size=(H, W), mode='bilinear')
        mask = torch.where(target > 0)
        return self.loss(pred[mask], target[mask])


class RMSE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(RMSE, self).__init__()

    def forward(self, pred, target, unsqueeze=True):
        if unsqueeze:
            target = target.unsqueeze(1)

        if not pred.shape == target.shape:
            print(pred.shape)
            print(target.shape)
            _, _, H, W = target.shape
            pred = F.upsample(pred, size=(H, W), mode='bilinear')

        mask = torch.where(target > 0)
        loss = torch.sqrt(torch.mean(torch.abs(target[mask] - pred[mask]) ** 2))
        return loss
