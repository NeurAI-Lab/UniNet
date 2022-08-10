import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, cfg, gamma=2.0, alpha=.25, size_average=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, ignore_index=-1):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        mask = target == ignore_index
        mask = mask.view(-1, 1)
        target[mask] = 0
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            self.alpha=self.alpha.type_as(input)
            select = (target != 0).long()
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * Variable(at)

        mask = mask.view(-1)
        logpt[mask] = 0
        pt[mask] = 0
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()