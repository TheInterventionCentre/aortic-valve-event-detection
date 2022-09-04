import torch
import numpy as np
from torch import nn

from evaluators.metric_base import MetricBase

########################################################################################################################
class Cross_entropy_loss(MetricBase):
    """ Wrapper around nn.BCELoss"""
    def __init__(self, name=None, reduction='mean'):
        super(Cross_entropy_loss, self).__init__(name, reduction, lower_is_better=False)
        """ Computes the cross entropy loss of an image of shape: batch_size x N x M

        args:
            pred: The predicted image of probabilities
            lab: The referance image
            reduce: how to combine batch size: 'mean', 'sum', 'None'
            balance_class: Whether to balance the class (can help if the region is small/large compare to the image size)

        return:
            loss
        """
        self.BCELoss       = nn.BCELoss(reduction='none')
        return

    def forward(self, pred, lab, weight=1):
        loss = self.BCELoss(pred, lab)
        if loss.ndim>1:
            dims = tuple(np.linspace(1, loss.ndim - 1, loss.ndim - 1).astype(int))
        return torch.mean(loss, dim=dims)

########################################################################################################################

class Regression(MetricBase):
    def __init__(self, norm='L1', name=None, reduction='mean'):
        super(Regression, self).__init__(name, reduction, lower_is_better=False)
        self.norm = norm
        return

    def forward(self, pred, label, patch_confidence=1.):
        if self.norm=='L1':
            loss = torch.abs(pred-label)*patch_confidence
        elif self.norm == 'L2':
            loss = (pred - label)**2 * patch_confidence
        else:
            raise ValueError('Unknown norm')
        return loss















