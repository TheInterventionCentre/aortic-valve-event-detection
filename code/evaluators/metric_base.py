import numpy as np
from abc import abstractmethod
import copy
import torch

class MetricBase():
    """ Abstract base class which all metrics and losses should inherit from. """
    def __init__(self, name, reduction, lower_is_better=True, requires_target=True):
        """
        Args:
            name: Name of the metric/loss
            reduction: How to treat the batch axis. Options, mean, sum, None
            lower_is_better: Decides whether or lower or higher value of the metric is better
            requires_target: Whether a target (label) is required for computing the metric.
        """

        self.results      = list()
        self.summary_tags = dict() # The summary_tags are used to track e.g. view (2CH/4CH), region (LV, LA), and
                                   # state (ED, ES). The summary_tags are hence used while creating the summary.
        self.lower_is_better = lower_is_better
        self.reduction = reduction
        self.name = name
        self.requires_target = requires_target
        return

    def __call__(self, pred, label, weight=None, summary_tags=None, **kwargs):
        try:
            if self.requires_target:
                if len(kwargs)==0:
                    batch_results = self.forward(pred, label)
                else:
                    batch_results = self.forward(pred, label, *kwargs.values())
            else:
                batch_results = self.forward(pred, *kwargs.values())
        except Exception as e:
            batch_results = torch.Tensor([float('NaN')])
            print(f'Error while computing {self.name}')
            print(e)
            raise ValueError('Error while computing loss function')

        self.results.extend(batch_results.detach())

        summary_copy = copy.deepcopy(summary_tags)
        for key in summary_tags.keys():
            if key in self.summary_tags.keys():
                self.summary_tags[key].extend(summary_copy[key])
            else:
                self.summary_tags[key] = summary_copy[key]

        weight = weight.to(batch_results.device)

        if self.reduction=='mean':
            if batch_results.ndim>1:
                out = (torch.unsqueeze(weight, dim=1) * batch_results).mean()
            else:
                out = (weight * batch_results).mean()
            # out = (weight*batch_results).mean()

        elif self.reduction=='sum':
            out = (weight * batch_results).sum()
        else:
            out = weight * batch_results
        return out

    def get_summary(self, reduction='mean'):
        """ Returns a dictionary of each loss case

        Args:
            reduction: how to reduce the metrics across the current epoch results. 'mean' | 'length' | None

        """
        summary_dict = {}
        for ii in range(len(self.results)):
            signature = self.summary_tags['signature'][ii]
            if signature not in summary_dict.keys():
                summary_dict[signature] = []
            summary_dict[signature].append(self.results[ii].cpu().numpy())

        # collaps by method
        if reduction is not None:
            for signature_key in summary_dict.keys():
                values = summary_dict[signature_key]
                if reduction == 'mean':
                    summary_dict[signature_key]= self.mean(values)
                elif reduction == 'length':
                    summary_dict[signature_key] = len(values)
        return summary_dict

    @abstractmethod
    def forward(self, pred, label, *kwargs):
        raise NotImplementedError
        # subclasses should implement this

    def epoch_reset(self):
        # reset results for next epoch
        self.results = list()
        self.summary_tags = dict()

    def absmean(self):
        if len(self.results) > 0:
            return np.mean(abs(np.array(self.results)))
        return None

    @staticmethod
    def mean(values):
        if len(values) > 0:
            mean = np.mean(np.array(values))
            if np.isnan(mean):
                print('nan values found, using nanmean')
                # logging.warning("Found NAN in results, using nanmean")
                mean = np.nanmean(np.array(values))
            return mean
        return None

    def median(self):
        if len(self.results) > 0:
            med = np.median(np.array(self.results))
            if np.isnan(med):
                # logging.warning("Found NAN in results, using nanmedian")
                med = np.nanmedian(np.array(self.results))
            return med
        return None

    def absmedian(self):
        if len(self.results) > 0:
            return np.median(abs(np.array(self.results)))
        return None

    def std(self):
        if len(self.results) > 0:
            return np.std(np.array(self.results))
        return None
