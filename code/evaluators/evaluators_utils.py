from utils.loader import load_class
import torch

class Losses_and_metrics():
    """ The class connects the signatures (e.g rnn_1d_conf_ed, cnn_1d_conf_ed) from cfg['model']['net1]['output_list_indices']
     together with targets (found from the dataloader) for the various losses and metrics.

     """
    def __init__(self, in_dict, mode='training', optimizer=None, is_loss=False):
        self.is_loss = is_loss
        self.in_dict = in_dict
        self.optimizer = optimizer
        self.mode = mode
        self.metric_dict = {}
        for key in in_dict.keys():
            self.metric_dict[key] = {}
            self.metric_dict[key]['class']      = load_class(self.in_dict[key])
            self.metric_dict[key]['signatures'] = self.in_dict[key].signatures
            self.metric_dict[key]['weight']     = self.in_dict[key].weight
            self.metric_dict[key]['calculate_during_train'] = self.in_dict[key].calculate_during_train
            self.metric_dict[key]['additional_inputs'] = self.in_dict[key].additional_inputs
        return

    def __call__(self, dataDict, predDict):
        """
         Args:
             dataDict: The dataDict from the dataloader
             predDict: The output dict from the model class.

         Returns:
            A dict with the average value of the metrics/losses.

         """
        out_dict = {}
        for metric_key in self.metric_dict.keys():
            for signature in self.metric_dict[metric_key]['signatures']:
                for module_key in predDict.keys():
                    label = dataDict[signature[0]]
                    pred = predDict[module_key][signature[1]]['value']
                    module_weight = predDict[module_key][signature[1]]['weight']

                    loss_weight      = self.metric_dict[metric_key]['weight']
                    weight = torch.tensor(module_weight * loss_weight)*dataDict['weights']
                    tag = f'{metric_key}_{module_key}_{signature[1]}'

                    batch_size   = pred.shape[0]
                    summary_tags = {'signature': [signature[1]]*batch_size}

                    #Compute metric only if in evaluation mode or 'calculate_during_train' is True
                    if self.mode == 'training':
                        if self.metric_dict[metric_key]['calculate_during_train'] is False:
                            continue

                    if self.metric_dict[metric_key]['additional_inputs'] is not None:
                        additional_inputs = {k:v for k,v in dataDict.items()
                                             if k in self.metric_dict[metric_key]['additional_inputs']}
                        out_dict[tag] = self.metric_dict[metric_key]['class'](pred, label, weight,
                                                                              summary_tags, **additional_inputs)
                    else:
                        out_dict[tag] = self.metric_dict[metric_key]['class'](pred, label, weight, summary_tags)

        if self.is_loss is not False:
            # out_dict['totalLoss'] = sum(torch.tensor(list(out_dict.values())) * dataDict['weights'])
            out_dict['totalLoss'] = sum(out_dict.values())

        #add loss value corresponding to the weight decay to the "loss dict" for visualization
        if self.optimizer is not None:
            out_dict['weight_decay'] = get_weight_decay_loss_from_optimizer(self.optimizer)
        return out_dict

    def get_loss_summary(self, reduction='mean'):
        summary_dict = {}
        for key in self.metric_dict.keys():
            summary_dict[key] = self.metric_dict[key]['class'].get_summary(reduction)
        return summary_dict

    def epoch_reset(self):
        for key in self.metric_dict.keys():
            self.metric_dict[key]['class'].epoch_reset()
        return


def get_weight_decay_loss_from_optimizer(optimizer):
    l2_reg = 0
    # numb_of_parameters = 0
    for param in optimizer.param_groups[0]['params']:
        l2_reg += param.pow(2).sum()
    return optimizer.param_groups[0]['weight_decay']*l2_reg
