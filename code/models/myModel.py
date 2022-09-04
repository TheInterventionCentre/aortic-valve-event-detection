from torch import nn
import torch
import torch.nn.functional as F
import collections
import numpy as np
from utils.loader import load_class
from pprint import pprint

class Model(nn.Module):
    """ A wrapper class around the neural network (nn.Module).

    - The network is specified within the config (cfg) dict by cfg['model']['net1]['type']
    - How the output tensor(s) are used is defined by cfg['model']['net1]['output_list_indices']

    Example of 'output_list_indices':

    output_list_indices:
        0:         # list index in "out_list"
            order:
                 [
                 [['LA', 'LV', 'MYO', 'background'], 'C'],
                 [['basal_left'], 'WH'],
                 [['basal_right'], 'WH']
                 ]
            weight: 1
            tag: 'final'
        1:         # list index in "out_list"
            order:
                 [
                 [['2CH', '4CH', 'APLAX'], 'C'],
                 ]
            weight: 1
            tag: 'final'

    Consider that the network defined by cfg['model']['net1]['type'] returns two tensors.
    The keys '0', '1' specifies the index in the list of tensors. Lets say the first tensor has
    shape of [batch=16, ch=6, Ny=128, Nx=128]. The 'order' specifies that the first four channels are used to
    output predictions of ['LA', 'LV', 'MYO', 'background']. The 'C' character within the list specifies that the
    softmax function should be applied along the channel dimension. For the next two elements in the 'order' list,
    the softmax function should be applied across the image's width and height (WH) dimension.
    The second tensor can, for instance, have a shape of [batch=16, ch=3], and again the softmax function will be applied
    along the channel dimension.
      """

    def __init__(self, cfg, to_ONNX=False):
        super(Model, self).__init__()
        self.to_ONNX = to_ONNX
        self.cfg = cfg
        self.start_epoch   = 0
        self.update_steps  = 0

        self.module_dict = collections.OrderedDict()
        for key in cfg.model.keys():
            self.module_dict[key] = load_class(cfg.model[key])
        self.nets = nn.ModuleList(self.module_dict.values())
        return

    def forward(self, input_data):
        output = input_data
        predDict = {}
        for key in self.cfg.model.keys():
            output = self.module_dict[key].forward(output)

            #hardcoded to ONNX. NB will output the first model key
            if self.to_ONNX:
                return output

            if isinstance(output, (list, tuple)):
                output_list = output
            else:
                output_list = [output]

            # Go through and matches the "order" keys with the losses and metrics signature keys.
            output_list_indices = self.cfg.model[key].output_list_indices
            for index_key in sorted(output_list_indices.keys()):
                order_list = output_list_indices[index_key]['order']
                output_tag = output_list_indices[index_key]['tag']
                weight     = output_list_indices[index_key]['weight']
                index_start = 0
                if output_tag not in predDict.keys():
                    predDict[output_tag] = {}
                for group in order_list:
                    indices = tuple(np.arange(0, len(group[0]), 1)+index_start)
                    index_start = index_start + len(group[0])
                    if group[1]=='C': #softmax over channels
                        after_activation = F.softmax(output_list[index_key][:,  indices], dim=1)
                    elif group[1]=='WH': #softmax over width and height
                        assert len(indices)==1
                        N, C, Ny, Nx = output_list[index_key].shape
                        after_activation = F.softmax(output_list[index_key][:,indices].view((N, -1)), dim=1).view(N, 1, Ny, Nx)
                    elif group[1] == 'softmax':  # softmax over channels, single key
                        after_activation = F.softmax(output_list[index_key], dim=1)
                    elif group[1] == 'sigmoid':  # sigmoid over channels, single key
                        after_activation = torch.sigmoid(output_list[index_key][:,indices])
                        # after_activation = torch.sigmoid(output_list[index_key])
                    else:
                        if len(group[0])==1:
                            after_activation = output_list[index_key]
                        else:
                            after_activation = output_list[index_key][:, indices]
                    for group_ind, group_instance in enumerate(group[0]):
                        predDict[output_tag][group_instance] = {}
                        if len(group[0])==1:
                            predDict[output_tag][group_instance]['value'] = after_activation
                        else:
                            predDict[output_tag][group_instance]['value'] = after_activation[:, group_ind]
                        predDict[output_tag][group_instance]['weight'] = weight

        return predDict
