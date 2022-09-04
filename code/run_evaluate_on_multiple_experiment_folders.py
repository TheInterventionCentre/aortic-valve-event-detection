import os
import ruamel.yaml as yaml
from box import Box
import copy
from pathlib import Path

from post_processing.post_processing_v2 import get_flags
from evaluate import run_evaluation

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    folder_path = Path('../experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00')

    flags = get_flags(head='att')
    # flags_list = get_flags(head='rnn')
    # flags_list = get_flags(head='cnn')

    for exp_ind, experiment_folder in enumerate(sorted(list(folder_path.glob('*')))):
        print(f'{exp_ind}  | {experiment_folder}')
        for experiment in sorted(list(experiment_folder.glob('*'))):
            print(experiment)
            if 'summary' in experiment.stem:
                continue
            cfg_path = experiment / 'config.yaml'
            try:
                with open(cfg_path, 'r') as file:
                    cfg = Box(yaml.safe_load(file))
            except Exception as e:
                print(e)
                raise ValueError('Error reading the config file')
            cfg.device = 'cuda:0'
            cfg_copy = copy.deepcopy(cfg)
            run_evaluation(cfg_copy, flags)
        a = 1
