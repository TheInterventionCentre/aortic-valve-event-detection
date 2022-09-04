import os
from box import Box
import torch
import copy
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
import time
import traceback

from train import run_training
from evaluate import run_evaluation
from utils.misc import parse_data_paths
from post_processing.post_processing_v2 import get_flags

def train_networks(device, flags, cfg_sub_list):
    """ Trains multiple networks based on 'cfg_sub_list' """
    for cfg in cfg_sub_list:
        cfg_copy = Box(copy.deepcopy(cfg))
        cfg_copy['device'] = device
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        try:
            cfg_copy = run_training(cfg_copy)
        except Exception as e:
            print(traceback.print_exc())
            print(e)
            print('Error during training')

        try:
            cfg_copy2 = copy.deepcopy(cfg_copy)
            run_evaluation(cfg_copy2, flags)
        except Exception as e:
            print(e)
            print('Error during evaluating')

    return

def run(cfg_list, flags, numb_workers=1, numb_gpus=1):
    """ This function splits the config list to each worker.

      numb_workers: number of parallel runs
      numb_workers: number of gpus

      Example: numb_workers=6, numb_gpus=3, each gpu will train two models in parallel.

      """
    processes = []
    print(f'numb_workers = {numb_workers}')
    print(f'numb_gpus = {numb_gpus}')
    numb_of_configs = len(cfg_list)
    for i in range(numb_workers):
        indices = np.arange(i, numb_of_configs, numb_workers)
        cfg_sub_list = [cfg_list[ind] for ind in indices]
        if len(cfg_sub_list)==0:
            continue
        device = f'cuda:{i%numb_gpus}'

        if numb_workers==1 and numb_gpus==1:
            train_networks(device, cfg_sub_list)
        else:
            try:
                process = multiprocessing.Process(target=train_networks, args=(device, flags, cfg_sub_list))
                processes.append(process)
                process.start()
            except Exception as e:
                print(e)
            time.sleep(2)
    for proc in processes:
        proc.join()
    print('Cross validation done')

def run_multiple_cross_validations(cfg,  flags, number_of_runs=1, numb_workers=1, numb_gpus=1):
    # from a cfg, run multiple cross validation experiments.
    # The seed for each cross validation runs is the same

    for start_seed in range(number_of_runs):
        new_cfg = copy.deepcopy(cfg)
        base = Path(new_cfg.experiment.save_dir).parent
        head = str(Path(new_cfg.experiment.save_dir).stem) + f'_{start_seed}'
        new_cfg.experiment.save_dir = str(Path(base) / head)

        # Set seeds
        new_cfg.seeds.use_given_seeds = True
        new_cfg.seeds.numpy = start_seed
        new_cfg.seeds.torch = start_seed
        new_cfg.seeds.torch_cuda = start_seed
        new_cfg.seeds.random = start_seed

        cfg_list = parse_data_paths(new_cfg)
        run(cfg_list, flags, numb_workers=numb_workers, numb_gpus=numb_gpus)

    return

########################################################################################################################
if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_default_m_json.yaml', help='Path to config file')

    args = vars(parser.parse_args())
    try:
        cfg = Box.from_yaml(Path(args['config_path']).read_text())
    except Exception as e:
        print(e)
        raise ValueError('Error reading the config file')

    #set model size
    k=1 #1,2,3
    downsampling_mode = 'maxpool' # 'maxpool' | 'maxBlurPool'
    norm_layer_mode = 'batchNorm' # 'batchNorm' | 'groupNorm'

    #update config file
    cfg.model.net1.kwargs.enc_chs         = [v*k for v in cfg.model.net1.kwargs.enc_chs]
    cfg.model.net1.kwargs.rnn_hidden_size = cfg.model.net1.kwargs.rnn_hidden_size*k
    cfg.model.net1.kwargs.downsampling_mode = downsampling_mode
    cfg.model.net1.kwargs.norm_layer_mode = norm_layer_mode

    # hard corded values: lr and noise
    lr = 0.001
    noise = 0.00

    save_dir = Path(cfg.experiment.save_dir)
    cfg.experiment.save_dir = Path(str(save_dir.parent)+ f'_k{int(k)}_{downsampling_mode}_{norm_layer_mode}_lr{lr:0.0e}_noise{noise:0.0e}') / save_dir.stem

    number_of_runs = 3 # number of complete cross validations with different seeds
    numb_workers = 2   # number of parallel processes
    numb_gpus = 1      # number of gpus, set os.environ["CUDA_VISIBLE_DEVICES"] accordingly

    flags = get_flags() #specifying the post processing parameters
    run_multiple_cross_validations(cfg, flags, number_of_runs=number_of_runs, numb_workers=numb_workers, numb_gpus=numb_gpus)
