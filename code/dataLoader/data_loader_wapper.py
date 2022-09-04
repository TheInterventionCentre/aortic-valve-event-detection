from torch.utils.data import DataLoader
from utils.loader import import_function
import torch
import numpy
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # print(worker_seed)
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    a = 1

def create_dataloaders(cfg):
    """ Creates the data loaders.

     Args:
         cfg: The configuration dict

     Returns:
         A dict with data loaders for each phase (train, val, test) specified in the config dict

     """
    dataLoaderDict = {}
    for phase, is_training in cfg.run.running_phases.items():
        print(phase)
        dataset_fn = import_function(cfg[phase].dataset.type)
        dataset    = dataset_fn(cfg, phase, **cfg[phase].dataset.kwargs)
        if 'batch_sampler' in cfg[phase].keys():
            sampler_fn = import_function(cfg[phase].sampler.type)
            batch_sampler_fn = import_function(cfg[phase].batch_sampler.type)
            batch_sampler = batch_sampler_fn(sampler_fn(dataset), **cfg[phase].batch_sampler.kwargs)
            dataLoaderDict[phase] = DataLoader(dataset, **cfg[phase].data_loader.kwargs, batch_sampler=batch_sampler, worker_init_fn=seed_worker,)
        else:
            dataLoaderDict[phase] = DataLoader(dataset, **cfg[phase].data_loader.kwargs, worker_init_fn=seed_worker)
    return dataLoaderDict

