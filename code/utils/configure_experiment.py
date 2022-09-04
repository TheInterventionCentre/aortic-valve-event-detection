from utils.misc import get_experiment_path
from pathlib import Path
import random
from utils.debug_mode import set_to_debug_mode
import numpy as np
import torch
import sys

def configure_experiment(cfg, debug_mode):
    """ This function updates and store the cfg (config) file. It also creates the experiment folder """

    # Set the random seed for reproducible experiments
    if cfg.seeds.use_given_seeds:
        np.random.seed(cfg.seeds['numpy'])
        torch.manual_seed(cfg.seeds['torch'])
        random.seed(cfg.seeds['random'])
        if 'cuda' in cfg.device:
            torch.cuda.manual_seed(cfg.seeds['torch_cuda'])
    else:
        seed = random.randrange(sys.maxsize)
        random.seed(seed)
        cfg.seeds['random'] = seed
        cfg.seeds['numpy'] = int(np.random.get_state()[1][0])
        cfg.seeds['torch'] = int(torch.initial_seed())
        cfg.seeds['torch_cuda'] = int(torch.cuda.initial_seed())

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    experiment_path = Path(get_experiment_path(cfg))
    experiment_path.mkdir(parents=True, exist_ok=True)
    cfg.experiment.experiment_path = str(experiment_path)
    config_path = experiment_path / 'config.yaml'
    # Path(config_path).write_text(cfg.to_yaml(indent=4))
    Path(config_path).write_text(cfg.to_yaml())

    #if in "debug_mode" update cfg
    if debug_mode:
        cfg = set_to_debug_mode(cfg)

    return cfg


