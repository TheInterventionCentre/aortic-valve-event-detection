import os
from box import Box
import argparse
from pathlib import Path

from dataLoader.data_loader_wapper import create_dataloaders
from models.myModel import Model
from trainer.trainer import Trainer
from evaluators.evaluators_utils import Losses_and_metrics
from evaluate import run_evaluation
from utils.saverRestorer import SaverRestorer
from utils.loader import load_class
from utils.configure_experiment import configure_experiment
from post_processing.post_processing_v2 import get_flags

def run_training(cfg, debug_mode=False):

    #configure experiment
    cfg = configure_experiment(cfg, debug_mode)

    # create your data loader
    dataloaders = create_dataloaders(cfg)

    # create an instance of the model
    model = Model(cfg)
    model.to(cfg.device)

    optimizer = load_class(cfg.optimizer, model.parameters())

    if 'scheduler' in cfg.keys():
        scheduler = load_class(cfg.scheduler, optimizer)
    else:
        scheduler = None

    losses  = Losses_and_metrics(cfg.losses, optimizer, is_loss=True)
    metrics = Losses_and_metrics(cfg.metrics)

    # create an instance of the saver and restorer class
    saveRestorer     = SaverRestorer(cfg, **cfg.saver_restorer)
    model, optimizer, scheduler = saveRestorer.restore(model, optimizer, scheduler)

    # tensorboard
    visualizer = load_class(cfg.visualizer, cfg)

    # create trainer and pass all the previous components to it
    trainer = Trainer(model, cfg, dataloaders, saveRestorer, losses, metrics, optimizer, scheduler, visualizer)

    # train model
    trainer.train()

    return cfg

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_default_m_json.yaml',help='Path to config file')

    args = vars(parser.parse_args())
    try:
        cfg = Box.from_yaml(Path(args['config_path']).read_text())
    except Exception as e:
        print(e)
        raise ValueError('Error reading the config file')

    debug_mode = False # True | False
    run_training(cfg, debug_mode)
    flags = get_flags()
    run_evaluation(cfg, flags)
