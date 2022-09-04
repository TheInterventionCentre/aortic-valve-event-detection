import os
import argparse
import ruamel.yaml as yaml
from box import Box

from dataLoader.data_loader_wapper import create_dataloaders
from models.myModel import Model
from trainer.evaluater import Evaluater
from utils.saverRestorer import SaverRestorer
from evaluators.evaluators_utils import Losses_and_metrics
from post_processing.post_processing_v2 import get_flags

def run_evaluation(cfg, flags):

    # Update config (cfg) to reflect evaluation mode
    cfg.saver_restorer.resume_training.restore_best_model  = True
    cfg.saver_restorer.resume_training.restore_last_epoch = False
    cfg.train.data_loader.kwargs.batch_size = 1
    cfg.val.data_loader.kwargs.batch_size   = 1
    cfg.test.data_loader.kwargs.batch_size  = 1
    cfg.test.data_loader.kwargs.num_workers = 0
    cfg.run.numb_of_epochs = 1
    cfg.run.running_phases['test'] = False
    cfg.run.running_phases.pop('train')
    cfg.run.running_phases.pop('val')
    cfg.run.mode = 'evaluation'

    # create your data loader
    dataloaders = create_dataloaders(cfg)

    # create an instance of the model
    model = Model(cfg)
    model.to(cfg.device)

    losses  = Losses_and_metrics(cfg.losses, cfg.run.mode, is_loss=True)
    metrics = Losses_and_metrics(cfg.metrics, cfg.run.mode)
    metrics_single = Losses_and_metrics(cfg.metrics, cfg.run.mode)

    # create an instance of the saver and restorer class
    saveRestorer = SaverRestorer(cfg, **cfg.saver_restorer)
    model, _, _  = saveRestorer.restore(model)

    # create trainer and pass all the previous components to it
    evaluater = Evaluater(model, cfg, dataloaders, saveRestorer, losses, metrics, metrics_single, flags)

    # here you trainer your model
    evaluater.evaluate()

    return

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,default='../experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00/net_102/2022-02-14_02-13-38_Self_attention_rnn_v3_BFL_BFL_species_Reg_ED_Reg_ES_s5_weight_decay0.0001/config.yaml',help='Path to config file')

    args = vars(parser.parse_args())
    try:
        with open(args['config_path'], 'r') as file:
            cfg = Box(yaml.safe_load(file))
    except Exception as e:
        print(e)
        raise ValueError('Error reading the config file')

    flags = get_flags(head='att')
    run_evaluation(cfg, flags)
    a = 1
