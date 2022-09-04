from tqdm import tqdm
import torch

from utils.printTqdm import printTDQM
from utils.misc import calculateAVGstats
from utils.lossDictChecker import variableChecker
from utils.misc import dictToCuda


class Trainer():
    def     __init__(self, model, cfg, dataloaders, saveRestorer, losses, metrics, optimizer, scheduler, visualizer):
        """ The class used to train the model """
        self.model         = model
        self.cfg           = cfg
        self.dataloaders   = dataloaders
        self.visualizer    = visualizer
        self.saveRestorer  = saveRestorer
        self.losses        = losses
        self.metrics       = metrics
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        return

    def train(self):
        for cur_epoch in range(self.model.start_epoch, self.cfg.run.numb_of_epochs):
            for key in self.cfg.run.running_phases.keys():
                phase   = key
                is_train = self.cfg.run.running_phases[key]
                if phase == 'train':
                    self.model.train()
                    avgLossesDict, avgMetricDict = self.run_epoch(phase, is_train, cur_epoch)
                else:
                    with torch.no_grad():
                        self.model.eval()
                        avgLossesDict, avgMetricDict = self.run_epoch(phase, is_train, cur_epoch)

            self.saveRestorer.save(cur_epoch, avgLossesDict, avgMetricDict, self.model, self.optimizer, self.scheduler,
                                  self.model.update_steps)

        return

    def run_epoch(self, phase, is_train, cur_epoch):
        # initialize average dict
        avgStats = calculateAVGstats()
        pbar = tqdm(self.dataloaders[phase], desc='', leave=False, mininterval=0.01, ncols=400)
        for cur_it, dataDict in enumerate(pbar):
            print(cur_it)
            dataDictCuda = dictToCuda(dataDict, self.cfg.device)
            data = [dataDictCuda[key] for key in self.cfg.model.net1.input_keys]
            predDict     = self.model.forward(data)

            lossesDict   = self.losses(dataDictCuda, predDict)
            metricDict   = self.metrics(dataDictCuda, predDict)

            if phase=='train':
                self.optimizer.zero_grad()
                lossesDict['totalLoss'].backward()
                self.optimizer.step()

            if self.scheduler is not None:
                if 'reference_keys' in self.cfg.scheduler:
                    rf_value = sum([metricDict[key] for key in self.cfg.scheduler.reference_keys])
                    self.scheduler.step(rf_value)
                else:
                    self.scheduler.step()

            variableChecker(lossesDict, predDict, dataDictCuda, self.model)

            batch_size = dataDict['acc_mag'].shape[0]
            printTDQM(phase, lossesDict, metricDict, {}, pbar, cur_it, cur_epoch, len(self.dataloaders[phase]), batch_size)
            avgStats.update(lossesDict, metricDict, batch_size)

        # Logger - logging values to tensorboard
        avgLossesDict, avgMetricDict = avgStats.getAVGstats()
        varsDict = {}
        gradDict = {}
        for name, param in self.model.named_parameters():
            varsDict[name] = param.detach().cpu().numpy()
            gradDict[name + '_grad'] = param.grad.detach().cpu().numpy()
        if phase=='train':
            self.model.update_steps = self.model.update_steps + len(self.dataloaders['train'])
        self.visualizer.update(self.model.update_steps, phase, is_train, avgLossesDict,
                               avgMetricDict, predDict, dataDict, gradDict,
                               varsDict, self.optimizer, self.scheduler)

        return avgLossesDict, avgMetricDict
