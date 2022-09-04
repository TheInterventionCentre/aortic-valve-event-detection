import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from pathlib import Path
import pickle

from utils.colorbar import colorbar
from utils import misc

baseBackend = plt.get_backend()

class Logger():
    def __init__(self, cfg, image_frequency):
        self.cfg        = cfg
        self.loggerDict = {}
        self.image_frequency = image_frequency
        self.learning_curves = {}

        for key in self.cfg.run.running_phases.keys():
            phase         = key
            is_train = self.cfg.run.running_phases[key]

            path = Path(cfg.experiment.experiment_path) / 'tb' / (phase + str(int(is_train)))
            self.createDirs(path)
            self.clearTensorboardDir(path)

            # Creates logger instances
            tb_key = phase + str(int(is_train))  # tensorboard key
            self.loggerDict[tb_key] = Logger_tb(log_dir=path)
            self.learning_curves[tb_key] = {'save_path': Path(cfg.experiment.experiment_path) / (phase + str(int(is_train)))}
            self.learning_curves[tb_key]['losses'] = {}
            self.learning_curves[tb_key]['metrics'] = {}

    def update(self, epoch, mode, is_train_flag, lossesDict, accuracyDict, predDict, dataDict, gradDict, varsDict, optimizer, scheduler):
        """ """

        # Add additional metrics you want to log into the 'miscDict' e.g. the learning rate
        miscDict = {}
        if scheduler is not None:
            # miscDict['Current_lr'] = scheduler.get_last_lr()[0]
            miscDict['Current_lr'] = scheduler._last_lr[0]
        else:
            # miscDict['Current_lr'] = optimizer.optimizer.param_groups[0]['lr']
            miscDict['Current_lr'] = optimizer.param_groups[0]['lr']

        predDict = misc.dictToCpu(predDict)
        dataDict = misc.dictToCpu(dataDict)

        # Add data to tensorboard
        tb_key = mode + str(int(is_train_flag))  # tensorboard key
        for key, value in lossesDict.items():
            self.loggerDict[tb_key].log_scalar(tag='Losses/'+key, value=value, step=epoch)
            self.learning_curves[tb_key]['losses'][key] = self.learning_curves[tb_key]['losses'].get(key, []) + ([[epoch, value]])
        for key, value in accuracyDict.items():
            if key!='confusionMatrix':
                self.loggerDict[tb_key].log_scalar(tag='Accuracy/'+key, value=value, step=epoch)
                self.learning_curves[tb_key]['metrics'][key] = self.learning_curves[tb_key]['metrics'].get(key, []) + ([[epoch, value]])

        # save metrics_summary
        base_path = Path(self.learning_curves[tb_key]['save_path'])
        base_path.mkdir(parents=True, exist_ok=True)
        with open(base_path / f'learning_curves.pickle', 'wb') as handle:
            pickle.dump(self.learning_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if (epoch % self.image_frequency)==0:
            #meta data
            for key, value in miscDict.items():
                self.loggerDict[tb_key].log_scalar(tag='Misc/'+key, value=value, step=epoch)

            for key, value in gradDict.items():
                self.loggerDict[tb_key].log_histogram(tag='gradients/' + key, values=value, step=epoch)
            #
            for key, value in varsDict.items():
                self.loggerDict[tb_key].log_histogram(tag='variables/' + key, values=value, step=epoch)

            #images
            self.loggerDict[tb_key].log_custum_images(tag=mode, dataDict=dataDict,
                                                      predDict=predDict, accuracyDict=accuracyDict, step=epoch)

        return

    def createDirs(self, path):
        if self.cfg.saver_restorer.resume_training.restore_last_epoch != 1:
            if self.cfg.saver_restorer.resume_training.restore_best_model != 1:
                if not os.path.isdir(path):
                    os.makedirs(path)
        return

    def clearTensorboardDir(self, path):
        if self.cfg.saver_restorer.resume_training.restore_last_epoch != 1:
            if self.cfg.saver_restorer.resume_training.restore_best_model != 1:
                for f in path.glob('*'):
                    os.remove(f)
        return

########################################################################################################################
class Logger_tb():
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, filename_suffix=''):
        """Creates a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix, flush_secs=20)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """

        #Use tensorboardx or pytoch tensorboard
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        self.writer.flush()

    def log_custum_images(self, tag, dataDict, predDict, accuracyDict, step):
        # set to non interactive matplitlb backend
        # plt.switch_backend('Agg')

        """Logs a list of images."""
        sampleSize = 10
        if sampleSize>dataDict['acc_mag'].shape[0]:
            sampleSize = dataDict['acc_mag'].shape[0]

        for ii in range(sampleSize):
            fig = plt.figure(dpi=100, figsize=(8, 16))
            gs = fig.add_gridspec(10, 1)
            ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :]),
                  fig.add_subplot(gs[3, :]), fig.add_subplot(gs[4, :]), fig.add_subplot(gs[5, :]),
                  fig.add_subplot(gs[6:, :])]

            fig.suptitle({Path(dataDict["path"][ii]).stem})
            plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05, hspace=0.4, wspace=0.4)

            fov = predDict['']['fov']['value']
            dx = predDict['']['dx']['value']
            for module_key, axis_inds in zip(['rnn', 'att', 'cnn'], [[0,1], [2,3], [4,5]]):
                ax[axis_inds[0]].plot(dataDict['acc_mag'][ii], label='acc mag')
                ax[axis_inds[1]].plot(dataDict['lvp'][ii], label='lvp')

                ax[axis_inds[1]].set_xlim([0, dataDict['acc_mag'][ii].shape[0]])
                ax[axis_inds[0]].set_xlim([0, dataDict['acc_mag'][ii].shape[0]])

                for loc_key, c, c_ref in zip(['es', 'ed'], ['m', 'c'], ['k','r']):
                    loc = predDict[''][f'{module_key}_1d_loc_{loc_key}']['value'][ii]
                    conf =  predDict[''][f'{module_key}_1d_conf_{loc_key}']['value'][ii]
                    for axis_ind in axis_inds:
                        for kk in range(len(conf)):
                            if conf[kk]>0.5:
                                offset = kk*dx
                                est_pixel = offset + fov*loc[kk]
                                ax[axis_ind].axvline(x=est_pixel, color=c)
                    for axis_ind in axis_inds:
                        for jj in range(dataDict[f'number_of_{loc_key}'][ii]):
                            ax[axis_ind].axvline(x=dataDict[loc_key][ii][jj], color=c_ref)
                    ax[axis_inds[0]].set_title(module_key)
                    ax[axis_inds[1]].set_title(module_key)
                    ax[axis_inds[0]].legend()
                    ax[axis_inds[1]].legend()

            if 'attention_map' in predDict['']:
                attention_map = predDict['']['attention_map']['value'][ii]
                im = ax[6].imshow(attention_map)
                ax[6].set_aspect("auto")
                colorbar(im)
            # self.writer.add_figure(tag=f'Predictions {Path(dataDict["path"][ii]).stem}', figure=fig, global_step=step)
            self.writer.add_figure(tag=f'Predictions: ind {ii}', figure=fig, global_step=step)

        self.writer.flush()
        plt.switch_backend(baseBackend)
        return

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        self.writer.add_histogram(tag=tag, values=values, global_step=step, bins='tensorflow')
        self.writer.flush()

