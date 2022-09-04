import torch
import os
from pathlib import Path

class SaverRestorer():
    def __init__(self, cfg, reference_keys, resume_training, reference_dict='losses', lower_is_better=True):
        """ Class used for saving and loading network weights
        Args:
            cfg: (dict) configuration dictionary
            reference_keys: (list) The key(s) to be used for evaluating the "best" model
            resume_training: (dict)
            reference_dict: (dict) The dictionary used to extract values for evaluating the "best" model
            lower_is_better: (bool) Whether "best" model is given a large or a small metric/loss value.
        """

        self.cfg       = cfg
        self.save_dir  = Path(cfg.experiment.experiment_path) / 'model'
        self.reference_keys = reference_keys
        self.lower_is_better = lower_is_better
        self.resume_training = resume_training
        self.reference_dict = reference_dict
        if lower_is_better:
            self.referance_value = 1e8
        else:
            self.referance_value = -1e8
        self.removePreviousModel()
        return


    def save(self, epoch, losses, metrics, model, optimizer, scheduler, update_steps):

        if self.reference_dict=='losses':
            current_value = sum([losses[key] for key in self.reference_keys])
        elif self.reference_dict=='metrics':
            current_value = sum([metrics[key] for key in self.reference_keys])
        else:
            raise ValueError(f'Unknown "reference_dict" | {self.reference_dict}')

        #save as the last model
        self.removeLastModel()
        torch.save({
            'epoch': epoch,
            'update_steps': update_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, self.save_dir / f'last_epoch{epoch}.pt')

        # save as the best model
        update=False
        if self.lower_is_better:
            if self.referance_value>=current_value:
                update = True
        else:
            if self.referance_value<=current_value:
                update = True

        if update:
            self.removeBestModel()
            self.referance_value = current_value
            torch.save({
                'epoch': epoch,
                'update_steps': update_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, self.save_dir / f'best_epoch{epoch}.pt')
        return


    def restore(self, model, optimizer=None, scheduler=None):
        restore_dir = ''
        paths = self.save_dir.glob('*')
        if self.resume_training.restore_last_epoch == 1 and self.resume_training.restore_best_model != 1:
            for path in paths:
                if 'last_epoch' in str(path):
                    restore_dir = path
        elif self.resume_training.restore_last_epoch != 1 and self.resume_training.restore_best_model == 1:
            for path in paths:
                if 'best_epoch' in str(path):
                    restore_dir = path
        if restore_dir!='':
            checkpoint = torch.load(restore_dir, map_location=self.cfg['device'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.start_epoch = checkpoint['epoch'] + 1
            model.update_steps = checkpoint['update_steps'] + 1
        else:
            if self.resume_training.restore_last_epoch == 1 or self.resume_training.restore_best_model == 1:
                raise ValueError('Could not find the appropriate restore path')
        return model, optimizer, scheduler

    def removeLastModel(self):
        files = self.save_dir.glob('*')
        for f in files:
            if 'last_epoch' in str(f):
                os.remove(f)
        return

    def removeBestModel(self):
        files = self.save_dir.glob('*')
        for f in files:
            if 'best_epoch' in str(f):
                os.remove(f)
        return

    def removePreviousModel(self):
        if self.resume_training.restore_last_epoch != 1:
            if self.resume_training.restore_best_model != 1:
                # create directory if not existing
                if not os.path.isdir(self.save_dir):
                    os.makedirs(self.save_dir)
                # Remove old files
                files = self.save_dir.glob('*')
                for f in files:
                    os.remove(f)


