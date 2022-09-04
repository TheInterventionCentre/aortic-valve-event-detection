from tqdm import tqdm
import torch
import numpy as np
import pickle
from pathlib import Path

from utils.printTqdm import printTDQM
from utils.misc import dictToCuda
from utils.misc import dictToCpu
from post_processing.post_processing_v2 import postproc
from post_processing.utils import getErrors
from post_processing.stats_v4 import Stats

class Evaluater():
    def     __init__(self, model, cfg, dataloaders, saveRestorer, losses, metrics, metrics_single, flags):
        """ The class used to evaluate the model """
        self.model         = model
        self.cfg           = cfg
        self.dataloaders   = dataloaders
        self.saveRestorer  = saveRestorer
        self.losses        = losses
        self.metrics       = metrics
        self.metrics_single = metrics_single
        self.flags = flags
        return

    def evaluate(self):
        cur_epoch = 1
        for key in self.cfg.run.running_phases.keys():
            phase   = key
            is_train = self.cfg.run.running_phases[key]
            if phase == 'train':
                self.model.train()
                self.run_epoch(phase, is_train, cur_epoch)
            else:
                with torch.no_grad():
                    self.model.eval()
                    self.run_epoch(phase, is_train, cur_epoch)
        return

    def run_epoch(self, phase, is_train, cur_epoch):
        flags =  self.flags
        myStatCollection = {}

        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(6, 1, sharex=True)
        fig.set_size_inches(22, 13)
        plt.subplots_adjust(top=0.90, bottom=0.05, right=0.95, left=0.10, hspace=0.10, wspace=0.05)

        pbar = tqdm(self.dataloaders[phase], desc='', leave=False, mininterval=0.01, ncols=250)
        for cur_it, dataDict in enumerate(pbar):
            print(cur_it)
            nx = dataDict['acc_mag'].shape[1]

            dataDictCuda = dictToCuda(dataDict, self.cfg.device)
            animal_key = dataDict['animal_species'][0]
            data = [dataDictCuda[key] for key in self.cfg.model.net1.input_keys]
            predDict = self.model.forward(data)

            # To evaluate a long sequence with a model trained using sequences of 3000ms (pixels), the longer sequneces
            # is split to multiple sequence of 1500 pixels and passed to the model. The values are hard coded.
            FOV = 68 # The Model's receptive field.
            dx = 8   # Step length between (receptive field) patches in pixels.
            p_length = 1500 # Trained sequence length in pixels
            o_step = 19
            p_step = dx*o_step*2 + FOV # buffer to prevent RNN edge effects.

            seq_length = data[0].shape[1]
            N = int(np.ceil((seq_length-p_length) / (p_length-p_step)))+1

            for ff in range(N):
                p_start_ind = ff*(p_length-p_step)
                p_end_ind   = ff*(p_length-p_step) + p_length
                if p_end_ind>seq_length:
                    p_end_ind = seq_length
                o_start_ind  = int(p_start_ind / dx)
                if ff!=0:
                    o_start_ind += o_step
                o_end_ind = int((p_end_ind - FOV) / dx) + 1
                if ff<N-1:
                    o_end_ind -= o_step

                in_tensor = [data[0][0, p_start_ind:p_end_ind].unsqueeze(dim=0)]
                tmpDict = self.model.forward(in_tensor)
                if ff==0:
                    predDict = tmpDict
                else:
                    for key in predDict[''].keys():
                        if predDict[''][key]['value'].ndim==2 and predDict[''][key]['value'].shape[0]==1:
                            p = tmpDict[''][key]['value']
                            predDict[''][key]['value'] = torch.cat((predDict[''][key]['value'][0:1, 0:o_start_ind], p[0:1, o_step:]), dim=1)

            for loc in ['cnn', 'rnn', 'att']:
                predDict[''][f'{loc}_1d_loc_es']['value'] = torch.clamp(predDict[''][f'{loc}_1d_loc_es']['value'], min=0, max=1)
                predDict[''][f'{loc}_1d_loc_ed']['value'] = torch.clamp(predDict[''][f'{loc}_1d_loc_ed']['value'], min=0, max=1)

            if flags['linearLoc']:
                for loc in ['cnn', 'rnn', 'att']:
                    predDict[''][f'{loc}_1d_loc_es']['value'][:] = 0.5
                    predDict[''][f'{loc}_1d_loc_ed']['value'][:] = 0.5

            lossesDict   = self.losses(dataDictCuda, predDict)
            metricDict   = self.metrics(dataDictCuda, predDict)

            batch_size = dataDict['acc_mag'].shape[0]
            printTDQM(phase, lossesDict, metricDict, {}, pbar, cur_it, cur_epoch, len(self.dataloaders[phase]), batch_size)

            # Combine patch predictions to final es/ed estimates.
            predDict = dictToCpu(predDict)
            dataDict = dictToCpu(dataDict)
            ms_per_pixel = dataDict['ms_per_pixel'][0]
            output_dict = postproc(predDict, dataDict, ax, fig, flags, self.cfg.experiment['experiment_path'])

            for key in output_dict.keys():
                numbOfFound, numbOfMissed, numbOfnonTrue, distList, totalNumber = getErrors(output_dict[key]['est_pixels'],
                                                                                            output_dict[key]['est_confidence'],
                                                                                            output_dict[key]['label_pixels'],
                                                                                            ms_per_pixel,
                                                                                            flags['thresholds'],
                                                                                            nx)
                statCollection = {'totalNumber': totalNumber,
                                  'numbOfFound': numbOfFound,
                                  'numbOfMissed': numbOfMissed,
                                  'numbOfnonTrue': numbOfnonTrue,
                                  'distList': distList,
                                  'att_species': [int(np.round(predDict['']['att_species']['value'].mean()))],
                                  'species': [int(np.round(dataDict['lab_species'].mean()))],
                                  }

                print(key)
                print(statCollection)


                if animal_key not in myStatCollection:
                    myStatCollection[animal_key] = {}
                    for tmp_key in ['es', 'ed']:
                        myStatCollection[animal_key][tmp_key] = Stats()

                myStatCollection[animal_key][key].update('all', statCollection)
                p = dataDict['path'][0]
                myStatCollection[animal_key][key].update(Path(p).parent.stem, statCollection)

                if flags['debug']:
                    print('\n -----------------------------')
                    print(key)
                    print(f'numbOfFound={numbOfFound}')
                    print(f'numbOfMissed={numbOfMissed}')
                    print(f'numbOfnonTrue={numbOfnonTrue}')
                    print(f'totalNumber={totalNumber}')

            if flags['debug']:
                plt.show()
                fig_folder = Path(self.cfg.experiment.experiment_path) / phase / (f'figure_{flags["thresholds"]["accept_threshold"]*100:.0f}_{flags["thresholds"]["noDetectDist"]}ms_{flags["output_head"]}')
                fig_folder.mkdir(exist_ok=True)
                fig_path = fig_folder / (str(Path(dataDict['path'][0]).with_suffix('').name) +'.png')
                plt.savefig(fig_path, dpi=fig.dpi)
                for ii in range(len(ax)):
                    ax[ii].cla()

        for animal_key in myStatCollection.keys():
            for key in myStatCollection[animal_key].keys():
                print(f'animal={animal_key} | {key}')
                myStatCollection[animal_key][key].printResult()

                s1 = f'{animal_key}_{key}_{self.cfg.data_path.test[0]}_'
                s2 = f'thres_{int(flags["thresholds"]["accept_threshold"] * 100)}_dist_{flags["thresholds"]["noDetectDist"]}_head_{flags["output_head"]}'
                s3 = f"_densityWinSize_{flags['trigDensityWinSize']}_linearLoc_{int(flags['linearLoc'])}.csv"

                csv_path = Path(self.cfg.experiment.experiment_path) / phase / (s1+s2+s3)
                myStatCollection[animal_key][key].storeResultToExcel(csv_path)

        s2 = f'thres_{int(flags["thresholds"]["accept_threshold"] * 100)}_dist_{flags["thresholds"]["noDetectDist"]}_head_{flags["output_head"]}'
        s3 = f"_densityWinSize_{flags['trigDensityWinSize']}_linearLoc_{int(flags['linearLoc'])}.pkl"
        collection_path = Path(self.cfg.experiment.experiment_path) / phase / (self.cfg.data_path.test[0]+'_'+s2+s3)
        with open(collection_path, 'wb') as handle:
            pickle.dump(myStatCollection, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plt.close(fig)
        return
