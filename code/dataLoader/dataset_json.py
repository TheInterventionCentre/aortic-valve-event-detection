import numpy as np
import random
import json
import pandas as pd

from torch.utils.data import Dataset
from utils.FOV import validNetworkConfig
from utils.FOV import fov
from utils.FOV import get_unet_1d_param_vectors
from pathlib import Path
from utils.loader import load_class

########################################################################################################################
def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def modify_mv13_folder_structure():
    mv13_path = Path("../data_json/MV/MV13/")
    if list((mv13_path/"dobutamine").glob('*')) == []:
        (mv13_path/"baseline").mkdir(parents=True, exist_ok=True)
        (mv13_path/"MV1302_baseline_b_000.json").rename(mv13_path/"baseline"/"MV1302_baseline_b_000.json")
        (mv13_path/"MV1310_baseline_b_000.json").rename(mv13_path/"baseline"/"MV1310_baseline_b_000.json")
        (mv13_path/"MV1312_baseline_b_000.json").rename(mv13_path/"baseline"/"MV1312_baseline_b_000.json")
        
        (mv13_path/"crt").mkdir(parents=True, exist_ok=True)
        (mv13_path/"MV13_crt_b_000.json").rename(mv13_path/"crt"/"MV13_crt_b_000.json")
        (mv13_path/"MV1302_crt_b_000.json").rename(mv13_path/"crt"/"MV1302_crt_b_000.json")

        (mv13_path/"dobutamine").mkdir(parents=True, exist_ok=True)
        (mv13_path/"MV13_dobutamine_b_000.json").rename(mv13_path/"dobutamine"/"MV13_dobutamine_b_000.json")

        (mv13_path/"lbbb").mkdir(parents=True, exist_ok=True)
        (mv13_path/"MV13_lbbb_b_000.json").rename(mv13_path/"lbbb"/"MV13_lbbb_b_000.json")
        (mv13_path/"MV1359_lbbb_b_000.json").rename(mv13_path/"lbbb"/"MV1359_lbbb_b_000.json")
        (mv13_path/"MV1361_lbbb_b_000.json").rename(mv13_path/"lbbb"/"MV1361_lbbb_b_000.json")

    return

class myDataset(Dataset):
    def __init__(self, cfg, phase, seq_length_in_ms=1499, ms_per_pixel=2, species=()):
        self.cfg               = cfg
        self.phase             = phase #train, val, test
        self.seq_length_in_ms  = seq_length_in_ms
        if ms_per_pixel != 2:
            raise ValueError('ms_per_pixel=2 is currently supported only')
        
        #modify patient folder structure: data_json\MV\MV13
        modify_mv13_folder_structure()

        self.ms_per_pixel      = ms_per_pixel
        self.list_of_samples = self.get_sample_paths(cfg, phase, species)
        if phase !='test':
            random.shuffle(self.list_of_samples)

        if 'enc_chs' in cfg.model.net1.kwargs:
            filter_size, stride, dilation, padding = get_unet_1d_param_vectors(cfg.model.net1.kwargs.enc_chs,
                                                                               cfg.model.net1.kwargs.dilation,
                                                                               cfg.model.net1.kwargs.padding,
                                                                               cfg.model.net1.kwargs.filter_size)
        else:
            filter_size = cfg.model.net1.kwargs.filter_size
            stride = cfg.model.net1.kwargs.stride
            dilation = cfg.model.net1.kwargs.dilation
            padding = cfg.model.net1.kwargs.padding

        fovList, dxList = fov(filter_size, stride, dilation, padding)

        self.dx = dxList[-1]
        self.fov = fovList[-1]

        self.cnn_config = {}
        self.cnn_config['filter_size'] = filter_size
        self.cnn_config['stride']   = stride
        self.cnn_config['dilation'] = dilation
        self.cnn_config['padding']  = padding

        self.max_number_of_end_systole = 30

        self.transforms = {}
        if 'augmentation' in cfg[phase].keys():
            for transform in cfg[phase].augmentation:
                name = list(transform.keys())[0]
                transform_dict = list(transform.values())[0]
                transform_dict['function'] = load_class(transform_dict)
                self.transforms[name] = transform_dict
        return

    def get_sample_paths(self, cfg, phase, species):
        list_of_samples = []
        list_of_headers = cfg.data_path[phase]
        folder_paths    = pd.read_csv(cfg.experiment.root + cfg.data_path['csv'])
        for header in list_of_headers:
            list_of_individuals = list(folder_paths[header])
            for path_individual in list_of_individuals:
                if path_individual != path_individual: # is nan when unequal unequal number of patients in splits
                    continue
                tag = Path(path_individual).parent.stem
                if len(species)>0 and not tag in species:
                    continue
                path_individual = cfg.experiment.root + path_individual
                for path_intervention in Path(path_individual).glob('*'):
                    list_of_files = list(path_intervention.glob('*'))
                    list_of_samples.extend(list_of_files)
                    if len(list_of_samples)==0:
                        raise ValueError('Zero data paths were found')

        if phase == 'train':
            list_of_samples = list_of_samples
        return list_of_samples

    def __len__(self):
        return len(self.list_of_samples)

    def __getitem__(self, idx_tuple):
        if isinstance(idx_tuple, tuple):
            idx = idx_tuple[0]
            sequence_len_in_ms = idx_tuple[1]
        else:
            idx = idx_tuple
            sequence_len_in_ms = self.seq_length_in_ms

        path = self.list_of_samples[idx]
        #get data
        with open(path, 'r') as input_file:
            data = json.load(input_file)
        data['ms_per_pixel'] = 1000 / data['sample_rate']

        # get targets
        target_str = str(path).replace('data_json', 'targets_json')
        if "MKCMS" in target_str: # Need to modify the path to "targets_json" due to inconsistance with "data_json".
            if "baseline-closed-chest" in target_str:
                target_str = target_str.replace('-cc_b', '(cc)_b')
                target_str = target_str.replace('baseline-closed-chest', 'baseline(cc)')
            elif "-closed-chest" in target_str:
                target_str = target_str.replace('-closed-chest', '')

        with open(target_str, 'r') as input_file:
            target_data = json.load(input_file)
            data['es'] = target_data['es']
            data['ed'] = target_data['ed']

        #convert to numpy
        for key in ['acc_x', 'acc_y', 'acc_z', 'lvp', 'es', 'ed', 'ecg']:
            data[key] = np.array(data[key])
        data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
        data['dpdt'] = data['lvp'][1:] - data['lvp'][:-1]
        data['dpdt'] = np.append(data['dpdt'], data['dpdt'][-1]) #make all arrays of same length

        if 'ecg' not in data.keys():
            data['ecg'] = np.zeros(shape=data['acc_mag'].shape)

        if 'augmentation' in self.cfg[self.phase].keys():
            for transform_key, transform in self.transforms.items():
                probability = transform['probability']
                p = random.uniform(0, 1)
                if probability >= p:
                    data = transform['function'](data)

        if self.phase=='test':
            true_sequence_len_in_ms = data['acc_mag'].shape[0] * data['ms_per_pixel']
            shift = self.dx * self.ms_per_pixel
            if sequence_len_in_ms-shift>true_sequence_len_in_ms:
                sequence_len_in_ms = true_sequence_len_in_ms-shift

        #Find a valid sequence length based on wanted sequence length in ms and msPerPixel and the network configuration
        seq_length_in_pixels = int(sequence_len_in_ms / self.ms_per_pixel)
        layerSizes, is_valid, seqLenValid, layerSizesValid = validNetworkConfig(seq_length_in_pixels, self.cnn_config)
        if not is_valid:
            seq_length_in_pixels = seqLenValid
            layerSizes     = layerSizesValid

        # update based on wanted <ms_per_pixel>
        if data['ms_per_pixel'] != self.ms_per_pixel:
            data['es'] = data['es'] * data['ms_per_pixel'] / self.ms_per_pixel
            data['ed'] = data['ed'] * data['ms_per_pixel'] / self.ms_per_pixel

            N = data['acc_mag'].shape[0]
            new_time = np.arange(0, N*data['ms_per_pixel'], self.ms_per_pixel)
            old_time = np.linspace(0, (N-1)*data['ms_per_pixel'], N)
            for key in ['acc_mag', 'acc_x', 'acc_y', 'acc_z', 'lvp', 'ecg']:
                data[key] = np.interp(x=new_time, xp=old_time, fp=data[key])
            data['ms_per_pixel'] = self.ms_per_pixel

        # select region
        n_samples = data['acc_mag'].shape[0]
        if self.phase=='train':
            start_ind = random.randint(0, n_samples-seq_length_in_pixels-1)
            end_ind = start_ind + seq_length_in_pixels
        else:
            start_ind = 0
            end_ind = start_ind + seq_length_in_pixels

        # select pixels and update the labels ed and es
        for key in ['acc_x', 'acc_y', 'acc_z', 'lvp', 'dpdt', 'ecg', 'acc_mag']:
            if len(data[key].shape) == 0:
                continue
            if len(data[key].shape)==1:
                data[key] = data[key][start_ind:end_ind]
            elif len(data[key].shape)==2:
                data[key] = data[key][:, start_ind:end_ind]
            else:
                raise ValueError('unknown shape')

        for key in ['es', 'ed']:
            data[key] = data[key][data[key] > start_ind]
            data[key] = data[key][data[key] < end_ind]
            data[key] = data[key]-start_ind

        # Generate patch labels (confidence and location)
        fov = self.fov
        dx  = self.dx
        numbOfPredictions = int(layerSizes[-1])
        for key in ['es', 'ed']:
            data[f'patch_location_{key}']   = np.zeros(numbOfPredictions)
            data[f'patch_confidence_{key}'] = np.zeros(numbOfPredictions)
            data[f'patch_weight_{key}']     = np.zeros(numbOfPredictions)
            for ii in range(numbOfPredictions):
                minLoc = ii * dx
                maxLoc = (ii * dx) + fov
                condition = (data[key]>minLoc) * (data[key]<maxLoc)
                index = np.argwhere(condition)[:, 0]

                if len(index)==1:
                    data[f'patch_location_{key}'][ii] = (data[key][index] - minLoc)/fov
                    data[f'patch_confidence_{key}'][ii] = 1
                elif len(index)==0:
                    data[f'patch_location_{key}'][ii] = 0
                    data[f'patch_confidence_{key}'][ii] = 0
                else:
                    raise Exception('More than one trig within fov')
                if maxLoc<=seq_length_in_pixels:
                    data[f'patch_weight_{key}'][ii] = 1

        for key in ['es', 'ed']:
            data[f'number_of_{key}']     = data[key].shape[0]
            data[f'number_of_{key}_vec'] = data[key].shape[0] * np.ones(numbOfPredictions)
            if self.phase == 'test':
                data[key] = padarray(data[key], 200)
            else:
                data[key] = padarray(data[key], self.max_number_of_end_systole)

        del data['sample_rate']
        del data['dpdt']
        if 'rpeaks' in data.keys():
            del data['rpeaks']

        data['lab_species'] = np.ones(numbOfPredictions) if data['animal_species']=='dog' else np.zeros(numbOfPredictions)

        for key in ['acc_x', 'acc_y', 'acc_z', 'lvp', 'acc_mag', 'ed', 'es', 'patch_location_es',
                    'patch_confidence_es', 'patch_weight_es', 'patch_location_ed', 'patch_confidence_ed',
                    'patch_weight_ed', 'ecg', 'number_of_es_vec', 'number_of_ed_vec', 'lab_species']:
            data[key] = data[key].astype(np.float32)

        data['dx'] = self.dx
        data['fov'] = self.fov
        data['weights'] = 1
        data['path'] = str(path)

        return data

