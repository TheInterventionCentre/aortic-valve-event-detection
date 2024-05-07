import os
from pathlib import Path
import pandas as pd
import pickle
from collections import OrderedDict

from post_processing.stats_v4 import Stats
from post_processing.post_processing_v2 import get_flags

def findall(p, string):
    i = str(string).find(p)
    out = [i]
    while i != -1:
        i = str(string[(out[-1]+1):]).find(p)
        if i!=-1:
            out.append(i+out[-1]+1)
    return out

def find_pickle_file(file_paths, config):
    #Find path matching the config
    output_path = None
    for file_path in file_paths:
        indx = findall('_', file_path.stem)
        stem = file_path.stem
        split_id = stem[0:indx[0]]
        thres = config['thres']
        dist = config['dist']
        head = config['head']

        densityWinSize = config['densityWinSize']
        linearLoc = config['linearLoc']

        match = f'{split_id}_thres_{thres}_dist_{dist}_head_{head}_densityWinSize_{densityWinSize}_linearLoc_{int(linearLoc)}'
        if stem==match:
            output_path = file_path
            break
    if output_path is None:
        raise ValueError(f"Could not find config: thres:{thres} | dist:{dist} | head: {head}")
    return output_path, split_id

def get_save_file_name(output_path, config, species_key, event_key, extention='.csv'):
    thres = config['thres']
    dist = config['dist']
    head = config['head']
    densityWinSize = config['densityWinSize']
    linearLoc = config['linearLoc']
    file_name = f'{species_key}_{event_key}_thres_{thres}_dist_{dist}_head_{head}_densityWinSize_{densityWinSize}_linearLoc_{linearLoc}' + extention
    return output_path / file_name

def collector(folder_path, config):
    #load all stat collection (pickle files) from each cross validation
    myStatCollection = {}
    for experiment in sorted(list(folder_path.glob('*'))):
        if 'summary' in experiment.stem:
            continue
        file_paths = list((experiment / config['phase']).glob('*.pkl'))
        file_path, split_id = find_pickle_file(file_paths, config)
        with open(file_path, "rb") as input_file:
            data = pickle.load(input_file)
        myStatCollection[split_id] = data

    # collect result for split e.g. s0
    result_dict = {}
    for split_key in myStatCollection.keys():
        for animal_key in myStatCollection[split_key].keys():
            if animal_key not in result_dict.keys():
                result_dict[animal_key] = {'es': Stats(),
                                           'ed': Stats()}
            result_dict[animal_key]['es'] += myStatCollection[split_key][animal_key]['es']
            result_dict[animal_key]['ed'] += myStatCollection[split_key][animal_key]['ed']

    return result_dict

def store_result_to_latex(result_dict, folder_path, config):
    output_key = 'summary'
    output_path = folder_path / output_key
    output_path.mkdir(exist_ok=True)

    #create csv files
    for species_key in ['dog', 'pig']:
        event_key = 'AVO_AVC'
        file_path = get_save_file_name(output_path, config, species_key, event_key, extention='.txt')
        avo = result_dict[species_key]['ed']
        avc = result_dict[species_key]['es']
        combine_AVC_ACO_result_to_latex_table(avo, avc, file_path, species_key)

    return


def combine_AVC_ACO_result_to_latex_table(avo, avc, file_path, species_key):
    # Combine results from avo and avc for latex table

    if species_key == 'dog':
        ordering_keys = ['baseline', 'dobutamine', 'ischemia', 'rvp', 'lbbb', 'crt', 'lbbbdob', 'lbbbisc', 'lbbbloading', 'all']
    elif species_key == 'pig':
        #ordering_keys = ['baseline', 'baseline(cc)', 'adrenaline', 'dobutamine', 'esmolol', 'ischemia', 'ischemiadob', 'niprid', 'loading', 'unloading', 'all']
        ordering_keys = ['baseline', 'baseline-closed-chest', 'adrenaline', 'dobutamine', 'esmolol', 'ischemia', 'ischemiadob', 'niprid', 'loading-closed-chest', 'unloading-closed-chest', 'all']

    else:
        raise ValueError('Unknown species')

    #check that both "avo" and "avc" include the same keys
    if not sorted(list(avo.dictOfStatObjects.keys())) == sorted(list(avc.dictOfStatObjects.keys())):
        raise ValueError('Mismatch in interventions')

    columnsTitles = ['path', 'Total # of animals', 'Total # of sequences',
                     'number og events', 'True detections [#]','True detections [%]', 'False detections [#]',
                     'False detections [%]', 'error mean abs [ms]','error std [ms]']

    columnsTitles_ordered = []
    for idx, key in enumerate(ordering_keys):
        statRow = OrderedDict()

        avo_dict = avo.dictOfStatObjects[key].get_stat_dict_for_csv()
        avc_dict = avc.dictOfStatObjects[key].get_stat_dict_for_csv()

        for column_key in columnsTitles[0:3]:
            statRow[column_key] = avo_dict[column_key]
            columnsTitles_ordered.append(column_key)

        for column_key in columnsTitles[3:]:
            statRow['AVO_'+column_key] = avo_dict[column_key]
            columnsTitles_ordered.append('AVO_'+column_key)

        for column_key in columnsTitles[3:]:
            statRow['AVC_'+column_key] = avc_dict[column_key]
            columnsTitles_ordered.append('AVC_' + column_key)

        if idx==0:
            df = pd.DataFrame(columns=columnsTitles_ordered)
        df = df.append(statRow, ignore_index=True)

    formatDict = dict(zip(columnsTitles_ordered, (debug,) * len(columnsTitles_ordered)))
    df.to_latex(file_path, encoding='ascii', formatters=formatDict)
    df.to_csv(str(file_path).replace('.txt', '.csv'), sep=',', index=True)
    return


def debug(x):
    print(x)
    return str(x)

def collect_results_tables_VI_and_VII():
    folder_path = Path('../experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00/net_102')

    #convert the flags dict to the "config" dict
    flags = get_flags()  # specifying the post processing parameters
    config = {'thres': int(flags['thresholds']['accept_threshold']*100),
              'dist': int(flags['thresholds']['noDetectDist']),
              'head': flags['output_head'],                            # 'cnn' | 'rnn' | 'att'
              'densityWinSize': flags['trigDensityWinSize'],
              'linearLoc': int(flags['linearLoc']),                    # 0 | 1
              'phase': 'test'}

    result_dict = collector(folder_path, config)
    store_result_to_latex(result_dict, folder_path, config)

    return

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    folder_path = Path('../experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00/net_102')

    #convert the flags dict to the "config" dict
    flags = get_flags()  # specifying the post processing parameters
    config = {'thres': int(flags['thresholds']['accept_threshold']*100),
              'dist': int(flags['thresholds']['noDetectDist']),
              'head': flags['output_head'],                            # 'cnn' | 'rnn' | 'att'
              'densityWinSize': flags['trigDensityWinSize'],
              'linearLoc': int(flags['linearLoc']),                    # 0 | 1
              'phase': 'test'}

    result_dict = collector(folder_path, config)
    store_result_to_latex(result_dict, folder_path, config)
