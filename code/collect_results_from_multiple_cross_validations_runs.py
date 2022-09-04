from pathlib import Path
import numpy as np
from colorama import Fore, Style
import copy
from collect_results_from_single_cross_validations_run import collector

def copy_config(config):
    # duplicate config for individual elements
    config_list = []
    is_list = False
    for key, value in config.items():
        if isinstance(value, list):
            x_axis = {key: value}
            if is_list:
                raise ValueError('Multiple config lists found')
            is_list = True
            for ii in range(len(value)):
                d = copy.deepcopy(config)
                d[key] = config[key][ii]
                config_list.append(d)
    return config_list, x_axis

def collect_results_detection_distance_limit_with_variance(folder_path, config):
    config_list, x_axis = copy_config(config)

    plotter_list = []
    cross_validation_folders = []
    for cross_validation_folder in sorted(list(folder_path.glob('*'))):
        cross_validation_folders.append(str(cross_validation_folder))
        plotter_dict = {}
        # plotter_dict = {'pig': {'es':{'true_detection': [],
        #         #                               'false_detection': [],
        #         #                               'mse': 0,
        #         #                               'std': 0,
        #         #                               },
        #         #                         'ed': {'true_detection': [],
        #         #                                'false_detection': [],
        #         #                                'mse': 0,
        #         #                                'std': 0,
        #         #                                }
        #         #                         },
        #         #                'dog': {'es': {'true_detection': [],
        #         #                               'false_detection': [],
        #         #                               'mse': 0,
        #         #                               'std': 0,
        #         #                               },
        #         #                         'ed': {'true_detection': [],
        #         #                                'false_detection': [],
        #         #                                'mse': 0,
        #         #                                'std': 0,
        #         #                                },
        #         #                         },
        #         #                 }
        for tmp_config in config_list:
            result_dict = collector(cross_validation_folder, tmp_config)
            for animal_key in result_dict.keys():
                for event_key in result_dict[animal_key].keys():
                    true_events  = result_dict[animal_key][event_key].dictOfStatObjects['all'].totNumbOfFoundECG
                    events       = result_dict[animal_key][event_key].dictOfStatObjects['all'].totNumbOfTrueTrigs
                    false_events = result_dict[animal_key][event_key].dictOfStatObjects['all'].totNumbOfnonTrueTrig
                    distList = result_dict[animal_key][event_key].dictOfStatObjects['all'].totdistList
                    if hasattr(result_dict[animal_key][event_key].dictOfStatObjects['all'], 'species'):
                        species_lab = result_dict[animal_key][event_key].dictOfStatObjects['all'].species
                        species_pred = result_dict[animal_key][event_key].dictOfStatObjects['all'].att_species

                    if animal_key not in plotter_dict.keys():
                        plotter_dict[animal_key] =  {'es': {'true_detection': [],
                                                           'false_detection': [],
                                                           'mse': 0,
                                                           'std': 0,
                                                           'species_acc': 0},
                                                    'ed': {'true_detection': [],
                                                           'false_detection': [],
                                                           'mse': 0,
                                                           'std': 0,
                                                           'species_acc': 0,
                                                           }
                                                     }

                    plotter_dict[animal_key][event_key]['true_detection'].append(true_events/events)
                    plotter_dict[animal_key][event_key]['false_detection'].append(false_events/events)

                    plotter_dict[animal_key][event_key]['mse'] = np.mean(np.abs(distList))
                    plotter_dict[animal_key][event_key]['std'] = np.std(distList)

                    if hasattr(result_dict[animal_key][event_key].dictOfStatObjects['all'], 'species'):
                        plotter_dict[animal_key][event_key]['species_acc'] = 1-np.sum(np.abs(np.array(species_pred)-np.array(species_lab))) / len(species_lab)

        plotter_list.append(plotter_dict)
    return plotter_list, x_axis, cross_validation_folders

def find_avg_performing_model(plotter_list, x_axis, cross_validation_folders, config):
    #find the average performing model for each of the cross validation runs (seeds)

    n_configs = len(list(x_axis.values())[0])
    n_samples =  len(plotter_list)
    summary_dict = {}

    animals_keys = list(plotter_list[0].keys())

    for ii, animal_key in enumerate(animals_keys):
        summary_dict[animal_key] = {}
        for event_key in ['ed', 'es']:
            true_detection = np.zeros(shape=(n_samples, n_configs))
            false_detection = np.zeros(shape=(n_samples, n_configs))
            ms_mse = np.zeros(shape=(n_samples, n_configs))
            ms_std = np.zeros(shape=(n_samples, n_configs))
            species_acc = np.zeros(shape=(n_samples, n_configs))

            for ind in range(len(plotter_list)):
                true_detection[ind,:] = np.array(plotter_list[ind][animal_key][event_key]['true_detection'])*100
                false_detection[ind, :] = np.array(plotter_list[ind][animal_key][event_key]['false_detection']) * 100
                ms_mse[ind] = plotter_list[ind][animal_key][event_key]['mse']
                ms_std[ind] = plotter_list[ind][animal_key][event_key]['std']
                species_acc[ind] = plotter_list[ind][animal_key][event_key]['species_acc']
            summary_dict[animal_key][event_key] = {}
            summary_dict[animal_key][event_key]['true_detection'] = true_detection
            summary_dict[animal_key][event_key]['false_detection'] = false_detection
            summary_dict[animal_key][event_key]['mse'] = ms_mse
            summary_dict[animal_key][event_key]['std'] = ms_std
            summary_dict[animal_key][event_key]['species_acc'] = species_acc


    event_map = {'ed': 'AVO',
                 'es': 'AVC'}
    animal_map = {'pig': 'porcine',
                  'dog': 'canines'}

    #print avg result and best match
    all_avg_percentage = {'mean': [], 'TD_mean': [], 'FD_mean': [], 'error': []}
    all_avg_percentage_species = {}
    for animal_key in animals_keys:
        all_avg_percentage_species[animal_key] = {'mean': [], 'TD_mean': [], 'FD_mean': [], 'error': []}

    for mode in ['true_detection', 'false_detection']:
        for animal_key in animals_keys:
            for event_key in ['ed', 'es']:
                preds = summary_dict[animal_key][event_key][mode]
                avg = np.mean(summary_dict[animal_key][event_key][mode], axis=0)
                std = np.std(summary_dict[animal_key][event_key][mode], axis=0)
                max_vals = np.max(summary_dict[animal_key][event_key][mode], axis=0)
                min_vals = np.min(summary_dict[animal_key][event_key][mode], axis=0)
                n = len(summary_dict[animal_key][event_key][mode])

                print(f'\n ------ {animal_map[animal_key]} ------- {event_map[event_key]} -----{mode} -------')
                for ii in range(len(x_axis['dist'])):
                    d = x_axis['dist'][ii]
                    outstr1 = f'detection distance: {d} ms | min={min_vals[ii]:0.2f}% | | {Fore.RED}avg= {avg[ii]:0.2f}% (u={std[ii]/np.sqrt(n-1):0.2f}%) (s={std[ii]:0.2f}%) {Style.RESET_ALL}'
                    outstr2 = f'| max={max_vals[ii]:0.2f}'

                    outstr3 = ''
                    if mode == 'true_detection':
                        ms_mse = summary_dict[animal_key][event_key]['mse']
                        ms_std = summary_dict[animal_key][event_key]['std']
                        outstr3 = f' | mse={np.mean(ms_mse):0.2f}({np.std(ms_mse):0.2f}) | std={np.mean(ms_std):0.2f}({np.std(ms_std):0.2f})'
                        all_avg_percentage['mean'].append(100-preds[..., ii])
                        all_avg_percentage['TD_mean'].append(preds[..., ii])
                        all_avg_percentage['error'].append(ms_mse[..., ii])
                        all_avg_percentage_species[animal_key]['mean'].append(100-preds[..., ii])
                        all_avg_percentage_species[animal_key]['TD_mean'].append(preds[..., ii])
                        all_avg_percentage_species[animal_key]['error'].append(ms_mse[..., ii])
                    else:
                        all_avg_percentage['mean'].append( preds[..., ii] )
                        all_avg_percentage['FD_mean'].append( preds[..., ii] )
                        all_avg_percentage_species[animal_key]['mean'].append(preds[..., ii])
                        all_avg_percentage_species[animal_key]['FD_mean'].append(preds[..., ii])

                    outstr4 = f"  |  species_acc={np.mean( summary_dict[animal_key][event_key]['species_acc']):0.4f}"

                    print(outstr1+outstr2+outstr3+outstr4)

    outstring = '\nall' + '|  '
    for key in ['TD_mean', 'FD_mean', 'mean', 'error']:
        vals = all_avg_percentage[key]
        vals = np.stack(vals, axis=0)
        vals = np.mean(vals, axis=0)
        N = len(vals)
        outstring += f'& {np.mean(vals):0.2f} $\\pm$ {np.std(vals) / np.sqrt(N):0.2f} '
    outstring += '  \\\ '
    print(outstring)

    for animal_key in animals_keys:
        outstring = animal_key + '|  '
        for key in ['TD_mean', 'FD_mean', 'error']:
            vals = all_avg_percentage_species[animal_key][key]
            vals = np.stack(vals, axis=0)
            vals = np.mean(vals, axis=0)
            N = len(vals)
            outstring += f'& {np.mean(vals):0.2f} $\\pm$ {np.std(vals) / np.sqrt(N):0.2f} '
        outstring += '  \\\ '
        print(outstring)

    #find the "best" cross validation path
    print('\n')
    print(config)
    print('---------------------------------------------------------')
    vals = all_avg_percentage['mean']
    vals = np.stack(vals, axis=0)
    vals = np.mean(vals, axis=0)
    ind = np.argmin(vals)
    print(ind)
    print(cross_validation_folders[ind])
    outstring = 'all' + '|  '
    for key in ['TD_mean', 'FD_mean', 'mean', 'error']:
        vals = all_avg_percentage[key]
        vals = np.stack(vals, axis=0)
        vals = np.mean(vals, axis=0)
        outstring += f'& {vals[ind]:0.2f}   '
    outstring += '  \\\ '
    print(outstring)

    for animal_key in animals_keys:
        outstring = animal_key + '|  '
        for key in ['TD_mean', 'FD_mean', 'error']:
            vals = all_avg_percentage_species[animal_key][key]
            vals = np.stack(vals, axis=0)
            vals = np.mean(vals, axis=0)
            outstring += f'& {vals[ind]:0.2f}   '
        outstring += '  \\\ '
        print(outstring)

    return

if __name__ == '__main__':

    folder_path = Path('../experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00')

    config = {'thres': 40,
              'dist': [40],
              'head': 'att',
              'densityWinSize': 29,
              'linearLoc': 0,
              'phase': 'test'}

    plotter_list, x_axis, cross_validation_folders = collect_results_detection_distance_limit_with_variance(folder_path, config)
    find_avg_performing_model(plotter_list, x_axis, cross_validation_folders, config)
