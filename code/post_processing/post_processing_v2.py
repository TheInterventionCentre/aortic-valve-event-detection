from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from post_processing.utils import calculateEstimationDensity
from post_processing.utils import calculate_confidence_from_prediction_density
from post_processing.utils import refine_pixels_based_on_confidences


def get_flags(head='att'):
    """
    The function returns a dict specifying the post processing parameters.
    """

    flags = {}
    flags['thresholds'] = {}
    flags['thresholds']['uncertainBorderWidth'] = 300  # default 300ms. The distance from the beginning and end of the
                                                       # sequences not taking into account in the performace statistics.
    flags['thresholds']['noDetectDist'] = 40           # distance to no detection [ms]. Called "Detection distance limit"
    flags['thresholds']['accept_threshold'] = 0.40     # confidence score threshold
    flags['trigDensityWinSize'] = 29           # Length of the prediction density smoothing filter. Note: should be odd
    flags['linearLoc'] = False              # False = The model predict the location | True = The location is set to be in the middle of the patch (0.5).
    flags['debug'] = False # False | True
    flags['store_figures'] = False
    flags['output_head'] = head  # 'rnn', 'cnn', 'att'

    return flags

def postproc(predDict, dataDict, ax, fig, flags, folder_path):
    """
    Combine patch predictions to final es/ed estimates.
    """
    if flags['debug']:
        scatter_plotter = Scatter_plotter()
    sample_ind = 0
    ms_per_pixel = dataDict['ms_per_pixel'][sample_ind]

    output_dict = {}
    nx = dataDict['acc_mag'].shape[1]
    pred_keys = predDict[''].keys()
    for event_key in ['es', 'ed']:
        # find location and confidence prediction from predDict
        keys = []
        for pred_key in pred_keys:
            if (event_key in pred_key) and (flags['output_head'] in pred_key) and 'species' not in pred_key:
                keys.append(pred_key)
        if len(keys) == 0:
            continue

        key_location = [key for key in keys if 'loc' in key][0]
        key_confidence = [key for key in keys if 'conf' in key][0]
        location = predDict[''][key_location]
        confidence = predDict[''][key_confidence]
        numb_of_patches = confidence['value'].shape[1]
        confidence_list = []
        pixel_list = []
        for jj in range(numb_of_patches):
            dx = predDict['']['dx']['value']
            fov = predDict['']['fov']['value']
            offset = jj * dx
            c = confidence['value'][sample_ind, jj]
            if c > 0.5:
                pixel = offset + location['value'][sample_ind, jj] * fov
                pixel_list.append(pixel)
                confidence_list.append(c)

        density = calculateEstimationDensity(pixel_list, confidence_list, flags['trigDensityWinSize'], nx)
        est_pixels, est_confidence = calculate_confidence_from_prediction_density(density)

        output_dict[event_key] = {}
        output_dict[event_key]['est_pixels'] = est_pixels
        output_dict[event_key]['est_confidence'] = est_confidence

        pixels = dataDict[event_key][sample_ind, :]
        count = dataDict['number_of_'+ event_key][sample_ind]
        output_dict[event_key]['label_pixels'] = pixels[0:count]
        output_dict[event_key]['NN_pixels'] = pixel_list
        output_dict[event_key]['NN_confidence'] = confidence_list
        output_dict[event_key]['density'] = density


    #with refinement
    output_dict = refine_pixels_based_on_confidences(output_dict, ms_per_pixel, flags['thresholds']['accept_threshold'])

    #plot figure
    if flags['debug']:
        for event_key in ['es', 'ed']:
            color_dict = {'ed': 'red', 'es': 'm'}
            for p, c in zip(output_dict[event_key]['NN_pixels'], output_dict[event_key]['NN_confidence']):
                scatter_plotter.plot(ax[0], p, c, event_key)
            ax[0].set_xlim([0, nx - 1])
            ax[0].set_ylim([0, 1.1])
            ax[0].grid(True)
            ax[0].set_ylabel('window \nconfidences')

            ' ------------------plot initial estimated------------------'
            ax[1].plot(output_dict[event_key]['density'], 'r', label=f'density-{event_key}', color=color_dict[event_key])
            ax[1].plot(output_dict[event_key]['est_pixels'], output_dict[event_key]['est_confidence'], '.r', label=f'criterion-{event_key}', color=color_dict[event_key])

            #search for missing events
            for lab in output_dict[event_key]['label_pixels']:
                m = np.min(np.abs((lab-output_dict[event_key]['est_pixels'])))
                if m*ms_per_pixel>flags['thresholds']['noDetectDist']:
                    ax[1].axvline(x=lab, color=color_dict[event_key])

            #search for non true event
            for p,c in zip(output_dict[event_key]['est_pixels'], output_dict[event_key]['est_confidence']):
                m = np.min(np.abs((p - output_dict[event_key]['label_pixels'])))
                if m * ms_per_pixel > flags['thresholds']['noDetectDist']:
                    ax[1].plot(p, 2, '*', color=color_dict[event_key], markersize=10)

            ax[1].grid(True)
            ax[1].set_ylim([0, 2.5])
            ax[1].set_xlim([0, nx - 1])
            ax[1].axhline(y=flags['thresholds']['accept_threshold'], color='k', linestyle='-', label='accept threshold')
            ax[1].set_ylabel('window confidences')

            # ------------------plot refined estimated------------------
            # output_dict = refine_pixels_based_on_confidences(output_dict, ms_per_pixel, flags['thresholds']['accept_threshold'])
            ax[2].plot(output_dict[event_key]['density'], 'r', label=f'density-{event_key}', color=color_dict[event_key])
            ax[2].plot(output_dict[event_key]['est_pixels'], output_dict[event_key]['est_confidence'], '.r', label=f'criterion-{event_key}', color=color_dict[event_key])
            #search for missing events
            for lab in output_dict[event_key]['label_pixels']:
                m = np.min(np.abs((lab-output_dict[event_key]['est_pixels'])))
                if m*ms_per_pixel>flags['thresholds']['noDetectDist']:
                    ax[2].axvline(x=lab, color=color_dict[event_key])

            #search for non true event
            for p,c in zip(output_dict[event_key]['est_pixels'], output_dict[event_key]['est_confidence']):
                m = np.min(np.abs((p - output_dict[event_key]['label_pixels'])))
                if m * ms_per_pixel > flags['thresholds']['noDetectDist']:
                    ax[2].plot(p, 2, '*', color=color_dict[event_key], markersize=10)
            ax[2].grid(True)
            ax[2].set_ylim([0, 2.5])
            ax[2].set_xlim([0, nx - 1])
            ax[2].axhline(y=flags['thresholds']['accept_threshold'], color='k', linestyle='-', label='accept threshold')

        #plot the rest
        for hh in range(6):
            if hh==0:
                ax[hh].axvline(x=flags['thresholds']['uncertainBorderWidth'] / ms_per_pixel, color='k', linestyle='--',label='uncertainBorderWidth')
                ax[hh].axvline(x=nx - flags['thresholds']['uncertainBorderWidth'] / ms_per_pixel, color='k', linestyle='--',label='uncertainBorderWidth')
            else:
                ax[hh].axvline(x=flags['thresholds']['uncertainBorderWidth'] / ms_per_pixel, color='k', linestyle='--')
                ax[hh].axvline(x=nx - flags['thresholds']['uncertainBorderWidth'] / ms_per_pixel, color='k',linestyle='--')

        p = dataDict['path'][sample_ind]
        # fig.suptitle(f'{Path(p).parent.stem} - {Path(p).stem} \n\n Time between [sec]: {dataDict["time"][0,0]:0.1f} - {dataDict["time"][0,-1]:0.1f}')
        fig.suptitle(f'{Path(p).parent.stem} - {Path(p).stem}')
        scatter_plotter.add_legend()
        for ind, data_key in enumerate(['acc_mag', 'lvp', 'ecg']):
            if data_key in dataDict.keys():
                data = dataDict[data_key][sample_ind,:]
            else:
                continue
            if len(data.shape)==3:
                ax[3 + ind].imshow(data[0,:,:], label=data_key)
                ax[3 + ind].set_aspect('auto')
            else:
                if data_key=='lvp':
                    ax[3 + ind].plot(data[1:]-data[:-1], color='b', label='lv dp/dt')
                else:
                    ax[3+ind].plot(data, color='b', label=data_key)
            for key in color_dict.keys():
                pixels = dataDict[key][sample_ind, :]
                count = dataDict['number_of_'+key][sample_ind]
                for kk in range(count):
                    ax[3+ind].axvline(x=pixels[kk], color=color_dict[key])
            ax[3 + ind].legend(loc=1)
            ax[3 + ind].set_xlim([0, nx - 1])
        a = 1

    if flags['store_figures']:
        p = dataDict['path'][sample_ind]
        figure_path = Path(folder_path) / 'figures'
        figure_path.mkdir(exist_ok=True)
        path = figure_path / f'{Path(p).parent.stem} - {Path(p).stem}.png'
        plt.savefig(path)

    return output_dict

########################################################################################################################

class Scatter_plotter():
    def __init__(self):
        self.handles = []
        self.handle_names = []
        self.ax = None
        self.color_dict = {'ed': 'red', 'es': 'm'}
        return

    def plot(self, ax, pixel, y, color_key):
        handle = ax.scatter(pixel, y, clip_on=False, color=self.color_dict[color_key])
        self.ax = ax
        if color_key not in self.handle_names:
            self.handle_names.append(color_key)
            self.handles.append(handle)
        return

    def add_legend(self):
        if len(self.handles)>0:
            self.ax.legend(self.handles, self.handle_names, bbox_to_anchor=(0.1, 1.1, 0.8, .102), ncol=6, mode="expand", borderaxespad=0.)
        return