from pathlib import Path
import pickle
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_pickle_data_to_json(folder_path):
    for series in Path(folder_path).glob('*'):
        for animal in series.glob('*'):
            for intervention in animal.glob('*'):
                for recording in intervention.glob('*'):
                    print(recording)
                    with open(str(recording), "rb") as input_file:
                        data = pickle.load(input_file)

                    # if 'ecg' in list(data.keys()):
                    #     if np.max(data['ecg'])==0 and np.min(data['ecg'])==0:
                    #         a = 1
                    # else:
                    #     a = 1


                    trimmed_dict = {}
                    headers = ['experiment_name', 'animal_species', 'sample_rate', 'intervention', 'identifier', 'acc_x', 'acc_y', 'acc_z', 'lvp' , 'ecg']
                    for key in headers:
                        trimmed_dict[key] = data[key]
                    parts = list(recording.parts)
                    parts[2] = 'data_json'
                    csv_file_path = '/'.join(parts[:-1]) + '/'
                    Path(csv_file_path).mkdir(parents=True, exist_ok=True)

                    json_save_name = csv_file_path + parts[-1] + '.json'
                    with open(json_save_name, "w") as outfile:
                        json.dump(trimmed_dict, outfile, cls=NumpyEncoder, indent=4)
    return


def convert_pickle_targets_to_json(folder_path):
    for series in Path(folder_path).glob('*'):
        for animal in series.glob('*'):
            for intervention in animal.glob('*'):
                for recording in intervention.glob('*'):
                    print(recording)
                    with open(str(recording), "rb") as input_file:
                        data = pickle.load(input_file)

                    trimmed_dict = {}
                    headers = ['ed', 'es']
                    for key in headers:
                        trimmed_dict[key] = data[key]
                    parts = list(recording.parts)
                    parts[2] = 'targets_json'
                    csv_file_path = '/'.join(parts[:-1]) + '/'
                    Path(csv_file_path).mkdir(parents=True, exist_ok=True)

                    json_save_name = csv_file_path + parts[-1] + '.json'
                    with open(json_save_name, "w") as outfile:
                        json.dump(trimmed_dict, outfile, cls=NumpyEncoder, indent=4)
    return

if __name__ == '__main__':
    folder_path = '../../data_pickle5'
    convert_pickle_data_to_json(folder_path)
    # convert_pickle_targets_to_json(folder_path)









