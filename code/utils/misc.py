import torch
from pprint import pprint
from datetime import datetime
from pathlib import Path
import copy

def dictToCpu(inDict):
    """ Returns a dict where all instances are converted to numpy """
    outDict = {}
    for key in inDict.keys():
        if type(inDict[key]) is dict:
            outDict[key] = dictToCpu(inDict[key])
        else:
            if isinstance(inDict[key], (torch.Tensor)):
                outDict[key] = inDict[key].detach().cpu().numpy()
            else:
                outDict[key] = inDict[key]
    return outDict

def dictToCuda(inDict, device):
    # Creates a new dictionary where the pytorch tensors are passed to the given device.
    # Instances which not are pytoch tensors will be kept at the CPU.
    outDict = {}
    for key in inDict.keys():
        if type(inDict[key]) is dict:
            outDict[key] = dictToCuda(inDict[key], device)
        else:
            if torch.is_tensor(inDict[key]):
                outDict[key] = inDict[key].to(device)
            else:
                outDict[key] = inDict[key]
    return outDict



########################################################################
def get_experiment_path(cfg):
    """ This function returns the experiment path based on the configuration file """
    folder = cfg.experiment.save_dir
    prefix = cfg.experiment.experiment_prefix
    keys   = cfg.experiment.experiment_suffix
    path   = ''

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for key in keys:
        element = cfg
        for key_element in key.split("."):
            if isinstance(element, list):
                for e in element:
                    if key_element in e.keys():
                        element = e
                        break
            element = element[key_element]

        if isinstance(element, str):
            # remove characters before "."
            element = element.split('.')[-1]
            path += '_' + element
        elif isinstance(element, int):
            path += '_' + key_element+str(element)
        elif isinstance(element, float):
            path += '_' + key_element+str(element)
        elif isinstance(element, list):
            path += '_'
            for elm in element:
                path += str(elm)
        elif isinstance(element, dict):
            path += ''
            for elKey, elVal in element.items():
                path += '_'+str(elKey)
        else:
            raise Exception('Unknown element in config')
        path += ''
    return Path(folder) / Path(date_time+prefix+path+'/')

########################################################################
def parse_data_paths(cfg):
    """ Duplicates the config dict with different sets of data_path.

    Args:
        cfg: standard config dict

    Returns
        cfg_list: A list of config dicts were each instance have different train/val/test data_path.

    The config dict should have the key 'data_path'. Example (yaml layout):
    data_path:
        csv: '../data_split_v5.csv'
        train: ['s1', 's2', 's3', 's4']
        val:   ['s5']
        test:  ['s0']

    The paths will be split based on s#. The initial split between train, val, and test will have no effect.
    """

    all_paths        = sorted([i for k, v in cfg['data_path'].items() if k in ['train', 'val', 'test'] for i in v])

    #find all combinations
    cfg_list = []
    for ind, test_id in enumerate(all_paths):

        test = all_paths[ind]
        if ind==0:
            val = all_paths[-1]
        else:
            val = all_paths[ind-1]

        train = copy.deepcopy(all_paths)
        train.remove(test)
        train.remove(val)

        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.data_path.train = train
        cfg_copy.data_path.val   = [val]
        cfg_copy.data_path.test  = [test]
        cfg_list.append(cfg_copy)

        # print path set
        print(f'------------ set={ind} --------------')
        print('test')
        pprint(test)
        print('val')
        pprint(val)
        print('train')
        pprint(train)

    return cfg_list


##################################################################################################################################
class calculateAVGstats():
    def __init__(self):
        self.is_initialized = 0
        return

    def initDictFromKeys(self, inDict, initType):
        newDict = {}
        for key in inDict.keys():
            if initType=='zero':
                newDict[key] = 0
            elif initType=='list':
                newDict[key] = []
        return newDict

    def update(self, lossesDict, accuracyDict, batchSize):
        if self.is_initialized == 0:
            self.is_initialized = 1
            self.sampleCounter = 0
            self.avgLossesDict = self.initDictFromKeys(lossesDict, 'zero')
            self.avgAccuracyDict = self.initDictFromKeys(accuracyDict, 'zero')

            # these dictioneries holds information of the statistics from all epochs (rounds between "getAVGstats" is called)
            self.avgLossesDictList = self.initDictFromKeys(lossesDict, 'list')
            self.avgAccuracyDictList = self.initDictFromKeys(accuracyDict, 'list')

        #Updates all values by adding with new values from the current batch
        self.sampleCounter = self.sampleCounter + batchSize
        self.avgLossesDict   = self.sum2Dict(self.avgLossesDict, lossesDict, batchSize)      #dict sum
        self.avgAccuracyDict = self.sum2Dict(self.avgAccuracyDict, accuracyDict, batchSize)  #dict sum
        return

    def getAVGstats(self):
        # return the result by dividing on the numb of samples
        dictList = [self.avgLossesDict, self.avgAccuracyDict]
        for currentDicy in dictList:
            keys = currentDicy.keys()
            for key in keys:
                if 'confusionMatrix' not in key:
                    currentDicy[key] = currentDicy[key].item() / self.sampleCounter

        avgLossesDictOut   = dict(self.avgLossesDict)
        avgAccuracyDictout = dict(self.avgAccuracyDict)



        #add to the "DictLists"
        for key in self.avgLossesDictList.keys():
            self.avgLossesDictList[key].append(avgLossesDictOut.get(key))

        for key in self.avgAccuracyDictList.keys():
            self.avgAccuracyDictList[key].append(avgAccuracyDictout[key])

        # initialize the dicts by removing all data from them
        self.avgLossesDict   = self.initDictFromKeys(self.avgLossesDict, 'zero')
        self.avgAccuracyDict = self.initDictFromKeys(self.avgAccuracyDict, 'zero')
        self.sampleCounter = 0
        return avgLossesDictOut, avgAccuracyDictout

    def sum2Dict(self, orgDict, newDict, batchSize):
        outDict = {}
        for key in orgDict.keys():
            if 'confusionMatrix' not in key:
                # outDict[key] = orgDict[key] + newDict[key].data * batchSize
                if torch.is_tensor(newDict[key]):
                    outDict[key] = orgDict[key] + newDict[key].detach()* batchSize
                else:
                    outDict[key] = orgDict[key] + newDict[key] * batchSize
                # newDict[key].cpu().data.numpy()
                # newDict[key].item()
            else:
                outDict[key] = orgDict[key] + newDict[key].detach()
        return outDict