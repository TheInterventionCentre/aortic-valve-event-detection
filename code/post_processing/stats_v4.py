import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Stats():
    def __init__(self, dictOfStatObjects=None):
        if dictOfStatObjects is None:
            self.dictOfStatObjects = {}
        else:
            self.dictOfStatObjects = dictOfStatObjects
        return

    def update(self, tag, statCollection):
        #update with new statcollection
        if tag in self.dictOfStatObjects.keys():
            self.dictOfStatObjects[tag].update(statCollection)
        else:
            new_obj = StatObject(tag)
            new_obj.update(statCollection)
            self.dictOfStatObjects[tag] = new_obj
        return

    def printResult(self):
        totNumbSeq = 0
        for obj in self.dictOfStatObjects.values():
            obj.printResult()
            totNumbSeq += obj.totNumbOfSeq

        print('\n----------------')
        print(f'totNumbSeq for all locations== {totNumbSeq}')
        return

    def plot_histogram(self, name, max_value=100):

        for obj in self.listOfStatObjects:
            if obj.tag=='all':
                dist = obj.totdistList[0][0]

        fig, ax = plt.subplots(1,1)
        ax.hist(dist, bins=np.arange(0,max_value,1), label=f'# End-systole {len(dist)}')
        ax.legend()
        ax.set_xlabel('End-systole distance error  [ms]')
        ax.set_ylabel('Count [#]')
        ax.set_title(name.with_suffix('').name)
        plt.savefig(name)
        plt.close(fig)
        return

    def __add__(self, other):
        #copy object if self if empty
        if len(self.dictOfStatObjects) == 0:
            return Stats(other.dictOfStatObjects)

        #Update stats for each intervention
        for new_obj_key in other.dictOfStatObjects.keys():
            if new_obj_key in self.dictOfStatObjects.keys():
                self.dictOfStatObjects[new_obj_key].totNumbOfTrueTrigs += other.dictOfStatObjects[new_obj_key].totNumbOfTrueTrigs
                self.dictOfStatObjects[new_obj_key].totNumbOfFoundECG += other.dictOfStatObjects[new_obj_key].totNumbOfFoundECG
                self.dictOfStatObjects[new_obj_key].totNumbOfMissedECG += other.dictOfStatObjects[new_obj_key].totNumbOfMissedECG
                self.dictOfStatObjects[new_obj_key].totNumbOfnonTrueTrig += other.dictOfStatObjects[new_obj_key].totNumbOfnonTrueTrig
                self.dictOfStatObjects[new_obj_key].totNumbOfSeq += other.dictOfStatObjects[new_obj_key].totNumbOfSeq
                self.dictOfStatObjects[new_obj_key].totdistList += other.dictOfStatObjects[new_obj_key].totdistList
                if hasattr(other.dictOfStatObjects[new_obj_key], 'species'):
                    self.dictOfStatObjects[new_obj_key].species += other.dictOfStatObjects[new_obj_key].species
                    self.dictOfStatObjects[new_obj_key].att_species += other.dictOfStatObjects[new_obj_key].att_species

            else:
                self.dictOfStatObjects[new_obj_key] = other.dictOfStatObjects[new_obj_key]

        return Stats(self.dictOfStatObjects)

    def storeResultToExcel(self, fileName):

        for idx, obj in enumerate(self.dictOfStatObjects.values()):
            statRow = obj.get_stat_dict_for_csv()
            if idx==0:
                df = pd.DataFrame(columns=list(statRow.keys()))
            df = df.append(statRow, ignore_index=True)
            # df = df.concat(statRow, ignore_index=True)

        if len(self.dictOfStatObjects.values())>0:
            fileName.parent.mkdir(exist_ok=True)
            df.to_csv(fileName, sep=',', index=True)
        return


    def storeResultToLatex(self, fileName):
        columnsTitles = ['Location',
                        'Sub location',
                        'Modality',
                        'Total # of seq',
                        '# of bad seq',
                        '# of bad seq2',
                        '# of correct seq.',
                        '# of correct seq2',
                        '# True ECG trig',
                        '# of found ECG trig',
                        '# of found ECG trig2',
                        '# of nonTrue ECG trig',
                        '# of nonTrue ECG trig2',
                         'Accuracy 1',
                         'Accuracy 2']
        df = pd.DataFrame()
        for obj in self.dictOfStatObjects.values():
                statRow = obj.get_stat_dict_for_csv()
                df = df.append(statRow, ignore_index=True)

        df.reindex(columns=columnsTitles)
        def debug(x):
            print(x)
            return str(x)
        formatDict = dict(zip(columnsTitles, (debug,)*len(columnsTitles)))
        df.to_latex(fileName, index=False, encoding='ascii', formatters=formatDict)
        return

#######################################################################################################################
class StatObject():
    def __init__(self, tag):
        self.tag      = tag
        self.totNumbOfTrueTrigs    = 0
        self.totNumbOfFoundECG     = 0
        self.totNumbOfMissedECG    = 0
        self.totNumbOfnonTrueTrig  = 0
        self.totNumbOfSeq          = 0
        self.totdistList           = []
        self.att_species = []
        self.species = []
        return

    def update(self, statCollection):
        self.totNumbOfTrueTrigs   += statCollection['totalNumber']
        self.totNumbOfFoundECG    += statCollection['numbOfFound']
        self.totNumbOfMissedECG   += statCollection['numbOfMissed']
        self.totNumbOfnonTrueTrig += statCollection['numbOfnonTrue']
        self.totdistList += statCollection['distList']
        self.totNumbOfSeq += 1
        self.att_species += statCollection['att_species']
        self.species += statCollection['species']
        return

    def get_stat_dict_for_csv(self):
        statDict = {}
        statDict['path']                    = f'{self.tag}'
        statDict['Total # of animals']      = f'{self.totNumbOfSeq:.0f}'
        statDict['Total # of sequences']    = f'{self.totNumbOfSeq:.0f}'
        statDict['number og events']       = f'{self.totNumbOfTrueTrigs:.0f}'
        statDict['True detections [#]']     = f'{self.totNumbOfFoundECG:.0f}'
        statDict['True detections [%]']     = f'{self.totNumbOfFoundECG/self.totNumbOfTrueTrigs*100:.1f}'
        statDict['False detections [#]']    = f'{self.totNumbOfnonTrueTrig:.0f}'
        statDict['False detections [%]']    = f'{self.totNumbOfnonTrueTrig/self.totNumbOfTrueTrigs*100:.1f}'
        statDict['error mean abs [ms]']     = f'{abs(np.mean(np.abs(self.totdistList))):.1f}'
        statDict['error std [ms]']          = f'{np.sqrt(np.mean(np.asarray(self.totdistList) ** 2)):.1f}'
        return statDict

    def printResult(self):
        print('\n\n------------------------------------------------')
        print(self.tag)
        print(f'TrueTrigs = {self.totNumbOfTrueTrigs}')
        print(f'FoundECG = {self.totNumbOfFoundECG} ({self.totNumbOfFoundECG/self.totNumbOfTrueTrigs*100:.2f}%)')
        print(f'MissedECG = {self.totNumbOfMissedECG} ({self.totNumbOfMissedECG/self.totNumbOfTrueTrigs*100:.2f}%)')
        print(f'nonTrueTrig = {self.totNumbOfnonTrueTrig} ({self.totNumbOfnonTrueTrig/self.totNumbOfTrueTrigs*100:.2f}%)')
        print(f'meanAbs={abs(np.mean(self.totdistList)):.1f}ms')
        print(f'std = {np.sqrt(np.mean(np.asarray(self.totdistList) ** 2)):.1f}ms')
        print(f'Total number of sequences = {self.totNumbOfSeq}')
        return