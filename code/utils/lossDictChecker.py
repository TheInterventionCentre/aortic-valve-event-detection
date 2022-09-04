import torch

def lossDictChecker(inputDict):
    flag = 0
    for key in inputDict.keys():
        if torch.isnan(inputDict[key]):
            flag = 1
    return


def variableChecker(inputDict, predDict, dataDictCuda, net):
    flag = 0
    #check if any weight parameters or weight gradients are nan
    for name, param in net.named_parameters():
        if torch.isnan(param.grad).sum()>0:
            raise ValueError('Found network gradients to be NaN')
            flag = 1

    for name, param in net.named_parameters():
        if torch.isnan(param.data).sum()>0:
            raise ValueError('Found network weights to be NaN')
            flag = 1

    return flag

