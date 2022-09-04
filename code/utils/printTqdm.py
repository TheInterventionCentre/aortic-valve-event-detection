def printTDQM(mode,_lossesDict, _metricDict, _miscDict, pbar, ii, epoch, numbTrainSamples, trainBatchSize):

    currentDicts = [_lossesDict]
    currentDictsname = ['Loss: |  ', 'Acc:  |  ', '  misc  |  ']
    printStr = ''
    for kk in range(len(currentDicts)):
        currentDict = currentDicts[kk]
        printStr = printStr + f' {epoch} | {mode} | {currentDictsname[kk]}'
        keys = currentDict.keys()
        for key in keys:
            if key not in ['confusionMatrix']:
                if isinstance(currentDict[key], dict):
                    subKeys = currentDict[key].keys()
                    for subKey in subKeys:
                        printStr = printStr + f'{key} - {subKey:23}  = {currentDict[key][subKey]:.3f} | '
                else:
                    printStr = printStr + f'{key} = {currentDict[key]:.3f} | '

    text_string = printStr + f'{ii/(numbTrainSamples/trainBatchSize):.3f}'
    if len(text_string)> pbar.ncols:
        pbar.set_description(text_string[:pbar.ncols-40])
    else:
        pbar.set_description(text_string)
    pbar.refresh()
    return