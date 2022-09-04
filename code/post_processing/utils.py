import numpy as np
from scipy.signal import tukey

def calculateConfidenceMeasuresAndReturnECGTrigs(confidence, flags, msPerPixel):
    '''
    Calculate for each region:
    - area
    - max heigh
    - centerOfMass
    :return:
    '''
    threshold            = flags['thresholds']['accept_threshold']

    nx = confidence.shape[0]

    areaList = []
    maxheighList = []
    centerOfMassList = []
    stdList = []
    criterionlist = []
    maxHeightIndList = []

    nonZeroInds = np.nonzero(confidence)[0]
    newArea = 1
    for ii in range(len(nonZeroInds)):
        if newArea == 1:
            tmpConfidence = []
            tmpInds       = []
        newArea = 0
        ind = nonZeroInds[ii]
        # tmpInds.append(bin_edges[ind])
        tmpInds.append(ind)
        tmpConfidence.append(confidence[ind])

        if ii + 1 < len(nonZeroInds):
            if nonZeroInds[ii + 1] > nonZeroInds[ii] + 1:
                newArea = 1
                area, maxheight, std, centerOfMass, maxHeightInd = calcStats(tmpInds, tmpConfidence)
                criterion = area/np.maximum(np.sqrt(std+0.1), 3)
                # criterion = maxheight
                if criterion > threshold and centerOfMass>=0 and centerOfMass<nx:
                    areaList.append(area)
                    maxheighList.append(maxheight)
                    centerOfMassList.append(centerOfMass)
                    stdList.append(std)
                    criterionlist.append(criterion)
                    maxHeightIndList.append(maxHeightInd)
        elif ii + 1 == len(nonZeroInds): #last index in the sequence
            newArea = 1
            area, maxheight, std, centerOfMass, maxHeightInd = calcStats(tmpInds, tmpConfidence)
            # criterion = area / np.maximum(np.sqrt(std+0.001), 3)
            criterion = area / np.sqrt(std + 0.001)
            # criterion = maxheight
            if criterion > threshold and centerOfMass >= 0 and centerOfMass < nx:
                areaList.append(area)
                maxheighList.append(maxheight)
                centerOfMassList.append(centerOfMass)
                stdList.append(std)
                criterionlist.append(criterion)
                maxHeightIndList.append(maxHeightInd)

    confidenceDict = {}
    confidenceDict['area']         = np.array(areaList)
    confidenceDict['maxheigh']     = np.array(maxheighList)
    confidenceDict['centerOfMass'] = np.array(centerOfMassList)
    confidenceDict['std']          = np.array(stdList)
    confidenceDict['criterion']    = np.array(criterionlist)
    confidenceDict['maxHeightInd'] = np.array(maxHeightIndList)
    return confidenceDict['centerOfMass'], confidenceDict


def calculate_confidence_from_prediction_density(density):
    nx = density.shape[0]
    centerOfMassList = []
    criterionlist = []

    nonZeroInds = np.nonzero(density)[0]
    newArea = 1
    for ii in range(len(nonZeroInds)):
        if newArea == 1:
            tmpConfidence = []
            tmpInds = []
        newArea = 0
        ind = nonZeroInds[ii]
        tmpInds.append(ind)
        tmpConfidence.append(density[ind])

        if ii + 1 < len(nonZeroInds):
            if nonZeroInds[ii + 1] > nonZeroInds[ii] + 1:
                newArea = 1
                area, maxheight, std, centerOfMass, maxHeightInd = calcStats(tmpInds, tmpConfidence)
                criterion = area / np.maximum(np.sqrt(std + 0.1), 3)
                # criterion = maxheight
                if criterion > 0 and centerOfMass >= 0 and centerOfMass < nx:
                    centerOfMassList.append(centerOfMass)
                    criterionlist.append(criterion)

        elif ii + 1 == len(nonZeroInds):  # last index in the sequence
            newArea = 1
            area, maxheight, std, centerOfMass, maxHeightInd = calcStats(tmpInds, tmpConfidence)
            criterion = area / np.sqrt(std + 0.001)

            if criterion > 0 and centerOfMass >= 0 and centerOfMass < nx:
                centerOfMassList.append(centerOfMass)
                criterionlist.append(criterion)

    return np.array(centerOfMassList), np.array(criterionlist)

def refine_pixels_based_on_confidences(output_dict, ms_per_pixel, initial_threshold=0.4):
    #post prosessing code for cleaning predictions

    refine_dict = {}
    refine_dict['es'] = {}
    refine_dict['ed'] = {}

    # threshold predictios with initial threshold
    for key in ['es', 'ed']:
        inds = np.where(output_dict[key]['est_confidence']>initial_threshold)[0]
        refine_dict[key]['est_confidence'] = output_dict[key]['est_confidence'][inds]
        refine_dict[key]['est_pixels']     = output_dict[key]['est_pixels'][inds]

    #remove close predictions, keep highest conf
    min_lim_pixels = 300 / ms_per_pixel
    for key in ['es', 'ed']:
        pixels = refine_dict[key]['est_pixels']
        confidence = refine_dict[key]['est_confidence']
        pixels_out, confidence_out = remove_close_predictions(pixels, confidence, min_lim_pixels)
        refine_dict[key]['est_pixels'] = pixels_out
        refine_dict[key]['est_confidence'] = confidence_out

    # If 2 consecutive are found, add a prediction with lower confidence.
    #if missing event between two other, add the one with highest conf, if noe est event, padd?
    #add 'es' first
    for mother_key, child_key in [['ed', 'es'], ['es', 'ed']]:
        for region_ind in range(len(refine_dict[mother_key]['est_pixels'])-1):
            region_start = refine_dict[mother_key]['est_pixels'][region_ind]
            region_end   = refine_dict[mother_key]['est_pixels'][region_ind+1]

            #most likely a wrong estimate
            if (region_end-region_start)<min_lim_pixels:
                continue

            #check if event is between "region_start" and "region_end"
            refine_ind = np.where((refine_dict[child_key]['est_pixels']>region_start) * (refine_dict[child_key]['est_pixels']<region_end))[0]
            if len(refine_ind)==0:
                output_ind = np.where((output_dict[child_key]['est_pixels']>region_start) * (output_dict[child_key]['est_pixels']<region_end))[0]
                if len(output_ind)!=0:
                    ind_to_max_confidence = output_ind[np.argmax(output_dict[child_key]['est_confidence'][output_ind])]

                    #add and sort
                    new_pixels = output_dict[child_key]['est_pixels'][ind_to_max_confidence]
                    new_conf = output_dict[child_key]['est_confidence'][ind_to_max_confidence]

                    refine_dict[child_key]['est_pixels']     = np.append(refine_dict[child_key]['est_pixels'],new_pixels)
                    refine_dict[child_key]['est_confidence'] = np.append(refine_dict[child_key]['est_confidence'],new_conf)

                    sort_inds = np.argsort(refine_dict[child_key]['est_pixels'])
                    refine_dict[child_key]['est_pixels'] = refine_dict[child_key]['est_pixels'][sort_inds]
                    refine_dict[child_key]['est_confidence'] = refine_dict[child_key]['est_confidence'][sort_inds]

    #copy over to output_dict
    for key in ['es', 'ed']:
        output_dict[key]['est_pixels'] = refine_dict[key]['est_pixels']
        output_dict[key]['est_confidence'] = refine_dict[key]['est_confidence']

    return output_dict

def remove_close_predictions(pixels, confidence, min_lim_pixels):
    # The function removes the pixels and the corresponding confidence of pixels with closer distance than "min_lim_pixels"
    out_pixels = []
    out_conf = []
    # find group of close predictions
    ii = 0
    N = pixels.shape[0]
    groups = []
    group = []
    while ii < N - 1:
        if pixels[ii + 1] - pixels[ii] < min_lim_pixels:
            if ii not in group:
                group.append(ii)
            group.append(ii + 1)
            ii += 1
            if ii == N - 1:
                groups.append(group)
        else:
            if len(group) > 0:
                groups.append(group)
            group = []
            ii += 1
    # find max confidence in each group
    for group in groups:
        group_ind = np.argmax(confidence[group])
        out_ind = group[group_ind]
        out_pixels.append(pixels[out_ind])
        out_conf.append(confidence[out_ind])

    # add pixel and conf values outside of the groups
    all_group_inds = [item for sublist in groups for item in sublist]
    for ii in range(N):
        if ii not in all_group_inds:
            out_pixels.append(pixels[ii])
            out_conf.append(confidence[ii])

    # sort the out arrays
    out_pixels = np.array(out_pixels)
    out_conf = np.array(out_conf)

    sort_inds = np.array(np.argsort(out_pixels))
    out_pixels = out_pixels[sort_inds]
    out_conf = out_conf[sort_inds]
    return out_pixels, out_conf


def calcStats(tmpInds, tmpConfidence):
    area      = np.sum(tmpConfidence)
    maxheight = np.max(tmpConfidence)
    centerOfMass = 0
    std          = 0

    maxHeightInd = int(tmpInds[np.argmax(tmpConfidence)])
    for ii in range(len(tmpInds)):
        centerOfMass += tmpInds[ii]*tmpConfidence[ii]
    centerOfMass = centerOfMass/area
    for ii in range(len(tmpInds)):
        std += tmpConfidence[ii]*(tmpInds[ii]-centerOfMass)**2
    std = std/area
    return area, maxheight, std, int(centerOfMass), maxHeightInd

########################################################################################################################
def getErrors(est_pixels, est_confidence, label_pixels, msPerPixel, thresholds, Nx):
    uncertainBorderWidth = thresholds['uncertainBorderWidth']
    noDetectDist         = thresholds['noDetectDist']

    numbOfFound   = 0  # number of found ecg trig
    numbOfMissed  = 0  # ECG trig that was not detected
    numbOfnonTrue = 0  # estimated trigs that was fals
    totalNumber   = 0
    distList = []

    if len(est_pixels) == 0:
        for ii in range(len(label_pixels)):
            if uncertainBorderWidth < label_pixels[ii] * msPerPixel and label_pixels[ii] * msPerPixel < Nx * msPerPixel - uncertainBorderWidth:
                totalNumber += 1
                numbOfMissed = numbOfMissed + 1
    else:
        for ii in range(len(label_pixels)):
            minAbsDistance = min(abs(est_pixels - label_pixels[ii])) * msPerPixel
            argInd = np.argmin(abs(est_pixels - label_pixels[ii]))
            minDistance = (est_pixels[argInd] - label_pixels[ii]) * msPerPixel

            if minAbsDistance < noDetectDist:
                if uncertainBorderWidth < label_pixels[ii]*msPerPixel and label_pixels[ii]*msPerPixel<Nx*msPerPixel-uncertainBorderWidth:
                    numbOfFound = numbOfFound + 1
                    distList.append(minDistance)
                    totalNumber += 1

            else:
                if uncertainBorderWidth < label_pixels[ii]*msPerPixel and label_pixels[ii]*msPerPixel<Nx*msPerPixel-uncertainBorderWidth:
                    numbOfMissed = numbOfMissed + 1
                    totalNumber += 1

        # find estimated trigs that failed
        if len(label_pixels)==0:
            print('No labels....')
            for ii in range(len(est_pixels)):
                if uncertainBorderWidth < est_pixels[ii]*msPerPixel and est_pixels[ii]*msPerPixel<Nx*msPerPixel-uncertainBorderWidth:
                    numbOfnonTrue = numbOfnonTrue + 1
            return 0, 0, numbOfnonTrue, [], 0

        # find Non-True
        for ii in range(len(est_pixels)):
            minAbsDistance = min(abs(est_pixels[ii] - label_pixels)) * msPerPixel
            if minAbsDistance > noDetectDist:
                if uncertainBorderWidth < est_pixels[ii]*msPerPixel and est_pixels[ii]*msPerPixel<Nx*msPerPixel-uncertainBorderWidth:
                    numbOfnonTrue = numbOfnonTrue + 1
    return numbOfFound, numbOfMissed, numbOfnonTrue, distList, totalNumber

########################################################################################################################
def movingaverage(values, window, winType='rect', padding='original'):

    if window%2== 0:
        raise Exception('window size is even, must be odd')

    if winType=='rect':
        weights = np.repeat(1.0, window)/window
    if winType == 'tukey':
        weights = tukey(M=window, alpha=0.8)
        weights = weights/np.sum(weights)

    sma = np.convolve(values, weights, 'valid')
    oneSide = int(np.floor(window/2))

    #fix that the expectced value at the edges are lower
    leftSide = np.zeros(oneSide)
    for ind in range(oneSide):
        leftSide[ind] = np.sum(values[0:oneSide+ind] * weights[oneSide-ind+1:] / np.sum(weights[oneSide-ind+1:]))

    rightSide = np.zeros(oneSide)
    for ind in range(oneSide):
        rightSide[ind] = np.sum(values[-oneSide-ind:] * weights[0:oneSide+ind] / np.sum(weights[0:oneSide+ind]))
    rightSide = np.flipud(rightSide)

    return np.concatenate((leftSide, sma, rightSide))

########################################################################################################################
def calculateEstimationDensity(pixel_list, confidence_list, filter_len, nx):

    #sort the ecg trig pixel locations
    pixels        = np.array(pixel_list)
    sortInds      = np.argsort(pixels)
    pixels_sorted = np.sort(pixels)

    #scaling the confidence
    confidence          = 2*(np.array(confidence_list)-0.5)
    confidence_sorted   = confidence[sortInds]

    #find limits
    density, _ = np.histogram(a=pixels_sorted, bins=nx, range=(0, nx-1), weights=confidence_sorted)

    density_avg_conf   = movingaverage(density, filter_len, 'tukey')
    return density_avg_conf