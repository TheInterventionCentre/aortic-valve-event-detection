import numpy as np


def get_unet_1d_param_vectors(enc_chs, dilation=1, padding=0, filter_size=3):
    N = len(enc_chs)
    if not isinstance(filter_size, list):
        filter_size_vec = []
        for ii in range(N):
            filter_size_vec.append(filter_size)
            filter_size_vec.append(filter_size)
            if ii<N-1:
                filter_size_vec.append(2)
        filter_size = filter_size_vec

    if not isinstance(dilation, list):
        dilation_vec = []
        for ii in range(N):
            dilation_vec.append(dilation)
            dilation_vec.append(dilation)
            if ii < N - 1:
                dilation_vec.append(1)
        dilation = dilation_vec

    if not isinstance(padding, list):
        padding_vec = []
        for ii in range(N):
            padding_vec.append(padding)
            padding_vec.append(padding)
            if ii < N - 1:
                padding_vec.append(0)
        padding = padding_vec

    stride_vec = []
    for ii in range(N):
        stride_vec.append(1)
        stride_vec.append(1)
        if ii < N - 1:
            stride_vec.append(2)
    stride = stride_vec

    return filter_size, stride, dilation, padding


def fov(filter_size, stride, dilation, padding):
    # Inputs:
    # dict "CNN_config" with:
    #   f (list): Filter size for each layer
    #   s (list): Stride for each layer
    #   d (list): Dilation factor for each layer
    #   p (list): padding for each layer

    # Output
    # R: The calculated receptive field for each layer as a numpy array
    # dx: Calculate the number of spatial samples between the consecutive spatial features.

    f = filter_size
    s = stride
    d = dilation
    p = padding

    dx = np.cumproduct(np.array(s))

    #modify filter size based on dilation factors
    for ii in range(len(d)):
        if f[ii]%2==1:
            f[ii] = (int((f[ii]-1)/2) * d[ii])*2 + 1
        else:
            if d[ii] >1:
                raise ValueError('Cannot use dialtion factor above 1 with even filter size')

    # #FOV boarders
    # R_border = [1]
    # for kk in range(len(s)):
    #     S = 1
    #     for ii in range(kk):
    #         S = S * s[ii]
    #     fov = R_border[-1] + (f[kk]-p[kk] - 1) * S
    #     R_border.append(fov)

    #FOV
    R = [1]
    for kk in range(len(s)):
        S = 1
        for ii in range(kk):
            S = S * s[ii]
        fov = R[-1] + (f[kk] - 1) * S
        R.append(fov)
    return np.array(R), dx


########################################################################################################################

def validNetworkConfig(imgSize, CNN_config, lowerBound=False, recursive=True):
    f = CNN_config['filter_size']
    s = CNN_config['stride']
    d = CNN_config['dilation']
    p = CNN_config['padding']
    #modify filter size based on dilation factors
    for ii in range(len(d)):
        if f[ii]%2==1:
            f[ii] = (int((f[ii]-1)/2) * d[ii])*2 + 1
        else:
            if d[ii] >1:
                raise ValueError('Cannot use dialtion factor above 1 with even filter size')

    out = imgSize
    layerSizes = []
    layerSizes.append(imgSize)
    for ii in range(len(f)):
        out = (out -f[ii] + 2*p[ii]) / s[ii] + 1
        layerSizes.append(out)

    layerSizes = np.array(layerSizes)
    #Is valid NN
    is_valid = 1
    for val in layerSizes:
        if not val.is_integer():
            is_valid = 0

    #Find closest accepted sequence size
    seqLenValid     = imgSize
    layerSizesValid = layerSizes
    iter    = 0
    foundClosest = True
    if is_valid == 0 and recursive:
        while foundClosest:
            iter = iter + 1
            out = imgSize + iter
            layerSizesValid, tmpValid, _, _ = validNetworkConfig(out, CNN_config, recursive=False)
            if tmpValid and not lowerBound:
                foundClosest = False
                seqLenValid = out
            if foundClosest: #added last, if error remove this if statement.
                out = imgSize - iter
                layerSizesValid, tmpValid, _, _ = validNetworkConfig(out, CNN_config, recursive=False)
                if tmpValid:
                    foundClosest = False
                    seqLenValid = out
    return layerSizes, is_valid, seqLenValid, layerSizesValid
