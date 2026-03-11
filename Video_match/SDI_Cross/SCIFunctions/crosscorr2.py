import numpy as np
from SDI_Cross.utility_dataset import circshift

def crosscorr2(array1, array2):

    if (array1.shape[0]<array2.shape[0])or(array1.shape[1]<array2.shape[1]):
        temp = array1
        array1 = array2
        array2 = temp

    array1 = array1 - np.mean(array1)
    # array1 = array1 / np.linalg.norm(array1)
    array2 = array2 - np.mean(array2)
    # array2 = array2 / np.linalg.norm(array2)
    array2 = np.pad(array2,((0,array1.shape[0]-array2.shape[0]),(0,array1.shape[1]-array2.shape[1])),'constant') # constant edge


    # tilted_array2 = np.flipud(np.fliplr(array2))
    tilted_array2 = np.rot90(array2,2)
    TA = np.fft.fft2(tilted_array2)
    FA = np.fft.fft2(array1)
    FF = FA * TA
    ret = np.real(np.fft.ifft2(FF))
    ret = circshift(ret, 1, 1)
    return ret.astype('float32')

