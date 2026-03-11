import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
# 用快速傅里叶变换计算两个矩阵各种平移下的相关性值（未归一化）
# 其中ret 最右下角的值代表不平移，在两个矩阵匹配的情况下应该显著大于其它位置的值
def crosscorr(array1, array2):
    array1 = array1 - np.mean(array1)
    array2 = array2 - np.mean(array2)
    tilted_array2 = np.flipud(np.fliplr(array2))
    TA = np.fft.fft2(tilted_array2)
    FA = np.fft.fft2(array1)
    FF = FA * TA
    ret = np.real(np.fft.ifft2(FF))

    return ret.astype('float32')


def crosscorr3(array1, array2):

    array1 = array1.astype(np.double)
    array2 = array2.astype(np.double)
    array1 = array1 - array1.mean()
    array2 = array2 - array2.mean()

    ############### End of filtering
    normalizator = np.sqrt(np.sum(np.power(array1, 2)) * np.sum(np.power(array2, 2)))
    tilted_array2 = np.fliplr(array2);
    del array2
    tilted_array2 = np.flipud(tilted_array2)
    TA = np.fft.fft2(tilted_array2);
    del tilted_array2
    FA = np.fft.fft2(array1);
    del array1
    AC = np.multiply(FA, TA);
    del FA, TA

    if normalizator == 0:
        ret = None
    else:
        ret = np.real(np.fft.ifft2(AC)) / normalizator
    return ret
