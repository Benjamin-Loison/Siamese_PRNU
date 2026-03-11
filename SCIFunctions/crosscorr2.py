import numpy as np
##   u代表原矩阵，shiftnum1代表行，shiftnum2代表列。
def circshift(u,shiftnum1,shiftnum2):
    h,w = u.shape
    if shiftnum1 < 0:
        u = np.vstack((u[-shiftnum1:,:],u[:-shiftnum1,:]))
    else:
        u = np.vstack((u[(h-shiftnum1):,:],u[:(h-shiftnum1),:]))
    if shiftnum2 > 0:
        u = np.hstack((u[:, (w - shiftnum2):], u[:, :(w - shiftnum2)]))
    else:
        u = np.hstack((u[:,-shiftnum2:],u[:,:-shiftnum2]))
    return u


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

