# utilities for reading device PRNUs and image noise residuals
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import os



# vision_dev_list = 'D02_Apple_iPhone4s;D04_LG_D290;D05_Apple_iPhone5c;D06_Apple_iPhone6;D07_Lenovo_P70A;D08_Samsung_GalaxyTab3;' \
#                          'D10_Apple_iPhone4s;D11_Samsung_GalaxyS3;D12_Sony_XperiaZ1Compact;D14_Apple_iPhone5c;D15_Apple_iPhone6;D16_Huawei_P9Lite;D17_Microsoft_Lumia640LTE;' \
#                          'D20_Apple_iPadMini;D21_Wiko_Ridge4G;D22_Samsung_GalaxyTrendPlus;D23_Asus_Zenfone2Laser;D25_OnePlus_A3000;D27_Samsung_GalaxyS5;' \
#                          'D28_Huawei_P8;D29_Apple_iPhone5;D30_Huawei_Honor5c;D31_Samsung_GalaxyS4Mini;D32_OnePlus_A3003;D33_Huawei_Ascend;D35_Samsung_GalaxyTabA'
# vision_dev_list = vision_dev_list.split(';')

dresden_dev_list = 'D01_Olympus_mju_1050SW_0;D02_Olympus_mju_1050SW_1;D03_Olympus_mju_1050SW_2;D04_Olympus_mju_1050SW_3;D05_Olympus_mju_1050SW_4;'\
                        'D06_Panasonic_DMC-FZ50_0;D07_Panasonic_DMC-FZ50_1;D08_Panasonic_DMC-FZ50_2;D09_Nikon_D200_0;D10_Nikon_D200_1;'\
                        'D11_Pentax_OptioA40_0;D12_Pentax_OptioA40_1;D13_Pentax_OptioA40_2;D14_Pentax_OptioA40_3;'\
                        'D15_Praktica_DCZ5.9_0;D16_Praktica_DCZ5.9_1;D17_Praktica_DCZ5.9_2;D18_Praktica_DCZ5.9_3;D19_Praktica_DCZ5.9_4;'\
                        'D20_Ricoh_GX100_0;D21_Ricoh_GX100_1;D22_Ricoh_GX100_2;D23_Ricoh_GX100_3;D24_Ricoh_GX100_4;'\
                        'D25_Rollei_RCP-7325XS_0;D26_Rollei_RCP-7325XS_1;D27_Rollei_RCP-7325XS_2;'\
                        'D28_Kodak_M1063_0;D29_Kodak_M1063_1;D30_Kodak_M1063_2;D31_Kodak_M1063_3;D32_Kodak_M1063_4;'\
                        'D33_Samsung_L74wide_0;D34_Samsung_L74wide_1;D35_Samsung_L74wide_2;'\
                        'D36_Samsung_NV15_0;D37_Samsung_NV15_1;D38_Samsung_NV15_2;D39_Sony_DSC-H50_0;D40_Sony_DSC-H50_1'
dresden_dev_list = dresden_dev_list.split(';')




dresden_prnu_dir = '/home/seamus20/matlab_wyl/Camera_Dresden_PRNU_50_npy/%s.npy'
# dresden_prnu_dir = '/home/seamus20/matlab_wyl/Camera_Dresden_PRNU_50_WIENER_npy/%s.npy'
# dresden_prnu_dir = '/home/seamus20/matlab_wyl/Dresden_40cameras_PRNU_FUSION_npy/%s.npy'




def preprocessing_function(x):
    return x / (np.sqrt(np.mean(np.square(x))) + np.finfo(float).eps)


def load_prnu(dev_id, size=None):
    # load device PRNU and crop it
    # dev_id = device identifier
    # size = crop size, if None the PRNU is not cropped

    # PRNU of devices were saved as .npy files as they were computed following the implementation at:
    # https://github.com/polimi-ispl/prnu-python

    if np.in1d(dev_id, dresden_dev_list):
        try:
            with h5py.File(dresden_prnu_dir % dev_id, 'r') as f:
                prnu = np.copy(f['prnu'].value).transpose((1, 0))
        except:
            prnu = np.load(dresden_prnu_dir % dev_id)

    prnu = np.copy(prnu, order='C')
    if size is not None:
        r0 = (prnu.shape[0] - size) // 2
        r1 = (prnu.shape[1] - size) // 2
        prnu = prnu[r0:(r0 + size), r1:(r1 + size)]
        assert (prnu.shape == (size, size))
    prnu=prnu.astype(np.float32)
    assert (prnu.dtype == np.float32)
    assert (not (np.isfortran(prnu)))

    return prnu


def load_res(list_noises, size=None, dtype=np.float32):
    # load image noise residual and crop it
    # dev_id = device identifier
    # size = crop size, if None the residue is not cropped

    # noise residuals of every image were saved as .npy files as they were computed following the implementation at:
    # https://github.com/polimi-ispl/prnu-python

    shape_img = None
    crop_frame = [0, 0]
    list_res = list()
    for index, filename in tqdm(enumerate(list_noises)):

        res = np.load(filename)
        res = np.copy(res, order='C')

        if shape_img is None:
            shape_img = (res.shape[0], res.shape[1])
            if size is not None:
                ##此处用于测试移位场景，手动添加移位像素值
                crop_frame[0] = ((res.shape[0] - size) // 2)+10
                crop_frame[1] = ((res.shape[1] - size) // 2)+10

        # check if the img is rotated by 90 deg
        elif shape_img[0] == res.shape[1] and shape_img[1] == res.shape[0]:
            res = res.T
        else:
            assert (shape_img[0] == res.shape[0])
            assert (shape_img[1] == res.shape[1])

        if size is not None:
            res = res[crop_frame[0]:(crop_frame[0] + size), crop_frame[1]:(crop_frame[1] + size)]
            assert (res.shape[0] == size)
            assert (res.shape[1] == size)

        assert (not (np.isfortran(res)))
        assert (res.dtype == dtype)

        res = np.expand_dims(res, 0)
        list_res.append(res)
    res = np.concatenate(list_res, 0)
    assert (not (np.isfortran(res)))
    assert (res.dtype == dtype)

    return res

# def load_img(list_noises, size=None, dtype=np.float32):
#     shape_img = None
#     crop_frame = [0, 0]
#     list_img = list()
#     for index, filename in tqdm(enumerate(list_noises)):
#
#         img_name = filename.split('/')[-1][:-4]
#         if img_name[0] == 'D':
#             # D:\\Datasets\\Vision_res_npy\\D25_OnePlus_A3000\\D25_I_nat_0012.npy
#             # /home/seamus20/z_Datasets/Vision_res_npy/D25_OnePlus_A3000/D25_I_nat_0012.npy
#             img_path = os.path.join('/home/seamus20/VISION',filename.split('/')[5],'images/nat',img_name+'.jpg')
#         else:
#             # /home/seamus20/z_Datasets/Dresden_res_npy/Praktica_DCZ5.9_0/Praktica_DCZ5.9_0_33694.npy
#             img_path = os.path.join('/home/seamus20/z_Datasets/Dresden',filename.split('/')[5][:-2]+'(5)',filename.split('/')[5],img_name+'.JPG')
#
#         Probeimage = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         print(img_path)
#         Probeimagefloat = np.float32(Probeimage)
#         print(Probeimagefloat.shape)
#
#         if shape_img is None:
#             shape_img = (Probeimagefloat.shape[0], Probeimagefloat.shape[1])
#             if size is not None:
#                 crop_frame[0] = (Probeimagefloat.shape[0] - size) // 2
#                 crop_frame[1] = (Probeimagefloat.shape[1] - size) // 2
#
#         # check if the img is rotated by 90 deg
#         elif shape_img[0] == Probeimagefloat.shape[1] and shape_img[1] == Probeimagefloat.shape[0]:
#             Probeimagefloat = Probeimagefloat.T
#         else:
#             assert (shape_img[0] == Probeimagefloat.shape[0])
#             assert (shape_img[1] == Probeimagefloat.shape[1])
#
#         if size is not None:
#             Probeimagefloat = Probeimagefloat[crop_frame[0]:(crop_frame[0] + size), crop_frame[1]:(crop_frame[1] + size)]
#             assert (Probeimagefloat.shape[0] == size)
#             assert (Probeimagefloat.shape[1] == size)
#
#         assert (not (np.isfortran(Probeimagefloat)))
#         assert (Probeimagefloat.dtype == dtype)
#
#         Probeimagefloat = np.expand_dims(Probeimagefloat, 0)
#         list_img.append(Probeimagefloat)
#     img = np.concatenate(list_img, 0)
#     assert (not (np.isfortran(img)))
#     assert (img.dtype == dtype)
#
#     return img

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def evaluation(y_test, y_predict):
    accuracy = classification_report(y_test, y_predict, output_dict=True)['accuracy']
    s = classification_report(y_test, y_predict, output_dict=True)['weighted avg']
    precision = s['precision']
    recall = s['recall']
    f1_score = s['f1-score']
    kappa=cohen_kappa_score(y_test, y_predict)
    return accuracy, precision, recall, f1_score, kappa


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
