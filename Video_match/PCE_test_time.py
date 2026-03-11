import os
import numpy as np
from SDI_Cross.my_show import myshow
from utility_dataset_dresden import load_prnu, load_res
from tqdm import tqdm
import time
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2
from SDI_Cross.src.pce import PCE


# --------------------------------------------------- Main -------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # define the list of test devices
    def_dev_list = 'D01_Samsung_GalaxyS3Mini;D02_Apple_iPhone4s;D03_Huawei_P9;D04_LG_D290;D05_Apple_iPhone5c;D06_Apple_iPhone6;D07_Lenovo_P70A;D08_Samsung_GalaxyTab3;' \
               'D10_Apple_iPhone4s;D11_Samsung_GalaxyS3;D12_Sony_XperiaZ1Compact;D14_Apple_iPhone5c;D15_Apple_iPhone6;D16_Huawei_P9Lite;D17_Microsoft_Lumia640LTE;' \
               'D20_Apple_iPadMini;D21_Wiko_Ridge4G;D22_Samsung_GalaxyTrendPlus;D23_Asus_Zenfone2Laser;D25_OnePlus_A3000;D26_Samsung_GalaxyS3Mini;D27_Samsung_GalaxyS5;' \
               'D28_Huawei_P8;D29_Apple_iPhone5;D30_Huawei_Honor5c;D31_Samsung_GalaxyS4Mini;D32_OnePlus_A3003;D33_Huawei_Ascend;D35_Samsung_GalaxyTabA;Praktica_DCZ5.9_0;' \
               'Praktica_DCZ5.9_1;Praktica_DCZ5.9_2;Praktica_DCZ5.9_3;Praktica_DCZ5.9_4;Olympus_mju_1050SW_0;Olympus_mju_1050SW_1'
    parser.add_argument('--list_dev_test', type=str, default=def_dev_list) # list of test devices
    parser.add_argument('--image_size', type=int, default=256)  # minimum size of the image
    parser.add_argument('--crop_size', type=int, default=256) # size of central patch
    parser.add_argument('--output_file', type=str, default='./output_Pce_shift/results_256_256——111111.npz')  # output file with result

    config, _ = parser.parse_known_args()


    list_dev = config.list_dev_test.split(';')
    output_file = config.output_file
    image_size = config.image_size
    crop_size = config.crop_size
    num_dev = len(list_dev)

    print("Preparing test data loader")
    # list with the device PRNUs
    list_prnu = [load_prnu(item, image_size) for item in list_dev]
    list_prnu = np.stack(list_prnu, 0)

    # list with the noise residuals of the images
    list_content = [np.load('Noises_lists/test/%s.npy' % item).tolist() for item in list_dev]

    print('Starting Test')
    score_mat = [None for x in range(num_dev)]

    time_list = []

    for indexD in range(num_dev):  # loop on devices
        list_residuals = list_content[indexD]  # list with the residuals of the device
        num_residuals = len(list_residuals)

        # initialize the matrix with the scores
        score_mat[indexD] = [None for x in range(num_residuals)]

        for indexR in tqdm(range(num_residuals)):  # loop on the residuals of the device

            residue = load_res([list_residuals[indexR], ], crop_size)  # load the residue
            score_mat[indexD][indexR] = np.nan * np.ones((num_dev, 1))
            res_processed = residue[0, :, :]
            data = np.ones((num_dev, image_size, image_size))
            data_shift_range = [None for x in range(num_dev)]
            for prnuD in range(num_dev):
                data[prnuD, :, :] = crosscorr2(list_prnu[prnuD, :, :], res_processed)
                data_shift_range[prnuD] = max([list_prnu[prnuD,:,:].shape[0] - res_processed.shape[0], list_prnu[prnuD,:,:].shape[1] - res_processed.shape[1]], [10, 10])
            start_time = time.time()
            for i in range(num_dev):
                valid_shift_range = data_shift_range[i]
                C = data[i, :, :]
                mypce = PCE(C,shift_range=valid_shift_range)
                score_mat[indexD][indexR][i, 0] = mypce

            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / num_dev

            time_list.append(elapsed_time*1000.)

    # save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    score_mat_array = np.array([None, score_mat])
    np.savez(output_file, list_dev=list_dev, score_mat=score_mat_array, time_list=time_list)

    myshow(file_result=config.output_file)