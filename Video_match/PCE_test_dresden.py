import os
import numpy as np
from SDI_Cross.my_show import myshow
from utility_dataset_dresden import load_prnu, load_res
from tqdm import tqdm
import time
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2
from SDI_Cross.SCIFunctions.crosscorr import crosscorr


from SDI_Cross.src.pce import PCE


##测试PCE代码

# --------------------------------------------------- Main -------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # define the list of test devices
    def_dev_list = 'D01_Olympus_mju_1050SW_0;D02_Olympus_mju_1050SW_1;D03_Olympus_mju_1050SW_2;D04_Olympus_mju_1050SW_3;D05_Olympus_mju_1050SW_4;'\
                        'D06_Panasonic_DMC-FZ50_0;D07_Panasonic_DMC-FZ50_1;D08_Panasonic_DMC-FZ50_2;D09_Nikon_D200_0;D10_Nikon_D200_1;'\
                        'D11_Pentax_OptioA40_0;D12_Pentax_OptioA40_1;D13_Pentax_OptioA40_2;D14_Pentax_OptioA40_3;'\
                        'D15_Praktica_DCZ5.9_0;D16_Praktica_DCZ5.9_1;D17_Praktica_DCZ5.9_2;D18_Praktica_DCZ5.9_3;D19_Praktica_DCZ5.9_4;'\
                        'D20_Ricoh_GX100_0;D21_Ricoh_GX100_1;D22_Ricoh_GX100_2;D23_Ricoh_GX100_3;D24_Ricoh_GX100_4;'\
                        'D25_Rollei_RCP-7325XS_0;D26_Rollei_RCP-7325XS_1;D27_Rollei_RCP-7325XS_2;'\
                        'D28_Kodak_M1063_0;D29_Kodak_M1063_1;D30_Kodak_M1063_2;D31_Kodak_M1063_3;D32_Kodak_M1063_4;'\
                        'D33_Samsung_L74wide_0;D34_Samsung_L74wide_1;D35_Samsung_L74wide_2;'\
                        'D36_Samsung_NV15_0;D37_Samsung_NV15_1;D38_Samsung_NV15_2;D39_Sony_DSC-H50_0;D40_Sony_DSC-H50_1'
    parser.add_argument('--list_dev_test', type=str, default=def_dev_list) # list of test devices
    parser.add_argument('--image_size', type=int, default=256)  # minimum size of the image
    parser.add_argument('--crop_size', type=int, default=256) # size of central patch
    parser.add_argument('--output_file', type=str, default='./output_Pce/results_256_256-dre-dwt1111.npz')  # output file with result

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
    list_content = [np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_DWT/test/%s.npy' % item).tolist() for item in list_dev]##dwt提取的测试PRNU
    # list_content = [np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_SPNCNN/test/%s.npy' % item).tolist() for item in list_dev]##spncnn提取的测试PRNU
    # list_content = [np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_FUSION/test/%s.npy' % item).tolist() for item in list_dev]##DFPRNU-NET提取的测试PRNU


    print("num_dev:",num_dev)

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
                data_shift_range[prnuD] = max([list_prnu[prnuD,:,:].shape[0] - res_processed.shape[0], list_prnu[prnuD,:,:].shape[1] - res_processed.shape[1]], [0, 0])
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