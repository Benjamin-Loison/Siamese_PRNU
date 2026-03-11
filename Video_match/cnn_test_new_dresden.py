import os
import numpy as np
import torch
from torch.autograd import Variable
from SDI_Cross.my_show import myshow
from utility_dataset_dresden import load_prnu, load_res, preprocessing_function

from tqdm import tqdm
from SDI_Cross.arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep, reparameterize_model
from SDI_Cross.arch.resnet import resnet34
from SDI_Cross.arch.mobilenetv3 import mbv3_small
from mobilenet_v3_eca import mbv3_small_eca

import time
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2
from SDI_Cross.SCIFunctions.crosscorr import crosscorr

###测试匹配网络代码


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# --------------------------------------------------- Main -------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # define the list of test devices
    def_dev_list ='D01_Olympus_mju_1050SW_0;D02_Olympus_mju_1050SW_1;D03_Olympus_mju_1050SW_2;D04_Olympus_mju_1050SW_3;D05_Olympus_mju_1050SW_4;'\
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
    parser.add_argument('--crop_size', type=int, default=205) # size of central patch
    parser.add_argument('--model_dir', type=str, default='./weight_Eff_repB0_dresden/Pixle_256_205-fusion_3')  # directory with1 CNN weights
    parser.add_argument('--output_file', type=str, default='./output_Eff_repB0_dresden/results_256_205-fusion_3_1.npz')  # output file with result
    parser.add_argument('--gpu', type=str, default='1') # gpu to be used
    parser.add_argument('--base_network', type=str, default='Eff_repB0') # CNN architecture to be used
                                                  # |EffB0|Eff_repB0|ShuffV2|ResNet|MBVitxxs|MBNetV2|MBNetV3
                                                # Eff_pcB0|Eff_pc_ecaB0    0.34-0.64   78.8%-85.6%   0.968-0.982
                                                                                      #  81.7%         0.982
                                                                                      #  81.9%         0.980
#验证集和训练集上的正确率要测试一下

    config, _ = parser.parse_known_args()
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu  # set the GPU device


    list_dev = config.list_dev_test.split(';')
    output_file = config.output_file
    image_size = config.image_size
    crop_size = config.crop_size
    model_dir = config.model_dir
    num_dev = len(list_dev)
    base_network = config.base_network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the model used

    if config.base_network == 'Eff_repB0':
        model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    elif config.base_network == 'ResNet':
        model = resnet34()
    elif config.base_network == 'MBNetV3':
        model = mbv3_small()
    elif config.base_network == 'MBNetV3_ECA':
        model = mbv3_small_eca()



    pretrained_model = torch.load(model_dir + '/checkpoint.pt')
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    if config.base_network == 'Eff_repB0':
        model = reparameterize_model(model)
    model.eval()
    model = model.to(device)

    print("Preparing test data loader")
    # list with the device PRNUs
    list_prnu = [load_prnu(item, image_size) for item in list_dev]
    list_prnu = np.stack(list_prnu, 0)

    # list with the noise residuals of the images
    list_content = [np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_FUSION/test/%s.npy' % item).tolist() for item in list_dev]
    # list_content = [np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_DWT/test/%s.npy' % item).tolist() for item in list_dev]
    # list_content = [np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_SPNCNN/test/%s.npy' % item).tolist() for item in list_dev]


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
            # res_processed = preprocessing_function(residue[0, :, :].astype(np.float32))
            res_processed = residue[0, :, :].astype(np.float32)

            data = np.ones((num_dev, image_size, image_size))
            for prnuD in range(num_dev):
                # data[prnuD,:,:] = crosscorr2(preprocessing_function(list_prnu[prnuD,:,:].astype(np.float32)), res_processed)#原始
                data[prnuD,:,:] = preprocessing_function(crosscorr2(preprocessing_function(list_prnu[prnuD,:,:].astype(np.float32)), res_processed))#PRNU和C都进行归一化

                # data[prnuD,:,:] = crosscorr2(list_prnu[prnuD,:,:].astype(np.float32), res_processed)#去掉归一化

            # res_processed = np.tile(res_processed[None, :, :], (num_dev, 1, 1))
            # data = np.stack((list_prnu, res_processed), 1)
            data = torch.Tensor(data[:,None,:,:])
            data = Variable(data)
            data = data.cuda()
            # print(data.shape()) ( num_dev, 1, image_size, image_size )

            start_time = time.time()

            # predict
            result = model(data).cpu().detach().numpy()
            score_mat[indexD][indexR][:, 0] = softmax(result)[:, 1]

            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / num_dev

            time_list.append(elapsed_time*1000.)

    # save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    score_mat_array = np.array([None, score_mat])
    np.savez(output_file, list_dev=list_dev, score_mat=score_mat_array, time_list=time_list)

    myshow(file_result=config.output_file)