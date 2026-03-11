import os
import numpy as np
import torch
from torch.autograd import Variable
from SDI_Cross.my_show import myshow
# from utility_dataset_new import load_prnu, load_res, preprocessing_function
from SDI_Cross.utility_dataset import load_prnu, load_res, preprocessing_function

from tqdm import tqdm
# from arch.effnet import EfficientNet
# from arch.effnet_pconv import EfficientNet_pconv
# from arch.effnet_pconv_eca import EfficientNet_pconv_eca
from SDI_Cross.arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep, reparameterize_model
# from arch.shufflenetv2 import ShuffleNetV2
from SDI_Cross.arch.resnet import resnet34
from SDI_Cross.arch.mobilenetv3 import mbv3_small
from mobilenet_v3_eca import mbv3_small_eca



# from arch.mobilevit import mobilevit_xs,mobilevit_xxs
# from arch.mobilenetv2 import MobileNetV2
# from arch.mobilenetv3 import mbv3_small
# from arch.PCN import Pcn
import time
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2
from SDI_Cross.SCIFunctions.crosscorr import crosscorr


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# --------------------------------------------------- Main -------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # define the list of test devices
    def_dev_list ='D04_LG_D290;D05_Apple_iPhone5c;D06_Apple_iPhone6;D07_Lenovo_P70A;D08_Samsung_GalaxyTab3;' \
                         'D10_Apple_iPhone4s;D11_Samsung_GalaxyS3;D12_Sony_XperiaZ1Compact;D14_Apple_iPhone5c;D15_Apple_iPhone6;D16_Huawei_P9Lite;D17_Microsoft_Lumia640LTE;' \
                         'D20_Apple_iPadMini;D21_Wiko_Ridge4G;D22_Samsung_GalaxyTrendPlus;D23_Asus_Zenfone2Laser;D25_OnePlus_A3000;D27_Samsung_GalaxyS5;' \
                         'D28_Huawei_P8;D29_Apple_iPhone5;D30_Huawei_Honor5c;D31_Samsung_GalaxyS4Mini;D32_OnePlus_A3003;D33_Huawei_Ascend;D35_Samsung_GalaxyTabA;Praktica_DCZ5.9_0;' \
                         'Praktica_DCZ5.9_1;Praktica_DCZ5.9_2;Praktica_DCZ5.9_3;Praktica_DCZ5.9_4;Olympus_mju_1050SW_0;Olympus_mju_1050SW_1'
    parser.add_argument('--list_dev_test', type=str, default=def_dev_list) # list of test devices
    parser.add_argument('--image_size', type=int, default=256)  # minimum size of the image
    parser.add_argument('--crop_size', type=int, default=256) # size of central patch
    parser.add_argument('--model_dir', type=str, default='./weight_ResNet/Pixle_256_256-201')  # directory with1 CNN weights
    parser.add_argument('--output_file', type=str, default='./output_ResNet/results_256_256-201.npz')  # output file with result
    parser.add_argument('--gpu', type=str, default='0') # gpu to be used
    parser.add_argument('--base_network', type=str, default='ResNet') # CNN architecture to be used
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
    list_content = [np.load('SDI_Cross/Noises_lists/test/%s.npy' % item).tolist() for item in list_dev]

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