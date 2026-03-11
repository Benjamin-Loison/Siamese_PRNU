import glob
import os
import time
import cv2
import numpy as np
import torch

from SDI_Cross.src.pce import PCE

from SDI_Cross.arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep, reparameterize_model
import scipy.io as sio
from torch.autograd import Variable
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2


# from SDI_Cross.SCIFunctions.PCE1 import PCE1
# from SDI_Cross.SCIFunctions.crosscorr import crosscorr
# from SDI_Cross.arch.resnet import resnet34
# from SDI_Cross.arch.mobilenetv3 import mbv3_small


def preprocessing_function(x):
    return x / (np.sqrt(np.mean(np.square(x))) + np.finfo(float).eps)


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)
def main():

    # 加载预训练的网络模型
    model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    model_path = '/home/seamus20/z_Project/SDI_Cross/weight_Eff_repB0/Pixle_256_256/checkpoint.pt'

    pretrained_model = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    model = reparameterize_model(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    # fp = sio.loadmat(os.path.join('/home/seamus20/matlab_wyl/VISION_PRNU_LIUYANG', 'D17_Microsoft_Lumia640LTE.mat'))
    # ref_prnu = fp['PRNU']
    # fp = np.load(os.path.join('/home/seamus20/matlab_wyl/VISION_PRNU_FUSION_npy', 'D02_Apple_iPhone4s.npy'))
    # fp = np.load(os.path.join('/home/seamus20/matlab_wyl/Dresden_40cameras_PRNU_FUSION_npy', 'D20_Ricoh_GX100_0.npy'))

    # ref_prnu = fp
    fp = np.load(os.path.join('/home/seamus20/z_Project/SDI_Cross/VISION_PRNU_npy', 'D02_Apple_iPhone4s.npy'))
    ref_prnu = fp



    # ref_prnu=ref_prnu.astype(np.float32)

    # noisex = sio.loadmat(os.path.join('/home/seamus20/matlab_wyl', 'D02_nat_10.mat'))
    # noise = noisex['PRNU']
    # noise = np.load(os.path.join('/home/seamus20/matlab_wyl/VISION_every_image_npy/D02_Apple_iPhone4s', 'D02_I_nat_0040.npy'))
    # noise = np.load(os.path.join('/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras/D20_Ricoh_GX100_0', 'Ricoh_GX100_0_36112.npy'))

    noise = np.load(os.path.join('/home/seamus20/z_Datasets/Vision_res_npy/D02_Apple_iPhone4s', 'D02_I_nat_0041.npy'))
    noise=noise.astype(np.float32)

    crop_size = 128  #中线点两侧分别裁剪的像素值
    ncc_range = 0
    y_size = ref_prnu.shape[0] // 2
    x_size = ref_prnu.shape[1] // 2

    y_size =128
    x_size =128

    ref_prnu = ref_prnu[(y_size - crop_size - ncc_range):(y_size + crop_size + ncc_range),
                  (x_size - crop_size - ncc_range):(x_size + crop_size + ncc_range)]
    noise = noise[(y_size - crop_size):(y_size + crop_size), (x_size - crop_size):(x_size + crop_size)]


    # ref_prnu = preprocessing_function(ref_prnu)
    res_processed = preprocessing_function(noise)

    # data_1 = crosscorr(ref_prnu,noise)
    data_2 = crosscorr2(ref_prnu,res_processed)

    # data2 = crosscorr2(ref_prnu,noise)
    # 此处需要强调 PCE配套crosscorr2   PCE1配套crosscorr   需要配套使用才行
    # data2 = crosscorr2(preprocessing_function(ref_prnu),noise)#1和2差距很大
    # data2 = crosscorr2(preprocessing_function(ref_prnu),noise)
    # PCE_1 = PCE1(data_1)

    PCE_2 = PCE(data_2)

    # print(f'PCE_crosscorr: {PCE_1}')
    print(f'PCE_crosscorr2: {PCE_2}')



    data = torch.Tensor(data_2[None, None, :, :])
    data = Variable(data)
    data = data.cuda()

    model.eval()
    with torch.no_grad():
        result = model(data).cpu().detach().numpy()
        print(result)
        # predict=np.argmax(result, axis=0)
        # print(predict)

        score = softmax(result)[:, 1]
        score = score.item()

    print(f'Similarity: {score}')

if __name__ == '__main__':
    main()