import glob
import random
import scipy.io as sio
import h5py
import os
import cv2
import numpy as np
from scipy.io import loadmat

from DnCNN import *
import torch
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
# 以下为自己编写的函数
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from SCIFunctions.NoiseExtractDL import NoiseExtractDL

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

# postprocess = True
postprocess = False

def crop_image(image):#从输入图像中裁剪四个不同的图像块。

    B = 256
    centerY = np.round(image.shape[0] / 2).astype(int)
    centerX = np.round(image.shape[1] / 2).astype(int)
    patch1 = image[centerY-B: centerY, centerX-B: centerX]#左上角
    patch2 = image[centerY-B: centerY, centerX: centerX+B]#右上角
    patch3 = image[centerY: centerY+B, centerX - B: centerX]#左下角
    patch4 = image[centerY: centerY+B, centerX: centerX + B]#右下角

    return patch1,patch2,patch3,patch4

def ROC_prefcurve(PCE):
    intra_pce = np.array(PCE['intra_pce']).flatten()
    inter_pce = np.array(PCE['inter_pce']).flatten()
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]
    true_lable = np.ones(t1)
    false_lable = np.zeros(t2)

    lable = np.concatenate([true_lable, false_lable])
    value = np.concatenate([intra_pce, inter_pce])

    fpr, tpr, _ = roc_curve(lable, value)
    roc_auc = auc(fpr, tpr)

    return fpr,tpr,roc_auc

def main():
    # ffdnet_intra_pce = []
    # ffdnet_inter_pce = []
    adnet_intra_pce = []
    adnet_inter_pce = []
    # dncnn_intra_pce = []
    # dncnn_inter_pce = []

    ## 1 Load saved models
    print('Loading model ...\n')

    # # FFDNet
    # net_FFDNet = FFDNet(num_input_channels=1)
    # model_FFDNet = nn.DataParallel(net_FFDNet, device_ids=device_ids).cuda()
    # model_FFDNet.load_state_dict(torch.load('Mymodels/FFDNet.pth'))
    # model_FFDNet.eval()
    #
    # # ADNet
    model_ADNet = DnCNN(channels=1)
    model_ADNet = nn.DataParallel(model_ADNet, device_ids=device_ids).cuda()
    pretrained_model = torch.load('logs_nat/DnCNN_SPN_MSE.pth')
    model_ADNet.load_state_dict(pretrained_model)
    model_ADNet.eval()

    # DnCNN
    # net_DnCNN = DnCNN(channels=1, num_of_layers=17)
    # model_DnCNN = nn.DataParallel(net_DnCNN, device_ids=device_ids).cuda()
    # dict_DnCNN = torch.load('Mymodels/net2.pth')
    # model_DnCNN.load_state_dict(dict_DnCNN)
    # model_DnCNN.eval()

    k = 0
    root_dir = '/home/seamus20/z_Datasets/Dresden'
    camera_models = ['Olympus_mju_1050SW(5)', 'Panasonic_DMC-FZ50(3)', 'Nikon_D200(2)', 'Pentax_OptioA40(4)',
                     'Praktica_DCZ5.9(5)', 'Ricoh_GX100(5)', 'Rollei_RCP-7325XS(3)', 'Kodak_M1063(5)',
                     'Samsung_L74wide(3)', 'Samsung_NV15(3)', 'Sony_DSC-H50(2)']
    DID_files = sorted(glob.glob(os.path.join('Camera_Dresden_overlay', '*.mat')))
    for i in range(11):
        camera_model = camera_models[i]
        for j in range(int(camera_model[-2])):  # 将字符串 camera_model 的倒数第二个字符（通常是相机模型的编号）转换为整数
            img_num = 0
            ## 2 read the RP image

            RPname = DID_files[k]
            mat_data = loadmat(RPname)
            RP = np.array(mat_data['average_image'])

            # RP = cv2.imread(PRNU, cv2.IMREAD_GRAYSCALE)  #转换为灰度图像
            # RP = (np.float32(RP) - 127.5)/32.5  #这行代码将读取的灰度图像进行了归一化处理

            ## 3 read the probe image
            camera_name = camera_model[:-3] + '_' + str(j)  # 即camera model文件夹下，同设备不同相机的文件名
            # img_files = sorted(glob.glob(os.path.join(root_dir, camera_model, camera_name, '*.JPG')))
            img_files = sorted(glob.glob(os.path.join(root_dir, camera_model, camera_name, '*.JPG')),
                               reverse=True)  # 倒序

            # 正样本（类内）
            p = 0
            while img_num < 50:
                imx = img_files[p]
                p = p + 1
                Probeimage = cv2.imread(imx, cv2.IMREAD_GRAYSCALE)
                if Probeimage.shape[0] > Probeimage.shape[1]:  # 如果图像的高度大于宽度，跳过这层循环
                    continue

                # 裁剪图片与指纹计算PCE,并存入list中
                img_patch1, img_patch2, img_patch3, img_patch4 = crop_image(Probeimage)
                imgfloat_patch1, imgfloat_patch2, imgfloat_patch3, imgfloat_patch4 = np.float32(img_patch1), np.float32(
                    img_patch2), np.float32(img_patch3), np.float32(img_patch4)
                RP_patch1, RP_patch2, RP_patch3, RP_patch4 = crop_image(RP)

                _, PRNU1, _ = NoiseExtractDL(RP_patch1, model_ADNet, postprocess)
                _, PRNU2, _ = NoiseExtractDL(RP_patch2, model_ADNet, postprocess)
                _, PRNU3, _ = NoiseExtractDL(RP_patch3, model_ADNet, postprocess)
                _, PRNU4, _ = NoiseExtractDL(RP_patch4, model_ADNet, postprocess)

                ## 4 Extract noisex form probe
                # _, NoisexnpDnCNN1, _ = NoiseExtractDL(img_patch1, model_DnCNN, postprocess)
                # _, NoisexnpFFDNet1, _ = NoiseExtractFFD(img_patch1, 3 / 255, True, model_FFDNet, postprocess)
                _, NoisexnpADNet1, _ = NoiseExtractDL(img_patch1, model_ADNet, postprocess)

                # _, NoisexnpDnCNN2, _ = NoiseExtractDL(img_patch2, model_DnCNN, postprocess)
                # _, NoisexnpFFDNet2, _ = NoiseExtractFFD(img_patch2, 3 / 255, True, model_FFDNet, postprocess)
                _, NoisexnpADNet2, _ = NoiseExtractDL(img_patch2, model_ADNet, postprocess)

                # _, NoisexnpDnCNN3, _ = NoiseExtractDL(img_patch3, model_DnCNN, postprocess)
                # _, NoisexnpFFDNet3, _ = NoiseExtractFFD(img_patch3, 3 / 255, True, model_FFDNet, postprocess)
                _, NoisexnpADNet3, _ = NoiseExtractDL(img_patch3, model_ADNet, postprocess)

                # _, NoisexnpDnCNN4, _ = NoiseExtractDL(img_patch4, model_DnCNN, postprocess)
                # _, NoisexnpFFDNet4, _ = NoiseExtractFFD(img_patch4, 3 / 255, True, model_FFDNet, postprocess)
                _, NoisexnpADNet4, _ = NoiseExtractDL(img_patch4, model_ADNet, postprocess)

                ## 5 Compute the correlation between probe and RP
                KI_1 = imgfloat_patch1 * PRNU1
                # C_DnCNN1 = crosscorr(NoisexnpDnCNN1, KI_1)
                # PCE_value_DnCNN1 = PCE1(C_DnCNN1)
                # C_FFDNet1 = crosscorr(NoisexnpFFDNet1, KI_1)
                # PCE_value_FFDNet1 = PCE1(C_FFDNet1)
                C_ADNet1 = crosscorr(NoisexnpADNet1, KI_1)
                PCE_value_ADNet1 = PCE1(C_ADNet1)

                KI_2 = imgfloat_patch2 * PRNU2
                # C_DnCNN2 = crosscorr(NoisexnpDnCNN2, KI_2)
                # PCE_value_DnCNN2 = PCE1(C_DnCNN2)
                # C_FFDNet2 = crosscorr(NoisexnpFFDNet2, KI_2)
                # PCE_value_FFDNet2 = PCE1(C_FFDNet2)
                C_ADNet2 = crosscorr(NoisexnpADNet2, KI_2)
                PCE_value_ADNet2 = PCE1(C_ADNet2)

                KI_3 = imgfloat_patch3 * PRNU3
                # C_DnCNN3 = crosscorr(NoisexnpDnCNN3, KI_3)
                # PCE_value_DnCNN3 = PCE1(C_DnCNN3)
                # C_FFDNet3 = crosscorr(NoisexnpFFDNet3, KI_3)
                # PCE_value_FFDNet3 = PCE1(C_FFDNet3)
                C_ADNet3 = crosscorr(NoisexnpADNet3, KI_3)
                PCE_value_ADNet3 = PCE1(C_ADNet3)

                KI_4 = imgfloat_patch4 * PRNU4
                # C_DnCNN4 = crosscorr(NoisexnpDnCNN4, KI_4)
                # PCE_value_DnCNN4 = PCE1(C_DnCNN4)
                # C_FFDNet4 = crosscorr(NoisexnpFFDNet4, KI_4)
                # PCE_value_FFDNet4 = PCE1(C_FFDNet4)
                C_ADNet4 = crosscorr(NoisexnpADNet4, KI_4)
                PCE_value_ADNet4 = PCE1(C_ADNet4)

                # ffdnet_intra_pce.append(PCE_value_FFDNet1)
                # ffdnet_intra_pce.append(PCE_value_FFDNet2)
                # ffdnet_intra_pce.append(PCE_value_FFDNet3)
                # ffdnet_intra_pce.append(PCE_value_FFDNet4)
                #
                adnet_intra_pce.append(PCE_value_ADNet1)
                adnet_intra_pce.append(PCE_value_ADNet2)
                adnet_intra_pce.append(PCE_value_ADNet3)
                adnet_intra_pce.append(PCE_value_ADNet4)

                # dncnn_intra_pce.append(PCE_value_DnCNN1)
                # dncnn_intra_pce.append(PCE_value_DnCNN2)
                # dncnn_intra_pce.append(PCE_value_DnCNN3)
                # dncnn_intra_pce.append(PCE_value_DnCNN4)

                img_num = img_num + 1
                print('Processing intra ---- {}/50:'.format(img_num) + camera_name)
                # print('PCE_value_FFDNet:')
                # print(PCE_value_FFDNet1, PCE_value_FFDNet2, PCE_value_FFDNet3, PCE_value_FFDNet4)
                print('PCE_value_ADNet:')
                print(PCE_value_ADNet1, PCE_value_ADNet2, PCE_value_ADNet3, PCE_value_ADNet4)
                # print('PCE_value_DnCNN:')
                # print(PCE_value_DnCNN1, PCE_value_DnCNN2, PCE_value_DnCNN3, PCE_value_DnCNN4)

            # 负样本（类间）
            r = list(range(0, i)) + list(range(i + 1, 11))  # 生成一个不包括i的0-10之间的列表
            camera_model_idx_list = random.sample(r, 3)  # 列表中选择3个不同的相机模型索引作为负样本

            for m in range(3):
                print('+++++++++++++++++++++++++++')
                print(m)
                inter_img_num = 0
                camera_name = camera_models[camera_model_idx_list[m]][:-3] + '_' + str(0)
                inter_img_files = sorted(
                    glob.glob(os.path.join(root_dir, camera_models[camera_model_idx_list[m]], camera_name, '*.JPG')),
                    reverse=True)

                if len(inter_img_files) == 0:
                    print('========空list' + camera_models[camera_model_idx_list[m]], camera_name)
                while inter_img_num < 50:

                    imx = inter_img_files.pop(0)

                    Probeimage = cv2.imread(imx, cv2.IMREAD_GRAYSCALE)
                    if Probeimage.shape[0] > Probeimage.shape[1]:
                        continue

                    # 裁剪图片与指纹计算PCE,并存入list中
                    img_patch1, img_patch2, img_patch3, img_patch4 = crop_image(Probeimage)
                    imgfloat_patch1, imgfloat_patch2, imgfloat_patch3, imgfloat_patch4 = np.float32(
                        img_patch1), np.float32(img_patch2), np.float32(img_patch3), np.float32(img_patch4)

                    ## 4 Extract noisex form probe
                    # _, NoisexnpDnCNN1, _ = NoiseExtractDL(img_patch1, model_DnCNN, postprocess)
                    # _, NoisexnpFFDNet1, _ = NoiseExtractFFD(img_patch1, 3 / 255, True, model_FFDNet, postprocess)
                    _, NoisexnpADNet1, _ = NoiseExtractDL(img_patch1, model_ADNet, postprocess)

                    # _, NoisexnpDnCNN2, _ = NoiseExtractDL(img_patch2, model_DnCNN, postprocess)
                    # _, NoisexnpFFDNet2, _ = NoiseExtractFFD(img_patch2, 3 / 255, True, model_FFDNet, postprocess)
                    _, NoisexnpADNet2, _ = NoiseExtractDL(img_patch2, model_ADNet, postprocess)

                    # _, NoisexnpDnCNN3, _ = NoiseExtractDL(img_patch3, model_DnCNN, postprocess)
                    # _, NoisexnpFFDNet3, _ = NoiseExtractFFD(img_patch3, 3 / 255, True, model_FFDNet, postprocess)
                    _, NoisexnpADNet3, _ = NoiseExtractDL(img_patch3, model_ADNet, postprocess)

                    # _, NoisexnpDnCNN4, _ = NoiseExtractDL(img_patch4, model_DnCNN, postprocess)
                    # _, NoisexnpFFDNet4, _ = NoiseExtractFFD(img_patch4, 3 / 255, True, model_FFDNet, postprocess)
                    _, NoisexnpADNet4, _ = NoiseExtractDL(img_patch4, model_ADNet, postprocess)

                    ## 5 Compute the correlation between probe and RP
                    KI_1 = imgfloat_patch1 * PRNU1
                    # C_DnCNN1 = crosscorr(NoisexnpDnCNN1, KI_1)
                    # PCE_value_DnCNN1 = PCE1(C_DnCNN1)
                    # C_FFDNet1 = crosscorr(NoisexnpFFDNet1, KI_1)
                    # PCE_value_FFDNet1 = PCE1(C_FFDNet1)
                    C_ADNet1 = crosscorr(NoisexnpADNet1, KI_1)
                    PCE_value_ADNet1 = PCE1(C_ADNet1)

                    KI_2 = imgfloat_patch2 * PRNU2
                    # C_DnCNN2 = crosscorr(NoisexnpDnCNN2, KI_2)
                    # PCE_value_DnCNN2 = PCE1(C_DnCNN2)
                    # C_FFDNet2 = crosscorr(NoisexnpFFDNet2, KI_2)
                    # PCE_value_FFDNet2 = PCE1(C_FFDNet2)
                    C_ADNet2 = crosscorr(NoisexnpADNet2, KI_2)
                    PCE_value_ADNet2 = PCE1(C_ADNet2)

                    KI_3 = imgfloat_patch3 * PRNU3
                    # C_DnCNN3 = crosscorr(NoisexnpDnCNN3, KI_3)
                    # PCE_value_DnCNN3 = PCE1(C_DnCNN3)
                    # C_FFDNet3 = crosscorr(NoisexnpFFDNet3, KI_3)
                    # PCE_value_FFDNet3 = PCE1(C_FFDNet3)
                    C_ADNet3 = crosscorr(NoisexnpADNet3, KI_3)
                    PCE_value_ADNet3 = PCE1(C_ADNet3)

                    KI_4 = imgfloat_patch4 * PRNU4
                    # C_DnCNN4 = crosscorr(NoisexnpDnCNN4, KI_4)
                    # PCE_value_DnCNN4 = PCE1(C_DnCNN4)
                    # C_FFDNet4 = crosscorr(NoisexnpFFDNet4, KI_4)
                    # PCE_value_FFDNet4 = PCE1(C_FFDNet4)
                    C_ADNet4 = crosscorr(NoisexnpADNet4, KI_4)
                    PCE_value_ADNet4 = PCE1(C_ADNet4)

                    # ffdnet_inter_pce.append(PCE_value_FFDNet1)
                    # ffdnet_inter_pce.append(PCE_value_FFDNet2)
                    # ffdnet_inter_pce.append(PCE_value_FFDNet3)
                    # ffdnet_inter_pce.append(PCE_value_FFDNet4)
                    #
                    adnet_inter_pce.append(PCE_value_ADNet1)
                    adnet_inter_pce.append(PCE_value_ADNet2)
                    adnet_inter_pce.append(PCE_value_ADNet3)
                    adnet_inter_pce.append(PCE_value_ADNet4)

                    # dncnn_inter_pce.append(PCE_value_DnCNN1)
                    # dncnn_inter_pce.append(PCE_value_DnCNN2)
                    # dncnn_inter_pce.append(PCE_value_DnCNN3)
                    # dncnn_inter_pce.append(PCE_value_DnCNN4)

                    inter_img_num = inter_img_num + 1
                    print('Processing inter ---- {}/2, {}/50:'.format(m, inter_img_num) + camera_name)
                    # print('PCE_value_FFDNet:')
                    # print(PCE_value_FFDNet1, PCE_value_FFDNet2, PCE_value_FFDNet3, PCE_value_FFDNet4)
                    print('PCE_value_ADNet:')
                    print(PCE_value_ADNet1, PCE_value_ADNet2, PCE_value_ADNet3, PCE_value_ADNet4)
                    # print('PCE_value_DnCNN:')
                    # print(PCE_value_DnCNN1, PCE_value_DnCNN2, PCE_value_DnCNN3, PCE_value_DnCNN4)

            k = k + 1

    # sio.savemat('FFDNet_PCE_0.8978.mat', {'intra_pce': ffdnet_intra_pce, 'inter_pce': ffdnet_inter_pce})
    sio.savemat('Dresden_PCE_dict_256/SPN_PCE_two.mat', {'intra_pce': adnet_intra_pce, 'inter_pce': adnet_inter_pce})

    ADNet_PCE = sio.loadmat('Dresden_PCE_dict_256/SPN_PCE_two.mat')
    X, Y, AUC = ROC_prefcurve(ADNet_PCE)
    print(AUC)


if __name__ == '__main__':
    main()