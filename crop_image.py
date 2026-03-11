import numpy as np
import scipy.io
import glob
import random
import os
import cv2
import numpy as np
from scipy.io import loadmat
from models_dude_nat import DnCNN
import torch
from sklearn.metrics import roc_curve, auc
import torch.nn as nn

from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from SCIFunctions.NoiseExtractDL import NoiseExtractDL

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

# postprocess = True
postprocess = False
model_ADNet = DnCNN(channels=1)
model_ADNet.cuda()
# pretrained_model = torch.load('logs_nat/Siamese_PRNU_distance_0.5_8_27.047347688343216.pth')
# pretrained_model = torch.load('logs_nat/Siamese_distance_100image _3_AUC=0.9225006103515625.pth')
pretrained_model = torch.load('logs_nat/dude_nat_70.pth')
model_ADNet.load_state_dict(pretrained_model)
model_ADNet.eval()

def crop_image(image):#

    B = 100
    centerY = np.round(image.shape[0] / 2).astype(int)
    centerX = np.round(image.shape[1] / 2).astype(int)
    patch1 = image[centerY-B: centerY, centerX-B: centerX]#
    patch2 = image[centerY-B: centerY, centerX: centerX+B]#
    patch3 = image[centerY: centerY+B, centerX - B: centerX]#左下角
    patch4 = image[centerY: centerY+B, centerX: centerX + B]#右下角

    return patch1,patch2,patch3,patch4

RP_path='Camera_Dresden_overlay/D01_Olympus_mju_1050SW_0.mat'
mat_data = scipy.io.loadmat(RP_path)
RP = np.array(mat_data['average_image'])

imx='/home/seamus20/Dresden/Olympus_mju_1050SW(5)/Olympus_mju_1050SW_0/Olympus_mju_1050SW_0_23690.JPG'

RP_patch1, RP_patch2, RP_patch3, RP_patch4 = crop_image(RP)


Probeimage = cv2.imread(imx, cv2.IMREAD_GRAYSCALE)
if Probeimage.shape[0] > Probeimage.shape[1]:  #
    print("null")

img_patch1, img_patch2, img_patch3, img_patch4 = crop_image(Probeimage)

imgfloat_patch1, imgfloat_patch2, imgfloat_patch3, imgfloat_patch4 = np.float32(img_patch1),np.float32(img_patch2),np.float32(img_patch3),np.float32(img_patch4)

_, NoisexnpADNet1, _ = NoiseExtractDL(img_patch1, model_ADNet, postprocess)
_, NoisexnpADNet2, _ = NoiseExtractDL(img_patch2, model_ADNet, postprocess)
_, NoisexnpADNet3, _ = NoiseExtractDL(img_patch3, model_ADNet, postprocess)
_, NoisexnpADNet4, _ = NoiseExtractDL(img_patch4, model_ADNet, postprocess)

_, PRNU1, _ = NoiseExtractDL(RP_patch1, model_ADNet, postprocess)
_, PRNU2, _ = NoiseExtractDL(RP_patch2, model_ADNet, postprocess)
_, PRNU3, _ = NoiseExtractDL(RP_patch3, model_ADNet, postprocess)
_, PRNU4, _ = NoiseExtractDL(RP_patch4, model_ADNet, postprocess)


# PRNU1=RP_patch1
# PRNU2=RP_patch2
# PRNU3=RP_patch3
# PRNU4=RP_patch4

## 5 Compute the correlation between probe and RP
KI_1 = imgfloat_patch1 * PRNU2
C_ADNet1 = crosscorr(NoisexnpADNet1, KI_1)
PCE_value_ADNet1 = PCE1(C_ADNet1)

KI_2 = imgfloat_patch2 * PRNU1
C_ADNet2 = crosscorr(NoisexnpADNet2, KI_2)
PCE_value_ADNet2 = PCE1(C_ADNet2)

KI_3 = imgfloat_patch3 * PRNU1
C_ADNet3 = crosscorr(NoisexnpADNet3, KI_3)
PCE_value_ADNet3 = PCE1(C_ADNet3)

KI_4 = imgfloat_patch4 * PRNU1
C_ADNet4 = crosscorr(NoisexnpADNet4, KI_4)
PCE_value_ADNet4 = PCE1(C_ADNet4)

print('PCE_value_ADNet:')
print(PCE_value_ADNet1, PCE_value_ADNet2, PCE_value_ADNet3, PCE_value_ADNet4)