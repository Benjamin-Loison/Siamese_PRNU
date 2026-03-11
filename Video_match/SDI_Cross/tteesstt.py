import cv2
import numpy as np
from torch.autograd import Variable
from utility_dataset import load_prnu, load_res, preprocessing_function
from SCIFunctions.crosscorr2 import crosscorr2
from SCIFunctions.crosscorr import crosscorr,crosscorr3
from SCIFunctions.PCE1 import PCE1
from src.pce import PCE
import scipy.io as sio
import numpy as np
import torch
from arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep,reparameterize_model

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


# A=np.array([[1,-2,-3],[-2,3,-4],[4,-5,6],[14,-15,16],[24,-25,26]])
# B=np.array([[1,-2,-3],[-2,3,-4],[24,-25,26]])
#
# print(max([A.shape[0]-B.shape[0],A.shape[1]-B.shape[1]],[0,0]))

# C=crosscorr2(A,B)
# print(np.around(C,4))

print('read the RP image ...\n')
# RP = sio.loadmat('./samples/D01_Samsung_GalaxyS3Mini.mat')['PRNU'][:256,:256]
RP = preprocessing_function(np.load('/home/seamus20/z_Project/SDI_Cross/VISION_PRNU_npy/D01_Samsung_GalaxyS3Mini.npy'))[:1024,:1024]

print('read the probe noise ...\n')
# Noise = sio.loadmat('./samples/D01_I_nat_0130.mat')['Noisex']
Noise = preprocessing_function(np.load('/home/seamus20/z_Datasets/Vision_res_npy/D02_Apple_iPhone4s/D02_I_nat_0008.npy'))[:1024,:1024]


print('Compute the correlation between probe and RP ...\n')

valid_shift_range = max([RP.shape[0]-Noise.shape[0],RP.shape[1]-Noise.shape[1]],[0,0])

C = crosscorr2(RP, Noise)
# PCE_value_DnCNN = PCE(C,shift_range=valid_shift_range)
# print("PCE of model_DnCNN")
# print(PCE_value_DnCNN)

model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
pretrained_model = torch.load('./weight_Eff_repB0/Pixle_256_256/checkpoint.pt')
model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
model = reparameterize_model(model)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

data = torch.Tensor(C[None, None, :, :])
data = Variable(data)
data = data.cuda()
result = model(data).cpu().detach().numpy()
print(softmax(result))