import h5py
import os
import cv2
import numpy as np
from scipy.io import loadmat
from models_dude_nat import DnCNN
import torch
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
# 以下为自己编写的函数
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from NoiseExtract_wyl import NoiseExtract_wyl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
    # 1 Load saved models
    print('Loading model ...\n')
    model_ADNet = DnCNN(channels=1)
    model_ADNet.cuda()
    pretrained_model = torch.load('logs_nat/dude_nat_70.pth')
    model_ADNet.load_state_dict(pretrained_model)
    model_ADNet.eval()
    #dataset
    data = h5py.File('/home/seamus20/matlab_wyl/Train_Dataset/VISION_train_40_50_4197800.mat', 'r')#
    PRNU = np.array(data['PRNU'])
    img = np.array(data['inputs'])
    print(len(img))

    # for i in range(0, len(img), batch_size):
    #     batch_images = img[i:i + batch_size]
    #     _, Noisexnp_SPNCNN, _ = NoiseExtract_wyl(img, model_ADNet, False)
     #     all_Noisexnp_SPNCNN.append(Noisexnp_SPNCNN)
     #     print(all_Noisexnp_SPNCNN.shape)

    # Create an empty list to store Noisexnp_SPNCNN
    all_Noisexnp_SPNCNN = []

    batch_size=1000
    numofbatch = len(img) // batch_size
    for i in range(numofbatch+1):   # 包括最后一个不完整的批次
        batch_begin = i * batch_size
        batch_end = min(batch_begin+batch_size,len(img)) # 防止索引越界
        batch_images = img[batch_begin:batch_end, :, :, :]

        with torch.no_grad():
            _, Noisexnp_SPNCNN, _ = NoiseExtract_wyl(batch_images, model_ADNet, False)
            print(np.shape(Noisexnp_SPNCNN))

        all_Noisexnp_SPNCNN.extend(Noisexnp_SPNCNN)
        print(np.shape(all_Noisexnp_SPNCNN))


    # Convert the list to a NumPy array
    # all_Noisexnp_SPNCNN = np.concatenate(all_Noisexnp_SPNCNN)
    all_Noisexnp_SPNCNN = np.array(all_Noisexnp_SPNCNN)

    # 定义保存路径，例如放在 "dataset" 文件夹下
    save_dir = 'REF_PRNU'
    save_path = os.path.join(save_dir, f'REF_PRNU_dude_{len(all_Noisexnp_SPNCNN)}.npy')    # 如果文件夹不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存 NumPy 数组到文件
    np.save(save_path, all_Noisexnp_SPNCNN)

if __name__ == '__main__':
    main()


