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
from NoiseExtract_wyl import NoiseExtract_wyl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
def main():
    # 1 Load saved models
    print('Loading model ...\n')
    #SPNCNN_MSE
    model_ADNet = DnCNN_SPN(channels=1)
    # model_ADNet.cuda()
    model_ADNet = nn.DataParallel(model_ADNet, device_ids=device_ids).cuda()
    pretrained_model = torch.load('logs_nat/DnCNN_SPN_MSE.pth')
    model_ADNet.load_state_dict(pretrained_model)
    model_ADNet.eval()
    #dataset
    data = h5py.File('/home/seamus20/matlab_wyl/Train_Dataset/VISION_train_40_50_4197800.mat', 'r')#
    PRNU = np.array(data['PRNU'])
    img = np.array(data['inputs'])
    print(len(img))

    # Create an empty list to store Noisexnp_SPNCNN
    all_Noisexnp_SPNCNN = []

    # for i in range(0, len(img), batch_size):
    #     batch_images = img[i:i + batch_size]
    #     _, Noisexnp_SPNCNN, _ = NoiseExtract_wyl(img, model_ADNet, False)
     #     all_Noisexnp_SPNCNN.append(Noisexnp_SPNCNN)
     #     print(all_Noisexnp_SPNCNN.shape)

    save_dir = 'REF_PRNU'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = 1000
    numofbatch = len(img) // batch_size

    for i in range(numofbatch + 1):  # 包括最后一个不完整的批次
        batch_begin = i * batch_size
        batch_end = min(batch_begin + batch_size, len(img))  # 防止索引越界
        batch_images = img[batch_begin:batch_end, :, :, :]

        with torch.no_grad():
            _, Noisexnp_SPNCNN, _ = NoiseExtract_wyl(batch_images, model_ADNet, False)
            print(f'Noisexnp_SPNCNN={np.shape(Noisexnp_SPNCNN)}')

        all_Noisexnp_SPNCNN.extend(Noisexnp_SPNCNN)
        print(f'all={np.shape(all_Noisexnp_SPNCNN)}')

        # 每500个批次保存一个文件
        if (i % 500 == 0 and i >= 500) or i == numofbatch:
            # Convert the list to a NumPy array
            all_Noisexnp_SPNCNN_array = np.array(all_Noisexnp_SPNCNN)

            # 保存 NumPy 数组到文件
            save_path = os.path.join(save_dir, f'REF_PRNU_spncnn_{i}.npy')
            np.save(save_path, all_Noisexnp_SPNCNN_array)

            # 清空列表，准备保存下一个文件
            all_Noisexnp_SPNCNN = []

        # 合并所有保存的文件
    merged_array = np.concatenate(
        [np.load(os.path.join(save_dir, f'REF_PRNU_spncnn_{i}.npy')) for i in range(500, numofbatch + 1, 500)])

    # 定义最终的保存路径
    final_save_path = os.path.join(save_dir, f'REF_PRNU_spncnn_merged.npy')
    np.save(final_save_path, merged_array)


if __name__ == '__main__':
    main()