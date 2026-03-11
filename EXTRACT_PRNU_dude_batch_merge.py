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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # 1 Load saved models
    print('Loading model ...\n')
    model_ADNet = DnCNN(channels=1)
    model_ADNet.cuda()
    pretrained_model = torch.load('logs_nat/dude_nat_70.pth')
    model_ADNet.load_state_dict(pretrained_model)
    model_ADNet.eval()

    # dataset
    data = h5py.File('/home/seamus20/matlab_wyl/Train_Dataset/VISION_train_40_50_4197800.mat', 'r')
    PRNU = np.array(data['PRNU'])
    img = np.array(data['inputs'])
    print(len(img))

    # Create an empty list to store Noisexnp_SPNCNN
    all_Noisexnp_SPNCNN = []

    batch_size = 1000
    numofbatch = len(img) // batch_size

    save_dir = 'REF_PRNU'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(numofbatch + 1):   # 包括最后一个不完整的批次
        batch_begin = i * batch_size
        batch_end = min(batch_begin + batch_size, len(img))  # 防止索引越界
        batch_images = img[batch_begin:batch_end, :, :, :]

        with torch.no_grad():
            _, Noisexnp_SPNCNN, _ = NoiseExtract_wyl(batch_images, model_ADNet, False)
            print(f'Noisexnp_SPNCNN={np.shape(Noisexnp_SPNCNN)}')

        all_Noisexnp_SPNCNN.extend(Noisexnp_SPNCNN)
        print(f'all={np.shape(all_Noisexnp_SPNCNN)}')

        # 每500个批次保存一个文件
        if (i % 1000 == 0 and i >= 1000) or i == numofbatch:
            # Convert the list to a NumPy array
            all_Noisexnp_SPNCNN_array = np.array(all_Noisexnp_SPNCNN)

            # 保存 NumPy 数组到文件
            save_path = os.path.join(save_dir, f'REF_PRNU_dude_{i}.npy')
            np.save(save_path, all_Noisexnp_SPNCNN_array)

            # 清空列表，准备保存下一个文件
            all_Noisexnp_SPNCNN = []

    # 合并所有保存的文件
    merged_array = np.concatenate([np.load(os.path.join(save_dir, f'REF_PRNU_dude_{i}.npy')) for i in range(1000, numofbatch + 1, 1000)])

    # 定义最终的保存路径
    final_save_path = os.path.join(save_dir, f'REF_PRNU_dude_merged.npy')
    np.save(final_save_path, merged_array)

if __name__ == '__main__':
    main()
