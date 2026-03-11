
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data


class DatasetDnCNNRho(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self):
        super(DatasetDnCNNRho, self).__init__()

        # data = h5py.File('data/VISIONdemotriplettest_device=20_image_40_60_128_250.mat', 'r')#正例、128*128、250patch
        # data = h5py.File('data/VISIONdemotriplettest_positive_device=20_patch_128_128_256.mat', 'r')#正例、256*256、2560patch

        # data = h5py.File('data/Dresdendemotriplettest_positive_patch_128_256.mat', 'r')
        #
        # data = h5py.File('data/Dresdentest_positive_patch_128_5120.mat', 'r')  # 同设备，不同相机

        data = h5py.File('data/Dresdentest_positive_patch_128_256.mat', 'r')  # 正例、128*128、256patch dresden
        # data = h5py.File('Dresden_test/Dresdentest_positive_patch_100image_128_256.mat', 'r')  # 单张参考 单张测试



        self.PRNUs = np.array(data['PRNUV'])
        self.inputs = np.array(data['inputsV'])
        data.close()


    def __getitem__(self, index):

        """
        # --------------------------------
        # get L/H image pairs
        # --------------------------------
        """
        PRNU = self.PRNUs[index]
        img = self.inputs[index]

        PRNU = torch.from_numpy(np.ascontiguousarray(PRNU))
        img = torch.from_numpy(np.ascontiguousarray(img))


        return {'PRNU': PRNU, 'img': img}

    def __len__(self):
        return int(len(self.PRNUs))

