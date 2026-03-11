
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data


class dataset_prnu_negative(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self):
        super(dataset_prnu_negative, self).__init__()

        # data = h5py.File('data/VISIONdemotriplettest_negative_device=17_image_40_60_128_250.mat', 'r')#验证集
        # data = h5py.File('data/VISIONdemotriplettest_negative_device=17_patch_128_128_256.mat', 'r')#128*128*5120

        # data = h5py.File('data/Dresdendemotriplettest_negative_patch_128_256.mat', 'r')
        #
        # data = h5py.File('data/Dresdentest_negative_patch_128_5120.mat', 'r')  # 同设备，不同相机


        # data = h5py.File('Dresden_test/Dresdentest_negative_patch_100image_128_256.mat', 'r')  # 单张参考 单张测试
        # data = h5py.File('data/Dresdentest_negative_patch_128_256.mat', 'r')  # 负例、128*128、256patch dresden

        data = h5py.File('Dresden_test/Dresdentest_negative_PRNU-IMAGES_128_256.mat', 'r')  # 负例、128*128、256patch dresden


        self.PRNUs = np.array(data['PRNUV'])
        self.inputs = np.array(data['inputsV'])
        data.close()


    def __getitem__(self, index):

        """
        # --------------------------------
        # get L/H image pairs
        # --------------------------------
        """
        ne_PRNU = self.PRNUs[index]
        ne_img = self.inputs[index]

        ne_PRNU = torch.from_numpy(np.ascontiguousarray(ne_PRNU))
        ne_img = torch.from_numpy(np.ascontiguousarray(ne_img))


        return {'ne_PRNU': ne_PRNU, 'ne_img': ne_img}

    def __len__(self):
        return int(len(self.PRNUs))

