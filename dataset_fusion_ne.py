
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data


class dataset_fusion_ne(data.Dataset):
    def __init__(self):
        super(dataset_fusion_ne, self).__init__()
        data = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_NEGATIVE-CENTER-nowiener222128_1_4000.mat','r')  # large




        # data = h5py.File('Dresden_test/DRESDEN_TEST_NE_PRNU128_256_120.mat', 'r')
        # data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_NEGATIVE128_1_1600.mat', 'r')#large
        # data = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_NEGATIVE-CENTER-nowiener196_1_4000.mat','r')  # large


        # data = h5py.File('Fusion_test_nowiener/DRESDEN_lower-left_NEGATIVE-CENTER-nowiener160_1_4000.mat','r')  # large

        # data = h5py.File('Fusion_test_nowiener/DRESDEN_TEST_LARGE_NEGATIVE-CENTER-nowiener128_1_4000.mat',
        #                   'r')  # test rep

        # data = h5py.File('Fusion_test_nowiener/DRESDEN_image_ne256_1_800.mat', 'r')  # large
        # data = h5py.File('Fusion_test/DRESDEN_TEST_LARGE_NEGATIVE-CENTER-wiener128_1_2000.mat', 'r')  # wiener


        self.PRNUV = np.array(data['PRNUV'])
        # self.PRNUO = np.array(data['PRNUO'])
        self.inputsV = np.array(data['inputsV'])
        self.inputs = np.array(data['inputs'])
        data.close()

    def __getitem__(self, index):
        """
        # --------------------------------
        # get L/H image pairs
        # --------------------------------
        """
        PRNU = self.PRNUV[index]
        # PRNUO = self.PRNUO[index]
        img = self.inputsV[index]
        imgf = self.inputs[index]

        # PRNUO = torch.from_numpy(np.ascontiguousarray(PRNUO))
        PRNU = torch.from_numpy(np.ascontiguousarray(PRNU))
        img = torch.from_numpy(np.ascontiguousarray(img))
        imgf = torch.from_numpy(np.ascontiguousarray(imgf))
        return {'ne_PRNU': PRNU,'ne_img': img,'ne_img_fusion':imgf}

    def __len__(self):
        return int(len(self.PRNUV))

