
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data


class dataset_fusion(data.Dataset):
    def __init__(self):
        super(dataset_fusion, self).__init__()
        data = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_POSITIVE-CENTER-nowiener128_1_4000.mat','r')  # large


        # data = h5py.File('Dresden_test/DRESDEN_TEST_POS_PRNU128_256_120.mat', 'r')
        # data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_POSITIVE128_1_1600.mat', 'r')#large
        # data = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_POSITIVE-CENTER-nowiener96_1_4000.mat', 'r')  # large

        # data = h5py.File('Fusion_test_nowiener/DRESDEN_lower-left_POSITIVE-CENTER-nowiener160_1_4000.mat', 'r')  # large


        # data = h5py.File('Fusion_test_nowiener/DRESDEN_image_po256_1_800.mat', 'r')  # large

        # data = h5py.File('Fusion_test_nowiener/DRESDEN_TEST_LARGE_POSITIVE-CENTER-nowiener128_1_4000.mat', 'r')  # test rep

        # data = h5py.File('Fusion_test/DRESDEN_TEST_LARGE_POSITIVE-CENTER-wiener128_1_2000.mat', 'r')  # wien  er


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

        PRNU = torch.from_numpy(np.ascontiguousarray(PRNU))
        # PRNUO = torch.from_numpy(np.ascontiguousarray(PRNUO))
        img = torch.from_numpy(np.ascontiguousarray(img))
        imgf = torch.from_numpy(np.ascontiguousarray(imgf))

        return {'PRNU': PRNU, 'img': img,'img_fusion':imgf}

    def __len__(self):
        return int(len(self.PRNUV))

