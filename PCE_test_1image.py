import os
import argparse
from torch.utils.data import DataLoader
from SCIFunctions.NoiseExtractDL import NoiseExtractDL
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from dataset_dncnn_rho import DatasetDnCNNRho
import torch.optim as optim
from torch.autograd import Variable
from ROC_CURVE_DRAW import *
from datasetnegative import datanegative
from models_dude_nat import DnCNN
import h5py
import random
from utils import *
import numpy as np
import os
import scipy.io as sio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs_nat", help='path of log files')

opt = parser.parse_args()

def data_augmentation(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def main():
    # Load dataset
    print('Loading dataset ...\n')

    test_dataset = DatasetDnCNNRho()
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    test_dataset_ne = datanegative()
    test_loader_ne = DataLoader(test_dataset_ne, batch_size=1,
                                shuffle=False, num_workers=1,
                                drop_last=False, pin_memory=True)

    # Build model
    model = DnCNN(channels=1)
    #load pth
    # model.load_state_dict(torch.load('logs_nat/Siamese_PRNU_distance_0.5_8_27.047347688343216.pth')) #100张参考 1张测试Siamese_distance_150image_18_AUC=0.9225311279296875
    model.load_state_dict(torch.load('logs_nat/Siamese_distance_100image _3_AUC=0.9225006103515625.pth'))

    # model.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))
    # model.load_state_dict(torch.load('logs_nat/Siamese_distance_150image_17_AUC=0.92242431640625.pth'))
    # Move to GPU
    # model = nn.DataParallel(model, device_ids=[0,1])
    model.cuda()
    model.eval()
    # validate

    pos_pce_values = []
    neg_pce_values = []
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1

        PRNUV = Variable(torch.FloatTensor(test_data['PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['img']).cuda())
        with torch.no_grad():
            EPRNU = model(imgV)
            PRNUV1 = model(PRNUV)
        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV1 = PRNUV1.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV1 = PRNUV1.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()

        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV1

        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        pos_pce_values.append(PCE)
        print(f"PCE for block {idx}: {PCE}")
        avg_pce += PCE

    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))

    model.eval()

    # validate
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader_ne:
        idx += 1

        PRNUV = Variable(torch.FloatTensor(test_data['ne_PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['ne_img']).cuda())
        with torch.no_grad():
            EPRNU = model(imgV)
            PRNUV1 = model(PRNUV)
        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV1 = PRNUV1.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV1 = PRNUV1.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()

        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV1

        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_values.append(PCE)
        print(f"PCE for block {idx}: {PCE}")
        avg_pce += PCE

    avg_pce = avg_pce / idx
    print("\n ne_pce_val: %.6f" % (avg_pce))

    # 计算AUC值
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    print("AUC:", OUR_auc)

    # 创建 "pce文件夹" 文件夹，如果它不存在
    # save_folder = "PCE_save"
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)


    # # 在第一个代码中，保存数组到 指定 文件夹下的 .mat 文件
    # save_path = os.path.join(save_folder, 'Dresden_distance_pce_values_same_model_256.mat')
    # sio.savemat(save_path, {'positive_pce': pos_pce_values,'negative_pce': neg_pce_values})




if __name__ == "__main__":
    main()

