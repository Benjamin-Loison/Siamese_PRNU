import os
import argparse
import time
import h5py
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
import torch.optim as optim
from torch.autograd import Variable
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from models_dude_nat import DnCNN
from models_dude_nat import ImageFusionNet
import random
from utils import *
from noiseprint_tensor import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():

    # data = h5py.File('Dresden_test/DRESDEN_TEST_POS_PRNU128_256_120.mat', 'r')
    data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE_POSITIVE128_1_1600.mat', 'r')#large

    img_f = np.array(data['inputs'])
    img=np.array(data['inputsV'])


    # data1 = h5py.File('Dresden_test/DRESDEN_TEST_NE_PRNU128_256_120.mat', 'r')
    data1 = h5py.File('Dresden_test/DRESDEN_TEST_LARGE_NEGATIVE128_1_1600.mat', 'r')#large

    ne_img_f=np.array(data1['inputs'])
    ne_img=np.array(data1['inputsV'])

    # Load dataset
    print('Loading dataset ...\n')
    # 'path to your prepared data'

#init_validate_dataset
    test_dataset = dataset_fusion()
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    test_dataset_ne = dataset_fusion_ne()
    test_loader_ne = DataLoader(test_dataset_ne, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)


    # test_dataset_prnu_positive=dataset_prnu_positive()
    # test_loader_prnu_positive=DataLoader(test_dataset_prnu_positive, batch_size=1,
    #                             shuffle=False, num_workers=1,
    #                             drop_last=False, pin_memory=True)
    # test_dataset_prnu_negative=dataset_prnu_negative()
    # test_loader_prnu_negative=DataLoader(test_dataset_prnu_negative, batch_size=1,
    #                             shuffle=False, num_workers=1,
    #                             drop_last=False, pin_memory=True)

    # dude_net
    moudle_dncnn = DnCNN(channels=1)
    moudle_dncnn.cuda()
    moudle_dncnn.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))
    # fusion_net
    module_fusion = ImageFusionNet()
    module_fusion.cuda()
    module_fusion.load_state_dict(torch.load('logs_nat/ImageFusion-Net*image-softmax-epoch_161_0.9271527777777778.pth'))
    # module_fusion.load_state_dict(torch.load('logs_nat/ImageFusion-Net*image-softmax-epoch_108_0.9184722222222222.pth'))


    # validate
    pos_pce_values = []
    neg_pce_values = []
    pos_pce_fusion=[]
    neg_pce_fusion=[]
    pos_pce_overlay=[]
    neg_pce_overlay = []

#positive
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)
            # PRNUV1 = moudle_dncnn(PRNUV)
        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        # PRNUV = PRNUV.detach()[0].float().cpu()
        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        pos_pce_values.append(PCE)
        avg_pce += PCE
        print(f"PRNU PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))

    #overlay
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['PRNUO']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)
            PRNUV1 = moudle_dncnn(PRNUV)
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
        pos_pce_overlay.append(PCE)
        avg_pce += PCE
        print(f"OVERLAY PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))

#fusion
    avg_pce = 0.0
    idx = 0
    num= int(len(img) )
    for i in range(num):
        idx += 1

        PRNUBbegin = i
        PRNUBend = PRNUBbegin+1
        ImgBbegin = i  * 50
        ImgBend = ImgBbegin +  50
        imgV = img[PRNUBbegin:PRNUBend, :, :, :]
        PRNUB = img_f[ImgBbegin:ImgBend, :, :, :]

        imgV = Variable(torch.FloatTensor(imgV).cuda())
        PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())
        # print(PRNUB.shape)
        # print(imgV.shape)

        with torch.no_grad():

            EPRNU = moudle_dncnn(imgV)
            PRNUB1=module_fusion(PRNUB)
            PRNUV=moudle_dncnn(PRNUB1)


        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV = PRNUV.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        pos_pce_fusion.append(PCE)
        avg_pce += PCE
        print(f"FUSION PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_fusion_val: %.6f" % (avg_pce))




#####negative
    # validate
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader_ne:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['ne_PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['ne_img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)
            # PRNUV1 = moudle_dncnn(PRNUV)
        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        # PRNUV1 = PRNUV1.detach()[0].float().cpu()
        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_values.append(PCE)
        avg_pce += PCE
        print(f"PRNU NE PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))

    #overlay
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader_ne:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['ne_PRNUO']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['ne_img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)
            PRNUV1 = moudle_dncnn(PRNUV)
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
        neg_pce_overlay.append(PCE)
        avg_pce += PCE
        print(f"OVERLAY NE PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))

#FUSION
    avg_pce = 0.0
    idx = 0
    num= int(len(ne_img) )
    for i in range(num):
        idx += 1

        PRNUBbegin = i
        PRNUBend = PRNUBbegin+1
        ImgBbegin = i  * 50
        ImgBend = ImgBbegin +  50
        imgV = ne_img[PRNUBbegin:PRNUBend, :, :, :]
        PRNUB = ne_img_f[ImgBbegin:ImgBend, :, :, :]

        imgV = Variable(torch.FloatTensor(imgV).cuda())
        PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())
        # print(PRNUB.shape)
        # print(imgV.shape)

        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)
            # PRNUB = PRNUB.squeeze(1)
            PRNUB1 = module_fusion(PRNUB)
            # PRNUB1 = PRNUB1.unsqueeze(1)
            PRNUV = moudle_dncnn(PRNUB1)

        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV = PRNUV.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_fusion.append(PCE)
        avg_pce += PCE
        print(f"FUSION PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_fusion_val: %.6f" % (avg_pce))
    #
    # 计算AUC值
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    print("PRNU_AUC:", OUR_auc)

    OUR_PCE_OVERLAY = {'positive_pce': pos_pce_overlay, 'negative_pce': neg_pce_overlay}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE_OVERLAY)
    # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    print("OVERLAY_AUC:", OUR_auc)

    # 计算AUC值
    OUR_PCE_prnu = {'positive_pce': pos_pce_fusion, 'negative_pce': neg_pce_fusion}
    OUR_prnu_fpr, OUR_prnu_tpr, OUR_prnu_auc = ROC_prefcurve(OUR_PCE_prnu)
    print("FUSION_AUC: %.6f" % ( OUR_prnu_auc))


if __name__ == "__main__":
    main()
