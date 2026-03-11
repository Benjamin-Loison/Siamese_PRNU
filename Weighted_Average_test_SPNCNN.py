import os
import h5py
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from torch.autograd import Variable
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from DnCNN import *
from utils import *

from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

##测试代码，测试SPNCNN方法


def main():
#
# #test-dataset
#     #positive
#     # data = h5py.File('Dresden_test/DRESDEN_TEST_POS_PRNU128_256_120.mat', 'r')
#     # data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_POSITIVE128_1_1600.mat', 'r')#large wiener
#     data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE_POSITIVE64_1_1600.mat', 'r')#large
#     img_f = np.array(data['inputs'])
#     img=np.array(data['inputsV'])
#
#     #negative
#     # data1 = h5py.File('Dresden_test/DRESDEN_TEST_NE_PRNU128_256_120.mat', 'r')
#     # data1 = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_NEGATIVE128_1_1600.mat', 'r')#large wiener
#     data1 = h5py.File('Dresden_test/DRESDEN_TEST_LARGE_NEGATIVE64_1_1600.mat', 'r')#large
#
#     ne_img_f=np.array(data1['inputs'])
#     ne_img=np.array(data1['inputsV'])

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


    # SPNCNN_MSE
    moudle_spncnn = DnCNN_SPN(channels=1)
    # model_ADNet.cuda()
    moudle_spncnn = nn.DataParallel(moudle_spncnn, device_ids=device_ids).cuda()
    pretrained_model = torch.load('logs_nat/DnCNN_SPN_MSE.pth')
    moudle_spncnn.load_state_dict(pretrained_model)
    moudle_spncnn.eval()

# validate
    pos_pce_values = []
    neg_pce_values = []

#positive_prnu
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_spncnn(imgV)

        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        # PRNUV = PRNUV.detach()[0].float().cpu()
        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        PRNUV = PRNUV*255

        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        pos_pce_values.append(PCE)
        avg_pce += PCE
        # print(f"avg_pce= {avg_pce}")
        print(f"PRNU PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))


#####negative_prnu
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader_ne:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['ne_PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['ne_img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_spncnn(imgV)

        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        # PRNUV1 = PRNUV1.detach()[0].float().cpu()
        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        PRNUV = PRNUV*255
        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_values.append(PCE)
        avg_pce += PCE
        print(f"PRNU NE PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))



    #计算AUC值_PRNU
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    print("SPNCNN_AUC:", OUR_auc)


    # # 创建 "pce文件夹" 文件夹，如果它不存在
    save_folder = "PCE_SAVE_Dresden_Surround_SPNCNN"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
# # #保存权重用于后续ROC曲线绘制
    save_path = os.path.join(save_folder, 'Dresden_SPNCNN-nowiener-center_128_pce_values_upper-left2.mat')
    sio.savemat(save_path, {'positive_pce': pos_pce_values,'negative_pce': neg_pce_values})

if __name__ == "__main__":
    main()
