import os
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from utils import *
import CameraFingerprint.Functions as Fu
import CameraFingerprint.Filter as Ft

from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT


##测试代码，测试dwt方法

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
def main():

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
# validate
    pos_pce_values = []
    neg_pce_values = []

#positive_prnu
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1
        PRNUV = test_data['PRNU']
        imgV = test_data['img']
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        # PRNUV = PRNUV*255
        img = (imgV * 255).astype(np.uint8)
        # print(img.dtype)
        EPRNUV = Ft.NoiseExtractFromImage(img, sigma=2.)
        EPRNUV = Fu.WienerInDFT(EPRNUV, np.std(EPRNUV))
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

        PRNUV = test_data['ne_PRNU']
        imgV = test_data['ne_img']

        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # PRNUV = PRNUV*255
        img = (imgV * 255).astype(np.uint8)
        EPRNUV = Ft.NoiseExtractFromImage(img, sigma=2.)
        EPRNUV = Fu.WienerInDFT(EPRNUV, np.std(EPRNUV))
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

    #计算AUC值_PRNU
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    print("Conventional_AUC:", OUR_auc)


    # # 创建 "pce文件夹" 文件夹，如果它不存在
    save_folder = "PCE_SAVE_Dresden_Surround_Conventional"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
#保存权重用于后续ROC曲线绘制
    # save_path = os.path.join(save_folder, 'Dresden_Conventional-nowiener-center_128_pce_values_upper-left2.mat')
    # sio.savemat(save_path, {'positive_pce': pos_pce_values,'negative_pce': neg_pce_values})

if __name__ == "__main__":
    main()
