import os
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from utils import *
import CameraFingerprint.Functions as Fu
import CameraFingerprint.Filter as Ft
from SCIFunctions.crosscorr2 import crosscorr2
from torch.autograd import Variable



from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT


from match_network.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep,reparameterize_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


##论文第四章最后一个相机源识别实验测试代码，测试DWT方法+相似性得分计算网络


def preprocessing_function(x):
    return x / (np.sqrt(np.mean(np.square(x))) + np.finfo(float).eps)


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = '/home/seamus20/z_Project/Video_match/weight_Eff_repB0_dresden/Pixle_128_128-spncnn_1'
    model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    pretrained_model = torch.load(model_dir + '/checkpoint.pt')
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    model = reparameterize_model(model)
    model.eval()
    model = model.to(device)


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

        # EPRNUV=preprocessing_function(EPRNUV)

        # KI = imgV * PRNUV
        KI =   PRNUV


        C = preprocessing_function(crosscorr2(preprocessing_function(KI), EPRNUV))
        data = torch.Tensor(C[None, None, :, :])
        data = Variable(data)
        data = data.cuda()
        model.eval()
        with torch.no_grad():
            result = model(data).cpu().detach().numpy()
        score = softmax(result)[:, 1]
        score1 = score.item()

        pos_pce_values.append(score1)
        avg_pce += score1
        # print(f"avg_pce= {avg_pce}")
        print(f"PRNU PCE for block {idx}: {score1}")
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

        # EPRNUV=preprocessing_function(EPRNUV)
        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV
        # KI =   PRNUV

        C = preprocessing_function(crosscorr2(preprocessing_function(KI), EPRNUV))
        data = torch.Tensor(C[None, None, :, :])
        data = Variable(data)
        data = data.cuda()
        model.eval()
        with torch.no_grad():
            result = model(data).cpu().detach().numpy()
        score = softmax(result)[:, 1]
        score1 = score.item()

        neg_pce_values.append(score1)
        avg_pce += score1
        print(f"PRNU NE PCE for block {idx}: {score1}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))

    #计算AUC值_PRNU
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    print("Conventional_AUC:", OUR_auc)


    # # 创建 "pce文件夹" 文件夹，如果它不存在
    save_folder = "SCORE_SAVE_Dresden_REP"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
#保存权重用于后续ROC曲线绘制
    save_path = os.path.join(save_folder, 'Dresden_DWT_128_REP_scores.mat')
    sio.savemat(save_path, {'positive_pce': pos_pce_values,'negative_pce': neg_pce_values})

if __name__ == "__main__":
    main()
