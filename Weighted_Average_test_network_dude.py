import os
import h5py
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.crosscorr2 import crosscorr2
from SCIFunctions.PCE1 import PCE1
from src.pce import PCE2
from torch.autograd import Variable
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from models_dude_nat import DnCNN
from models_dude_nat import Weight_Get
from model_conv import Weight_Get_conv7
from model_attention import Weight_Get_conv7_se
from utils import *


from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT

from match_network.mobilenet_v3_eca import mbv3_small_eca
from match_network.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep,reparameterize_model

##论文第四章最后一个相机源识别实验测试代码，测试DFPRNU-Net方法+相似性得分计算网络

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


    # dude_net
    moudle_dncnn = DnCNN(channels=1)
    moudle_dncnn.cuda()
    moudle_dncnn.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))
    # fusion_net
    module_fusion = Weight_Get_conv7_se()
    module_fusion.cuda()
    module_fusion.load_state_dict(torch.load('Fusion_Weights/conv7-se_37_0.9316666666666666.pth')) #0.9157
    #

    # module_fusion.load_state_dict(torch.load('Fusion_Weights/conv7-se_1*1_16_0.9315972222222223.pth'))  #

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dir = '/home/seamus20/z_Project/Video_match/weight_MBNetV3_ECA/Pixle_128_128-601'
    # model_dir = '/home/seamus20/z_Project/Video_match/weight_Eff_repB0_dresden/Pixle_128_128-fusion'
    # model = mbv3_small_eca()
    # pretrained_model = torch.load(model_dir + '/checkpoint.pt')
    # model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    # model.eval()
    # model = model.to(device)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = '/home/seamus20/z_Project/Video_match/weight_Eff_repB0_dresden/Pixle_128_128-dude'
    model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    pretrained_model = torch.load(model_dir + '/checkpoint.pt')
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    model = reparameterize_model(model)
    model.eval()
    model = model.to(device)

    #
    # model1_dir = '/home/seamus20/z_Project/Video_match/weight_MBNetV3_ECA/Pixle_128_128-dude-prnu-2'
    # model1 = mbv3_small_eca()
    # pretrained_model = torch.load(model1_dir + '/checkpoint.pt')
    # model1.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    # model1.eval()
    # model1 = model1.to(device)


# validate
    pos_pce_values = []
    neg_pce_values = []
    pos_pce_fusion=[]
    neg_pce_fusion=[]

#positive_prnu
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)

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
        # KI =  PRNUV
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
    print("\n score_val: %.6f" % (avg_pce))


#####negative_prnu
    avg_pce = 0.0
    idx = 0
    for test_data in test_loader_ne:
        idx += 1
        PRNUV = Variable(torch.FloatTensor(test_data['ne_PRNU']).cuda())
        imgV = Variable(torch.FloatTensor(test_data['ne_img']).cuda())
        with torch.no_grad():
            EPRNU = moudle_dncnn(imgV)

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
        # KI =  PRNUV

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
    print("\n score_val: %.6f" % (avg_pce))


    #计算AUC值_FUSION
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    print("PRNU_AUC:", OUR_auc)

    # 创建 "pce文件夹" 文件夹，如果它不存在
    save_folder = "SCORE_SAVE_Dresden_REP"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    #保存权重用于后续ROC曲线绘制
    save_path = os.path.join(save_folder, 'Dresden_dude_128_REP_scores.mat')
    sio.savemat(save_path, {'positive_pce': pos_pce_values,'negative_pce': neg_pce_values})

if __name__ == "__main__":
    main()
