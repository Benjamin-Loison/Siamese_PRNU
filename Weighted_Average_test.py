import os
import h5py
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from torch.autograd import Variable
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from models_dude_nat import DnCNN
from models_dude_nat import Weight_Get
from utils import *

##测试提参考网络代码


from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():


# Load dataset
    print('Loading dataset ...\n')
# 'path to your prepared data'

#test-dataset
    #positive
    # data = h5py.File('Dresden_test/DRESDEN_TEST_POS128_256_120.mat', 'r')
    # data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_POSITIVE128_1_1600.mat', 'r')#large wiener
    data = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_POSITIVE-CENTER-nowiener128_1_4000.mat', 'r')#large
    img_f = np.array(data['inputs'])
    img=np.array(data['inputsV'])


   #negative
    # data1 = h5py.File('Dresden_test/DRESDEN_TEST_NE128_256_120.mat', 'r')
    # data1 = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_NEGATIVE128_1_1600.mat', 'r')#large wiener
    data1 = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_NEGATIVE-CENTER-nowiener222128_1_4000.mat', 'r')#large

    ne_img_f=np.array(data1['inputs'])
    ne_img=np.array(data1['inputsV'])



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
    module_fusion = Weight_Get()
    module_fusion.cuda()
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average_25_0.9325694444444443.pth'))
    #
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average-large_5_0.9294444444444445.pth'))#flat first
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average-wiener_48_0.9302777777777776.pth'))#flat
    # module_fusion.load_state_dict(torch.load('Fusion_Weights/weight_average-wiener_50flat_check+_49_0.9300694444444445.pth'))#flat check
    #
    module_fusion.load_state_dict(torch.load('logs_nat/weight_average-wiener_50nat_9_0.929513888888889.pth'))
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average-wiener_50nat_9_0.929513888888889.pth'))

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
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        pos_pce_values.append(PCE)
        avg_pce += PCE
        # print(f"avg_pce= {avg_pce}")
        print(f"PRNU PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))


#positive_fusion
    avg_pce = 0.0
    idx = 0
    num= int(len(img) )
    for i in range(num):
        idx += 1

        PRNUBbegin = i
        PRNUBend = PRNUBbegin+1
        ImgBbegin = i  * 50
        ImgBend = ImgBbegin + 50
        imgV = img[PRNUBbegin:PRNUBend, :, :, :]
        PRNUB = img_f[ImgBbegin:ImgBend, :, :, :]

        imgV = Variable(torch.FloatTensor(imgV).cuda())
        PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())

        # print(PRNUB.shape)

        with torch.no_grad():
            #提取参考PRNU
            weight1 = module_fusion(PRNUB)
            out_img1 = moudle_dncnn(PRNUB)
            ref1 = weight1 * out_img1
            PRNUV = ref1.sum(dim=0, keepdim=True)
            #提取测试PRNU
            EPRNU = moudle_dncnn(imgV)

            # 计算PRNUV的均值，并进行零均值化操作
            mean_value = torch.mean(PRNUV, dim=(2, 3))
            # mean_value = torch.mean(PRNUV)
            PRNUV = PRNUV - mean_value



            # # 计算PRNUV的行均值
            # row_mean = torch.mean(PRNUV, dim=2, keepdim=True)
            # # 减去行均值
            # PRNUV = PRNUV - row_mean
            #
            # # 计算PRNUV的列均值
            # column_mean = torch.mean(PRNUV, dim=3, keepdim=True)
            # # 减去列均值
            # PRNUV = PRNUV - column_mean
            #



        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV = PRNUV.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()

        #conventional PRNU enhance
        # PRNUV = Zeromean(PRNUV)
        # std = np.std(PRNUV)
        # PRNUV = WienerInDFT(PRNUV, std)
        # -----------------------
        # calculate PCE
        # -----------------------
        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        pos_pce_fusion.append(PCE)
        avg_pce += PCE
        # print(f"avg_pce= {avg_pce}")
        print(f"FUSION PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_fusion_val: %.6f" % (avg_pce))




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
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_values.append(PCE)
        avg_pce += PCE
        print(f"PRNU NE PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_val: %.6f" % (avg_pce))


#negative_fusion
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

        with torch.no_grad():
            weight1 = module_fusion(PRNUB)
            out_img1 = moudle_dncnn(PRNUB)
            ref1 = weight1 * out_img1
            PRNUV = ref1.sum(dim=0, keepdim=True)
            EPRNU = moudle_dncnn(imgV)

            # 计算PRNUV的均值
            mean_value = torch.mean(PRNUV, dim=(2, 3))
            # mean_value = torch.mean(PRNUV)
            # 零均值化操作
            PRNUV = PRNUV - mean_value

            # # 计算PRNUV的行均值
            # row_mean = torch.mean(PRNUV, dim=2, keepdim=True)
            # # 减去行均值
            # PRNUV = PRNUV - row_mean
            #
            # # 计算PRNUV的列均值
            # column_mean = torch.mean(PRNUV, dim=3, keepdim=True)
            # # 减去列均值
            # PRNUV = PRNUV - column_mean



        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV = PRNUV.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()
        # -----------------------
        # calculate PCE
        # -----------------------
        # PRNUV = Zeromean(PRNUV)
        # std = np.std(PRNUV)
        # PRNUV = WienerInDFT(PRNUV, std)

        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_fusion.append(PCE)
        avg_pce += PCE
        print(f"FUSION NE PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_fusion_val: %.6f" % (avg_pce))


    #计算AUC值_PRNU
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    print("PRNU_AUC:", OUR_auc)

    # OUR_PCE_OVERLAY = {'positive_pce': pos_pce_overlay, 'negative_pce': neg_pce_overlay}
    # OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE_OVERLAY)
    # # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    # print("OVERLAY_AUC:", OUR_auc)

    #计算AUC值_PRNU
    OUR_PCE_prnu = {'positive_pce': pos_pce_fusion, 'negative_pce': neg_pce_fusion}
    OUR_prnu_fpr, OUR_prnu_tpr, OUR_prnu_auc = ROC_prefcurve(OUR_PCE_prnu)
    print("FUSION_AUC: %.6f" % ( OUR_prnu_auc))

    # # 创建 "pce文件夹" 文件夹，如果它不存在
    save_folder = "PCE_SAVE_Dresden_Surround"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
#保存权重用于后续ROC曲线绘制
    save_path = os.path.join(save_folder, 'Dresden_PRNU-nowiener-center_128_pce_values_upper-left2.mat')
    sio.savemat(save_path, {'positive_pce': pos_pce_values,'negative_pce': neg_pce_values})

    save_path = os.path.join(save_folder, 'Dresden_FUSION-nat-nowiener-center-128_pce_values_upper-left2.mat')
    sio.savemat(save_path, {'positive_pce': pos_pce_fusion, 'negative_pce': neg_pce_fusion})

if __name__ == "__main__":
    main()
