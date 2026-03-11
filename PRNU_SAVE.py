import os
import h5py
from torch.utils.data import DataLoader
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from torch.autograd import Variable
from dataset_fusion import dataset_fusion
from dataset_fusion_ne import dataset_fusion_ne
from models_dude_nat import DnCNN
from models_dude_nat import Weight_Get_conv5_batch
from utils import *

from SCIFunctions.NoiseExtractDL import NoiseExtractDL

from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():


# Load dataset
    print('Loading dataset ...\n')
# 'path to your prepared data'

#test-dataset
    #positive
    # data = h5py.File('Dresden_test/DRESDEN_TEST_POS128_256_120.mat', 'r')
    # data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_POSITIVE128_1_1600.mat', 'r')#large wiener
    # data = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_POSITIVE-CENTER-nowiener96_1_4000.mat', 'r')#large
    data = h5py.File('Fusion_test_nowiener/DRESDEN_image_po256_1_800.mat', 'r')  # large

    img_f = np.array(data['inputs'])
    img=np.array(data['inputsV'])

   #negative
    # data1 = h5py.File('Dresden_test/DRESDEN_TEST_NE128_256_120.mat', 'r')
    # data1 = h5py.File('Dresden_test/DRESDEN_TEST_LARGE-new_NEGATIVE128_1_1600.mat', 'r')#large wiener
    # data1 = h5py.File('Fusion_test_nowiener/DRESDEN_upper-left_NEGATIVE-CENTER-nowiener196_1_4000.mat', 'r')#large
    data1 = h5py.File('Fusion_test_nowiener/DRESDEN_image_ne256_1_800.mat', 'r')  # large

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
    module_fusion = Weight_Get_conv5_batch()
    module_fusion.cuda()
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average_25_0.9325694444444443.pth'))
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average-large_5_0.9294444444444445.pth'))#flat first
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average-wiener_48_0.9302777777777776.pth'))#flat
    # module_fusion.load_state_dict(torch.load('Fusion_Weights/weight_average-wiener_50flat_check+_49_0.9300694444444445.pth'))#flat check
    # module_fusion.load_state_dict(torch.load('logs_nat/weight_average-wiener_50nat_9_0.929513888888889.pth'))
    module_fusion.load_state_dict(torch.load('Fusion_Weights/conv5_11_0.9290277777777779.pth'))

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

        with torch.no_grad():
            out_img_list = []
            # 循环遍历每张图像
            for j in range(50):
                # 选择当前的图像
                PRNUB_j = PRNUB[j:j + 1, :, :, :]  # 确保使用 j:j+1 来正确地索引
                # 使用你的模块处理当前的图像
                out_img1 = moudle_dncnn(PRNUB_j)
                # 将处理后的图像从 Variable 转换为 Tensor，并从 CUDA 移动到 CPU
                out_img_list.append(out_img1)
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            # 将列表中的图像拼接成一个批次
            out_img_batch = torch.cat(out_img_list, dim=0)

            #提取参考PRNU
            weight1 = module_fusion(PRNUB)
            # print(weight1.device)  # 将输出张量所在的设备
            # print(out_img_batch.device)  # 将输出张量所在的设备


            ref1 = weight1 * out_img_batch
            PRNUV = ref1.sum(dim=0, keepdim=True)
            #提取测试PRNU
            EPRNU = moudle_dncnn(imgV)

            # 计算PRNUV的均值，并进行零均值化操作
            mean_value = torch.mean(PRNUV, dim=(2, 3))
            PRNUV = PRNUV - mean_value

        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV = PRNUV.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()

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


###negative_prnu
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

            out_img_list = []
            # 循环遍历每张图像
            for j in range(50):  # 注意这里的索引变量应更改为 j 以避免与外部循环的 i 冲突
                # 选择当前的图像
                PRNUB_j = PRNUB[j:j + 1, :, :, :]  # 确保使用 j:j+1 来正确地索引
                # 使用你的模块处理当前的图像
                out_img1 = moudle_dncnn(PRNUB_j)  # 确保模块名称是正确的，比如 module_dncnn
                # 将处理后的图像添加到列表中
                out_img_list.append(out_img1)
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            # 将列表中的图像拼接成一个批次
            out_img_batch = torch.cat(out_img_list, dim=0)

            # 提取参考PRNU
            weight1 = module_fusion(PRNUB)
            ref1 = weight1 * out_img_batch

            PRNUV = ref1.sum(dim=0, keepdim=True)
            EPRNU = moudle_dncnn(imgV)

            # 计算PRNUV的均值
            mean_value = torch.mean(PRNUV, dim=(2, 3))
            # 零均值化操作
            PRNUV = PRNUV - mean_value

        imgV = imgV.detach()[0].float().cpu()
        EPRNU = EPRNU.detach()[0].float().cpu()
        PRNUV = PRNUV.detach()[0].float().cpu()

        EPRNUV = EPRNU.squeeze().float().cpu().numpy()
        PRNUV = PRNUV.squeeze().float().cpu().numpy()
        imgV = imgV.squeeze().float().cpu().numpy()

        KI = imgV * PRNUV
        C = crosscorr(EPRNUV, KI)
        PCE = PCE1(C)
        # 将当前 PCE 值添加到数组中
        neg_pce_fusion.append(PCE)
        avg_pce += PCE
        print(f"FUSION NE PCE for block {idx}: {PCE}")
    avg_pce = avg_pce / idx
    print("\n pce_fusion_val: %.6f" % (avg_pce))


    #计算AUC值_
    OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
    # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    print("PRNU_AUC:", OUR_auc)

    #计算AUC值_dude
    OUR_PCE_prnu = {'positive_pce': pos_pce_fusion, 'negative_pce': neg_pce_fusion}
    OUR_prnu_fpr, OUR_prnu_tpr, OUR_prnu_auc = ROC_prefcurve(OUR_PCE_prnu)
    print("FUSION_AUC: %.6f" % ( OUR_prnu_auc))

    # # 创建 "pce文件夹" 文件夹，如果它不存在
    save_folder = "PCE_SAVE_Dresden_Surround"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


if __name__ == "__main__":
    main()
