import os
import argparse

import torch
from torch.utils.data import DataLoader
from SCIFunctions.NoiseExtractDL import NoiseExtractDL
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
import torch.optim as optim
from torch.autograd import Variable

from dataset_dncnn_rho import DatasetDnCNNRho
from datasetnegative import datanegative
from models_dude_nat import DnCNN
import h5py
import random
from utils import *
# from test_loss import *
from NoisePrint_torch import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#一机多卡设置
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'#设置所有可以使用的显卡，共计四块
# device_ids = [0,1]#选中其中两块
# model = nn.DataParallel(model, device_ids=device_ids)#并行使用两块
# #net = torch.nn.Dataparallel(model)  # 默认使用所有的device_ids
# model = model.cuda()

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
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
    resume = True
    rangeepochs = range(opt.epochs)
    # Load dataset
    print('Loading dataset ...\n')
    # 'path to your prepared data'
    # data = h5py.File('data/VISION_distance_40_64.mat', 'r')
    # data = h5py.File('data/VISION_distance_step=10_40_64_119168.mat', 'r')
    # data = h5py.File('data/VISION_distance_100image_40_64_198592.mat', 'r')#负优化，反auc数据集
    data = h5py.File('data/VISION_train_step=20_mod50_40_64_112128.mat', 'r')#



    # data = h5py.File('data/VISION_distance_40_64_95360.mat', 'r')
    PRNU = np.array(data['PRNU'])
    img = np.array(data['inputs'])

    test_dataset = DatasetDnCNNRho()
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)


    test_dataset_ne = datanegative()
    test_loader_ne = DataLoader(test_dataset_ne, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    print("# of training samples: %d\n" % int(len(img)))


    # Build model
    model = DnCNN(channels=1)
    # model.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    # model = nn.DataParallel(model, device_ids=[0,1])
    model.cuda()
    criterion.cuda()

    if resume:
        model.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))
        # model.load_state_dict(torch.load('logs_nat/Siamese_mloss_100image_1_AUC=0.1571044921875.pth'))#负优化，反auc


        # rangeepochs = range(20, 40)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    batchsize = opt.batchSize
    numofbatch = int(len(PRNU) / batchsize)
    max_pce = 60
    for epoch in rangeepochs:
        # for epoch in range(opt.epochs):
        if epoch <= 30:
            current_lr = opt.lr
        # if epoch > 20 and epoch <= 40:
        #     current_lr = opt.lr / 10.
        if epoch > 30 and epoch <= 60:
            current_lr = opt.lr / 10.
        if epoch > 60 and epoch <= 70:
            current_lr = opt.lr / 100.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        index = list(range(0, len(PRNU)))
        # random.shuffle(index)

        # 训练数据需要置乱
        shuffedPRNU = PRNU[index, :, :, :]
        shuffedImg = img[index, :, :, :]

        # group_indices = np.eye(batchsize)  # 对角线为 1，其余位置为 0
        # group_indices[group_indices == 0] = -1  # 将对角线以外的元素设置为 -1
        group_indices = torch.eye(batchsize)  # 创建一个对角线为 1 的单位矩阵
        group_indices[group_indices == 0] = -1  # 将对角线以外的元素设置为 -1
        # group_indices=torch.Tensor(group_indices)

        total_loss = 0.0
        batch_count = 0

        for i in range(numofbatch):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            Bbegin = i * batchsize
            Bend = Bbegin + batchsize
            PRNUB = shuffedPRNU[Bbegin:Bend, :, :, :]
            ImgB = shuffedImg[Bbegin:Bend, :, :, :]

            # 训练数据需要增广
            augmod = random.randint(0, 3)

            temptrans = np.transpose(ImgB, (2, 3, 0, 1))
            temptrans = data_augmentation(temptrans, augmod)
            ImgB = np.transpose(temptrans, (2, 3, 0, 1)).copy()

            temptrans = np.transpose(PRNUB, (2, 3, 0, 1))
            temptrans = data_augmentation(temptrans, augmod)
            PRNUB = np.transpose(temptrans, (2, 3, 0, 1)).copy()

            ImgB = Variable(torch.FloatTensor(ImgB).cuda())
            PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())


            #siamese network
            out_prnu = model(PRNUB)
            out_img= model(ImgB)
            # print("Shape of out_prnu:", out_prnu.shape)
            # print("Shape of out_img:", out_img.shape)




            # 假设 device 是你选择的GPU设备
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # 将 distances、probabilities、group_indices 移动到 GPU 上

            distances = calculate_distance(out_prnu,out_img)
            probabilities = calculate_probabilities(distances)
            batch_loss = calculate_batch_loss(probabilities, group_indices)

            #谱均衡化
            # power_spectrum_density1 = calculate_power_spectrum_density(out_prnu)
            # regularization_term1 = calculate_regularization_term(power_spectrum_density1)
            #
            # power_spectrum_density2 = calculate_power_spectrum_density(out_img)
            # regularization_term2 = calculate_regularization_term(power_spectrum_density2)
            #
            # regularization_term = regularization_term1 + regularization_term2
            # # print(regularization_term)
            # regularization_weight=0#正则化权重
            # batch_loss_tensor = torch.tensor(batch_loss, requires_grad=True)
            # total_loss = batch_loss_tensor - regularization_weight * regularization_term

            # 将 batch_loss 转换为 PyTorch 张量
            loss=batch_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # results
            if (i+1) % 100 == 0:
                print("[epoch %d][%d/%d]   loss: %.7f" %
                      (epoch + 1, i + 1, numofbatch, loss.item()))

        # the end of each epoch
        model.eval()

        avg_loss = total_loss / batch_count
        print("\n[epoch %d] Average Loss: %.7f" % (epoch + 1, avg_loss))

        pos_pce_values = []
        neg_pce_values = []
        # validate
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
            pos_pce_values.append(PCE)
            avg_pce += PCE
        avg_pce = avg_pce / idx
        print("\n[epoch %d] pce_val: %.6f" % (epoch + 1, avg_pce))

        #negative sample
        avg_pce_ne = 0.0
        idx1 = 0
        for test_data_ne in test_loader_ne:
            idx1 += 1

            ne_PRNUV = Variable(torch.FloatTensor(test_data_ne['ne_PRNU']).cuda())
            ne_imgV = Variable(torch.FloatTensor(test_data_ne['ne_img']).cuda())
            with torch.no_grad():
                EPRNU = model(ne_imgV)
                PRNUV1 = model(ne_PRNUV)
            ne_imgV = ne_imgV.detach()[0].float().cpu()
            EPRNU = EPRNU.detach()[0].float().cpu()
            PRNUV1 = PRNUV1.detach()[0].float().cpu()

            EPRNUV = EPRNU.squeeze().float().cpu().numpy()
            PRNUV1 = PRNUV1.squeeze().float().cpu().numpy()
            ne_imgV = ne_imgV.squeeze().float().cpu().numpy()

            # -----------------------
            # calculate PCE
            # -----------------------
            KI = ne_imgV * PRNUV1
            C = crosscorr(EPRNUV, KI)
            PCE = PCE1(C)
            neg_pce_values.append(PCE)
            avg_pce_ne += PCE
        avg_pce_ne = avg_pce_ne / idx1
        print("\n[epoch %d] ne_pce_val: %.6f" % (epoch + 1, avg_pce_ne))

        #计算AUC值
        OUR_PCE = {'positive_pce': pos_pce_values, 'negative_pce': neg_pce_values}
        OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
        # dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
        print("\n[epoch %d] AUC: %.6f" % (epoch + 1, OUR_auc))

        # 计算AUC 保留7位小数
        OUR_auc = round(OUR_auc, 7)
        # 计算平均损失
        avg_loss = round(avg_loss, 7)


        # # save model
        model_name = 'Siamese_mloss_random' + '_' + str(epoch + 1) + '_AUC=' + str(OUR_auc) + '_loss=' + str(avg_loss)+'.pth'
        torch.save(model.state_dict(), os.path.join(opt.outf, model_name))
    torch.save(model.state_dict(), os.path.join(opt.outf, 'Siamese_PRNU.pth'))


if __name__ == "__main__":
    main()


