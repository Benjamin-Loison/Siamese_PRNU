import os
import argparse
import h5py
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
import torch.optim as optim
from torch.autograd import Variable
from models_dude_nat import DnCNN
from DnCNN import *
from models_dude_nat import Weight_Get
import random
from utils import *
from noiseprint_tensor import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=20, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
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
    rangeepochs = range(opt.epochs)
    # Load dataset
    print('Loading dataset ...\n')
    # training dataset
    data = h5py.File('/home/seamus20/matlab_wyl/Train_Dataset/VISION_train_40_50_4197800.mat', 'r')#big
    # data = h5py.File('/home/seamus20/matlab_wyl/Train_Dataset/VISION_train_40_50_6554700.mat', 'r')#large

    img = np.array(data['inputs'])
    PRNU = np.array(data['PRNU'])

    dude_fingerprint = np.load('REF_PRNU/Fingerprint_dude_83956.npy')
    # spncnn_noise = np.load('REF_PRNU/REF_PRNU_spncnn_merged.npy')

    #validate dataset
    data_pos = h5py.File('Dresden_test/DRESDEN_TEST_POS128_256_120.mat', 'r')#positive 128*128
    img_f = np.array(data_pos['inputs'])
    img_test = np.array(data_pos['inputsV'])

    data_ne = h5py.File('Dresden_test/DRESDEN_TEST_NE128_256_120.mat', 'r')#negative
    ne_img_f = np.array(data_ne['inputs'])
    ne_img_test = np.array(data_ne['inputsV'])

    print("# of training samples: %d\n" % int(len(PRNU)))
    print("# of validate samples: %d\n" % int(len(img_test)))

    # dude net
    moudle_dncnn = DnCNN(channels=1)
    moudle_dncnn.cuda()
    moudle_dncnn.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))
    for param in moudle_dncnn.parameters():
        param.requires_grad = False

    # weighted_average net
    module_fusion = Weight_Get()
    module_fusion.cuda()
    # module_fusion.load_state_dict(torch.load('logs_nat/ImageFusion-Net*image-softmax-epoch_161_0.9271527777777778.pth'))
    module_fusion.load_state_dict(torch.load('logs_nat/weight_average_25_0.9325694444444443.pth'))

    # SPN_CNN
    moudle_spncnn = DnCNN_SPN(channels=1)
    # moudle_spncnn.cuda()
    device_ids = [0]
    moudle_spncnn = nn.DataParallel(moudle_spncnn, device_ids=device_ids).cuda()
    moudle_spncnn.load_state_dict(torch.load('logs_nat/DnCNN_SPN_MSE.pth'))
    for param in moudle_spncnn.parameters():
        param.requires_grad = False


    # Optimizer
    optimizer_fusion = optim.Adam(module_fusion.parameters(), lr=opt.lr)
    # optimizer_dncnn = optim.Adam(moudle_dncnn.parameters(), lr=opt.lr)

    # training_dncnn
    batchsize = opt.batchSize
    numofbatch = int(len(PRNU) / batchsize)
    max_AUC = 0.5

    for epoch in rangeepochs:
        # for epoch in range(opt.epochs):
        if epoch <= 30:
            current_lr = opt.lr
        if epoch > 30 and epoch <= 50:
            current_lr = opt.lr / 10.
        if epoch > 50 and epoch <= 100:
            current_lr = opt.lr / 100.
        # set learning rate
        # for param_group in optimizer_dncnn.param_groups:
        #     param_group["lr"] = current_lr
        # print('learning rate %f' % current_lr)
        for param_group in optimizer_fusion.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)


        for i in range(numofbatch):
            # training step
            module_fusion.train()
            # moudle_dncnn.train()

            module_fusion.zero_grad()
            optimizer_fusion.zero_grad()

            PRNUBbegin = i * batchsize
            PRNUBend = PRNUBbegin + batchsize
            ImgBbegin= i * batchsize*50
            ImgBend= ImgBbegin+ batchsize*50
            PRNUB = PRNU[PRNUBbegin:PRNUBend, :, :, :]
            ImgB = img[ImgBbegin:ImgBend, :, :, :]

            # 训练数据需要增广
            augmod = random.randint(0, 3)
            temptrans = np.transpose(ImgB, (2, 3, 0, 1))
            temptrans = data_augmentation(temptrans, augmod)
            ImgB = np.transpose(temptrans, (2, 3, 0, 1)).copy()
            temptrans = np.transpose(PRNUB, (2, 3, 0, 1))
            temptrans = data_augmentation(temptrans, augmod)
            PRNUB = np.transpose(temptrans, (2, 3, 0, 1)).copy()
            #把训练数据放到gpu上
            ImgB = Variable(torch.FloatTensor(ImgB).cuda())
            PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())
            # siamese network

            weight = module_fusion(ImgB)
            out_img = moudle_spncnn(ImgB)
            # print(out_img.shape)
            ref = weight * out_img
            # ref_weight = torch.sum(ref, dim=0, keepdim=True)
            ref_reshaped = ref.view(-1, 50, 1, 40, 40)
            # 沿着新的维度求和
            ref_summed = ref_reshaped.sum(dim=1, keepdim=False)
            # # 打印结果的维度
            # print(ref_summed.shape)
            loss = mloss(ref_summed, PRNUB)
            loss.backward()
            optimizer_fusion.step()

            # results
            if (i+1) % 100 == 0:
                print("[epoch %d][%d/%d]   loss: %.7f" %
                      (epoch + 1, i + 1, numofbatch, loss.item()))

        if (epoch + 1) % 1 == 0:
            torch.save(module_fusion.state_dict(),os.path.join(opt.outf, f'weight_average_spncnn-prnu-epoch{epoch + 1}.pth'))

        module_fusion.eval()

        pos_pce_fusion = []
        neg_pce_fusion = []

        avg_pce = 0.0
        idx = 0
        num = int(len(img_test))
        for i in range(num):
            idx += 1
            PRNUBbegin = i
            PRNUBend = PRNUBbegin + 1
            ImgBbegin = i * 50
            ImgBend = ImgBbegin + 50
            imgV = img_test[PRNUBbegin:PRNUBend, :, :, :]
            PRNUB = img_f[ImgBbegin:ImgBend, :, :, :]
            imgV = Variable(torch.FloatTensor(imgV).cuda())
            PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())

            with torch.no_grad():
                weight1 = module_fusion(PRNUB)
                out_img1 = moudle_spncnn(PRNUB)
                ref1 = weight1 * out_img1
                ref_reshaped1 = ref1.view(-1, 50, 1, 128, 128)
                PRNUV = ref_reshaped1.sum(dim=1, keepdim=False)
                EPRNU = moudle_spncnn(imgV)

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
            # print(f"PCE for block {idx}: {PCE}")
        avg_pce = avg_pce / idx
        print("\n pce_fusion_val: %.6f" % (avg_pce))

        avg_pce = 0.0
        idx = 0
        num = int(len(ne_img_test))
        for i in range(num):
            idx += 1

            PRNUBbegin = i
            PRNUBend = PRNUBbegin + 1
            ImgBbegin = i * 50
            ImgBend = ImgBbegin + 50
            imgV = ne_img_test[PRNUBbegin:PRNUBend, :, :, :]
            PRNUB = ne_img_f[ImgBbegin:ImgBend, :, :, :]

            imgV = Variable(torch.FloatTensor(imgV).cuda())
            PRNUB = Variable(torch.FloatTensor(PRNUB).cuda())
            # print(PRNUB.shape)
            # print(imgV.shape)

            with torch.no_grad():
                weight1 = module_fusion(PRNUB)
                out_img1 = moudle_spncnn(PRNUB)
                ref1 = weight1 * out_img1
                ref_reshaped1 = ref1.view(-1, 50, 1, 128, 128)
                PRNUV = ref_reshaped1.sum(dim=1, keepdim=False)
                EPRNU = moudle_spncnn(imgV)

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
            # print(f"PCE for block {idx}: {PCE}")
        avg_pce = avg_pce / idx
        print("\n pce_fusion_val: %.6f" % (avg_pce))


        OUR_PCE_prnu = {'positive_pce': pos_pce_fusion, 'negative_pce': neg_pce_fusion}
        OUR_prnu_fpr, OUR_prnu_tpr, OUR_prnu_auc = ROC_prefcurve(OUR_PCE_prnu)
        print("\nAUC_fusion: %.6f" % (OUR_prnu_auc))


        if OUR_prnu_auc > max_AUC:
            max_AUC = OUR_prnu_auc
            moudle_name = 'weight_average-large' + '_' + str(epoch + 1) + '_' + str(max_AUC) + '.pth'
            torch.save(module_fusion.state_dict(), os.path.join(opt.outf, moudle_name))
            # torch.save(module_fusion.state_dict(),os.path.join(opt.outf, f'ImageFusion-Net*image-softmax-epoch{epoch + 1}.pth'))


        # # save model
        # model_name = 'Fusion_net +' + '_' + str(epoch + 1) +  str(loss.item()) + '.pth'
        # torch.save(moudle_dncnn.state_dict(), os.path.join(opt.outf, model_name))
    # torch.save(moudle_dncnn.state_dict(), os.path.join(opt.outf, 'Siamese_PRNU.pth'))
    torch.save(module_fusion.state_dict(), os.path.join(opt.outf, 'weight_average-spncnn-prnu.pth'))

if __name__ == "__main__":
    main()
