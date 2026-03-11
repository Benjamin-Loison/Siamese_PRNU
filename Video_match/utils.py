import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


def mloss(x, y):
    muX = torch.mean(x, dim=(2, 3))
    muY = torch.mean(y, dim=(2, 3))

    muXexpand = torch.unsqueeze(torch.unsqueeze(muX, 2), 3)
    muYexpand = torch.unsqueeze(torch.unsqueeze(muY, 2), 3)
    Xzero = x - muXexpand
    Yzero = y - muYexpand

    term1 = torch.sum(Xzero * Yzero, dim=(2, 3))
    term2 = torch.sum(torch.pow(Xzero, 2), dim=(2, 3))
    term3 = torch.sum(torch.pow(Yzero, 2), dim=(2, 3))

    sqrt2 = torch.sqrt(term2)
    sqrt3 = torch.sqrt(term3)
    lossi = 1 - (term1 / (sqrt2 * sqrt3 + 1e-10))

    loss = torch.mean(lossi) + 50 * torch.mean(torch.pow((x-y), 2))
    # loss = torch.mean(lossi)
    return loss

def euclidean_distance(x, y):
    # 计算两个张量之间的欧几里得距离，这里计算第三和第四个维度的距离
    return torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=(2, 3)))

def triplet_loss(anchor, positive, negative, margin=0.085):
    # 计算锚样本与正例样本之间的欧几里得距离
    distance_positive = euclidean_distance(anchor, positive)
    # 计算锚样本与负例样本之间的欧几里得距离
    distance_negative = euclidean_distance(anchor, negative)
    # 计算三元组损失，如果输入的值大于等于 0，则返回原始值；如果输入的值小于 0，则返回 0。
    loss = torch.mean(torch.relu(distance_positive - distance_negative + margin))
    return loss

class DistanceBasedLogisticLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.0):
        super(DistanceBasedLogisticLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, x1, x2, y):
        # 计算样本之间的距离
        distance = torch.norm(x1 - x2, dim=1)
        # 将距离映射到概率
        probability = 1 / (1 + torch.exp(-self.alpha * (distance - self.beta)))
        # 计算损失值
        loss = -y * torch.log(probability) - (1 - y) * torch.log(1 - probability)
        return loss.mean()



def triplet_loss_correlation(anchor, positive, negative, margin=0.08):
    # 计算锚样本与正例样本之间的欧几里得距离
    distance_positive = mloss(anchor, positive)
    # 计算锚样本与负例样本之间的欧几里得距离
    distance_negative = mloss(anchor, negative)
    # 计算三元组损失，如果输入的值大于等于 0，则返回原始值；如果输入的值小于 0，则返回 0。
    loss = torch.mean(torch.relu(  distance_positive -distance_negative    + margin))

    return loss



def ROC_prefcurve(PCE):
    intra_pce = np.array(PCE['positive_pce']).flatten()
    inter_pce = np.array(PCE['negative_pce']).flatten()
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]
    true_lable = np.ones(t1)
    false_lable = np.zeros(t2)

    lable = np.concatenate([true_lable, false_lable])
    value = np.concatenate([intra_pce, inter_pce])

    fpr, tpr, _ = roc_curve(lable, value)
    roc_auc = auc(fpr, tpr)

    return fpr,tpr,roc_auc
