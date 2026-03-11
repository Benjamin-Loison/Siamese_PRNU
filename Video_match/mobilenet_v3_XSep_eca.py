import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from thop import profile, clever_format

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ECA_Block(nn.Module):#使用ECA模块来替代SE模块
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Block, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


# DWConv
class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


# XSepConv
class XSepBlock(nn.Module):
    ''' 1*1 expand + Improved Symmetric Padding + 2*2 DW + 1*K DW + k*1 DW + 1*1 out'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, ith):
        super(XSepBlock, self).__init__()
        self.stride = stride
        self.ith = ith

        # 1*1 expand
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # 2*2 DW
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=2, stride=1, padding=0, groups=expand_size,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        # 1*k DW
        self.conv3 = nn.Conv2d(expand_size, expand_size, kernel_size=(1, kernel_size), stride=stride,
                               padding=(0, kernel_size // 2), groups=expand_size, bias=False)
        self.bn3 = nn.BatchNorm2d(expand_size)
        self.nolinear3 = nolinear
        # k*1 DW
        self.conv4 = nn.Conv2d(expand_size, expand_size, kernel_size=(kernel_size, 1), stride=stride,
                               padding=(kernel_size // 2, 0), groups = expand_size, bias = False)
        self.bn4 = nn.BatchNorm2d(expand_size)
        # SE
        self.se = semodule
        self.nolinear_se = nolinear
        # 1*1 out
        self.conv5 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        x1 = x
        if self.ith % 4 == 0:
            x1 = nn.functional.pad(x1, (0, 1, 1, 0), mode="constant", value=0)  # right top
        elif self.ith % 4 == 1:
            x1 = nn.functional.pad(x1, (0, 1, 0, 1), mode="constant", value=0)  # right bottom
        elif self.ith % 4 == 2:
            x1 = nn.functional.pad(x1, (1, 0, 1, 0), mode="constant", value=0)  # left top
        elif self.ith % 4 == 3:
            x1 = nn.functional.pad(x1, (1, 0, 0, 1), mode="constant", value=0)  # left bottom
        else:
            raise NotImplementedError('ith layer is not right')

        # 1*1 expand
        out = self.nolinear1(self.bn1(self.conv1(x1)))
        # 2*2 DW
        out = self.nolinear2(self.bn2(self.conv2(out)))
        # 1*k DW
        out = self.nolinear3(self.bn3(self.conv3(out)))
        # k*1 DW
        out = self.bn4(self.conv4(out))
        # SE
        if self.se != None:
            out = self.nolinear_se(self.se(out))
        # 1*1 out
        out = self.bn5(self.conv5(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


# 按照论文设置改的，可能会有出入
class XSepMobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=2):
        super(XSepMobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),  # 下采样层使用DW，因为kernel_size < 7
            XSepBlock(5, 40, 240, 40, hswish(), SeModule(40), 1, 1),
            XSepBlock(5, 40, 240, 40, hswish(), SeModule(40), 1, 2),
            XSepBlock(5, 40, 120, 48, hswish(), SeModule(48), 1, 3),
            XSepBlock(5, 48, 144, 48, hswish(), SeModule(48), 1, 4),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),  # 下采样层使用DW，因为kernel_size < 7
            XSepBlock(3, 96, 576, 96, hswish(), SeModule(96), 1, 1),
            XSepBlock(3, 96, 576, 96, hswish(), SeModule(96), 1, 2),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


def test():
    net = XSepMobileNetV3_Small()
    input = torch.randn(1, 1, 256, 256)
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
