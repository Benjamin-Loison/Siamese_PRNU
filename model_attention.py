import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.nn import init

##提参考网络测试

class ECA_Block(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Block, self).__init__()
        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)
        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v

# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)



##提参考网络最终版本
class Weight_Get_conv7_se(nn.Module):
    def __init__(self):
        super(Weight_Get_conv7_se, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        #ECA模块
        # self.CAM = ECA_Block(64)
        self.CAM = SE_Block(64)

        self.conv7 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(64, 1, kernel_size=1, padding=0)#使用1*1卷积融合特征


        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            batch_output = torch.relu(self.conv3(batch_output))
            batch_output = torch.relu(self.conv4(batch_output))
            batch_output = torch.relu(self.conv5(batch_output))
            batch_output = torch.relu(self.conv6(batch_output))

            se_output = self.CAM(batch_output)
            batch_output = self.conv7(se_output)  # 最后一层生成权重
            batch_output = F.softmax(batch_output, dim=0)  # 归一化权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output

class Weight_Get_convx_se(nn.Module):
    def __init__(self):
        super(Weight_Get_convx_se, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)


        # self.CAM = ECA_Block(64)
        self.CAM = SE_Block(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.CAM_1 = SE_Block(96)
        self.conv13 = nn.Conv2d(96, 1, kernel_size=1, padding=0)

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

        # # 初始化权重
        # nn.init.kaiming_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.kaiming_normal_(self.conv3.weight)
        # nn.init.kaiming_normal_(self.conv4.weight)
        # nn.init.kaiming_normal_(self.conv5.weight)
        # nn.init.kaiming_normal_(self.conv6.weight)
        # nn.init.xavier_uniform_(self.conv7.weight)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]

            identity = batch_input

            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            batch_output = torch.relu(self.conv3(batch_output))
            batch_output = torch.relu(self.conv4(batch_output))
            batch_output = torch.relu(self.conv5(batch_output))
            batch_output = torch.relu(self.conv6(batch_output))
            se_output = self.CAM(batch_output)

            batch_output = torch.relu(self.conv7(se_output))
            batch_output = torch.relu(self.conv8(batch_output))
            batch_output = torch.relu(self.conv9(batch_output))
            batch_output = torch.relu(self.conv10(batch_output))
            batch_output = torch.relu(self.conv11(batch_output))
            batch_output = torch.relu(self.conv12(batch_output))
            se_output1 = self.CAM_1(batch_output)
            batch_output = torch.relu(self.conv13(se_output1))

            batch_output = batch_output + identity

            batch_output = F.softmax(batch_output, dim=0)  # 得到每张图片的权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output


if __name__ == '__main__':
    print(Weight_Get_convx_se())
    input = torch.randn(1000, 1, 40, 40)  # 随机生成一个输入特征图
    output = Weight_Get_convx_se(input)
    print(output.shape)