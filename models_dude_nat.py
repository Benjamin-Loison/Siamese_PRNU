import torch
import torch.nn as nn
from deform_conv_v2 import DeformConv2d
from natten import NeighborhoodAttention2D as NA
from timm.models.layers import DropPath
import torch.nn.functional as F
import math


##此代码包括提测试的网络dfprnu-net

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7, dilation=1, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = kernel_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NA(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)


    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 96
        groups = 1
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(
            DeformConv2d(inc=channels, outc=features, kernel_size=kernel_size, padding=padding, bias=False, modulation=True), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            NATBlock(dim=features,
                     num_heads=6,
                     kernel_size=7,
                     dilation=2,
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop=0.0,
                     attn_drop=0.0,
                     drop_path=0.0,
                     norm_layer=nn.LayerNorm)
            ,nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            NATBlock(dim=features,
                     num_heads=6,
                     kernel_size=7,
                     dilation=2,
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop=0.0,
                     attn_drop=0.0,
                     drop_path=0.0,
                     norm_layer=nn.LayerNorm)
            ,nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            NATBlock(dim=features,
                     num_heads=6,
                     kernel_size=7,
                     dilation=2,
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop=0.0,
                     attn_drop=0.0,
                     drop_path=0.0,
                     norm_layer=nn.LayerNorm)
            ,nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=groups,
                                  bias=False)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(2 * features)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.ReLU(inplace=True))
        self.conv2_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.ReLU(inplace=True))
        self.conv2_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.ReLU(inplace=True))
        self.conv2_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0, groups=groups,
                                  bias=False)
        self.conv3_1 = nn.Conv2d(in_channels=2 * features, out_channels=1, kernel_size=1, padding=0, groups=groups,
                                 bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0, groups=groups, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        # print x1.size()
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.conv2_5(x2)
        x2 = self.conv2_6(x2)
        x2 = self.conv2_7(x2)
        x2 = self.conv2_8(x2)
        x2 = self.conv2_9(x2)
        x2 = self.conv2_10(x2)
        x2 = self.conv2_11(x2)
        x2 = self.conv2_12(x2)
        x2 = self.conv2_13(x2)
        x2 = self.conv2_14(x2)
        x2 = self.conv2_15(x2)
        x2 = self.conv2_16(x2)
        # print x2.size()
        x3 = torch.cat([x1, x2], 1)
        x3 = self.BN(x3)
        x3 = self.ReLU(x3)
        x3 = self.conv3_1(x3)
        x4 = torch.cat([x, x3], 1)
        x4 = self.conv3_2(x4)

        return x4


#权重矩阵
# class ImageFusionNet(nn.Module):
#     def __init__(self):
#         super(ImageFusionNet, self).__init__()
#         # 定义一个可学习的权重参数
#         self.weights = nn.Parameter(torch.randn(1, 1, 1, 1), requires_grad=True)
#
#     def forward(self, input_images):
#         # 输入数据形状：(num, 1, 40, 40)
#
#         # 使用可学习的权重对每张输入图像进行加权合成
#         fused_image = torch.sum(input_images * self.weights, dim=0)
#
#         # 输出图像形状：(1, 1, 40, 40)
#         return fused_image
#


# class ImageFusionNet(nn.Module):
#     def __init__(self):
#         super(ImageFusionNet, self).__init__()
#         # 定义多个卷积层
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
#     def forward(self, x):
#         # 获取输入数据的形状
#         num, channels, height, width = x.shape
#         # x = x.transpose(0, 1)
#         # 初始化结果张量
#         results = []
#         for i in range(0, num, 50):
#             batch_input  = x[i:i+50]  # 取出每批次的50张图片
#             batch_output = torch.relu(self.conv1(batch_input))
#             batch_output = torch.relu(self.conv2(batch_output))
#             batch_output = torch.relu(self.conv3(batch_output))
#             batch_output = batch_input*batch_output
#             # 计算每批次的平均值并添加到结果列表中
#             batch_avg = torch.mean(batch_output, dim=0, keepdim=True)
#             results.append(batch_avg)
#         # 拼接结果列表，得到最终输出
#         output = torch.cat(results, dim=0)
#         return output
#
#将前面的图片进行加权
class ImageFusionNet(nn.Module):
    def __init__(self):
        super(ImageFusionNet, self).__init__()
        # 定义多个卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 获取输入数据的形状
        num, channels, height, width = x.shape
        # 初始化结果张量
        results = []
        for i in range(0, num, 50):
            batch_input  = x[i:i+50]  # 取出每批次的50张图片
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            # batch_output = torch.relu(self.conv3(batch_output))
            batch_output = self.conv3(batch_output)
            # 归一化权重
            # batch_output = F.normalize(batch_output, p=1, dim=0)  # 归一化通道维度
            batch_output = F.softmax(batch_output, dim=0)
            # print(batch_output)
            batch_output = batch_input * batch_output
            batch_sum = torch.sum(batch_output, dim=0, keepdim=True)
            results.append(batch_sum)
        # 拼接结果列表，得到最终输出
        output = torch.cat(results, dim=0)
        return output

#根据输入图片的质量，来对使用输入图像提取的噪声残差进行加权
class Weight_Get(nn.Module):
    def __init__(self):
        super(Weight_Get, self).__init__()
        # 定义多个卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # 初始化网络权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, x):
        # 获取输入数据的形状
        num, channels, height, width = x.shape
        # 初始化结果张量
        results = []
        for i in range(0, num, 50):
            batch_input  = x[i:i+50]  # 取出每批次的50张图片
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            # batch_output = torch.relu(self.conv3(batch_output))
            batch_output = self.conv3(batch_output)
            # 生成每张图片的权重
            batch_output = F.softmax(batch_output, dim=0)
            results.append(batch_output)
        # 拼接结果列表，得到最终输出
        output = torch.cat(results, dim=0)
        return output


class Weight_Get_conv4(nn.Module):
    def __init__(self):
        super(Weight_Get_conv4, self).__init__()
        # 定义多个卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # 初始化网络权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, x):
        # 获取输入数据的形状
        num, channels, height, width = x.shape
        # 初始化结果张量
        results = []
        for i in range(0, num, 50):
            batch_input  = x[i:i+50]  # 取出每批次的50张图片
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            batch_output = torch.relu(self.conv3(batch_output))
            # batch_output = torch.relu(self.conv3(batch_output))
            batch_output = self.conv4(batch_output)
            # 生成每张图片的权重
            batch_output = F.softmax(batch_output, dim=0)
            results.append(batch_output)
        # 拼接结果列表，得到最终输出
        output = torch.cat(results, dim=0)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class Weight_Get_conv5(nn.Module):
    def __init__(self):
        super(Weight_Get_conv5, self).__init__()
        # 初始卷积层保持不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 添加一个新的卷积层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))  # 新添加的层
            batch_output = torch.relu(self.conv3(batch_output))
            batch_output = torch.relu(self.conv4(batch_output))
            batch_output = self.conv5(batch_output)  # 最后一层生成权重
            batch_output = F.softmax(batch_output, dim=0)  # 归一化权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output

class Weight_Get_conv5_batch(nn.Module):
    def __init__(self):
        super(Weight_Get_conv5_batch, self).__init__()
        # 初始卷积层保持不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]
            result1 = []
            # 进一步将50张图像分批次处理
            for j in range(batch_input.size(0)):
                sub_batch_input = batch_input[j:j + 1]
                # 通过卷积层
                sub_batch_output = torch.relu(self.conv1(sub_batch_input))
                sub_batch_output = torch.relu(self.conv2(sub_batch_output))
                sub_batch_output = torch.relu(self.conv3(sub_batch_output))
                sub_batch_output = torch.relu(self.conv4(sub_batch_output))
                sub_batch_output = self.conv5(sub_batch_output)
                # sub_batch_output = sub_batch_output.view(1, 128, 128)
                sub_batch_output = sub_batch_output.squeeze(0)  # 移除单维度
                # print(f"sub_batch_output shape: {sub_batch_output.shape}")
                result1.append(sub_batch_output)
            result1_stacked = torch.stack(result1)
            # print(f"result1_stacked shape: {result1_stacked.shape}")
            batch_output = F.softmax(result1_stacked, dim=0)  # 归一化权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output


class Weight_Get_conv5_bn(nn.Module):
    def __init__(self):
        super(Weight_Get_conv5_bn, self).__init__()
        # 初始卷积层保持不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]
            batch_output = F.relu(self.bn1(self.conv1(batch_input)))
            batch_output = F.relu(self.bn2(self.conv2(batch_output)))
            batch_output = F.relu(self.bn3(self.conv3(batch_output)))
            batch_output = F.relu(self.bn4(self.conv4(batch_output)))
            batch_output = self.conv5(batch_output)  # 最后一层生成权重
            batch_output = F.softmax(batch_output, dim=0)  # 归一化权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output



class FusionNet_BN(nn.Module):
    def __init__(self):
        super(FusionNet_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input  = x[i:i+50]
            batch_output = torch.relu(self.bn1(self.conv1(batch_input)))
            batch_output = torch.relu(self.bn2(self.conv2(batch_output)))
            batch_output = self.conv3(batch_output)
            batch_output = F.softmax(batch_output, dim=0)
            batch_output = batch_input * batch_output
            batch_sum = torch.sum(batch_output, dim=0, keepdim=True)
            results.append(batch_sum)
        output = torch.cat(results, dim=0)
        return output


class Weight_Get_BN(nn.Module):
    def __init__(self):
        super(Weight_Get_BN, self).__init__()
        # 定义多个卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    def forward(self, x):
        # 获取输入数据的形状
        num, channels, height, width = x.shape
        # 初始化结果张量
        results = []
        for i in range(0, num, 50):
            batch_input  = x[i:i+50]  # 取出每批次的50张图片
            batch_output = torch.relu(self.bn1(self.conv1(batch_input)))#BN一般位于激活函数之前
            batch_output = torch.relu(self.bn2(self.conv2(batch_output)))
            batch_output = torch.relu(self.conv3(batch_output))
            # batch_output = self.conv3(batch_output)
            # 归一化权重
            # batch_output = F.normalize(batch_output, p=1, dim=0)  # 归一化通道维度
            batch_output = F.softmax(batch_output, dim=0)
            results.append(batch_output)
        # 拼接结果列表，得到最终输出
        output = torch.cat(results, dim=0)
        return output

'''-------------一、SE模块-----------------------------'''
from torch import nn

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


class Weight_Get_conv5_SENET(nn.Module):
    def __init__(self):
        super(Weight_Get_conv5_SENET, self).__init__()
        # 初始卷积层保持不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 添加一个新的卷积层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #SENET模块
        self.SE = SE_Block(64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        num, channels, height, width = x.shape
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            batch_output = torch.relu(self.conv3(batch_output))
            batch_output = torch.relu(self.conv4(batch_output))
            se_output = self.SE(batch_output)
            # batch_output = batch_output*se_output
            batch_output = self.conv5(se_output)  # 最后一层生成权重
            batch_output = F.softmax(batch_output, dim=0)  # 归一化权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output




if __name__ == '__main__':
    model = DnCNN(channels=1)
    print(model)