import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class QualityWeightNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 编码路径提取质量特征
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.se1 = SEBlock(64)  # 注入SE模块
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.se2 = SEBlock(128)
        # 解码路径生成权重分布
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        num, channels, height, width = x.shape
        # 初始化结果张量
        results = []
        for i in range(0, num, 50):
            batch_input = x[i:i + 50]  # 取出每批次的50张图片
            batch_output = F.relu(self.conv1(batch_input))
            batch_output = self.se1(batch_output)
            batch_output = self.pool1(batch_output)
            batch_output = F.relu(self.conv2(batch_output))
            batch_output = self.se2(batch_output)
            batch_output = F.relu(self.deconv1(batch_output))
            batch_output = self.conv3(batch_output)
            batch_output = F.softmax(batch_output, dim=0)
            results.append(batch_output)
            # 拼接结果列表，得到最终输出
        output = torch.cat(results, dim=0)
        return output


