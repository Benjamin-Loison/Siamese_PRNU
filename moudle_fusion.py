import torch
import torch.nn as nn
import torch.nn.functional as F
class ImageFusionNet(nn.Module):
    def __init__(self):
        super(ImageFusionNet, self).__init__()
        # 定义多个卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # 获取输入数据的形状
        num, channels, height, width = x.shape

        # 初始化结果张量
        results = []
        for i in range(0, num, 50):
            batch_input  = x[i:i+50]  # 取出每批次的50张图片
            batch_output = torch.relu(self.conv1(batch_input))
            batch_output = torch.relu(self.conv2(batch_output))
            batch_output = self.conv3(batch_output)

            # 归一化权重
            # batch_output = F.normalize(batch_output, p=1, dim=0)  # 归一化通道维度
            batch_output = F.softmax(batch_output, dim=0)
            batch_output = batch_input * batch_output
            batch_sum = torch.sum(batch_output, dim=0, keepdim=True)
            results.append(batch_sum)
        # 拼接结果列表，得到最终输出
        output = torch.cat(results, dim=0)
        return output