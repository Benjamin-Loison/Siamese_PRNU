import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class Weight_Get_conv7(nn.Module):
    def __init__(self):
        super(Weight_Get_conv7, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(96, 1, kernel_size=3, padding=1)


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
            batch_output = self.conv7(batch_output)  # 最后一层生成权重
            batch_output = F.softmax(batch_output, dim=0)  # 归一化权重
            results.append(batch_output)
        output = torch.cat(results, dim=0)
        return output