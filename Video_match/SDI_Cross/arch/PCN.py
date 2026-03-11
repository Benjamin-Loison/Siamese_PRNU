import numpy as np
import torch
import torch.nn as nn
from thop import profile

from SDI_Cross.arch.bipooling import PairwiseCorrPooling
from timm.models.layers import trunc_normal_

class Pcn(nn.Module):
    def __init__(self):
        super(Pcn, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.MaxPool2d(kernel_size=3))

        layers.append(nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=0))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.MaxPool2d(kernel_size=3))

        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=0))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.MaxPool2d(kernel_size=3))

        layers.append(PairwiseCorrPooling())

        layers.append(nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, padding=0))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Softmax(dim=1))     #####

        self.pcn = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.pcn(x)
        return out

if __name__ == '__main__':
    model = Pcn()
    x = torch.randn(3,1,256,256)
    flops, params = profile(model, inputs=(x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


    # y = model(x)
    # criterion = nn.CrossEntropyLoss()
    # target = torch.tensor([[0., 1.],
    #                        [1., 0.],
    #                        [0., 1.]])
    # print(y)
    # device_pred = [np.argmax(item, axis=0) for item in y.detach().numpy()]
    # device_true = [np.argmax(item, axis=0) for item in target.detach().numpy()]
    # print(device_pred)
    # print(device_true)
    #
    # accuracy, precision, recall, f1_score, kappa = evaluation(device_pred, device_true)
    # print(accuracy, precision, recall, f1_score, kappa)