import torch.nn as nn


class PairwiseCorrPooling(nn.Module):

    def __init__(self):
        super(PairwiseCorrPooling, self).__init__()

    def forward(self, x):
        halfdim3 = x.shape[1] // 2
        # print(x.get_shape(), x.dtype)
        y = x[:, :halfdim3, :, :] * x[:, halfdim3:, :, :]
        # print(y.get_shape(), y.dtype)
        return y