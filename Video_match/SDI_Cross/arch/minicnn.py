from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from thop import profile

from arch.bipooling import PairwiseCorrPooling
from timm.models.layers import trunc_normal_

class MobileOneBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = 0,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1,
                 activation: bool = True) -> None:

        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = padding

        if activation:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

            # Re-parameterizable skip connection
        self.rbr_skip = None
        if out_channels == in_channels and stride == 1 :
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)

        # Re-parameterizable conv branches
        rbr_conv = list()
        for _ in range(self.num_conv_branches):
            rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
        self.rbr_conv = nn.ModuleList(rbr_conv)

        # Re-parameterizable scale branch
        self.rbr_scale1 = None
        self.rbr_scale3 = None
        if kernel_size > 1:
            self.rbr_scale1 = self._conv_bn(kernel_size=1,
                                           padding=0)
            self.rbr_scale3 = self._conv_bn(kernel_size=3,
                                            padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        """ Apply forward pass. """

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale1 is not None:
            scale_out += self.rbr_scale1(x)
        if self.rbr_scale3 is not None:
            scale_out += self.rbr_scale3(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(out)

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale1')
        self.__delattr__('rbr_scale3')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale1 = kernel_scale3 = 0
        bias_scale1 = bias_scale3 = 0
        if self.rbr_scale1 is not None:
            kernel_scale1, bias_scale1 = self._fuse_bn_tensor(self.rbr_scale1)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale1 = torch.nn.functional.pad(kernel_scale1,
                                                   [pad, pad, pad, pad])
        if self.rbr_scale3 is not None:
            kernel_scale3, bias_scale3 = self._fuse_bn_tensor(self.rbr_scale3)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2 - 1
            kernel_scale3 = torch.nn.functional.pad(kernel_scale3,
                                                   [pad, pad, pad, pad])
        kernel_scale = kernel_scale1 + kernel_scale3
        bias_scale = bias_scale1 + bias_scale3


        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list



class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
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
    model = MiniCNN()
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