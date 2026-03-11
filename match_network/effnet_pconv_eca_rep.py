# structural re-parameterization technique
import copy
from functools import partial

from torch import Tensor
import math
from thop import profile,clever_format
import torch
import torch.nn as nn
from match_network.layers import Flatten
from match_network.layers import ECABlock
from match_network.layers import MobileOneBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import List


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, kernel_size, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.kernel_size = kernel_size
        self.dim_untouched = dim - self.dim_conv3
        padding = (kernel_size - 1) // 2
        # self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, stride=1, padding=padding, bias=False)
        self.partial_conv3 = MobileOneBlock(self.dim_conv3, self.dim_conv3, kernel_size, stride=1,activation=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 kernel_size,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.kernel_size = kernel_size

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = nn.ModuleList([
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ])

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            kernel_size,
            pconv_fw_type
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x



class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = MobileOneBlock(in_, mid_, kernel_size=1) if expand != 1 else nn.Identity()

        self.depth_wise_conv = MobileOneBlock(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride, groups=mid_)

        self.eca = ECABlock(mid_)

        self.project_conv = MobileOneBlock(mid_, out_, kernel_size=1, activation=False)

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        # self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.eca(x)
        x = self.project_conv(x)
        if self.skip:
            # x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, drop_connect_ratio)]
        for i in range(1, num_repeat):
            # layers.append(MBConv(out_, out_, expand, kernel, 1, skip, drop_connect_ratio))
            layers.append(MLPBlock(dim=out_,n_div=4,mlp_ratio=2,drop_path=0.,layer_scale_init_value=0,kernel_size = kernel,
                                   act_layer=partial(nn.ReLU, inplace=True),norm_layer=nn.BatchNorm2d,pconv_fw_type='split_cat'))#  split_cat     slicing
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet_pconv_eca_rep(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=2):
        super().__init__()
        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        # self.stem = conv_bn_act(2, renew_ch(32), kernel_size=3, stride=2, bias=False)
        self.stem = MobileOneBlock(1, renew_ch(32), kernel_size=3, stride=2)

        self.blocks = nn.Sequential(
            # #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, drop_connect_rate),
            MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, drop_connect_rate),
            MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, drop_connect_rate),
            MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, drop_connect_rate),
            MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, drop_connect_rate)
        )

        self.head = nn.Sequential(
            MobileOneBlock(renew_ch(320), renew_ch(1280), kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            Flatten(),
            nn.Linear(renew_ch(1280), num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        head = self.head(x)
        return head

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


if __name__ == "__main__":
    print("Efficient B0 Summary")
    net = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    input = torch.randn(1, 1, 256, 256)
    net = reparameterize_model(net)
    flops, params = profile(net, inputs=(input,))

    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)