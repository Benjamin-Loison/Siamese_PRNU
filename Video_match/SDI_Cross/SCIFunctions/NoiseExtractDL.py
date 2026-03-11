import numpy as np
import torch
from torch.autograd import Variable
# user defined
from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.WienerInDFT import WienerInDFT
import time


def NoiseExtractDL(imorig, model, postprocess, swin=False):

    Img = np.float32(imorig)/255.
    Img = np.expand_dims(Img, 0)
    Img = np.expand_dims(Img, 1)

    # the probe image is used as noisy image
    INoisy = torch.Tensor(Img)
    INoisy = Variable(INoisy)
    INoisy = INoisy.cuda()

    with torch.no_grad(): # this can save much memory
        if swin:
            _, _, h_old, w_old = INoisy.size()
            h_pad = (h_old // 8 + 1) * 8 - h_old
            w_pad = (w_old // 8 + 1) * 8 - w_old
            INoisy = torch.cat([INoisy, torch.flip(INoisy, [2])], 2)[:, :, :h_old + h_pad, :]
            INoisy = torch.cat([INoisy, torch.flip(INoisy, [3])], 3)[:, :, :, :w_old + w_pad]
        Out = model(INoisy)
        if swin:
            Out = Out[..., :h_old , :w_old ]
        # since we changed the output from denoised image into residual

    Noisex = Out.cpu().numpy().squeeze()
    Noisexnp = Noisex
    stdValue = np.std(Noisex)
    # remove NUA, we find it is only needed in the RP side
    if postprocess:
        Noisex = ZeroMeanTotal(Noisex)
        std = np.std(Noisex)
        Noisex = WienerInDFT(Noisex, std)

    return Noisex, Noisexnp, stdValue