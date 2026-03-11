import h5py
import numpy as np
from utils import *

# data = h5py.File('data/VISIONdemotriplet_40_64.mat', 'r')
# ne_inputs = np.array(data['ne_inputs'])
# print(ne_inputs.shape)
# inputs = np.array(data['inputs'])
# print(inputs.shape)
# PRNU = np.array(data['PRNU'])
# print(PRNU.shape)

import torch

def mloss1(x, y):
    muX = torch.mean(x, dim=(0, 1))
    muY = torch.mean(y, dim=(0, 1))

    muXexpand = torch.unsqueeze(torch.unsqueeze(muX, 0), 1)
    muYexpand = torch.unsqueeze(torch.unsqueeze(muY, 0), 1)
    Xzero = x - muXexpand
    Yzero = y - muYexpand

    term1 = torch.sum(Xzero * Yzero, dim=(0, 1))
    term2 = torch.sum(torch.pow(Xzero, 2), dim=(0, 1))
    term3 = torch.sum(torch.pow(Yzero, 2), dim=(0, 1))

    sqrt2 = torch.sqrt(term2)
    sqrt3 = torch.sqrt(term3)
    lossi = 1 - (term1 / (sqrt2 * sqrt3 + 1e-10))

    # loss = torch.mean(lossi) + 50 * torch.mean(torch.pow((x-y), 2))
    loss = torch.mean(lossi)
    return loss

a = np.array([[0.2, 0.2],
              [-0.2, -0.2]])

b = np.array([[1, 1.2],
              [-1.1, -0.9]])

c = np.array([[6, 1],
              [-3, -2]])

# Convert numpy arrays to torch tensors and set the data type to float
a_tensor = torch.tensor(a).float()
b_tensor = torch.tensor(b).float()
c_tensor = torch.tensor(c).float()

# Calculate the losses
loss_ab = mloss1(a_tensor, b_tensor)
loss_ac = mloss1(a_tensor, c_tensor)

print("Loss between a and b:", loss_ab)
print("Loss between a and c:", loss_ac)
