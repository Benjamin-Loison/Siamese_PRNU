import h5py
import numpy as np

data = h5py.File('/home/seamus20/matlab_wyl/Train_fusion_dude/VISION_fusion-train_40_50_5429000.mat', 'r')#
PRNU = np.array(data['IMG'])
img = np.array(data['inputs'])
print(PRNU.shape)
print(img.shape)