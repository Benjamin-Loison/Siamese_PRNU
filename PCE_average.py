import numpy as np
import scipy.io as sio
from utils import *
##此代码用于加载存放PCE的mat文件，并进行显示

# 加载.mat文件
# data = sio.loadmat('PCE_Save_Dresden/Dresden_FUSION-nat-no-wiener-center_128_pce_values.mat')
# data = sio.loadmat('PCE_SAVE_Dresden_Surround_conv7+se/Dresden_PRNU-nowiener-center_128_pce_values_lower-left1.mat')

data = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_PRNU-nowiener-center_128_pce_values_upper-left2.mat')
# data = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_FUSION-nat-wiener-center-128_pce_values_upper-left.mat')

# data = sio.loadmat('/home/seamus20/z_Project/Siamese_PRNU/PCE_SAVE_Dresden_Surround_conv7+se/Dresden_FUSION-nat-nowiener-center-128_pce_values_upper-left2.mat')
# data = sio.loadmat('PCE_SAVE_Dresden_Surround_conv7+se/Dresden_FUSION-nat-nowiener-center-160_pce_values_lower-left.mat')



OUR_PCE = {'positive_pce':data['positive_pce'], 'negative_pce': data['negative_pce']}
OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
print("AUC:", OUR_auc)

# 提取正负样本的pce值
positive_pce = data['positive_pce']
negative_pce = data['negative_pce']

# 计算均值
positive_mean = np.mean(positive_pce)
negative_mean = np.mean(negative_pce)

print("正样本的均值：", positive_mean)
print("负样本的均值：", negative_mean)



