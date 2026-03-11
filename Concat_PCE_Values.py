from utils import *
import os
##合并四个位置不同的PCE数值

# # #DWT-nowiener
# data1 = sio.loadmat('PCE_SAVE_Dresden_Surround_Conventional/Dresden_Conventional-nowiener-center_128_pce_values_lower-left1.mat')
# data2 = sio.loadmat('PCE_SAVE_Dresden_Surround_Conventional/Dresden_Conventional-nowiener-center_128_pce_values_lower-right1.mat')
# data3 = sio.loadmat('PCE_SAVE_Dresden_Surround_Conventional/Dresden_Conventional-nowiener-center_128_pce_values_upper-left2.mat')
# data4 = sio.loadmat('PCE_Save_Dresden/Dresden_Conventional-no-wiener-center_128_pce_values.mat')

# #spn-nowiener
# data1 = sio.loadmat('PCE_SAVE_Dresden_Surround_SPNCNN/Dresden_SPNCNN-nowiener-center_128_pce_values_lower-left1.mat')
# data2 = sio.loadmat('PCE_SAVE_Dresden_Surround_SPNCNN/Dresden_SPNCNN-nowiener-center_128_pce_values_lower-right1.mat')
# data3 = sio.loadmat('PCE_SAVE_Dresden_Surround_SPNCNN/Dresden_SPNCNN-nowiener-center_128_pce_values_upper-left2.mat')
# data4 = sio.loadmat('PCE_Save_Dresden/Dresden_SPNCNN-no-wiener-center_128_pce_values.mat')

# # dude-nowiener
# data1 = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_PRNU-nowiener-center_128_pce_values_lower-right1.mat')
# data2 = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_PRNU-nowiener-center_128_pce_values_lower-left1.mat')
# data3 = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_PRNU-nowiener-center_128_pce_values_upper-left2.mat')
# data4 = sio.loadmat('PCE_Save_Dresden/Dresden_PRNU-no-wiener-center_128_pce_values.mat')

# # #our-nowiener
data1 = sio.loadmat('PCE_SAVE_Dresden_Surround_conv7+se/Dresden_FUSION-nat-nowiener-center-64_pce_values_lower-right.mat')
data2 = sio.loadmat('PCE_SAVE_Dresden_Surround_conv7+se/Dresden_FUSION-nat-nowiener-center-64_pce_values_upper-left.mat')
data3 = sio.loadmat('PCE_SAVE_Dresden_Surround_conv7+se/Dresden_FUSION-nat-nowiener-center-64_pce_values_lower-left.mat')
data4 = sio.loadmat('PCE_SAVE_Dresden_Surround_conv7+se/Dresden_FUSION-nat-nowiener-center-64_pce_values_upper-right.mat')



# fusion_rep
data1 = sio.loadmat('SCORE_SAVE_Dresden_REP/Dresden_FUSION_64_REP_scores_upper_right.mat')
data2 = sio.loadmat('SCORE_SAVE_Dresden_REP/Dresden_FUSION_64_REP_scores_lower_left.mat')
data3 = sio.loadmat('SCORE_SAVE_Dresden_REP/Dresden_FUSION_64_REP_scores_upper_left.mat')
data4 = sio.loadmat('SCORE_SAVE_Dresden_REP/Dresden_FUSION_64_REP_scores_lower_right.mat')




# 获取变量数据
pos_pce_values_1 = data1['positive_pce']
neg_pce_values_1 = data1['negative_pce']
pos_pce_values_2 = data2['positive_pce']
neg_pce_values_2 = data2['negative_pce']
pos_pce_values_3 = data3['positive_pce']
neg_pce_values_3 = data3['negative_pce']
pos_pce_values_4 = data4['positive_pce']
neg_pce_values_4 = data4['negative_pce']

# 将数据合并在一起
pos_pce_values_combined = np.concatenate((pos_pce_values_1, pos_pce_values_2, pos_pce_values_3,pos_pce_values_4), axis=1)
neg_pce_values_combined = np.concatenate((neg_pce_values_1, neg_pce_values_2, neg_pce_values_3,neg_pce_values_4), axis=1)

# 构建保存路径
# # 创建 "pce文件夹" 文件夹，如果它不存在
save_folder = "SCORE_SAVE_Dresden_REP"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_filename = 'Dresden_FUSION_64_REP_scores_all.mat'  # 文件名
save_path = os.path.join(save_folder, save_filename)
# 保存合并后的数据为新的MATLAB文件
combined_data = {
    'positive_pce': pos_pce_values_combined,
    'negative_pce': neg_pce_values_combined
}
sio.savemat(save_path, combined_data)

OUR_PCE = {'positive_pce':combined_data['positive_pce'], 'negative_pce': combined_data['negative_pce']}
OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(OUR_PCE)
print("AUC:", OUR_auc)