import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve, confusion_matrix


##此代码用于kappa系数计算
def Confusion_Matrix(PCE):
    # 提取正负样本的PCE值
    intra_pce = np.array(PCE['positive_pce']).flatten()
    inter_pce = np.array(PCE['negative_pce']).flatten()

    # 计算总样本数
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]

    # 构建真实标签和预测值的数组
    true_label = np.concatenate([np.ones(t1), np.zeros(t2)])  # 1 表示正样本，0 表示负样本
    pce_scores = np.concatenate([intra_pce, inter_pce])

    # 对PCE值进行排序并计算ROC曲线
    fpr, tpr, thresholds = roc_curve(true_label, pce_scores)

    # 通过ROC曲线找到最佳阈值
    acc = (tpr + (1 - fpr)) / 2
    index_max = np.argmax(acc)
    threshold = thresholds[index_max]

    # 应用阈值，得到预测标签
    pred_label = (pce_scores >= threshold).astype(int)

    # 计算混淆矩阵
    conf_mat = confusion_matrix(true_label, pred_label)

    # 计算观察一致性（P_o）
    Po = np.trace(conf_mat) / np.sum(conf_mat)

    # 计算预期一致性（P_e）
    Pe = np.sum(np.sum(conf_mat, axis=0) * np.sum(conf_mat, axis=1)) / np.sum(conf_mat) ** 2

    # 计算Kappa系数
    kappa = (Po - Pe) / (1 - Pe)

    return kappa

# 示例用法
if __name__ == '__main__':
    ADNet_PCE = sio.loadmat('PCE_SAVE_Dresden_64/Dresden_FUSION-nat-no-wiener-center-64-all_conv7+se.mat')
    kappa = Confusion_Matrix(ADNet_PCE)
    print("Kappa Coefficient:", kappa)
