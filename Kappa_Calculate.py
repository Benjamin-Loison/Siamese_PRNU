import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def Confusion_Matrix(PCE):
    intra_pce = np.array(PCE['positive_pce']).flatten()
    inter_pce = np.array(PCE['negative_pce']).flatten()
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]
    true_lable = np.ones(t1)
    false_lable = np.zeros(t2)

    lable = np.concatenate([true_lable, false_lable])
    value = np.concatenate([intra_pce, inter_pce])

    fpr, tpr, thresholds = roc_curve(lable, value)
    acc = (tpr + (1 - fpr)) / 2
    index_max = np.argmax(acc)
    threshold = thresholds[index_max]

    y_true = np.array([0] * len(inter_pce) + [1] * len(intra_pce)) # 真实标签
    pce_scores = np.concatenate((inter_pce, intra_pce)) # 清洗数据

    # 应用阈值，得到预测
    y_pred = (pce_scores >= threshold).astype(int)

    # 计算混淆矩阵
    conf_mat = confusion_matrix(y_true, y_pred)
    # 计算观察一致性（P_o）
    Po = np.trace(conf_mat) / float(np.sum(conf_mat))
    # 计算预期一致性（P_e）
    Pe = np.sum(np.sum(conf_mat, axis=0) * np.sum(conf_mat, axis=1)) / float(np.sum(conf_mat) ** 2)
    # 计算Kappa系数
    kappa = (Po - Pe) / (1 - Pe)
    print("Kappa Coefficient: ", kappa)

    # 绘制混淆矩阵
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Predict Negative', 'Predict Positive'],
    yticklabels=['Actual Negative', 'Actual Positive'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    ADNet_PCE = sio.loadmat('PCE_SAVE_Dresden_128/DWT-no-wiener-center-128-all2222.mat')
    Confusion_Matrix(ADNet_PCE)