import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

def calculate_threshold(intra_pce, inter_pce):
    """
    计算最优阈值，使得准确率最高
    :param intra_pce: 正样本的PCE得分
    :param inter_pce: 负样本的PCE得分
    :return: 最优阈值
    """
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]
    true_labels = np.ones(t1)
    false_labels = np.zeros(t2)

    labels = np.concatenate([true_labels, false_labels])
    values = np.concatenate([intra_pce, inter_pce])

    fpr, tpr, thresholds = roc_curve(labels, values)
    acc = (tpr + (1 - fpr)) / 2
    index_max = np.argmax(acc)
    return thresholds[index_max]


def plot_confusion_matrix(y_true, y_pred ,save_path='confusion_matrix_DWT.svg'):
    """
    绘制混淆矩阵的热力图
    :param y_true: 真实标签
    :param y_pred: 预测标签
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测反例', '预测正例'],
                yticklabels=['真实反例', '真实正例'])
    plt.ylabel('真实值')
    plt.xlabel('预测值')
    plt.title('混淆矩阵')

    # 保存为 SVG 文件
    plt.savefig(save_path, format='svg')
    plt.show()


if __name__ == '__main__':
    try:
        # 从.mat文件中加载PCE数据
        ADNet_PCE = sio.loadmat('PCE_SAVE_Dresden_128/DWT-no-wiener-center-128-all2222.mat')
        # 提取正样本和负样本的PCE得分
        intra_pce = np.array(ADNet_PCE['positive_pce']).flatten()
        inter_pce = np.array(ADNet_PCE['negative_pce']).flatten()

        # 计算最优阈值
        threshold = calculate_threshold(intra_pce, inter_pce)

        # 生成真实标签
        y_true = np.array([0] * len(inter_pce) + [1] * len(intra_pce))
        # 合并正、负样本的PCE得分
        pce_scores = np.concatenate((inter_pce, intra_pce))

        # 根据阈值生成预测标签
        y_pred = (pce_scores >= threshold).astype(int)

        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, y_pred)

    except FileNotFoundError:
        print("指定的.mat文件未找到，请检查文件路径。")
    except Exception as e:
        print(f"发生未知错误: {e}")
