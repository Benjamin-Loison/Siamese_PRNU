import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#此代码用于ROC曲线绘制
# # 设置全局字体大小
# plt.rcParams.update({'font.size': 16})

def ROC_prefcurve(PCE):
    intra_pce = np.array(PCE['positive_pce']).flatten()
    inter_pce = np.array(PCE['negative_pce']).flatten()
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]
    true_label = np.ones(t1)
    false_label = np.zeros(t2)

    labels = np.concatenate([true_label, false_label])
    values = np.concatenate([intra_pce, inter_pce])

    fpr, tpr, _ = roc_curve(labels, values)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


if __name__ == '__main__':
    B = 128
    Proposed_PCE = sio.loadmat('PCE_SAVE_Dresden_128/Dresden_FUSION-nat-no-wiener-center-128-all_conv7+se.mat')
    dude_PCE = sio.loadmat('PCE_SAVE_Dresden_128/Dresden_PRNU-no-wiener-center-128-all.mat')
    SPNCNN_PCE = sio.loadmat('PCE_SAVE_Dresden_128/SPNCNN-no-wiener-center-128-all.mat')
    PRNU_PCE = sio.loadmat('PCE_SAVE_Dresden_128/DWT-no-wiener-center-128-all.mat')

    # Calculate ROC curves and AUC values for each method
    OUR_fpr, OUR_tpr, OUR_auc = ROC_prefcurve(Proposed_PCE)
    dude_fpr, dude_tpr, dude_auc = ROC_prefcurve(dude_PCE)
    spn_fpr, spn_tpr, spn_auc = ROC_prefcurve(SPNCNN_PCE)
    prnu_fpr, prnu_tpr, prnu_auc = ROC_prefcurve(PRNU_PCE)

    # Create the ROC curve plot
    plt.figure(figsize=(9, 7))

    # Plot ROC curves with customized markers and line styles
    plt.plot(OUR_fpr, OUR_tpr, color='red', label='PROPOSED (AUC = {:.4f})'.format(OUR_auc),
             linewidth=1.5, marker='o', markersize=4.0, markevery=1000)
    plt.plot(dude_fpr, dude_tpr, color='blue', linestyle='-', label='DFPRNU-NET (AUC = {:.4f})'.format(dude_auc),
             linewidth=1.5, marker='s', markersize=4.0, markevery=1000)
    plt.plot(spn_fpr, spn_tpr, color='orange', linestyle='--', label='SPNCNN (AUC = {:.4f})'.format(spn_auc),
             linewidth=1.5, marker='^', markersize=4.0, markevery=1000)
    plt.plot(prnu_fpr, prnu_tpr, color='purple', linestyle=':', label='DWT (AUC = {:.4f})'.format(prnu_auc),
             linewidth=1.5, marker='D', markersize=4.0, markevery=1000)

    # 添加辅助网格线，间隔为0.1，增强可读性
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0.0, 1.05, 0.1), fontsize=14)  # x轴每隔0.1绘制网格线
    plt.yticks(np.arange(0.0, 1.05, 0.1),fontsize=14)  # y轴每隔0.1绘制网格线


    # Define x and y limits, labels, title, and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.2, 1.0])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.title('ROC Curve for Dresden Dataset ({}x{} Image Patch)'.format(B, B), fontsize=16)
    plt.legend(loc="lower right", fontsize=15)

    # Save the figure in high quality
    plt.savefig(
        'Dresden_ROC_IMAGE_svg/meihua_Dresden_16k_ROC_Curve_CENTER_se+conv7_{}x{}.svg'.format(B, B),
        dpi=2000,
        bbox_inches='tight'
    )

    # Display the plot
    plt.show()
