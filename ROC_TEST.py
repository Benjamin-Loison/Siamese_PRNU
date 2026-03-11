import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def ROC_prefcurve(PCE):
    intra_pce = np.array(PCE['positive_pce']).flatten()
    inter_pce = np.array(PCE['negative_pce']).flatten()
    t1 = intra_pce.shape[0]
    t2 = inter_pce.shape[0]
    true_lable = np.ones(t1)
    false_lable = np.zeros(t2)

    lable = np.concatenate([true_lable, false_lable])
    value = np.concatenate([intra_pce, inter_pce])

    fpr, tpr, _ = roc_curve(lable, value)
    roc_auc = auc(fpr, tpr)

    return fpr,tpr,roc_auc

if __name__ == '__main__':
    B = 128
    Proposed_PCE = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_FUSION-nat-nowiener-center-128_pce_values_lower-right1.mat')
    dude_PCE = sio.loadmat('PCE_SAVE_Dresden_Surround/Dresden_PRNU-nowiener-center_128_pce_values_lower-right1.mat')
    SPNCNN_PCE = sio.loadmat('PCE_SAVE_Dresden_Surround_SPNCNN/Dresden_SPNCNN-nowiener-center_128_pce_values_lower-right1.mat')
    PRNU_PCE = sio.loadmat('PCE_SAVE_Dresden_Surround_Conventional/Dresden_Conventional-nowiener-center_128_pce_values_lower-right1.mat')



    OUR_fpr,OUR_tpr,OUR_auc=ROC_prefcurve(Proposed_PCE)
    dude_fpr,dude_tpr,dude_auc=ROC_prefcurve(dude_PCE)
    spn_fpr, spn_tpr, spn_auc = ROC_prefcurve(SPNCNN_PCE)
    prnu_fpr, prnu_tpr, prnu_auc = ROC_prefcurve(PRNU_PCE)



    # 创建一个新的图表
    plt.figure(figsize=(8, 6))

    # plt.plot(OUR_fpr, OUR_tpr, color='red', label='Proposed AUC  = %0.4f' % OUR_auc)
    # plt.plot(dude_fpr, dude_tpr, color='green',  linestyle='--',label='dude_nat AUC  = %0.4f' % dude_auc)
    # 绘制 ROC 曲线
    # plt.plot(OUR_fpr_1, OUR_tpr_1, color='purple', label='Proposed_one (AUC = {:.4f})'.format(OUR_auc_1))

    plt.plot(OUR_fpr, OUR_tpr, color='red', label='Proposed (AUC = {:.4f})'.format(OUR_auc))
    plt.plot(dude_fpr, dude_tpr, color='blue', linestyle='-', label='Dudenet (AUC = {:.4f})'.format(dude_auc))
    plt.plot(spn_fpr, spn_tpr, color='orange', linestyle='--', label='SPNCNN (AUC = {:.4f})'.format(spn_auc))
    plt.plot(prnu_fpr, prnu_tpr, color='purple', label='DWT (AUC = {:.4f})'.format(prnu_auc))


    plt.xlim([0.0, 1.0])
    plt.ylim([0.2, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dresden dataset ({}x{})'.format(B, B))
    plt.legend(loc="lower right",fontsize = 12)
    plt.savefig('Dresden_ROC_IMAGE/Dresden_4k_ROC_Curve_CENTER_NOwiener1 {}x{}.jpg'.format(B,B), dpi=1000)
    plt.show()