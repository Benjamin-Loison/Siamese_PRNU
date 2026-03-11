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
    B = 64
    Proposed_PCE = sio.loadmat('PCE_SAVE_Dresden_64/Dresden_FUSION-nat-no-wiener-center-64-all_conv7+se.mat')
    dude_PCE = sio.loadmat('PCE_SAVE_Dresden_64/Dresden_PRNU-no-wiener-center-64-all.mat')
    SPNCNN_PCE = sio.loadmat('PCE_SAVE_Dresden_64/SPNCNN-no-wiener-center-64-all.mat')
    PRNU_PCE = sio.loadmat('PCE_SAVE_Dresden_64/DWT-no-wiener-center-64-all.mat')



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

    plt.plot(OUR_fpr, OUR_tpr, color='red', label='PROPOSED (AUC = {:.4f})'.format(OUR_auc))
    plt.plot(dude_fpr, dude_tpr, color='blue', linestyle='-', label='DFPRNU-NET (AUC = {:.4f})'.format(dude_auc))
    plt.plot(spn_fpr, spn_tpr, color='orange', linestyle='--', label='SPNCNN (AUC = {:.4f})'.format(spn_auc))
    plt.plot(prnu_fpr, prnu_tpr, color='purple', label='DWT (AUC = {:.4f})'.format(prnu_auc))


    plt.xlim([0.0, 1.0])
    plt.ylim([0.2, 1.05])
    plt.xlabel('FPR',fontsize = 12)
    plt.ylabel('TPR',fontsize = 12)
    plt.title('ROC CURVE FOR DRESDEN DATASET ({}x{} IMAGE PATCH)'.format(B, B),fontsize = 11)
    plt.legend(loc="lower right",fontsize = 11)
    plt.savefig('Dresden_ROC_IMAGE_svg/Dresden_16k_ROC_Curve_CENTER_no_wiener_se+conv7_{}x{}.svg'.format(B, B), dpi=1500,
                bbox_inches='tight')
    plt.show()