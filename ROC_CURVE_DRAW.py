import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def ROC_prefcurve(PCE):
    intra_pce = np.array(PCE['intra_pce']).flatten()
    inter_pce = np.array(PCE['inter_pce']).flatten()
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
    B = 256
    OUR_PCE = sio.loadmat('Dresden_PCE_dict_256/Proposed_PCE.mat')
    # OUR_PCE_one = sio.loadmat('Dresden_PCE_dict_128/Proposed_PCE_one_others.mat')
    # dude_PCE = sio.loadmat('Dresden_PCE_dict_128/dude_PCE_others.mat')
    dude_PCE_two = sio.loadmat('Dresden_PCE_dict_256/dude_PCE_two.mat')
    SPN_CNN_PCE = sio.loadmat('Dresden_PCE_dict_256/SPN_PCE_two.mat')


    # dude_rdd_PCE = sio.loadmat('Dresen_PCE_dict_{}/dude_nat_PCE.mat'.format(str(B)))
    # DWT_PCE = sio.loadmat('Dresen_PCE_dict_{}/DWT_PCE.mat'.format(str(B)))
    # xDnCNN_PCE = sio.loadmat('Dresen_PCE_dict_{}/xDnCNN_PCE.mat'.format(str(B)))
    # SPN_PCE = sio.loadmat('Daxing_PCE_dict_{}/SPN_PCE.mat'.format(str(B)))

    OUR_fpr,OUR_tpr,OUR_auc=ROC_prefcurve(OUR_PCE)
    # OUR_fpr_1,OUR_tpr_1,OUR_auc_1=ROC_prefcurve(OUR_PCE_one)
    # dude_fpr,dude_tpr,dude_auc=ROC_prefcurve(dude_PCE)
    dude_fpr_2,dude_tpr_2,dude_auc_2=ROC_prefcurve(dude_PCE_two)
    SPN_CNN_fpr,SPN_CNN_tpr,SPN_CNN_auc=ROC_prefcurve(SPN_CNN_PCE)


    # dude_rdd_fpr,dude_rdd_tpr,dude_rdd_auc=ROC_prefcurve(dude_rdd_PCE)
    # DWT_fpr,DWT_tpr,DWT_auc=ROC_prefcurve(DWT_PCE)
    # xDnCNN_fpr,xDnCNN_tpr,xDnCNN_auc=ROC_prefcurve(xDnCNN_PCE)
    # SPN_fpr,SPN_tpr,SPN_auc=ROC_prefcurve(SPN_PCE)

    # 创建一个新的图表
    plt.figure(figsize=(8, 6))

    # plt.plot(OUR_fpr, OUR_tpr, color='red', label='Proposed AUC  = %0.4f' % OUR_auc)
    # plt.plot(dude_fpr, dude_tpr, color='green',  linestyle='--',label='dude_nat AUC  = %0.4f' % dude_auc)


    # 绘制 ROC 曲线
    # plt.plot(OUR_fpr_1, OUR_tpr_1, color='purple', label='Proposed_one (AUC = {:.4f})'.format(OUR_auc_1))
    # plt.plot(dude_fpr, dude_tpr, color='green', linestyle='--', label='dude_nat_one (AUC = {:.4f})'.format(dude_auc))
    plt.plot(OUR_fpr, OUR_tpr, color='red', label='Proposed (AUC = {:.4f})'.format(OUR_auc))
    plt.plot(dude_fpr_2, dude_tpr_2, color='orange', linestyle='--', label='dude_nat (AUC = {:.4f})'.format(dude_auc_2))
    plt.plot(SPN_CNN_fpr, SPN_CNN_tpr, color='blue', linestyle='--', label='SPN_CNN (AUC = {:.4f})'.format(SPN_CNN_auc))



   #  plt.plot(DHDN_fpr, DHDN_tpr, color='black', label='DHDN AUC = %0.4f' % DHDN_auc)
   #  plt.plot(DWT_fpr, DWT_tpr, color='blue', label='DWT AUC  = %0.4f' % DWT_auc)
   #  plt.plot(xDnCNN_fpr, xDnCNN_tpr, color='green',  linestyle='--',label='xDnCNN AUC  = %0.4f' % xDnCNN_auc)
    # plt.plot(SPN_fpr, SPN_tpr, label='SPN AUC  = %0.4f' % SPN_auc)

    plt.xlim([-0.1, 1.0])
    plt.ylim([0.4, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dresden ({}x{})'.format(B, B))
    plt.legend(loc="lower right",fontsize = 12)
    plt.savefig('ROC_IMAGE/Dresden_ROC_Curve {}x{}.jpg'.format(B,B), dpi=2500)
    plt.show()