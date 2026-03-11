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
    DHDN_PCE = sio.loadmat('Dresen_PCE_dict_{}/DHDN_PCE.mat'.format(str(B)))
    dude_rdd_PCE = sio.loadmat('Dresen_PCE_dict_{}/dude_nat_PCE.mat'.format(str(B)))
    DWT_PCE = sio.loadmat('Dresen_PCE_dict_{}/DWT_PCE.mat'.format(str(B)))
    xDnCNN_PCE = sio.loadmat('Dresen_PCE_dict_{}/xDnCNN_PCE.mat'.format(str(B)))
    # SPN_PCE = sio.loadmat('Daxing_PCE_dict_{}/SPN_PCE.mat'.format(str(B)))

    DHDN_fpr,DHDN_tpr,DHDN_auc=ROC_prefcurve(DHDN_PCE)
    dude_rdd_fpr,dude_rdd_tpr,dude_rdd_auc=ROC_prefcurve(dude_rdd_PCE)
    DWT_fpr,DWT_tpr,DWT_auc=ROC_prefcurve(DWT_PCE)
    xDnCNN_fpr,xDnCNN_tpr,xDnCNN_auc=ROC_prefcurve(xDnCNN_PCE)
    # SPN_fpr,SPN_tpr,SPN_auc=ROC_prefcurve(SPN_PCE)



    plt.plot(dude_rdd_fpr, dude_rdd_tpr, color='red', label='Ours AUC  = %0.4f' % dude_rdd_auc)
   # plt.plot(Restormer_fpr, Restormer_tpr, color='yellow',  linestyle='--',label='dude_nat AUC  = %0.4f' % Restormer_auc)
    plt.plot(DHDN_fpr, DHDN_tpr, color='black', label='DHDN AUC = %0.4f' % DHDN_auc)
    plt.plot(DWT_fpr, DWT_tpr, color='blue', label='DWT AUC  = %0.4f' % DWT_auc)
    plt.plot(xDnCNN_fpr, xDnCNN_tpr, color='green',  linestyle='--',label='xDnCNN AUC  = %0.4f' % xDnCNN_auc)
    # plt.plot(SPN_fpr, SPN_tpr, label='SPN AUC  = %0.4f' % SPN_auc)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.65, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Dresden {}*{} pixel'.format(str(B),str(B)))
    plt.legend(loc="lower right",fontsize = 12)
    plt.savefig('img_result2/Dresden{}.jpg'.format(str(B)), dpi=800)
    plt.show()
