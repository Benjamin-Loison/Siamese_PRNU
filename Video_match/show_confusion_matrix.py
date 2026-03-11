import os
from SDI_Cross.utility_show import plot_confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import logging
import utils_logger


def softmax(x):
    ex=np.exp(x)
    return ex/ex.sum()

def myshow(file_result=''):
    logger_name = file_result.split('/')[1][7:]+'_'+file_result.split('/')[2].split('.')[0]
    utils_logger.logger_info(logger_name, os.path.join('./test_logs/', logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # file_result = 'output_Eff_pconv_eca_repB4/results_80.npz'
    # load results
    result = np.load(file_result,allow_pickle=True)
    dev_list  = result['list_dev'].tolist()

    # score_mat[dev_i][res_i][dev_j,1] = score between residue of device i (i.e., res_i) and prnu of device i (i.e., dev_j)
    score_mat = result['score_mat'][1]
    dev_labels = [x for x in dev_list]
    # caculate Acs
    Acs = 0
    count = 0
    for i in range(len(dev_list)):
        for j in range(len(score_mat[i])):
            if file_result.split('/')[1].split('_')[-1] == 'Pcn':
                temp = score_mat[i][j]
            else:
                temp = softmax(score_mat[i][j])
            Acs += temp[i,:]
            count += 1
    Acs = Acs/count
    print('Acs score: %5.5f' % Acs)

    num_dev = len(dev_list)
    score_img = [np.concatenate(item ,-1) for item in score_mat] # score_img[dev_i][dev_j,res_i]

    device_true = [index*np.ones(item.shape[-1]) for index, item in enumerate(score_img)]
    device_pred = [np.argmax(item, axis=0) for item in score_img]
    device_true = np.concatenate(device_true,-1)
    device_pred = np.concatenate(device_pred,-1)
    #
    # avg_time
    elapsed_time_list = result['time_list']
    avg_time = np.mean(elapsed_time_list[1:])
    # print(len(elapsed_time_list))
    print('avg_time: '+str(avg_time))

    cm = confusion_matrix(device_true, device_pred)
    # plot normalized confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # accuracy score
    skl_acc = accuracy_score(device_true, device_pred)
    # print('Classification on %d devices on single images' % num_dev)
    print('Accuracy score: %5.5f' % skl_acc)

    # matplotlib inline
    f = plt.figure(figsize=(20, 20), facecolor='white', edgecolor='black')
    plot_confusion_matrix(cm=cm, classes=dev_labels)
    plt.grid(color=[.7, .7, .7])

    plt.rcParams['axes.axisbelow'] = True
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    label_binary = [None,]*num_dev
    for index in range(num_dev):
        label_binary[index] = np.zeros(score_img[index].shape, np.int32)
        label_binary[index][index,:] = 1

    score_binary = np.concatenate(score_img ,-1).flatten()
    label_binary = np.concatenate(label_binary,-1).flatten()

    fpr, tpr, thresholds = roc_curve(label_binary,score_binary)
    acc = (tpr + (1-fpr))/2;
    auc = roc_auc_score(label_binary,score_binary)
    index_max = np.argmax(acc);

    fp_sort = sorted(fpr)
    tp_sort = sorted(tpr)
    idx_order = np.argsort(tpr)

    tpr_001_ind = [i for (i, val) in enumerate(fp_sort) if val >= 0.01][0]
    tpr_001 = tp_sort[tpr_001_ind]

    plt.savefig('confusion_matrix_256-205.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    print('AUC=%5.3f' % auc)

if __name__ == '__main__':
    # myshow(file_result = '/home/seamus20/z_Project/Video_match/output_Eff_repB0_dresden/results_256_256-fusion_shift30.npz')
    myshow(file_result = '/home/seamus20/z_Project/Video_match/output_Eff_repB0_dresden/results_256_205-fusion_3_0.666-0.949.npz')