import numpy as np
from sklearn.metrics import accuracy_score

file_result = 'cross/output_Pcn_cross/results_256.npz'
result = np.load(file_result,allow_pickle=True)
elapsed_time_list = result['time_list']
avg_time = np.mean(elapsed_time_list[1:])*1000
print(len(elapsed_time_list))
print('avg_time: '+str(avg_time))

# accuracy score
score_mat = result['score_mat'][1]
print(score_mat)
score_img = [np.concatenate(item ,-1) for item in score_mat]
device_true = [index*np.ones(item.shape[-1]) for index, item in enumerate(score_img)]
device_pred = [np.argmax(item, axis=0) for item in score_img]
device_true = np.concatenate(device_true,-1)
device_pred = np.concatenate(device_pred,-1)
skl_acc = accuracy_score(device_true, device_pred)
print('Accuracy score: %5.5f' % skl_acc)