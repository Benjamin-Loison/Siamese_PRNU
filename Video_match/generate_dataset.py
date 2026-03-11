# import os
# import numpy as np
# import glob
#
# # 设置数据集文件夹路径
# dataset_folder = 'path/to/your/dataset/folder'
#
# # 获取所有.npy文件的路径
# npy_files = sorted(glob.glob(os.path.join(dataset_folder, '*.npy')))
#
# # 打乱文件列表
# np.random.shuffle(npy_files)
#
# # 计算分割点
# split_idx = int(0.6 * len(npy_files))  # 训练集的分割点
# val_test_split_idx = int((0.2 + 0.6) * len(npy_files))  # 验证集和测试集的分割点
#
# # 分割数据集
# train_files = npy_files[:split_idx]
# val_files = npy_files[split_idx:val_test_split_idx]
# test_files = npy_files[val_test_split_idx:]
#
# # 将分割后的文件路径保存到不同的文件中
# with open('train_files.txt', 'w') as f:
#     for file_path in train_files:
#         f.write(file_path + '\n')
#
# with open('val_files.txt', 'w') as f:
#     for file_path in val_files:
#         f.write(file_path + '\n')
#
# with open('test_files.txt', 'w') as f:
#     for file_path in test_files:
#         f.write(file_path + '\n')
#
# print(f"训练集包含 {len(train_files)} 个文件，验证集包含 {len(val_files)} 个文件，测试集包含 {len(test_files)} 个文件。")

import os
import numpy as np
import glob


######数据集生成代码

# 设置包含多个数据集文件夹的根路径
root_path = '/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras_SPNCNN'

save_path = '/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_SPNCNN'


# 确保提供的路径存在
if not os.path.exists(root_path):
    print(f"提供的路径 {root_path} 不存在。")
else:
    # 遍历每个子文件夹
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            # 获取当前文件夹内所有.npy文件的路径
            npy_files = sorted(glob.glob(os.path.join(folder_path, '*.npy')))

            # 打乱文件列表
            # np.random.shuffle(npy_files)

            # 计算分割点
            total_files = len(npy_files)
            train_split = int(total_files * 0.6)  # 训练集占60%
            val_test_split = int(total_files * 0.8)  # 验证集和测试集的分割点，占80%

            # 分割数据集
            train_files = npy_files[:train_split]
            val_files = npy_files[train_split:val_test_split]
            test_files = npy_files[val_test_split:]

            # 为当前数据集创建train、valid、test子文件夹
            for split_name, file_list in zip(['train', 'valid', 'test'], [train_files, val_files, test_files]):
                split_save_path = os.path.join(save_path, split_name)
                if not os.path.exists(split_save_path):
                    os.makedirs(split_save_path)

                # 保存分割后的文件路径到.npy文件
                split_file_paths = os.path.join(split_save_path, f'{folder_name}.npy')
                np.save(split_file_paths, np.array(file_list))

            print(f"文件夹 '{folder_name}' 已处理完毕。")
