import os
import numpy as np
from scipy.io import loadmat

# 设置包含.mat文件的源文件夹路径
source_folder_path = '/home/seamus20/matlab_wyl/Camera_Dresden_PRNU_50_sort_wiener'
# 设置目标文件夹路径，用于保存.npy文件
target_folder_path = '/home/seamus20/matlab_wyl/Camera_Dresden_PRNU_50_WIENER_npy'

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder_path):
    if filename.endswith('.mat'):  # 确保是.mat文件
        # 构建完整的.mat文件路径
        mat_file_path = os.path.join(source_folder_path, filename)

        # 使用loadmat读取.mat文件，使用squeeze_me=True来删除单一维度
        mat_contents = loadmat(mat_file_path, squeeze_me=True)

        # 获取.mat文件中所有的变量名，它们是字典的键
        variable_names = mat_contents.keys()

        # 遍历变量名和对应的数据
        for var_name in variable_names:
            if not var_name.startswith('__'):  # 排除MATLAB的系统变量
                mat_data = mat_contents[var_name]

                # 构建原始.mat文件的名称（不包括扩展名）
                original_name = os.path.basename(mat_file_path)[:-4]
                # 构建.npy文件的路径，使用目标文件夹路径和原始.mat文件名
                npy_file_path = os.path.join(target_folder_path, original_name + '.npy')
                # 构建.npy文件的路径，使用目标文件夹路径和变量名

                # 将数据保存为.npy文件
                np.save(npy_file_path, mat_data)

                print(f'Converted {filename} to {os.path.basename(npy_file_path)}')

print("Conversion completed.")