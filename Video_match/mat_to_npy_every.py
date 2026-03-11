import os
import numpy as np
from scipy.io import loadmat

# 设置源文件夹路径和目标文件夹路径
source_folder_path = '/home/seamus20/matlab_wyl/Dresden_every_image_40cameras_DWT_mat'
target_folder_path = '/home/seamus20/matlab_wyl/Dresden_every_image_40cameras_DWT_npy'


def convert_mat_to_npy(source_path, target_path):
    # 确保目标文件夹存在
    os.makedirs(target_path, exist_ok=True)

    # 遍历源文件夹中的所有文件和文件夹
    for filename in os.listdir(source_path):
        file_path = os.path.join(source_path, filename)

        # 如果是文件夹，则递归调用
        if os.path.isdir(file_path):
            new_target_path = os.path.join(target_path, filename)
            convert_mat_to_npy(file_path, new_target_path)
        elif filename.endswith('.mat'):  # 确保是.mat文件
            mat_file_path = file_path
            npy_file_path = os.path.join(target_path, filename[:-4] + '.npy')

            # 读取.mat文件
            mat_contents = loadmat(mat_file_path, squeeze_me=True)
            variable_names = mat_contents.keys()

            # 遍历变量名和对应的数据
            for var_name in variable_names:
                if not var_name.startswith('__'):
                    mat_data = mat_contents[var_name]
                    np.save(npy_file_path, mat_data)
                    print(f'Converted {filename} to {os.path.basename(npy_file_path)}')


# 调用函数开始转换
convert_mat_to_npy(source_folder_path, target_folder_path)
print("Conversion completed.")