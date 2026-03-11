import numpy as np
import os
import time

from SDI_Cross.SCIFunctions.PCE1 import PCE1
from SDI_Cross.SCIFunctions.crosscorr import crosscorr
from glob import glob
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2
from SDI_Cross.SCIFunctions.crosscorr import crosscorr


from SDI_Cross.src.pce import PCE
# from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2

#D03_Huawei_P9
#D18_Apple_iPhone5c ValueError: operands could not be broadcast together with shapes (2448,3264) (3264,2448)

# 设置文件夹路径
a_folder = '/home/seamus20/matlab_wyl/Dresden_40cameras_PRNU_FUSION_npy'
b_folder = '/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras/D39_Sony_DSC-H50_0'

# 加载fp
fp_path = os.path.join(a_folder, 'D39_Sony_DSC-H50_0.npy')
fp = np.load(fp_path)

# 初始化PCE列表
PCE_list = []

# 获取b_folder中所有.npy文件的路径，并进行排序
npy_files = sorted(glob(os.path.join(b_folder, '*.npy')))

# # 遍历排序后的.npy文件
# for filename in npy_files:
#     # 加载噪声数据
#     noise = np.load(filename)
#
#     # 计算crosscorr
#     data1 = crosscorr2(fp, noise)
#     PCE_2 = PCE(data1)
#
#     # 将PCE值添加到列表
#     PCE_list.append(PCE_2)
#
#     # 打印结果
#     print(f'PCE for {os.path.basename(filename)}: {PCE_2}')
#
# # 将PCE列表转换为NumPy数组
# PCE_array = np.array(PCE_list)
start_time = time.time()
counter = 0
# 遍历排序后的.npy文件
for filename in npy_files:
    counter += 1
    print(f"Processing file {counter} of {len(npy_files)}: {filename}")


    noise = np.load(filename)

    # 计算crosscorr
    data1 = crosscorr2(fp, noise)  # 确保crosscorr2函数已定义
    PCE_2 = PCE(data1)  # 确保PCE函数已定义

    # 将PCE值添加到列表
    PCE_list.append(PCE_2)

    # 打印结果
    print(f'PCE for {os.path.basename(filename)}: {PCE_2}')

    # 如果PCE低于30，则删除文件
    if PCE_2 < 30:
        print(f'Deleting {os.path.basename(filename)} because PCE is below 30.')
        os.remove(filename)  # 删除文件

# 将PCE列表转换为NumPy数组（如果需要的话）
# PCE_array = np.array(PCE_list)
# 计算总时间
total_time = time.time() - start_time
print(f"Total time taken: {total_time:.2f} seconds")
print("Process completed.")


# # 保存PCE数组到.npy文件
# save_folder = '/home/seamus20/matlab_wyl/PCE_result'
# save_path = os.path.join(save_folder, 'PCE_results_D35_Samsung_GalaxyTabA.npy')
# np.save(save_path, PCE_array)
# print(f"PCE results saved to {save_path}")