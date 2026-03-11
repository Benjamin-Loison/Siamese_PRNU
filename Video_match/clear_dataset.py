import os
import re


###数据集清洗代码
# 设置A和B文件夹的路径
folder_a = '/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras_SPNCNN/D40_Sony_DSC-H50_1'
folder_b = '/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras/D40_Sony_DSC-H50_1'

# 定义一个函数，用于提取文件名不包含扩展名的最后四位
def extract_last_four(filename):
    basename = os.path.basename(filename)
    without_ext = os.path.splitext(basename)[0]  # 去除扩展名
    return without_ext[-5:]  # 返回最后四位

# 获取A和B文件夹中所有.npy文件的最后四位名称
files_a = {extract_last_four(file) for file in os.listdir(folder_a) if file.endswith('.npy')}
files_b = {extract_last_four(file) for file in os.listdir(folder_b) if file.endswith('.npy')}

# 找出只在A文件夹中的.npy文件的最后四位名称
files_to_delete = files_a - files_b

# 删除A文件夹中这些文件
for file_name in files_to_delete:
    # 构造完整的文件路径
    file_path = os.path.join(folder_a, f"Sony_DSC-H50_1_{file_name}.npy")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")

# 删除操作完成后，输出A文件夹内的文件数量
remaining_files_in_a = len(os.listdir(folder_a))
print(f"Remaining files in folder A: {remaining_files_in_a}")