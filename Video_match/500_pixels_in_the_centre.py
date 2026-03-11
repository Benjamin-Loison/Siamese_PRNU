import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def compute_pce_with_scaling(image, ref_prnu, rep_model, device):
    # 读取图像并提取中心500x500的块
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    half_block_size = 250  # 500 / 2

    # 提取中心500x500的块
    center_block = image[center_y - half_block_size:center_y + half_block_size,
                         center_x - half_block_size:center_x + half_block_size]

    # 子块大小
    sub_block_size = 400
    half_sub_block_size = sub_block_size // 2

    # 缩放范围和步长
    scale_range = np.arange(0.9, 1.1, 0.05)
    # 初始化PCE结果
    pce_results = []
    # 对500x500块的每个400x400子块进行处理
    for i in range(half_block_size * 2 - sub_block_size + 1):
        for j in range(half_block_size * 2 - sub_block_size + 1):
            sub_block = center_block[i:i + sub_block_size, j:j + sub_block_size]

            # 对每个400x400子块进行0.9-1.1的缩放
            for scale in scale_range:
                scaled_sub_block = cv2.resize(sub_block, (0, 0), fx=scale, fy=scale)

                # 计算与参考PRNU的PCE
                pce = calculate_pce(rep_model, ref_prnu, scaled_sub_block, device)
                pce_results.append((i, j, scale, pce))

                # 显示结果
                print(f'Sub-block at ({i}, {j}), Scale: {scale:.2f}, PCE: {pce:.2f}')

    # 返回或保存PCE结果
    np.save('pce_results.npy', pce_results)
    return pce_results

def calculate_pce(rep_model, ref_prnu, sub_block, device):
    # 确保参考PRNU和子块的大小匹配
    if ref_prnu.shape != sub_block.shape:
        min_shape = np.minimum(ref_prnu.shape, sub_block.shape)
        ref_prnu = ref_prnu[:min_shape[0], :min_shape[1]]
        sub_block = sub_block[:min_shape[0], :min_shape[1]]

    # 预处理输入数据以匹配模型的输入要求
    transform = transforms.ToTensor()
    ref_prnu = transform(ref_prnu).unsqueeze(0).to(device)
    sub_block = transform(sub_block).unsqueeze(0).to(device)

    # 使用模型计算PCE
    rep_model.eval()
    with torch.no_grad():
        pce = rep_model(ref_prnu, sub_block).item()

    return pce

# 示例使用
if __name__ == "__main__":
    # 加载图像和参考PRNU
    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    ref_prnu = cv2.imread('ref_prnu.jpg', cv2.IMREAD_GRAYSCALE)

    # 确保图像和参考PRNU加载正确
    if image is None or ref_prnu is None:
        print("Error loading image or ref_prnu")
    else:
        # 加载预训练的网络模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rep_model = torch.load('path_to_your_rep_model.pth')
        rep_model = rep_model.to(device)

        # 计算PCE
        pce_results = compute_pce_with_scaling(image, ref_prnu, rep_model, device)
        print("PCE计算完成，结果保存在'pce_results.npy'中")
