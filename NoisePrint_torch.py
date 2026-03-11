import time

import torch
def mloss(x, y):

    muX = torch.mean(x, dim=(2, 3))
    muY = torch.mean(y, dim=(2, 3))
    muXexpand = torch.unsqueeze(torch.unsqueeze(muX, 2), 3)
    muYexpand = torch.unsqueeze(torch.unsqueeze(muY, 2), 3)
    Xzero = x - muXexpand
    Yzero = y - muYexpand
    term1 = torch.sum(Xzero * Yzero, dim=(2, 3))
    term2 = torch.sum(torch.pow(Xzero, 2), dim=(2, 3))
    term3 = torch.sum(torch.pow(Yzero, 2), dim=(2, 3))
    sqrt2 = torch.sqrt(term2)
    sqrt3 = torch.sqrt(term3)
    lossi = 1 - (term1 / (sqrt2 * sqrt3 + 1e-10))
    loss = torch.mean(lossi) + 50 * torch.mean(torch.pow((x-y), 2))

    # loss = torch.mean(lossi)
    return loss

def calculate_distance(residual1, residual2):
    start_time = time.time()  # 记录开始时间

    n = 64
    distances = torch.zeros((n, n))  # 使用 PyTorch 的张量
    for i in range(n):
        for j in range(n):
            distances[i, j] = mloss(residual1[i:i + 1], residual2[j:j + 1])  # 使用切片来保持数据维度
            # print("distance=", i, j, distances[i, j])
    print(distances)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算执行时间
    print(f"calculate_distance执行时间: {elapsed_time} 秒")
    return distances


# # 计算距离
# group_indices = np.eye(64)  # 对角线为 1，其余位置为 0
# group_indices[group_indices == 0] = -1  # 将对角线以外的元素设置为 -1
def euclidean_distance(x, y):
    # 计算两个张量之间的欧几里得距离的平方，这里计算第三和第四个维度的距离
    return torch.sum(torch.pow(x - y, 2), dim=(-2, -1))
    # return np.sum(np.power(x - y, 2), axis=(-2, -1))  # 修改为 numpy 的计算方式

# def calculate_distance(residual1, residual2):
#     n = 64
#     distances = torch.zeros((n, n))  # 使用 PyTorch 的张量
#     for i in range(n):
#         for j in range(n):
#             distances[i, j] = mloss(residual1[i:i + 1], residual2[j:j + 1])  # 使用切片来保持数据维度
#             print(distances)
#             # print("distance=", i, j, distances[i, j])
#     return distances

# 计算概率分布
# def calculate_probabilities(distances):
#     n = distances.size(0)
#     probabilities = torch.zeros_like(distances)
#     for i in range(n):
#         for j in range(n):
#             probabilities[i, j] = torch.exp(-distances[i, j]) / (torch.sum(torch.exp(-distances[i])) + 1e-9)
#     return probabilities

def calculate_probabilities(distances):
    n = distances.shape[0]
    exp_distances = torch.exp(-distances)
    row_sum = torch.sum(exp_distances, dim=1, keepdim=True) + 1e-9
    probabilities = exp_distances / row_sum
    print(probabilities)
    return probabilities

# 计算单个补丁的损失
def calculate_patch_loss(probabilities, group_indices):
    positive_indices = torch.where(group_indices == 1)[0]
    loss = -torch.log(torch.sum(probabilities[positive_indices]) + 1e-9)

    return loss

# 计算整个小批量的损失
def calculate_batch_loss(probabilities_batch, group_indices_batch):
    batch_loss = 0
    for i in range(len(probabilities_batch)):
        patch_loss = calculate_patch_loss(probabilities_batch[i],group_indices_batch[i])
        batch_loss += patch_loss

    return batch_loss


# 计算功率谱密度矩阵
def calculate_power_spectrum_density(residuals):
    power_spectrum_density = torch.abs(torch.fft.fft2(residuals, dim=(-2, -1))) ** 2
    avg_power_spectrum_density = torch.mean(power_spectrum_density, dim=0)
    return avg_power_spectrum_density
# 计算正则化项
def calculate_regularization_term(power_spectrum_density):
    gm = torch.exp(torch.mean(torch.log(power_spectrum_density)))
    am = torch.mean(power_spectrum_density)
    return torch.log(gm / am)
# 计算完整的损失函数
def calculate_total_loss(probabilities_batch, group_indices_batch, residuals1, residuals2, regularization_weight):
    batch_loss = calculate_batch_loss(probabilities_batch, group_indices_batch)

    power_spectrum_density1 = calculate_power_spectrum_density(residuals1)
    regularization_term1 = calculate_regularization_term(power_spectrum_density1)

    power_spectrum_density2 = calculate_power_spectrum_density(residuals2)
    regularization_term2 = calculate_regularization_term(power_spectrum_density2)

    regularization_term = regularization_term1 + regularization_term2

    total_loss = batch_loss - regularization_weight * regularization_term
    return total_loss

# import numpy as np
# import torch
# import torch.nn as nn
#
# # 计算距离
# def euclidean_distance(x, y):
#     return torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=(-2, -1)))
#
# def calculate_distance(residual1, residual2):
#     n = residual1.shape[0]
#     distances = torch.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             distances[i, j] = euclidean_distance(residual1[i], residual2[j])
#     return distances
#
# # 计算概率分布
# def calculate_probabilities(distances):
#     n = distances.shape[0]
#     probabilities = torch.zeros_like(distances)
#     for i in range(n):
#         for j in range(n):
#             probabilities[i, j] = torch.exp(-distances[i, j]) / torch.sum(torch.exp(-distances[i]))
#     return probabilities
#
# # 计算单个补丁的损失
# def calculate_patch_loss(probabilities, group_indices):
#     loss = torch.sum(-torch.log(probabilities) * group_indices)
#     return loss
#
# # 计算整个小批量的损失
# def calculate_batch_loss(probabilities_batch, group_indices_batch):
#     batch_loss = 0
#     for i in range(len(probabilities_batch)):
#         patch_loss = calculate_patch_loss(probabilities_batch[i], group_indices_batch[i])
#         batch_loss += patch_loss
#     return batch_loss
