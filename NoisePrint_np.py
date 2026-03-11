import numpy as np
import math
import torch

# # 计算距离
def euclidean_distance(x, y):
    # 计算两个张量之间的欧几里得距离的平方，这里计算第三和第四个维度的距离
    return torch.sum(torch.pow(x - y, 2), dim=(-2, -1))

def calculate_distance(residual1,residual2):
    n =64
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = euclidean_distance(residual1[i:i+1],residual2[j:j+1])
    return distances
# 计算概率分布
def calculate_probabilities(distances):
    n = distances.shape[0]
    probabilities = np.zeros_like(distances)
    for i in range(n):
        for j in range(n):
            probabilities[i, j] = np.exp(-distances[i, j]) / (np.sum(np.exp(-distances[i])) +1e-9)
    return probabilities

# 计算单个补丁的损失
def calculate_patch_loss(probabilities, group_indices):
    positive_indices = np.where(group_indices == 1)[0]
    loss = -np.log(np.sum(probabilities[positive_indices])+ 1e-9)
    return loss

# 计算整个小批量的损失
def calculate_batch_loss(probabilities_batch, group_indices_batch):
    batch_loss = 0
    for i in range(len(probabilities_batch)):
        patch_loss = calculate_patch_loss(probabilities_batch[i], group_indices_batch[i])
        batch_loss += patch_loss
    return batch_loss

    out_prnu = model(PRNUB)
    out_img = model(ImgB)

    distances = calculate_distance(out_prnu,out_img)
    probabilities = calculate_probabilities(distances)
    batch_loss = calculate_batch_loss(probabilities, group_indices)

    batch_loss_tensor = torch.tensor(batch_loss, requires_grad=True)
    loss = batch_loss_tensor
    loss.backward()
    optimizer.step()



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


