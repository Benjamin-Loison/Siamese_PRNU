import torch
# 计算欧几里得距离的平方
def euclidean_distance(x, y):
    return torch.sum(torch.pow(x - y, 2), dim=(-2, -1))

def calculate_distance(residual1, residual2):
    n = 64
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = euclidean_distance(residual1[i], residual2[j])
            print(distances.shape)
    return distances

def calculate_probabilities(distances):
    n = distances.shape[0]
    probabilities = torch.zeros_like(distances)
    for i in range(n):
        for j in range(n):
            probabilities[i, j] = torch.exp(-distances[i, j]) / (torch.sum(torch.exp(-distances[i])) + 1e-9)
    return probabilities

def calculate_patch_loss(probabilities, group_indices):
    positive_indices = torch.where(group_indices == 1)[0]
    loss = -torch.log(torch.sum(probabilities[positive_indices]) + 1e-9)
    return loss

def calculate_batch_loss(probabilities_batch, group_indices_batch):
    batch_loss = 0
    for i in range(len(probabilities_batch)):
        patch_loss = calculate_patch_loss(probabilities_batch[i], group_indices_batch[i])
        batch_loss += patch_loss
    return batch_loss

def calculate_power_spectrum_density(residuals):
    power_spectrum_density = torch.abs(torch.fft.fft2(residuals, dim=(-2, -1))) ** 2
    avg_power_spectrum_density = torch.mean(power_spectrum_density, dim=0)
    return avg_power_spectrum_density

def calculate_regularization_term(power_spectrum_density):
    gm = torch.exp(torch.mean(torch.log(power_spectrum_density)))
    am = torch.mean(power_spectrum_density)
    return torch.log(gm / am)

def calculate_total_loss(probabilities_batch, group_indices_batch, residuals1, residuals2, regularization_weight):
    batch_loss = calculate_batch_loss(probabilities_batch, group_indices_batch)

    power_spectrum_density1 = calculate_power_spectrum_density(residuals1)
    regularization_term1 = calculate_regularization_term(power_spectrum_density1)

    power_spectrum_density2 = calculate_power_spectrum_density(residuals2)
    regularization_term2 = calculate_regularization_term(power_spectrum_density2)

    regularization_term = regularization_term1 + regularization_term2

    total_loss = batch_loss - regularization_weight * regularization_term
    return total_loss
