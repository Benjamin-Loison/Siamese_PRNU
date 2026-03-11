import torch
def euclidean_distance(x, y):
    # 计算两个张量之间的欧几里得距离的平方，这里计算第三和第四个维度的距离
    return torch.sum(torch.pow(x - y, 2), dim=(-2, -1))
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
#
def calculate_distance(residual1, residual2):
    n = 64
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = mloss(residual1[i:i+1], residual2[j:j+1])
    return distances

def calculate_probabilities(distances):
    n = distances.shape[0]
    exp_distances = torch.exp(-distances)
    row_sum = torch.sum(exp_distances, dim=1, keepdim=True) + 1e-9
    probabilities = exp_distances / row_sum
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


#
# def mloss(x, y):
#     muX = torch.mean(x, dim=(-2, -1))
#     muY = torch.mean(y, dim=(-2, -1))
#
#     muXexpand = torch.unsqueeze(torch.unsqueeze(muX, -2), -1)
#     muYexpand = torch.unsqueeze(torch.unsqueeze(muY, -2), -1)
#     Xzero = x - muXexpand
#     Yzero = y - muYexpand
#
#     term1 = torch.sum(Xzero * Yzero, dim=(-2, -1))
#     term2 = torch.sum(torch.pow(Xzero, 2), dim=(-2, -1))
#     term3 = torch.sum(torch.pow(Yzero, 2), dim=(-2, -1))
#
#     sqrt2 = torch.sqrt(term2)
#     sqrt3 = torch.sqrt(term3)
#     lossi = 1 - (term1 / (sqrt2 * sqrt3 + 1e-10))
#
#     loss = torch.mean(lossi) + 50 * torch.mean(torch.pow((x-y), 2))
#     loss = torch.mean(lossi)
    # return loss
#
# def calculate_distance(residual1, residual2):
#     n = 64
#     Reshape residual1 and residual2 to (n, 1, C, H, W)
    # residual1 = residual1.unsqueeze(1)
    # residual2 = residual2.unsqueeze(0)
    # Compute distances using broadcasting
    # distances = mloss(residual1, residual2)
    # return distances

