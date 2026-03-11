import time

import torch

def mloss(x, y):
    muX = torch.mean(x, dim=(3, 4))
    muY = torch.mean(y, dim=(3, 4))

    muXexpand = torch.unsqueeze(torch.unsqueeze(muX, 3), 4)
    muYexpand = torch.unsqueeze(torch.unsqueeze(muY, 3), 4)
    Xzero = x - muXexpand
    Yzero = y - muYexpand

    term1 = torch.sum(Xzero * Yzero, dim=(3, 4))
    term2 = torch.sum(torch.pow(Xzero, 2), dim=(3, 4))
    term3 = torch.sum(torch.pow(Yzero, 2), dim=(3, 4))

    sqrt2 = torch.sqrt(term2)
    sqrt3 = torch.sqrt(term3)
    lossi = 1 - (term1 / (sqrt2 * sqrt3 + 1e-10))

    loss = torch.mean(lossi) + 50 * torch.mean(torch.pow((x-y), 2))
    return loss


def calculate_distance(residual1, residual2):
    start_time = time.time()  # 记录开始时间

    n = residual1.shape[0]
    # 将 residual1 和 residual2 扩展为 (n,n, 1, 40, 40)
    residual1_expanded = residual1.unsqueeze(1).expand(-1, n, -1, -1, -1)
    residual2_expanded = residual2.unsqueeze(0).expand(n, -1, -1, -1, -1)
    print(residual1_expanded.shape)
    print(residual2_expanded.shape)

    # 计算距离矩阵
    muX = torch.mean(residual1_expanded, dim=(3, 4))
    muY = torch.mean(residual2_expanded, dim=(3, 4))
    print(muX.shape)
    print(muY.shape)

    # muX_expand = muX.unsqueeze(3).unsqueeze(4)
    # muY_expand = muY.unsqueeze(3).unsqueeze(4)
    muX_expand = torch.unsqueeze(torch.unsqueeze(muX, 3), 4)
    muY_expand = torch.unsqueeze(torch.unsqueeze(muY, 3), 4)
    print(muX_expand.shape)
    print(muY_expand.shape)

    Xzero = residual1_expanded - muX_expand
    Yzero = residual2_expanded - muY_expand
    term1 = torch.sum(Xzero * Yzero, dim=(3, 4))
    term2 = torch.sum(torch.pow(Xzero, 2), dim=(3, 4))
    term3 = torch.sum(torch.pow(Yzero, 2), dim=(3, 4))
    sqrt2 = torch.sqrt(term2)
    sqrt3 = torch.sqrt(term3)
    lossi = 1 - (term1 / (sqrt2 * sqrt3 + 1e-10))
    mse_loss = torch.mean(torch.pow(residual1 - residual2, 2))
    distances= lossi+50*mse_loss
    distances = distances.squeeze()
    print(distances.shape)
    print(distances)
    # 返回距离矩阵
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算执行时间
    print(f"calculate_distance执行时间: {elapsed_time} 秒")
    return distances




def calculate_probabilities(distances):

    start_time = time.time()  # 记录开始时间

    # n = distances.shape[0]
    # probabilities = torch.zeros_like(distances)
    # for i in range(n):
    #     for j in range(n):
    #         probabilities[i, j] = torch.exp(-distances[i, j]) / (torch.sum(torch.exp(-distances[i])) + 1e-9)
    exp_distances = torch.exp(-distances)
    row_sum = torch.sum(exp_distances, dim=1, keepdim=True) + 1e-9
    probabilities = exp_distances / row_sum

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"calculate_probabilities执行时间: {elapsed_time} 秒")
    print(probabilities)


    return probabilities



def calculate_patch_loss(probabilities, group_indices):
    positive_indices = torch.where(group_indices == 1)[0]
    loss = -torch.log(torch.sum(probabilities[positive_indices]) + 1e-9)
    return loss

def calculate_batch_loss(probabilities_batch, group_indices_batch):

    start_time = time.time()


    batch_loss = 0
    for i in range(len(probabilities_batch)):
        patch_loss = calculate_patch_loss(probabilities_batch[i], group_indices_batch[i])
        batch_loss += patch_loss

        end_time = time.time()
        elapsed_time = end_time - start_time
        if i == 63:
            print(f"calculate_batch_loss执行时间: {elapsed_time} 秒")

    return batch_loss



# def calculate_probabilities(distances):
#     n = distances.shape[0]
#     probabilities = torch.zeros(n, n)
#     for i in range(n):
#         for j in range(n):
#             probabilities[i, j] = torch.exp(-distances[i, j, 0]) / (torch.sum(torch.exp(-distances[i, :, 0])) + 1e-9)
#     return probabilities
