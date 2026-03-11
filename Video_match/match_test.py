import glob
import os
import time
import cv2
import numpy as np
import torch
from SDI_Cross.arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep, reparameterize_model
import scipy.io as sio
from torch.autograd import Variable
from SDI_Cross.SCIFunctions.crosscorr2 import crosscorr2

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)
def process_videos_and_compute_pce(visionPath, isPreCalculated=False):
    # 获取所有视频文件路径
    search_pattern = os.path.join(visionPath, '**/*tp')
    # 使用 glob.glob 函数查找匹配的文件
    videos = glob.glob(search_pattern, recursive=True)
    videos.sort()

    print(videos)
    print(len(videos))
    # 加载预训练的网络模型
    model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    model_path = './SDI_Cross/weight_Eff_repB0/Pixle_256_256/checkpoint.pt'
    pretrained_model = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_model.items()})
    model = reparameterize_model(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for i, videoPath in enumerate(videos):
        start_time = time.time()
        videoResPath = os.path.join(os.path.dirname(videoPath), os.path.basename(videoPath))
        print(f'Running video is {videoResPath}')

        if not isPreCalculated:
            fp = sio.loadmat(os.path.join(videoResPath, 'fp.mat'))
            ref_prnu = fp['cropped_prnu']

        compute_pce_with_scaling(videoPath,videoResPath, ref_prnu, model, device)
        print(f'Finished processing video {videoResPath} in {time.time() - start_time:.2f} seconds')


def compute_pce_with_scaling(videoPath,videoResPath, ref_prnu, model, device):
    # 使用 glob.glob 来找到所有匹配的.mat文件路径
    parent_directory = os.path.dirname(videoPath)
    search_pattern = os.path.join(parent_directory, '*.mat')
    frames = glob.glob(search_pattern)
    # 初始化一个空列表来存储过滤后的文件路径
    frames_path = []
    # 限制循环次数为 11 或者 frames 列表的长度，取较小值
    for i in range(min(11, len(frames))):
        # 检查文件名是否不是 'db.mat'
        if 'db.mat' not in frames[i]:
            # 构造完整的文件路径
            frames_path.append(os.path.join(frames[i]))
    frames_path.sort()
    print(frames_path)
    print(len(frames_path))


    # 读取图像并提取中心500x500的块
    crop_size = 500
    ncc_range = 65
    y_size = ref_prnu.shape[0] // 2
    x_size = ref_prnu.shape[1] // 2


    for a, frame_path in enumerate(frames_path[:11]):
        Noisex = sio.loadmat(frame_path)['PRNU_image']
        print(a+1)
        Fp_camera_1 = ref_prnu[(y_size - crop_size - ncc_range):(y_size + crop_size + ncc_range),
                      (x_size - crop_size - ncc_range):(x_size + crop_size + ncc_range)]
        Noisex_ref = Noisex[(y_size - crop_size):(y_size + crop_size), (x_size - crop_size):(x_size + crop_size)]


        print("Size of Noisex_ref:", Noisex_ref.shape)

        sub_block_size = 500

        # 定义缩放范围和步长
        scale_range = np.arange(0.9, 1.1, 0.05)

        score_results = []
        max_i = Noisex_ref.shape[0] - sub_block_size
        max_j = Noisex_ref.shape[1] - sub_block_size
        print(f"max_i={max_i},max_j={max_j}")

        # 确保索引不会超出边界
        for i in range(max_i + 1):
            for j in range(max_j + 1):
                sub_block = Noisex_ref[i:i + sub_block_size, j:j + sub_block_size]

                # 对每个子块进行0.9-1.1的缩放
                for scale in scale_range:
                    scaled_sub_block = cv2.resize(sub_block, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)#使用最近邻插值
                    data = crosscorr2(Fp_camera_1,scaled_sub_block)
                    score = calculate_score(model, data)
                    score_results.append((a+1,i, j, scale, score.item()))
                    print(f'Frame {a+1},Sub-block at ({i}, {j}), Scale: {scale:.4f}, Similarity: {score.item():.4f}')

    similarity_scores = np.array(score_results)[:, 4]  # 提取相似度得分
    max_similarity = np.max(similarity_scores)
    max_indices = np.where(similarity_scores == max_similarity)
    # 输出最大相似度得分及其对应的帧编号、子块位置和缩放比例
    print(f"Maximum Similarity Score: {max_similarity:.4f}")
    for index in max_indices[0]:
        frame_number, i, j, scale = score_results[index][:-1]
        print(f"Frame {frame_number}, Sub-block at ({i}, {j}), Scale: {scale:.4f}")


    file_name = 'Similarity_results.npy'
    full_file_path = os.path.join(videoPath, file_name)
    np.save(full_file_path, score_results)
    return score_results


def calculate_score(model, data):

    data = torch.Tensor(data[None, None, :, :])
    data = Variable(data)
    data = data.cuda()

    model.eval()
    with torch.no_grad():
        result = model(data).cpu().detach().numpy()
        score = softmax(result)[:, 1]
    # print(score)
    return score


if __name__ == "__main__":
    print("Script is running")
    visionPath = '/home/seamus20/matlab_wyl/VISION_VIDEO_dude/'
    process_videos_and_compute_pce(visionPath, isPreCalculated=False)
    print("Script is end")

