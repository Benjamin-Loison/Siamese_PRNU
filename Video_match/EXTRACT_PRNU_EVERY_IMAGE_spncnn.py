#此代码用于提取图像的测试PRNU并保存
import os
import cv2
import glob

import torch
from scipy.io import savemat
from torch.autograd import Variable
from DnCNN_SPN import DnCNN_SPN
import time
from utils import *
device_ids = [0]



###提取每张图像的PRNU，使用SPNCNN


def split_image(img, tile_size):
    img_height, img_width = img.shape
    # print(img.shape)
    tiles = []
    for i in range(0, img_height, tile_size):
        for j in range(0, img_width, tile_size):
            if i + tile_size > img_height:
                height = img_height - i
            else:
                height = tile_size
            if j + tile_size > img_width:
                width = img_width - j
            else:
                width = tile_size
            tile = img[i:i+height, j:j+width]
            tiles.append(tile)
    return np.array(tiles)

def stitch_tiles(tiles, tile_size, img_shape):
    img_height, img_width = img_shape

    stitched_img = np.zeros((img_height, img_width), dtype=tiles[0].dtype)
    idx = 0
    # print(tiles[0].dtype)
    # print(tiles[idx].shape)
    for i in range(0, img_height, tile_size):
        for j in range(0, img_width, tile_size):
            height = min(tile_size, img_height - i)
            width = min(tile_size, img_width - j)
            stitched_img[i:i+height, j:j+width] = tiles[idx][:height, :width]
            idx += 1
    return stitched_img



def PRNU_Extract_pinjie( model_dncnn, image_file):
    img = cv2.imread(image_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image) / 255.0

    tile_size = 1000
    tiles_list = split_image(gray_image, tile_size)
    # for stacked_group in tiles_list:
    #     print("堆叠后数组的形状:", stacked_group.shape)

    tensor_list = [torch.tensor(stacked_group) for stacked_group in tiles_list]
    tensor_list = [tensor.unsqueeze(0).unsqueeze(0) for tensor in tensor_list]#在前面加两个维度


    a = 1
    out_prnu_list = []
    for data in tensor_list:
        print(a)
        a = a + 1
        data = torch.tensor(data)
        data = data.cuda()
        with torch.no_grad():
            INoisy = data
            with torch.no_grad():  # this can save much memory
                img_PRNU = model_dncnn(INoisy)
            PRNU_PATCH = img_PRNU.cpu().numpy().squeeze()
            # print(PRNU_PATCH.shape)
            out_prnu_list.append(PRNU_PATCH)

    for prnu_patch in out_prnu_list:
        print(prnu_patch.shape)

    PRNU = stitch_tiles(out_prnu_list,tile_size,gray_image.shape)

    return PRNU, os.path.splitext(os.path.basename(image_file))[0]  # 移除文件扩展名


def main():
    # module_dncnn = DnCNN(channels=1)
    # module_dncnn.cuda()
    # module_dncnn.load_state_dict(torch.load('weights/dude_nat_70.pth'))

    # SPNCNN_MSE
    moudle_spncnn = DnCNN_SPN(channels=1)
    moudle_spncnn = nn.DataParallel(moudle_spncnn, device_ids=[0]).cuda()
    pretrained_model = torch.load('weights/DnCNN_SPN_MSE.pth')
    moudle_spncnn.load_state_dict(pretrained_model)
    moudle_spncnn.eval()

    # base_dir = '/home/seamus20/VISION'
    # base_dir = '/home/seamus20/matlab_wyl/Dresden_SDI_train'
    base_dir = '/home/seamus20/matlab_wyl/Camera_Dresden'

    # save_dir = '/home/seamus20/matlab_wyl/VISION_every_image_npy_patch_500'
    save_dir = '/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras_SPNCNN'

    # base_dir = '/home/seamus20/matlab_wyl/VISION/D02_Apple_iPhone4s/images/nat/'

    Cameras_dir =sorted( [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))])
    Cameras_dir_sorted = Cameras_dir
    # Cameras_dir_sorted = Cameras_dir_sorted[26:]

    for Camera_dir_name in Cameras_dir_sorted:
            folder_path = os.path.join(base_dir, Camera_dir_name)
            print(Camera_dir_name)
            # image_path_pattern = os.path.join(folder_path, 'images', 'nat','*.jpg')
            image_path_pattern = os.path.join(folder_path, '*.JPG')

            jpg_files = sorted(glob.glob(image_path_pattern))
            for image_file in jpg_files:

                start_time = time.time()
                prnu, image_name = PRNU_Extract_pinjie(moudle_spncnn, image_file)
                end_time = time.time()
                print(f"处理图像 {image_name} 用时: {end_time - start_time} 秒")

                save_folder_path = os.path.join(save_dir, Camera_dir_name)

                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                save_path = os.path.join(save_folder_path, f"{image_name}.npy")  # 以图像名字命名的.mat文件
                np.save(save_path, prnu)

                # savemat(save_path, {'PRNU_image': prnu})


if __name__ == "__main__":
    main()

