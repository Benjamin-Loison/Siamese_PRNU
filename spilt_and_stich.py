import os
import cv2
import h5py
import glob

from scipy.io import savemat

from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from torch.autograd import Variable
from models_dude_nat import DnCNN
from models_dude_nat import Weight_Get_conv5_batch
import time
from utils import *
from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.WienerInDFT import WienerInDFT
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

# def process_tiles(tiles, model):
#     processed_tiles = []
#     for tile in tiles:
#         tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).cuda()
#         with torch.no_grad():
#             output = model(tile_tensor)
#         processed_tile = output.cpu().numpy().squeeze(0).squeeze(0)
#         processed_tiles.append(processed_tile)
#     return np.array(processed_tiles)

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

def PRNU_Extract(Camera_dir_name, model_dncnn, model_fusion, tile_size):

    image_path_pattern = os.path.join('/home/seamus20/VISION', Camera_dir_name, 'images', 'flat', '*.jpg')
    jpg_files = sorted(glob.glob(image_path_pattern))
    image_files = []
    for file_path in jpg_files:
        img = cv2.imread(file_path)
        if img is not None:
            if img.shape[1] > img.shape[0]:
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # gray_image = np.expand_dims(np.float32(gray_image) / 255.0, axis=0)
                gray_image = np.float32(gray_image) / 255.0
                image_files.append(gray_image)
                if len(image_files) == 50:
                    break
    gray_images = np.array(image_files)

    # 对每张图像进行分块
    tiles_list = [split_image(img, tile_size) for img in gray_images]
    stacked_tiles = []
    # 计算 tiles_list 中图像块的最大数量
    max_num_tiles_per_image = max(len(tiles) for tiles in tiles_list)
    for i in range(max_num_tiles_per_image):
        tiles_at_index = [tiles[i] for tiles in tiles_list if i < len(tiles)]
        stacked_tiles.extend(tiles_at_index)
    # for tile in stacked_tiles:
    #     print("图像块的形状:", tile.shape)

    num_groups = len(stacked_tiles) // 50
    stacked_group_list = []
    for i in range(num_groups):
        start_idx = i * 50
        end_idx = start_idx + 50
        group_tiles = [stacked_tiles[idx] for idx in range(start_idx, end_idx)]
        stacked_group = np.stack(group_tiles, axis=0)
        stacked_group = np.expand_dims(stacked_group, axis=1)
        stacked_group_list.append(stacked_group)
    # for stacked_group in stacked_group_list:
    #     print("堆叠后数组的形状:", stacked_group.shape)
    tensor_list = [torch.tensor(stacked_group) for stacked_group in stacked_group_list]

    a=1
    out_prnu_list = []
    for tensor in tensor_list:
        print(a)
        a=a+1
        tensor = Variable(tensor)
        tensor = tensor.cuda()
        with torch.no_grad():
            out_img_list = []
            for i in range(50):
                INoisy = tensor[i:i + 1, :, :, :]
                with torch.no_grad():  # this can save much memory
                    img_PRNU = model_dncnn(INoisy)
                out_img_list.append(img_PRNU)
            torch.cuda.empty_cache()
            out_img_batch = torch.cat(out_img_list, dim=0)
            weight1 = model_fusion(tensor)
            ref1 = weight1 * out_img_batch
            PRNU_PATCH = ref1.sum(dim=0, keepdim=True)
            # print(PRNU_PATCH.shape)
            PRNU_PATCH = PRNU_PATCH.cpu().numpy().squeeze()
            print(PRNU_PATCH.shape)
            out_prnu_list.append(PRNU_PATCH)


    for prnu_patch in out_prnu_list:
        print(prnu_patch.shape)

    # print(gray_image.shape)
    # print(tile_size)
    # PRNU = [stitch_tiles(PRNU_PATCH,tile_size,gray_image.shape) for PRNU_PATCH in out_prnu_list]
    PRNU = stitch_tiles(out_prnu_list,tile_size,gray_image.shape)

    return PRNU

    # # 按位置处理所有图像块
    # processed_tiles = process_tiles(tiles_list, model_dncnn)
    # # 将处理后的小块拼接回原始图像大小的列表
    # stitched_images = [stitch_tiles(tiles, tile_size, img.shape[:2]) for img, tiles in zip(gray_images, processed_tiles)]

def main():
    # 加载模型
    module_dncnn = DnCNN(channels=1)
    module_dncnn.cuda()
    module_dncnn.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))

    module_fusion = Weight_Get_conv5_batch()
    module_fusion.cuda()
    module_fusion.load_state_dict(torch.load('Fusion_Weights/conv5_11_0.9290277777777779.pth'))

    base_dir = '/home/seamus20/VISION'
    Cameras_dir = [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))]
    for Camera_dir_name in Cameras_dir:
        folder_path = os.path.join(base_dir, Camera_dir_name)
        print(Camera_dir_name)
        start_time = time.time()
        PRNU = PRNU_Extract(folder_path, module_dncnn, module_fusion, tile_size=1000)
        end_time = time.time()
        print(f"处理文件夹 {Camera_dir_name} 用时: {end_time - start_time} 秒")
        save_path = os.path.join('/home/seamus20/matlab_wyl/VISION_PRNU_FUSION', os.path.basename(Camera_dir_name) + '.mat')
        savemat(save_path, {'PRNU': PRNU})

if __name__ == "__main__":
    main()

