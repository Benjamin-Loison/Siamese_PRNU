import os
import cv2
import h5py
import glob
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from torch.autograd import Variable
from models_dude_nat import DnCNN
from models_dude_nat import Weight_Get_conv5_batch
import time
from utils import *
from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.WienerInDFT import WienerInDFT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#这个函数的主要目的是使用深度学习模型来估计图像的噪声，并可选择进行后处理以进一步提高噪声估计的质量。
def NoiseExtractDL(imorig, model, postprocess):
    Img = np.float32(imorig)/255.
    Img = np.expand_dims(Img, 0)
    Img = np.expand_dims(Img, 1)
    # the probe image is used as noisy image
    INoisy = torch.Tensor(Img)
    INoisy = Variable(INoisy)
    INoisy = INoisy.cuda()
    with torch.no_grad(): # this can save much memory
        Out = model(INoisy)
        # since we changed the output from denoised image into residual
    # Noisex = Out.cpu().numpy().squeeze()
    Noisex = Out
    Noisexnp = Noisex
    stdValue = np.std(Noisex)
    # remove NUA, we find it is only needed in the RP side
    if postprocess:
        Noisex = ZeroMeanTotal(Noisex)
        std = np.std(Noisex)
        Noisex = WienerInDFT(Noisex, std)

    return Noisex, Noisexnp, stdValue

def main():

    # dude_net
    moudle_dncnn = DnCNN(channels=1)
    moudle_dncnn.cuda()
    moudle_dncnn.load_state_dict(torch.load('logs_nat/dude_nat_70.pth'))
    # fusion_net
    module_fusion = Weight_Get_conv5_batch()
    module_fusion.cuda()
    module_fusion.load_state_dict(torch.load('Fusion_Weights/conv5_11_0.9290277777777779.pth'))

    def PRNU_Extract(Camera_dir_name):
        image_path_pattern = os.path.join('/home/seamus20/VISION', Camera_dir_name, 'images', 'flat', '*.jpg')
        jpg_files = sorted(glob.glob(image_path_pattern))
        # 初始化一个空列表，用于存储符合条件的横向图像的路径
        image_files = []
        # 遍历jpg_files，筛选出50张横向图像
        for file_path in jpg_files:
            img = cv2.imread(file_path)
            if img is not None:
                # 检查图像是否是横向的（宽度大于高度）
                if img.shape[1] > img.shape[0]:
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray_image = np.float32(gray_image) / 255.0
                    image_files.append(gray_image)
                    if len(image_files) == 50:
                        break
        gray_images = np.expand_dims(np.array(image_files), axis=1)
        print(gray_images.shape)  # 应该输出 (50, 1, 1936, 2592)
        INoisy = torch.Tensor(gray_images)
        INoisy = Variable(INoisy)
        INoisy = INoisy.cuda()
        print(INoisy.shape)
        out_img_list = []
        # 循环遍历每张图像
        for i in range(50):
            INoisy1=INoisy[i:i + 1, :, :, :]
            with torch.no_grad():
                img_PRNU = moudle_dncnn(INoisy1)
            out_img_list.append(img_PRNU)
        torch.cuda.empty_cache()
        out_img_batch = torch.cat(out_img_list, dim=0)
        weight1 = module_fusion(INoisy)
        ref1 = weight1 * out_img_batch
        PRNU = ref1.sum(dim=0, keepdim=True)
        PRNU = PRNU.cpu().numpy().squeeze()
        return PRNU

    # 定义要遍历的目录路径
    base_dir = '/home/seamus20/VISION'

    # 使用os.path.join来构建完整的路径，并检查每个路径是否为文件夹
    Cameras_dir = [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))]
    for Camera_dir_name in Cameras_dir:
        folder_path = os.path.join(base_dir, Camera_dir_name)  # 构建完整的文件夹路径
        start_time = time.time()
        PRNU = PRNU_Extract(folder_path)
        end_time = time.time()
        print(f"处理文件夹 {Camera_dir_name} 所需的时间: {end_time - start_time} 秒")
        if not np.isnan(PRNU):
            # 构建保存.mat文件的完整路径
            save_path = os.path.join('F:\\', os.path.basename(Camera_dir_name) + '.mat')
            # 将PRNU特征保存为.mat文件
            np.save(save_path, PRNU)

    # # 获取'E:\VISION'目录下所有以'D'开头的文件夹
    # Cameras_dir_pattern = os.path.join('/home/seamus20', 'VISION', 'D*')
    # Cameras_dir = [d for d in glob.glob(Cameras_dir_pattern) if os.path.isdir(d)]
    #
    # for Camera_dir_name in Cameras_dir:
    #     start_time = time.time()
    #     PRNU = PRNU_Extract(Camera_dir_name)
    #     end_time = time.time()
    #
    #     print(f"Extraction time for {Camera_dir_name}: {end_time - start_time} seconds")


    from PIL import Image

    def load_images(image_folder):
        images = []
        for img_path in os.listdir(image_folder):
            with Image.open(os.path.join(image_folder, img_path)) as img:
                img = img.convert('RGB')
                images.append(np.array(img))
        return np.array(images)

    def split_image_into_tiles(img, tile_size):
        tiles = []
        img_height, img_width = img.shape[:2]
        # 计算完整块的数量以及最后一个块的尺寸
        full_rows, full_cols = img_height // tile_size, img_width // tile_size
        last_row_height, last_col_width = img_height % tile_size, img_width % tile_size

        # 分割完整块
        for i in range(full_rows):
            for j in range(full_cols):
                tiles.append(img[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size])

        # 分割最后一个行和/或列的不完整块
        if last_row_height > 0:
            for j in range(full_cols):
                tiles.append(img[-full_row_height:, j * tile_size:(j + 1) * tile_size])
        if last_col_width > 0:
            for i in range(full_rows):
                tiles.append(img[i * tile_size:(i + 1) * tile_size, -last_col_width:])

        # 如果图像的最后一个块既不在行上也不在列上
        if last_row_height > 0 and last_col_width > 0:
            tiles.append(img[-full_row_height:, -last_col_width:])

        return np.array(tiles)

    def process_tiles(model, tiles):
        tiles_tensor = torch.tensor(tiles).float() / 255.0  # Normalize pixel values
        tiles_tensor = tiles_tensor.permute(0, 3, 1, 2)  # Reshape to (batch_size, channels, height, width)
        with torch.no_grad():
            features = model(tiles_tensor)  # Forward pass through the model
        return features.numpy()

    def stitch_tiles(features, img_height, img_width, tile_size):
        stitched_features = np.zeros((img_height, img_width, features.shape[-1]))
        idx = 0
        # 拼接完整块
        for i in range(img_height // tile_size):
            for j in range(img_width // tile_size):
                stitched_features[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size, :] = features[
                    idx]
                idx += 1

        # 拼接最后一个行的不完整块（如果存在）
        last_row_height = img_height % tile_size if img_height % tile_size > 0 else tile_size
        for j in range(img_width // tile_size):
            stitched_features[-last_row_height:, j * tile_size:(j + 1) * tile_size, :] = features[idx]
            idx += 1

        # 拼接最后一个列的不完整块（如果存在）
        last_col_width = img_width % tile_size if img_width % tile_size > 0 else tile_size
        for i in range(img_height // tile_size):
            stitched_features[i * tile_size:(i + 1) * tile_size, -last_col_width:, :] = features[idx]
            idx += 1

        # 拼接最后一个块（如果存在）
        if img_height % tile_size > 0 and img_width % tile_size > 0:
            stitched_features[-last_row_height:, -last_col_width:, :] = features[idx]

        return stitched_features

    def main(image_folder, tile_size):
        images = load_images(image_folder)
        features_list = []

        for img in images:
            tiles = split_image_into_tiles(img, tile_size)
            features = process_tiles(model, tiles)
            stitched_features = stitch_tiles(features, *img.shape[:2], tile_size)
            features_list.append(stitched_features)

        return features_list

    # 指定图像文件夹路径和分块大小
    image_folder = 'path_to_your_image_folder'
    tile_size = 500

    # 运行主函数
    features_list = main(image_folder, tile_size)

if __name__ == "__main__":
    main()
