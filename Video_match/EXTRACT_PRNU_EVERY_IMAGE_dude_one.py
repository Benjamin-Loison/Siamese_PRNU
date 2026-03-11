#此代码用于提取图像的测试PRNU并保存
import os
import cv2
import glob
from scipy.io import savemat
from torch.autograd import Variable
from models_dude_nat import DnCNN
import time
from utils import *




def PRNU_Extract_one( model_dncnn, image_file):#此代码用于图像可以直接进行提取而不需要进行拼接的
    img = cv2.imread(image_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image) / 255.0
    gray_image = np.expand_dims(gray_image, axis=0)
    gray_image = np.expand_dims(gray_image, axis=1)
    INoisy = torch.Tensor(gray_image)
    INoisy = Variable(INoisy)
    INoisy = INoisy.cuda()
    with torch.no_grad():
        img_PRNU = model_dncnn(INoisy)
    img_PRNU = img_PRNU.cpu().numpy().squeeze()
    return img_PRNU, os.path.splitext(os.path.basename(image_file))[0]  # 移除文件扩展名



def main():
    module_dncnn = DnCNN(channels=1)
    module_dncnn.cuda()
    module_dncnn.load_state_dict(torch.load('weights/dude_nat_70.pth'))

    base_dir = '/home/seamus20/VISION'
    save_dir = '/home/seamus20/matlab_wyl/VISION_every_image_npy'
    # base_dir = '/home/seamus20/matlab_wyl/VISION/D02_Apple_iPhone4s/images/nat/'

    Cameras_dir =sorted( [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))])
    Cameras_dir_sorted = Cameras_dir

    for Camera_dir_name in Cameras_dir_sorted:
            folder_path = os.path.join(base_dir, Camera_dir_name)
            print(Camera_dir_name)
            image_path_pattern = os.path.join(folder_path, 'images', 'flat','*.jpg')
            jpg_files = sorted(glob.glob(image_path_pattern))
            for image_file in jpg_files:

                start_time = time.time()
                prnu, image_name = PRNU_Extract_one(module_dncnn, image_file)
                end_time = time.time()
                print(f"处理图像 {image_name} 用时: {end_time - start_time} 秒")
                # 保存为以图像名字命名的.mat文件

                save_folder_path = os.path.join(save_dir, Camera_dir_name)
                if not os.path.exists(save_folder_path):
                    # 如果文件夹不存在，使用os.makedirs()创建文件夹
                    os.makedirs(folder_path)

                save_path = os.path.join(save_folder_path, f"{image_name}.npy")  # 以图像名字命名的.mat文件
                np.save(save_path, prnu)

                # savemat(save_path, {'PRNU_image': prnu})


if __name__ == "__main__":
    main()

