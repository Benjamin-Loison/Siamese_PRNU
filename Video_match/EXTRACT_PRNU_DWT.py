import glob
import time
import CameraFingerprint.Functions as Fu
import CameraFingerprint.Filter as Ft
import os
import numpy as np

####提取每张图像的PRNU，使用DWT


def main():
    base_dir = '/home/seamus20/matlab_wyl/Camera_Dresden'

    save_dir = '/home/seamus20/matlab_wyl/Dresden_every_image_npy_40cameras_DWT'


    Cameras_dir =sorted( [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))])
    Cameras_dir_sorted = Cameras_dir
    print(Cameras_dir_sorted)

    for Camera_dir_name in Cameras_dir_sorted:
            folder_path = os.path.join(base_dir, Camera_dir_name)
            print(Camera_dir_name)
            image_path_pattern = os.path.join(folder_path, '*.JPG')
            jpg_files = sorted(glob.glob(image_path_pattern))
            for image_file in jpg_files:
                image_name = os.path.basename(image_file)
                start_time = time.time()
                Noisex = Ft.NoiseExtractFromImage(image_file, sigma=2.0)
                Noisex = Fu.WienerInDFT(Noisex, np.std(Noisex))
                end_time = time.time()
                print(f"处理图像 {image_name} 用时: {end_time - start_time} 秒")
                save_folder_path = os.path.join(save_dir, Camera_dir_name)

                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                save_path = os.path.join(save_folder_path, os.path.splitext(image_name)[0] + '.npy')  # 以图像名字命名的.npy文件
                np.save(save_path, Noisex)
if __name__ == "__main__":
    main()