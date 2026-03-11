import h5py
import os
import numpy as np
from models_dude_nat import DnCNN
import torch
from NoiseExtract_wyl import NoiseExtract_wyl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # 1 Load saved models
    print('Loading model ...\n')
    model_ADNet = DnCNN(channels=1)
    model_ADNet.cuda()
    pretrained_model = torch.load('logs_nat/dude_nat_70.pth')
    model_ADNet.load_state_dict(pretrained_model)
    model_ADNet.eval()

    #dataset
    data = h5py.File('Dresden_test/DRESDEN_TEST_LARGE_POSITIVE256_1_1600.mat', 'r')#large
    imgf = np.array(data['inputs'])
    print(f'len(imgf)={len(imgf)}')

    # Create an empty list to store Noisexnp_SPNCNN
    all_Noisexnp_SPNCNN = []
    batch_size = 40
    numofbatch = len(imgf) // batch_size
    print(f'numofbatch={numofbatch}')

    save_dir = 'REF_PRNU'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(numofbatch ):  #
        batch_begin = i * batch_size
        batch_end = batch_begin + batch_size
        batch_images = imgf[batch_begin:batch_end, :, :, :]

        with torch.no_grad():
            _, Noisexnp_SPNCNN, _ = NoiseExtract_wyl(batch_images, model_ADNet, False)
            print(f'Noisexnp_dude={np.shape(Noisexnp_SPNCNN)}')

        all_Noisexnp_SPNCNN.extend(Noisexnp_SPNCNN)
        print(f'all={np.shape(all_Noisexnp_SPNCNN)}')

        # 每500个批次保存一个文件
        if ((i+1)  % 400 == 0 and i+1 >= 400) or i == numofbatch-1:
            # Convert the list to a NumPy array
            all_Noisexnp_SPNCNN_array = np.array(all_Noisexnp_SPNCNN)
            # 保存 NumPy 数组到文件
            save_path = os.path.join(save_dir, f'Dresden_PRNU_dude_positive-256_{i+1}.npy')
            np.save(save_path, all_Noisexnp_SPNCNN_array)
            # 清空列表，准备保存下一个文件
            all_Noisexnp_SPNCNN = []

    # 合并所有保存的文件
    merged_array = np.concatenate(
        [np.load(os.path.join(save_dir, f'Dresden_PRNU_dude_positive-256_{i}.npy')) for i in range(400, numofbatch+1 , 400)])
    # 定义最终的保存路径
    final_save_path = os.path.join(save_dir, f'Dresden_PRNU_dude_256_merged_positive.npy')
    np.save(final_save_path, merged_array)


if __name__ == '__main__':
    main()



