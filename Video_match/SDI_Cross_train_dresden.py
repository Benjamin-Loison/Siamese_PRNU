import argparse
import logging
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # set the GPU device
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utility_dataset_dresden import load_prnu, load_res , preprocessing_function
from dataset_cross_new_dresden import DataGenerator
from SDI_Cross.pytorchtools import EarlyStopping
from sklearn.metrics import classification_report

# import SDI_Cross.utils_logger
from SDI_Cross import utils_logger

from SDI_Cross.arch.effnet_pconv_eca_rep import EfficientNet_pconv_eca_rep
from SDI_Cross.arch.resnet import resnet34
from SDI_Cross.arch.mobilenetv3 import mbv3_small
from mobilenet_v3_eca import mbv3_small_eca
from senet import SE_ResNet34



##训练匹配网络代码，Eff_repB0

seed = 42

def train_model(train_data_generator, valid_data_generator):

    # print('Starting training')
    logger.info('Starting training')

    # Build model
    if config.base_network == 'Eff_repB0':
        model = EfficientNet_pconv_eca_rep(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2, drop_connect_rate=0.2)
    elif config.base_network == 'ResNet':
        model = resnet34()
    elif config.base_network == 'MBNetV3':
        model = mbv3_small()
    elif config.base_network == 'MBNetV3_ECA':
        model = mbv3_small_eca()
    elif config.base_network == 'SE_ResNet34':
        model = SE_ResNet34()


    logger.info(model)
    criterion = nn.CrossEntropyLoss()
    # Move to GPU
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    os.makedirs(config.model_dir, exist_ok=True)  # Make model path

    early_stopping = EarlyStopping(patience=30, verbose=True)
    for epoch in range(1, config.num_epochs_train + 1):
        train_losses = []
        train_acc = []
        valid_losses = []
        valid_acc = []

        model.train()  # prep model for training
        for i in range(train_data_generator.__len__()):

            x_batch, y_batch = train_data_generator[i]
            x_batch, y_batch = torch.Tensor(x_batch).cuda(), torch.Tensor(y_batch).cuda()
            # print(x_batch)
            output = model(x_batch)
            # print(f'output={output}')
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            device_pred = [np.argmax(item, axis=0) for item in output.cpu().detach().numpy()]
            # print(f'device_pred={device_pred}')

            device_true = [np.argmax(item, axis=0) for item in y_batch.cpu().detach().numpy()]
            # print(f'device_true={device_true}')


            accuracy = classification_report(device_true, device_pred, output_dict=True)['accuracy']
            train_acc.append(accuracy)

        model.eval()
        for j in range(valid_data_generator.__len__()):
            x_batch, y_batch = valid_data_generator[j]
            x_batch, y_batch = torch.Tensor(x_batch).cuda(), torch.Tensor(y_batch).cuda()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            valid_losses.append(loss.item())
            device_pred = [np.argmax(item, axis=0) for item in output.cpu().detach().numpy()]
            device_true = [np.argmax(item, axis=0) for item in y_batch.cpu().detach().numpy()]
            accuracy = classification_report(device_true, device_pred, output_dict=True)['accuracy']
            valid_acc.append(accuracy)
        # calculate average loss over an epoch
        avg_train_loss = np.average(train_losses)
        avg_train_acc = np.average(train_acc)
        avg_valid_loss = np.average(valid_losses)
        avg_valid_acc = np.average(valid_acc)


        epoch_len = len(str(500))

        print_msg = (f'[{epoch:>{epoch_len}}/{500:>{epoch_len}}] ' +
                     f'train_loss: {avg_train_loss:.5f} ' +
                     f'train_acc: {avg_train_acc:.5f} ' +
                     f'valid_loss: {avg_valid_loss:.5f} ' +
                     f'valid_acc: {avg_valid_acc:.5f}')

        # print(print_msg)
        logger.info(print_msg)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_acc, model, save_path = config.model_dir, logger = logger)

        if early_stopping.early_stop:
            # print("Early stopping")
            logger.info("Early stopping")
            break

    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('checkpoint.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def_dev_list_train = 'D01_Olympus_mju_1050SW_0;D02_Olympus_mju_1050SW_1;D03_Olympus_mju_1050SW_2;D04_Olympus_mju_1050SW_3;D05_Olympus_mju_1050SW_4;'\
                        'D06_Panasonic_DMC-FZ50_0;D07_Panasonic_DMC-FZ50_1;D08_Panasonic_DMC-FZ50_2;D09_Nikon_D200_0;D10_Nikon_D200_1;'\
                        'D11_Pentax_OptioA40_0;D12_Pentax_OptioA40_1;D13_Pentax_OptioA40_2;D14_Pentax_OptioA40_3;'\
                        'D15_Praktica_DCZ5.9_0;D16_Praktica_DCZ5.9_1;D17_Praktica_DCZ5.9_2;D18_Praktica_DCZ5.9_3;D19_Praktica_DCZ5.9_4;'\
                        'D20_Ricoh_GX100_0;D21_Ricoh_GX100_1;D22_Ricoh_GX100_2;D23_Ricoh_GX100_3;D24_Ricoh_GX100_4;'\
                        'D25_Rollei_RCP-7325XS_0;D26_Rollei_RCP-7325XS_1;D27_Rollei_RCP-7325XS_2;'\
                        'D28_Kodak_M1063_0;D29_Kodak_M1063_1;D30_Kodak_M1063_2;D31_Kodak_M1063_3;D32_Kodak_M1063_4;'\
                        'D33_Samsung_L74wide_0;D34_Samsung_L74wide_1;D35_Samsung_L74wide_2;'\
                        'D36_Samsung_NV15_0;D37_Samsung_NV15_1;D38_Samsung_NV15_2;D39_Sony_DSC-H50_0;D40_Sony_DSC-H50_1'
    def_dev_list_valid = 'D01_Olympus_mju_1050SW_0;D02_Olympus_mju_1050SW_1;D03_Olympus_mju_1050SW_2;D04_Olympus_mju_1050SW_3;D05_Olympus_mju_1050SW_4;'\
                        'D06_Panasonic_DMC-FZ50_0;D07_Panasonic_DMC-FZ50_1;D08_Panasonic_DMC-FZ50_2;D09_Nikon_D200_0;D10_Nikon_D200_1;'\
                        'D11_Pentax_OptioA40_0;D12_Pentax_OptioA40_1;D13_Pentax_OptioA40_2;D14_Pentax_OptioA40_3;'\
                        'D15_Praktica_DCZ5.9_0;D16_Praktica_DCZ5.9_1;D17_Praktica_DCZ5.9_2;D18_Praktica_DCZ5.9_3;D19_Praktica_DCZ5.9_4;'\
                        'D20_Ricoh_GX100_0;D21_Ricoh_GX100_1;D22_Ricoh_GX100_2;D23_Ricoh_GX100_3;D24_Ricoh_GX100_4;'\
                        'D25_Rollei_RCP-7325XS_0;D26_Rollei_RCP-7325XS_1;D27_Rollei_RCP-7325XS_2;'\
                        'D28_Kodak_M1063_0;D29_Kodak_M1063_1;D30_Kodak_M1063_2;D31_Kodak_M1063_3;D32_Kodak_M1063_4;'\
                        'D33_Samsung_L74wide_0;D34_Samsung_L74wide_1;D35_Samsung_L74wide_2;'\
                        'D36_Samsung_NV15_0;D37_Samsung_NV15_1;D38_Samsung_NV15_2;D39_Sony_DSC-H50_0;D40_Sony_DSC-H50_1'

    parser.add_argument('--list_dev_train', type=str, default=def_dev_list_train)  # list of training devices
    parser.add_argument('--list_dev_valid', type=str, default=def_dev_list_valid)  # list of validation devices
    parser.add_argument('--image_size', type=int, default=128)  # minim um size of the image
    parser.add_argument('--crop_size', type=int, default=128)  # size of the central patch, to be cropped
    parser.add_argument('--random_cropping', type=int, default=40)  # 40 pixel for random cropping during training
    parser.add_argument('--model_dir', type=str, default='./weight_Eff_repB0_dresden/Pixle_128_128-spncnn')  # directory with CNN weights -101为cross2 -201为cross 601wei dresden
    parser.add_argument('--gpu', type=str, default='0')  # gpu to be used
    parser.add_argument('--num_epochs_train', type=int, default=500)  # number of training epochs
    parser.add_argument('--learning_rate', type=float, default=0.001)  # learning rate ***0.001效果最好
    parser.add_argument('--batch_size_train', type=int, default=80)  # batch size in training 必须是相机的两倍
    parser.add_argument('--batch_size_valid', type=int, default=80)  # batch size in validation
    parser.add_argument('--size_block', type=int, default=2)  # number of classes
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--base_network', type=str, default='Eff_repB0')  # CNN architecture to be used
                            # EffB0|Eff_repB0|ShuffV2|ResNet|MBVitxxs|MBNetV2|MBNetV3
                            # Eff_pcB0|Eff_pc_ecaB0
    config, _ = parser.parse_known_args()
    # if config.gpu is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu  # set the GPU device

    list_dev_train = def_dev_list_train.split(';')
    list_dev_valid = def_dev_list_valid.split(';')

    # setting log
    logger_name = config.base_network + '_' + str(config.image_size) + '_' + str(config.crop_size)
    utils_logger.logger_info(logger_name, os.path.join('./train_logs/', logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # print("Preparing train data loader")
    logger.info(config.base_network + '_' + str(config.image_size) + '_' + str(config.crop_size)+'_DWT')
    logger.info("Preparing train data loader")
    #
    train_prnu = [load_prnu(item, config.image_size)[None, :, :] for item in list_dev_train]
    print(f'train_prnu{train_prnu[0].shape}')      # train_prnu --> 36 1 720 720

    ##dude
    # train_data = [load_res(np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_FUSION/train/%s.npy' % item).tolist(), config.image_size) for item in list_dev_train]
    # dwt
    train_data = [load_res(np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_DWT/train/%s.npy' % item).tolist(), config.image_size) for item in list_dev_train]

    print(f'train_data{train_data[0].shape}')    # 36 N 720 720
    train_data_generator = DataGenerator(config.batch_size_train, train_prnu, train_data,
                                                   preprocessing_function0=preprocessing_function,
                                                   preprocessing_function1=preprocessing_function, random_crop=True,
                                                   size_crop=config.crop_size, horizontal_flip=True, rot_90=True,
                                                   seed=seed)
    train_data_generator.print_info(logger)

    # print("Preparing valid data loader")
    logger.info("Preparing valid data loader")

    valid_prnu = [load_prnu(item, config.image_size)[None, :, :] for item in list_dev_valid]
    ##dude
    # valid_data = [load_res(np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_FUSION/valid/%s.npy' % item).tolist(), config.image_size) for item in
    #               list_dev_valid]
    #dwt
    valid_data = [load_res(np.load('/home/seamus20/matlab_wyl/Train_nat_path_Dresden_40cameras_DWT/valid/%s.npy' % item).tolist(), config.image_size) for item in
                  list_dev_valid]

    # Generator of validation data
    valid_data_generator = DataGenerator(config.batch_size_valid, valid_prnu, valid_data,
                                                   preprocessing_function0=preprocessing_function,
                                                   preprocessing_function1=preprocessing_function, random_crop=True,
                                                   size_crop=config.crop_size, horizontal_flip=True, rot_90=True,
                                                   seed=seed + 1)

    valid_data_generator.print_info(logger)

    # training
    train_model(train_data_generator, valid_data_generator)
