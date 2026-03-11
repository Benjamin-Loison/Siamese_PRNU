import numpy as np

if __name__ == '__main__':
    def_dev_list = 'D01_Samsung_GalaxyS3Mini;D02_Apple_iPhone4s;D03_Huawei_P9;D04_LG_D290;D05_Apple_iPhone5c;D06_Apple_iPhone6;D07_Lenovo_P70A;D08_Samsung_GalaxyTab3;' \
                         'D10_Apple_iPhone4s;D11_Samsung_GalaxyS3;D12_Sony_XperiaZ1Compact;D14_Apple_iPhone5c;D15_Apple_iPhone6;D16_Huawei_P9Lite;D17_Microsoft_Lumia640LTE;' \
                         'D20_Apple_iPadMini;D21_Wiko_Ridge4G;D22_Samsung_GalaxyTrendPlus;D23_Asus_Zenfone2Laser;D25_OnePlus_A3000;D26_Samsung_GalaxyS3Mini;D27_Samsung_GalaxyS5;' \
                         'D28_Huawei_P8;D29_Apple_iPhone5;D30_Huawei_Honor5c;D31_Samsung_GalaxyS4Mini;D32_OnePlus_A3003;D33_Huawei_Ascend;D35_Samsung_GalaxyTabA;Praktica_DCZ5.9_0;' \
                         'Praktica_DCZ5.9_1;Praktica_DCZ5.9_2;Praktica_DCZ5.9_3;Praktica_DCZ5.9_4;Olympus_mju_1050SW_0;Olympus_mju_1050SW_1'
    list_dev = def_dev_list.split(';')
    train_num = test_num = valid_num = 0
    for item in list_dev:
        train_list = np.load('Noises_lists/train/%s.npy' % item).tolist()
        test_list = np.load('Noises_lists/test/%s.npy' % item).tolist()
        valid_list = np.load('Noises_lists/valid/%s.npy' % item).tolist()

        train_num += len(train_list)
        test_num += len(test_list)
        valid_num += len(valid_list)

    print('train_num: '+str(train_num))
    print('test_num '+str(test_num))
    print('valid_num '+str(valid_num))

