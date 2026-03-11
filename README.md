markdown
# Siamese_PRNU: 基于孪生网络的PRNU相机指纹识别

## 项目简介
本项目提出了一种基于孪生网络（Siamese Network）的PRNU（Photo Response Non-Uniformity）相机指纹识别方法，旨在提升相机来源认证、图像篡改检测等场景下的识别精度与鲁棒性。

## 核心功能
- 实现PRNU指纹的提取与特征表示
- 基于孪生网络的PRNU特征匹配与相似度计算
- 提供完整的实验流程与性能评估代码
- 支持视频/图像场景下的相机指纹验证

## 代码结构
Siamese_PRNU/
├── CameraFingerprint/ # PRNU 指纹提取核心模块
│ ├── Filter.py # 滤波处理相关函数
│ └── Functions.py # 特征计算核心函数
├── Video_match/ # 视频场景下的匹配模块
│ └── SDI_Cross/ # 跨场景匹配工具集
├── Caculate_PCE_Dude_nat2.py # PCE（Peak to Correlation Energy）计算
├── DnCNN.py # 去噪网络相关实现
└── Concat_PCE_Values.py # PCE 结果拼接与分析
plaintext

## 运行环境
- Python 3.7+
- 依赖库：numpy, opencv-python, torch, scipy, matplotlib

## 快速运行
1. 安装依赖：
```bash
pip install numpy opencv-python torch scipy matplotlib
执行核心实验脚本（以 PRNU 提取与匹配为例）：
bash
运行
python CameraFingerprint/Functions.py
实验说明
本项目的实验数据集与评估指标均遵循相关领域的主流标准，实验结果已通过统计学验证，具体细节见投稿论文。
免责声明
本代码仅用于学术研究目的，相关成果已投稿至期刊进行评审，未经允许请勿用于商业用途。
plaintext

