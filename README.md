markdown
# PRNU-Adapted Deep Fingerprint Learning and Reparameterized Correlation for Camera Source Identification

## 🌟 Overview

Reliable Camera Source Identification (CSI) is a cornerstone of digital forensics, yet it remains challenging due to low signal-to-noise ratios (SNR) and spatial misalignments. **DAF-Net-Rep** addresses these issues through a unified framework that combines quality-aware fingerprint estimation with a geometry-robust matching architecture.

### Key Features

1. Adaptive Weighting (AWA-Net): Moves beyond rigid statistical estimation by using a pixel-level quality assessment mechanism to fuse fragmented residuals into stable device fingerprints.
2. Structural Re-parameterization: Employs a multi-branch system during training for rich feature extraction, which is then collapsed into a single $3\times3$ layer for ultra-fast inference.
3. Geometry Resilience: Built-in robustness against spatial shifts and cropping by operating in the correlation domain rather than raw pixel residuals.

## 🏗️ Architecture
The framework operates in two collaborative stages:
1. Fingerprint Estimation (DAF-Net):Utilizes a dual-branch residual extraction network (DFPRNU-Net) to capture fine-grained sensor patterns while suppressing content interference.
2. Source Matching (RepEfficientNet): A high-performance similarity network that processes cross-correlation maps to ensure shift-invariance.



### Evaluation

1.Extract the reference PRNU camera fingerprint
Run the following command to extract the reference PRNU fingerprint of the camera:
```bash
python EXTRACT_PRNU_ALL_IMAGE.py
```
2. PRNU feature matching and verification
Based on the extracted reference PRNU, perform matching and performance evaluation:
```bash
python ./Video_match/cnn_test_new.py
```
---

## 📜 Citation

If you find this work helpful for your research, please cite:

```bibtex
@article{li2026prnu,
  title={PRNU-Adapted Deep Fingerprint Learning and Reparameterized Correlation for Camera Source Identification},
  author={Li, Jian and Zou, Fei and Ma, Bin and Li, Xiaolong and Qian, Zhenxing and Gao, Bo},
  journal={Preprint submitted to Elsevier},
  year={2026}
}
```

