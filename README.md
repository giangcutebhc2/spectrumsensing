# Introduction
This code is used for paper in Proceedings of The 12th International Symposium on Information and Communication Technology (SOICT 2023), December 7–8, 2023, Ho Chi Minh, Vietnam.

## Table of Contents

- [Abstract](#abstract)
- [Model Architecture](#modelarchitecture)
- [Training & Testing](#trainingtesting)
- [Results](#result)

## Abstract

This paper presents a deep learning approach for fifth-generation (5G) and Long-Term Evolution (LTE) signal discrimination, explicitly focusing on identifying modulated signals in next-generation wireless networks. The mixture of modulated signals, which is essentially difficult to discern in the form of a complex envelope, should be converted into a visually informative spectrogram image by applying the Fast Fourier transform (FFT). To segment spectral regions of 5G new radio (NR) and LTE in a spectrogram, we aptly improve DeepLabV3+, a deep encoder-decoder network for semantic segmentation, by incorporating an adaptive Atrous Spatial Pyramid Pooling (ASPP) block and an attention mechanism to accommodate intrinsic signal characteristics and amplify relevant features, respectively.
Besides increasing the learning efficiency in the encoder, the improvement enriches the recovery capability of crucial 5G and LTE details, thus resulting in more accurate signal identification in the spectrogram image. 
Relying on the simulation results benchmarked on a dataset consisting of spectral images containing both LTE and 5G signals, the new network demonstrated effectiveness when compared to the original version by increasing global accuracy, mean intersection-over-union (IoU), and mean boundary-F1-score (BFScore) up to 1.37%, 2.85% and 9.43% in that order. For medium SNR level, it can achieve 98.28% global accuracy and 96.66% mean IoU, while also showing robustness under various practical channel impairments.

## ModelArchitecture

![image](https://github.com/giangcutebhc2/spectrumsensing/assets/104675768/d4f9629d-7881-46e5-8647-d53fdfc3d699)

### Training & Testing

To train and test our trained model, please run the test file:
```sh
main.m
```
The model with the trained weights can be obtained through the commands:
```sh
load('improved_model.mat');
net = trainednetInfo{1,1};
```

## Results
![image](https://github.com/giangcutebhc2/spectrumsensing/assets/104675768/3797aa9c-611d-490a-a499-ee10edc5af8b)
