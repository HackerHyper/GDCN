## Generative Diffusion Contrastive Network for Multi-View Clustering
> **Authors:**
Jian Zhu, Xin Zou, Xi Wang, Ning Zhang, Bian Wu, Yao Yang, Ying Zhou, Lingfang Zeng, Chang Tang, Cheng Luo. 

This paper is submitted to the International Conference on Acoustics, Speech, and Signal Processing (ICASSP2026).
 [Generative Diffusion Contrastive Network for Multi-View Clustering](https://arxiv.org/abs/2509.09527)
 
## 1. Abstract
In recent years, Multi-View Clustering (MVC) has been significantly advanced under the influence of deep learning. By integrating heterogeneous data from multiple views, MVC enhances clustering analysis, making multi-view fusion critical to clustering performance. However, there is a problem of low-quality data in multi-view fusion. This problem primarily arises from two reasons: 1) Certain views are contaminated by noisy data. 2) Some views suffer from missing data. This paper proposes a novel Stochastic Generative Diffusion Fusion (SGDF) method to address this problem. SGDF leverages a multiple generative mechanism for the multi-view feature of each sample. It is robust to low-quality data. Building on SGDF, we further present the Generative Diffusion Contrastive Network (GDCN). Extensive experiments show that GDCN achieves the state-of-the-art results in deep MVC tasks. The source code is publicly available at https://github.com/HackerHyper/GDCN.

## 2.Requirements

pytorch==1.12.1

numpy>=1.21.6

scikit-learn>=1.0.2

## 3.Datasets

The Synthetic3d, Prokaryotic, and MNIST-USPS datasets are placed in "data" folder. The others dataset could be downloaded from [cloud](https://pan.baidu.com/s/1XNWW8UqTcPMkw9NpiKqvOQ). key: data

## 4.Usage

- an example for train a new model：

```bash
python train.py
```

- You can get the following output:

```bash
Epoch 285 Loss:14.258701
Epoch 286 Loss:14.277878
Epoch 287 Loss:14.265236
Epoch 288 Loss:14.249377
Epoch 289 Loss:14.250543
Epoch 290 Loss:14.231292
Epoch 291 Loss:14.204707
Epoch 292 Loss:14.229280
Epoch 293 Loss:14.224252
Epoch 294 Loss:14.226442
Epoch 295 Loss:14.225920
Epoch 296 Loss:14.204377
Epoch 297 Loss:14.204086
Epoch 298 Loss:14.188799
Epoch 299 Loss:14.208624
Epoch 300 Loss:14.205029
---------train over---------
Clustering results:
ACC = 0.9800 NMI = 0.9357 PUR=0.9800 ARI = 0.9506
```

  



## 5.Acknowledgments

Work&Code is inspired by [MFLVC](https://github.com/SubmissionsIn/MFLVC), [CONAN](https://github.com/Guanzhou-Ke/conan), [CoMVC](https://github.com/DanielTrosten/mvc) ... 

If you have any problems, contact me via qijian.zhu@outlook.com.

