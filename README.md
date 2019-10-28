# TAC-GAN

This repository is by [Yanwu Xu](http://xuyanwu.github.io)
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our NeurIPS2019 paper [Twin Auxiliary Classifiers GAN](https://arxiv.org/abs/1907.02690) by [Mingming Gong*](https://mingming-gong.github.io/), [Yanwu Xu*](http://xuyanwu.github.io), [Chunyuan Li](http://chunyuan.li/), [Kun Zhang](http://www.andrew.cmu.edu/user/kunz1/), [and Kayhan Batmanghelich†](https://kayhan.dbmi.pitt.edu/)

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/twin_ac/blob/master/figure/tac_gan_scheme.png">
</p>

# Visualization on Mixture of Gaussian

Visualize the biased reconstruction of AC-GAN and our TAC-GAN correction to this as well as Projection-GAN.

| Original |  TAC | AC | Projection
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](MOG/tac_gif_1D/o.gif)  |  ![](MOG/tac_gif_1D/twin_ac.gif) | ![](MOG/tac_gif_1D/ac.gif)  |  ![](MOG/tac_gif_1D/projection.gif)

## Experimemnts on real data
This implementation on cifar100 and Imagenet100 is based on [pytorch of BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) implementation
To prepare for the env for running our code. cd the repository and run 

```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch``` (This should work directly)

```conda env create -f environment.yml``` (Alternative)

### Overlap MNIST with '0' '1' and '0' '2'

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/twin_ac/blob/master/figure/overlap_MNIST.png">
</p>

### Cifar100 generation evaluation with Inception Score, FID and LPIPS, 32 resolution

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/twin_ac/blob/master/figure/cifar100.png">
</p>

### IMANGENET1000 generated images, 128 resolution

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/twin_ac/blob/master/figure/part_of_imagenet.png">
</p>

# To replicate our results of our NeuraIPS paper, do the follow:

## Simulation on MOG toy data

```
MOG
├── MOG_visualization.ipynb - Notebook to run 1-D MOG.
├── One_Dimensional_MOG.py - Script to run 1-D MOG.
└── Two_Dimensional_MOG.py - Script to run 2-D MOG.
```

## Experiments on real data
For the real data experiments, the code is based on [pytorch BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).

### Training data preparation

FIrstly, you need to transfer imagenet1000 image to HDF5 file, follow the command of [pytorch BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) implementation

### Running on Cifar100 and Imagenet1000

```
MOG
├── TAC-BigGAN
   ├── scripts
      ├── twin_ac_launch_cifar100_ema.sh - Script to run TAC-GAN on cifar100
      ├── twin_ac_launch_BigGAN_ch64_bs256x8.sh - Script to run TAC-GAN on Imagenet1000
```

if you want to change the weight of auxiliary classifier, you can modify the '--AC_weight' arguments in 'twin_ac_launch_cifar100_ema.sh' script. The same for AC-GAN and Projection-GAN, change script to 'ac_launch_cifar100_ema.sh' and 'projection_launch_cifar100_ema.sh' respectively.

# Acknowledgments

This work was partially supported by NIH Award Number 1R01HL141813-01, NSF 1839332 Tripod+X, and SAP SE. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research. We were also grateful for the computational resources provided by Pittsburgh SuperComputing grant number TG-ASC170024.
