# TAC-GAN

This is the official pytorch implemention of the NeurIPS2019 paper [Twin Auxiliary Classifiers GAN](https://arxiv.org/abs/1907.02690) by Mingming Gong*, Yanwu Xu*, Chunyuan Li, Kun Zhang, and Kayhan Batmanghelich

<p align="center">
  <img width="75%" height="%75" src="https://github.com/xuyanwu/TAC-GAN/blob/master/figure/tac_gan_scheme.png">
</p>

Visualize the biased reconstruction of AC-GAN and our TAC-GAN correction to this as well as Projection-GAN.

| Original |  TAC | AC | Projection
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](MOG/tac_gif_1D/o.gif)  |  ![](MOG/tac_gif_1D/twin_ac.gif) | ![](MOG/tac_gif_1D/ac.gif)  |  ![](MOG/tac_gif_1D/projection.gif)

## Experimemnts on real data
This implementation on cifar100 and Imagenet100 is based on [pytorch of BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) implementation

### Overlap MNIST with '0' '1' and '0' '2'

<p align="center">
  <img width="75%" height="%75" src="https://github.com/xuyanwu/TAC-GAN/blob/master/figure/overlap_MNIST.png">
</p>

### Cifar100 generation evaluation with Inception Score, FID and LPIPS, 32 resolution

<p align="center">
  <img width="75%" height="%75" src="https://github.com/xuyanwu/TAC-GAN/blob/master/figure/cifar100.png">
</p>

### IMANGENET1000 generated images, 128 resolution

<p align="center">
  <img width="75%" height="%75" src="https://github.com/xuyanwu/TAC-GAN/blob/master/figure/part_of_imagenet.png">
</p>

# To replicate our results of our NeuraIPS paper, do the follow:

## Simulation on MOG toy data
