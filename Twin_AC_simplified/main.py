
import argparse
import os

import torch

import torchvision.datasets as dset
from data_loader import Load_numpy_data, Load_gray_data,ILSVRC_HDF5

from model_resnet32 import SA_Generator, SA_Discriminator
from Resnet import multi_resolution_resnet, Resnet_mi
from biggan_model import Generator,Discriminator
from tensorboard_logger import configure, log_value
import torch.optim as optim
import numpy as np
from inception import load_inception_net



from train_test import train_g
from torchvision import datasets, transforms
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enbaled = True

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

#
def args():
    FLAG = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
    FLAG.add_argument('--dataset', default='OMNIGLOT',type=str, help='CIFAR10| CIFAR100 | MNIST | CUB | VGGFACE | IMAGENET100 | MNIST_overlap')
    FLAG.add_argument('--savingroot', default='../result', help='path to saving.')
    FLAG.add_argument('--dataroot', default='../data', help='path to dataset.')
    FLAG.add_argument('--manual_seed', default=42, help='manual seed.')
    FLAG.add_argument('--image_size', default=32,type=int, help='image size.')
    FLAG.add_argument('--batch_size', default=200,type=int,help='batch size.')
    FLAG.add_argument('--num_workers', default=32, help='num workers.')
    FLAG.add_argument('--iter', default=100000, type=int, help='num epoches, suggest MNIST: 20, CIFAR10: 500')
    FLAG.add_argument('--num_epoches_c', default=20, type=int, help='num epoches for training classifier')
    FLAG.add_argument('--num_D_steps', default=4, type=int, help='num_D_steps.')
    FLAG.add_argument('--num_G_steps', default=1, type=int, help='num_D_steps.')
    FLAG.add_argument('--nc', default=3, help='channel of input image; gray:1, RGB:3')
    FLAG.add_argument('--num_classes', default=1622,type=int, help='10,100,1000')
    FLAG.add_argument('--C_w', default=0.5, type=float, help='weight of classifier')
    FLAG.add_argument('--nz', default=128, help='length of noize.')

    FLAG.add_argument('--model_type', default='big', help='network structure, "sa" and "big", biggan give you amazing result')
    FLAG.add_argument('--loss_type', default='Twin_AC', type=str, help='conditional loss funtion, Projection | Twin_AC | AC')
    FLAG.add_argument('--SN', default=True, type=bool,help='SN in G')
    arguments = FLAG.parse_args()
    return arguments


##############################################################################

assert torch.cuda.is_available(), '[!] CUDA required!'


def train_gan(opt):

    os.makedirs(os.path.join(opt.savingroot,opt.dataset,'images'), exist_ok=True)
    os.makedirs(os.path.join(opt.savingroot,opt.dataset,'chkpts'), exist_ok=True)

    #Build networ
    if opt.model_type == 'sa':
        AC = False
        if opt.loss_type == 'Projection':
            AC = False
        elif opt.loss_type == 'Twin_AC':
            AC = True
        elif opt.loss_type == 'AC':
            AC = True
        netd_g = nn.DataParallel(
            SA_Discriminator(n_class=opt.num_classes, nc=opt.nc, AC=AC, Resolution=opt.image_size,ch=64).cuda())
        netg = nn.DataParallel(SA_Generator(n_class=opt.num_classes, code_dim=opt.nz, nc=opt.nc, SN=opt.SN,
                                            Resolution=opt.image_size,ch=32).cuda())
    elif opt.model_type == 'big':
        AC = False
        if opt.loss_type == 'Projection':
            AC = False
        elif opt.loss_type == 'Twin_AC':
            AC = True
        elif opt.loss_type == 'AC':
            AC = True

        netd_g = nn.DataParallel(Discriminator(n_classes=opt.num_classes,resolution=opt.image_size,AC=AC).cuda())
        netg = nn.DataParallel(Generator(n_classes=opt.num_classes,resolution=opt.image_size,SN=opt.SN).cuda())

    if opt.data_r == 'MNIST':
        dataset = dset.MNIST(root=opt.dataroot, download=True, transform=tsfm)
    elif opt.data_r == 'CIFAR10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)
    elif opt.data_r == 'CIFAR100':
        dataset = dset.CIFAR100(root=opt.dataroot, download=True, transform=tsfm)
    elif opt.data_r == 'CUB':
        dataset = dset.ImageFolder(root='/home/yanwuxu/CUB_200_2011_processed/ImageNet/ImageNet/', transform=tsfm)
    elif opt.data_r == 'VGGFACE':
        dataset = ILSVRC_HDF5(root='../data/VGGFACE64.hdf5', transform=tsfm)
    elif opt.data_r == 'IMAGENET100':
        dataset = Load_numpy_data(root='../data/ImageNet100.pt', transform=tsfm)
    elif opt.data_r == 'MNIST_overlap':
        dataset = Load_gray_data(root='../data/overlap_MNIST.pt', transform=tsfm)
    elif opt.data_r == 'OMNIGLOT':
        dataset = dset.Omniglot('../result',transform=tsfm,download=True)#Load_gray_data(root='omniglot.pt', transform=tsfm)

    print('training_start')
    print(opt.loss_type)


    step = 0

    train_g(netd_g,netg,dataset,step,opt)

if __name__ == '__main__':

    opt = args()
    print(opt)
    opt.data_r = opt.dataset

    tsfm = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5], [0.5]),
    ])


    torch.cuda.manual_seed(opt.manual_seed)

    opt.dataset = opt.dataset + '_' + opt.loss_type+ '_' + str(opt.C_w)

    if not os.path.exists(os.path.join(opt.savingroot, opt.dataset)):
        os.makedirs(os.path.join(opt.savingroot, opt.dataset))

    configure(os.path.join(opt.savingroot, opt.dataset, 'logs'),
              flush_secs=5)

    train_gan(opt)

