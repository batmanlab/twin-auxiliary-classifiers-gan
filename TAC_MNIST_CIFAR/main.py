
import argparse
import os

import torch

import torchvision.datasets as dset

from sn_net import SA_Generator32, SA_Discriminator32
from biggan_model import Generator,Discriminator,ema
from tensorboard_logger import configure
from train_test import train_g
from simple_model import D_MNIST,G_MNIST

from torchvision import transforms

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enbaled = True



#
def args():
    FLAG = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
    FLAG.add_argument('--dataset', default='MNIST', help='CIFAR10| CIFAR100 | MNIST')
    FLAG.add_argument('--savingroot', default='./result', help='path to saving.')
    FLAG.add_argument('--dataroot', default='./data', help='path to dataset.')
    FLAG.add_argument('--manual_seed', default=42, help='manual seed.')
    FLAG.add_argument('--image_size', default=32, help='image size.')
    FLAG.add_argument('--batch_size', default=100,help='batch size.')
    FLAG.add_argument('--num_workers', default=10, help='num workers.')
    FLAG.add_argument('--num_epoches', default=40, type=int, help='num epoches, suggest MNIST: 20, CIFAR10: 500')
    FLAG.add_argument('--num_D_steps', default=2, type=int, help='num_D_steps.')
    FLAG.add_argument('--num_G_steps', default=1, type=int, help='num_G_steps.')
    FLAG.add_argument('--nc', default=3, help='channel of input image; gray:1, RGB:3')
    FLAG.add_argument('--num_classes', default=10, help='10,100')
    FLAG.add_argument('--nz', default=128, help='length of noize.')

    FLAG.add_argument('--model_type', default='dc_gan', help='network structure,"dc_gan" "sa" and "big", biggan give you amazing result')
    FLAG.add_argument('--loss_type', default='Twin_AC', type=str, help='conditional loss funtion, Projection | Twin_AC | AC')
    FLAG.add_argument('--gan_loss_type', default='dc_gan', type=str,
                      help='gan loss funtion, hinge | dc_gan')
    arguments = FLAG.parse_args()
    return arguments


##############################################################################

assert torch.cuda.is_available(), '[!] CUDA required!'


def train_gan(opt):

    os.makedirs(os.path.join(opt.savingroot,opt.dataset,'images'), exist_ok=True)
    os.makedirs(os.path.join(opt.savingroot,opt.dataset,'chkpts'), exist_ok=True)

    if opt.data_r == 'MNIST':
        dataset = dset.MNIST(root=opt.dataroot, download=True, transform=tsfm)
        opt.nc = 1
        opt.num_classes = 10
    elif opt.data_r == 'CIFAR10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)
        opt.nc = 3
        opt.num_classes = 10
    elif opt.data_r == 'CIFAR100':
        dataset = dset.CIFAR100(root=opt.dataroot, download=True, transform=tsfm)
        opt.nc = 3
        opt.num_classes = 100

    AC = False
    if opt.loss_type == 'Projection':
        AC = False
    elif opt.loss_type == 'Twin_AC' or opt.loss_type == 'AC':
        AC = True

    #Build networ
    if opt.model_type == 'sa':
        netd_g = SA_Discriminator32(n_class=opt.num_classes,nc=opt.nc,AC=AC,ch=256).cuda()
        netg = SA_Generator32(n_class=opt.num_classes, code_dim=opt.nz,nc=opt.nc,ch=256).cuda()
        netg_ema = SA_Generator32(n_class=opt.num_classes, code_dim=opt.nz,nc=opt.nc,ch=256).cuda()
        ema_ = ema(netg, netg_ema)
    elif opt.model_type == 'dc_gan':
        netd_g = D_MNIST(n_class=opt.num_classes,nc=opt.nc,AC=AC,ch=256,SN=False).cuda()#SA_Discriminator32(n_class=opt.num_classes, nc=opt.nc, AC=AC, ch=64,SN=False).cuda()
        netg = G_MNIST(n_class=opt.num_classes, code_dim=opt.nz,nc=opt.nc,ch=256,SN=False).cuda()#SA_Generator32(n_class=opt.num_classes, code_dim=opt.nz, nc=opt.nc, ch=64,SN=False).cuda()
        netg_ema = G_MNIST(n_class=opt.num_classes, code_dim=opt.nz,nc=opt.nc,ch=256,SN=False).cuda()#SA_Generator32(n_class=opt.num_classes, code_dim=opt.nz, nc=opt.nc, ch=64,SN=False).cuda()
        ema_ = ema(netg, netg_ema)
    elif opt.model_type == 'big':

        netd_g = Discriminator(n_classes=opt.num_classes,resolution=opt.image_size,AC=AC).cuda()
        netg = Generator(n_classes=opt.num_classes,resolution=opt.image_size).cuda()
        netg_ema = Generator(n_classes=opt.num_classes,no_optim=True,resolution=opt.image_size).cuda()
        ema_ = ema(netg, netg_ema)

    print('training_start')
    print(opt.loss_type)
    step = 0

    for epoch in range(opt.num_epoches):
        print(f'Epoch {epoch:03d}.')

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size*opt.num_D_steps, shuffle=True, num_workers=opt.num_workers,drop_last=True)
        step = train_g(netd_g,netg,ema_,loader,epoch,step,opt)

if __name__ == '__main__':

    opt = args()
    print(opt)
    opt.data_r = opt.dataset



    if opt.data_r == 'MNIST':
        tsfm = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        tsfm = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    torch.cuda.manual_seed(opt.manual_seed)

    opt.dataset = opt.dataset + '_' + opt.loss_type

    if not os.path.exists(os.path.join(opt.savingroot, opt.dataset)):
        os.makedirs(os.path.join(opt.savingroot, opt.dataset))

    configure(os.path.join(opt.savingroot, opt.dataset, 'logs'),
              flush_secs=5)

    train_gan(opt)

