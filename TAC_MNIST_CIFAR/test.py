from biggan_model import Generator
import torch
import os
import torchvision

G = Generator(n_classes=10,no_optim=True,resolution=32).cuda()

dict_root = ('./G_ema.pth')
G.load_state_dict(torch.load(dict_root))

def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off

def denorm(x):
    return (x +1)/2

def test(netg):
    netg.eval()
    toggle_grad(netg,False)


    for i in range(10):
        fixed = torch.randn(10, 128).cuda()
        label = torch.ones(10).long().cuda()*i
        if i == 0:
            fixed_input = netg(fixed,label)
        else:
            fixed_input = torch.cat([fixed_input,netg(fixed,label)],dim=0)

    torchvision.utils.save_image(denorm(fixed_input.data), 'CIFAR10.jpg',nrow=10)
    toggle_grad(netg, True)

test(G)