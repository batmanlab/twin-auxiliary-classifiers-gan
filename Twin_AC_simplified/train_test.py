
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from tensorboard_logger import configure, log_value
import torchvision
import os
from tqdm import tqdm
from model_resnet32 import G_D
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import losses

def denorm(x):
    return (x +1)/2

def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off

class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj

def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float32)


    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_ = y_.to(device, torch.long)
    return z_, y_

def train_g(netd_g, netg, dataset, step, opt):
    noise, fake_label = prepare_z_y(G_batch_size=opt.batch_size, dim_z=opt.nz, nclasses=opt.num_classes)

    G_D_net = G_D(G=netg, D=netd_g)
    train = GAN_training_function(G=netg, D=netd_g, GD=G_D_net, z_=noise, y_=fake_label, config=opt)

    data_loader = sample_data(dataset, opt)

    # pbar = tqdm(enumerate(loader), dynamic_ncols=True)
    pbar = tqdm(range(opt.iter), dynamic_ncols=True)

    for _ in pbar:

        image_c,label = next(data_loader)

        # real_data0 = image_c[label==0][:24]
        # real_data1 = image_c[label==1][:24]
        #
        # torchvision.utils.save_image(denorm(real_data0),'real0.jpg', nrow=24)
        # torchvision.utils.save_image(denorm(real_data1), 'real1.jpg', nrow=24)

        netg.train()
        netd_g.train()
        image_c = image_c.cuda()
        label = label.cuda()

        metrics = train(image_c, label)

        step = step + 1

        G_loss = metrics['G_loss']
        D_loss_real = metrics['D_loss_real']
        D_loss_fake = metrics['D_loss_fake']
        C_loss = metrics['C_loss']


        pbar.set_description(
            (', '.join(['itr: %d' % step]
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]))
        )

        log_value('G_loss', G_loss, step)
        log_value('D_loss_real', D_loss_real, step)
        log_value('D_loss_fake', D_loss_fake, step)
        log_value('C_loss', C_loss, step)
        if step % 250 == 0:
            test(netg, step, opt)
        if step % 1000 == 0:
            torch.save({'G': netg.module.state_dict(),
                        'D': netd_g.module.state_dict(),
                        'step':step}, os.path.join(opt.savingroot, opt.dataset, f'chkpts/g_{step:03d}.pth'))

            # print(opt.C_w)
            #
            # test_ac(netg,netd_c)


    #######################
    # save image pre epoch
    #######################

    return step

def sample_data(dataset,opt):
    loader = DataLoader(dataset, batch_size=opt.batch_size*opt.num_D_steps, shuffle=True, num_workers=opt.num_workers,drop_last=True, pin_memory=True)
    # print(len(loader))
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(dataset, batch_size=opt.batch_size*opt.num_D_steps, shuffle=True, num_workers=opt.num_workers,drop_last=True, pin_memory=True)
            loader = iter(loader)
            yield next(loader)


def GAN_training_function(G, D, GD, z_, y_, config):
    def train(x, y):
        G.module.optim.zero_grad()
        D.module.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config.batch_size)
        y = torch.split(y, config.batch_size)
        counter = 0

        # Optionally toggle D and G's "require_grad"

        toggle_grad(D, True)
        toggle_grad(G, False)

        for step_index in range(config.num_D_steps):
            z_.sample_()
            y_.sample_()
            D_fake, D_real,mi, c_cls = GD(z_[:config.batch_size], y_[:config.batch_size],
                                x[counter], y[counter], train_G=False)

            D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
            if config.loss_type == 'Twin_AC':
                D_loss = (D_loss_real + D_loss_fake) + config.C_w*(F.cross_entropy(c_cls[D_fake.shape[0]:] ,y[counter]) + F.cross_entropy(mi[:D_fake.shape[0]] ,y_))
            elif config.loss_type == 'AC':
                D_loss = (D_loss_real + D_loss_fake) + config.C_w*F.cross_entropy(c_cls[D_fake.shape[0]:] ,y[counter])
            else:
                D_loss = (D_loss_real + D_loss_fake)
            (D_loss).backward()
            counter += 1
            D.module.optim.step()

        # Optionally toggle "requires_grad"
        toggle_grad(D, False)
        toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.module.optim.zero_grad()

        for step_index in range(config.num_G_steps):
            z_.sample_()
            y_.sample_()
            D_fake, mi,c_cls = GD(z_[:config.batch_size], y_[:config.batch_size], train_G=True)#D(fake_img, y_)
            G_loss = losses.generator_loss(D_fake)

            C_loss = 0
            MI_loss = 0
            if config.loss_type == 'Twin_AC':

                MI_loss = F.cross_entropy(mi, y_)
                C_loss = F.cross_entropy(c_cls, y_)

                ((G_loss - config.C_w*MI_loss + config.C_w*C_loss)).backward()
            elif config.loss_type == 'AC':

                C_loss = F.cross_entropy(c_cls, y_)

                ((G_loss + config.C_w*C_loss)).backward()
            else:
                (G_loss).backward()

        G.module.optim.step()
        
        out = {'G_loss': G_loss,
               'D_loss_real': D_loss_real,
               'D_loss_fake': D_loss_fake,
               'C_loss': C_loss,
               'MI_loss': MI_loss}
        # Return G's loss and the components of D's loss.
        return out

    return train


def test(netg,step,opt):
    netg.eval()
    toggle_grad(netg,False)


    for i in range(opt.num_classes):
        fixed = torch.randn(10, opt.nz).cuda()
        label = torch.ones(10).long().cuda()*i
        if i == 0:
            fixed_input = netg(fixed,label)
        else:
            fixed_input = torch.cat([fixed_input,netg(fixed,label)],dim=0)

    torchvision.utils.save_image(denorm(fixed_input.data), os.path.join(opt.savingroot,opt.dataset,f'images/fixed_{step:03d}.jpg'),nrow=10)
    toggle_grad(netg, True)

def test_acc(model, test_loader):
    toggle_grad(model,False)
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda().long()
            # if torch.sum(target == 1) >= 1:
            #     plt.figure(1)
            #     plt.imshow(np.transpose((data[target == 1][0].cpu().numpy() + 1) / 2, [1, 2, 0]))
            #     plt.show()
            _,output = model(data)
            # test_loss += F.nll_loss(output, target).sum().item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            c= 800
            if torch.sum(pred == c) >= 1:
                for j in range(torch.sum(pred == c)):
                    plt.figure(1)
                    plt.imshow(np.transpose((data[pred.squeeze() == c][0].cpu().numpy() + 1) / 2, [1, 2, 0]))
                    plt.show()

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)*1.0))

    toggle_grad(model, True)

    return correct / len(test_loader.dataset)*1.0

def test_ac(g,c):
    toggle_grad(g,False)
    toggle_grad(c,False)

    noise, fake_label = prepare_z_y(G_batch_size=100, dim_z=128, nclasses=2)

    num_0 = 0.0
    num_1 = 0.0
    num_2 = 0.0

    for _ in range(100):
        noise.sample_()
        fake_label.sample_()
        fake_img = g(noise,fake_label)
        cls = c(fake_img)
        pred = cls.max(1, keepdim=True)[1]
        num_0 += torch.sum(pred==0).float()
        num_1 += torch.sum(pred==1).float()
        num_2 += torch.sum(pred == 2).float()

    print(float(num_0/10000.0),float(num_1/10000.0),float(num_2/10000.0))



