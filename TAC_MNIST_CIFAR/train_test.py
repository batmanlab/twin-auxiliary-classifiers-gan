
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from tensorboard_logger import configure, log_value
import torchvision
import os
from tqdm import tqdm
from sn_net import G_D

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

def train_g(netd_g, netg,ema_, loader, epoch, step, opt):
    noise, fake_label = prepare_z_y(G_batch_size=opt.batch_size, dim_z=opt.nz, nclasses=opt.num_classes)

    G_D_net = G_D(G=netg, D=netd_g)
    train = GAN_training_function(G=netg, D=netd_g, GD=G_D_net, z_=noise, y_=fake_label, config=opt)

    if opt.gan_loss_type == 'hinge':
        losses.discriminator_loss = losses.loss_hinge_dis
        losses.generator_loss = losses.loss_hinge_gen
        print('using hinge loss')
    elif opt.gan_loss_type == 'dc_gan':
        losses.discriminator_loss = losses.loss_dcgan_dis
        losses.generator_loss = losses.loss_dcgan_gen
        print('using log likelyhood loss')

    pbar = tqdm(enumerate(loader), dynamic_ncols=True)

    for _, (image_c, label) in pbar:

        netg.train()
        netd_g.train()
        image_c = image_c.cuda()
        label = label.cuda()

        metrics = train(image_c, label)
        ema_.update(step)

        step = step + 1

        G_loss = metrics['G_loss']
        D_loss_real = metrics['D_loss_real']
        D_loss_fake = metrics['D_loss_fake']
        C_loss = metrics['C_loss']

        # print(', '.join(['%s : %+4.3f' % (key, metrics[key])
        #                    for key in metrics]), end=' ')


        pbar.set_description(
            (', '.join(['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]))
        )

        log_value('G_loss', G_loss, step)
        log_value('D_loss_real', D_loss_real, step)
        log_value('D_loss_fake', D_loss_fake, step)
        log_value('C_loss', C_loss, step)
        if step % 250 == 0:
            test(netg, step, opt)


    #######################
    # save image pre epoch
    #######################
    torch.save([netg.state_dict(),netd_g.state_dict()], os.path.join(opt.savingroot, opt.dataset, f'chkpts/g_{epoch:03d}.pth'))

    return step


def GAN_training_function(G, D, GD, z_, y_, config):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config.batch_size)
        y = torch.split(y, config.batch_size)
        counter = 0 

        # Optionally toggle D and G's "require_grad"

        toggle_grad(D, True)
        toggle_grad(G, False)


        for step_index in range(config.num_D_steps):
            # If accumulating gradients, loop multiple times before an optimizer step
            z_.sample_()
            y_.sample_()
            D_fake, D_real, mi, c_cls = GD(z_[:config.batch_size], y_[:config.batch_size],
                                           x[counter], y[counter], train_G=False)

            # Compute components of D's loss, average them, and divide by
            # the number of gradient accumulations
            D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
            C_loss = 0
            if config.loss_type == 'Twin_AC':
                C_loss += F.cross_entropy(c_cls[D_fake.shape[0]:], y[counter]) + F.cross_entropy(
                    mi[:D_fake.shape[0]], y_)
            elif config.loss_type == 'AC':
                C_loss += F.cross_entropy(c_cls[D_fake.shape[0]:], y[counter])
            D_loss = D_loss_real + D_loss_fake + C_loss
            D_loss.backward()
            counter += 1
            D.optim.step()

        # Optionally toggle "requires_grad"
        toggle_grad(D, False)
        toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()
        for step_index in range(config.num_G_steps):
            z_.sample_()
            y_.sample_()
            D_fake, mi, c_cls = GD(z_, y_, train_G=True)
            C_loss = 0
            MI_loss = 0
            if config.loss_type == 'AC' or config.loss_type == 'Twin_AC':
                C_loss = F.cross_entropy(c_cls, y_)
                if config.loss_type == 'Twin_AC':
                    MI_loss = F.cross_entropy(mi, y_)

            G_loss = losses.generator_loss(D_fake)
            C_loss = C_loss
            MI_loss = MI_loss
            # print(G_loss,C_loss,MI_loss)
            (G_loss + (C_loss - MI_loss)).backward()


            G.optim.step()

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item()),
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


