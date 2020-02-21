import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss



def loss_gan_dis(dis_fake, dis_real):
  dis_fake = F.sigmoid(dis_fake)
  dis_real = F.sigmoid(dis_real)
  L1 = F.binary_cross_entropy(dis_real, torch.ones(dis_real.size()[0]).cuda())
  L2 = F.binary_cross_entropy(dis_fake, torch.zeros(dis_fake.size()[0]).cuda())
  return L1, L2

def loss_gan_gen(dis_fake):
  dis_fake = F.sigmoid(dis_fake)
  loss = F.binary_cross_entropy(dis_fake, torch.ones(dis_fake.size()[0]).cuda())
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

# generator_loss = loss_gan_gen
# discriminator_loss = loss_gan_dis