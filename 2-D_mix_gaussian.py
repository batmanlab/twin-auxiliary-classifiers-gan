
import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.utils as utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard_logger import configure, log_value

from torchvision import datasets, transforms

import numpy as np
import scipy.stats as st

def plot_2d_density(data):
    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -3, 6
    ymin, ymax = -2.5, 2.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    ## Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k')
    # Label plot
    # ax.clabel(cset, inline=1, fontsize=10)
    # ax.set_xlabel('Y1')
    # ax.set_ylabel('Y0')
    plt.axis('equal')
    plt.show()
    return fig


def plot_density(flights,binwidth=0.1):
    ax = plt.subplot(1,1,1)

    # Draw the plot
    ax.hist(flights, bins=int(180 / binwidth),
            color='blue', edgecolor='black')

    # Title and labels
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size=30)
    ax.set_xlabel('Delay (min)', size=22)
    ax.set_ylabel('Flights', size=22)


    plt.tight_layout()
    plt.show()



class G_guassian(nn.Module):

    def __init__(self, nz,num_classes=2):
        super(G_guassian, self).__init__()


        self.embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=nz)

        self.decode = nn.Sequential(

            nn.Linear(nz*2,10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            # nn.Tanh(),
            nn.Linear(10, 2),
        )


        self.__initialize_weights()

    def forward(self, z, label, output=None):

        input = torch.cat([z,self.embed(label)],dim=1)
        x = input.view(input.size(0), -1)
        output = self.decode(x)

        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class D_guassian(nn.Module):

    def __init__(self, num_classes=10):
        super(D_guassian, self).__init__()

        self.encode = nn.Sequential(

            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            # nn.Tanh(),
        )
        self.gan_linear = nn.Linear(10, 1)
        self.aux_linear = nn.Linear(10, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input):

        x = self.encode(input)
        x = x.view(-1, 10)
        c = self.aux_linear(x)

        s = self.gan_linear(x)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

nz = 2
G = G_guassian(nz=nz,num_classes=2).cuda()

D = D_guassian(num_classes=2).cuda()

C = D_guassian(num_classes=2).cuda()

MI = D_guassian(num_classes=2).cuda()

optg = optim.Adam(G.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
optd = optim.Adam(D.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
optc = optim.Adam(C.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
optmi = optim.Adam(MI.parameters(), lr=0.002,
                      betas=(0.5, 0.999))

gap = 1
if not os.path.exists('2d_'+str(gap)):
    os.makedirs('2d_'+str(gap))
root = '2d_'+str(gap) + '/'
data1 = torch.randn(128000,2).cuda()
data2 = torch.randn(128000,2).cuda()
data2[:,0] = data2[:,0] + gap


# fig = plot_2d_density(data1.cpu().numpy())
# fig.savefig(root+'o_1.png')
# fig = plot_2d_density(data2.cpu().numpy())
# fig.savefig(root+'o_2.png')
# fig = plot_2d_density(torch.cat([data1,data2],dim=0).cpu().numpy())
# fig.savefig(root+'o_mixture.png')


def continus_cross_entropy(x, target):
    probt = F.softmax(x,dim=1)
    target = F.softmax(target,dim=1)

    out = torch.log(probt)
    loss = ((-target*out).sum(1)).mean()
    return loss

# plot_density(torch.cat([data1,data2],dim=0).cpu().numpy())

for i in range(1000):


    data = torch.cat([data1[128*i:128*i+128],data2[128*i:128*i+128]],dim=0).unsqueeze(dim=1)

    label1 = torch.zeros(128).cuda()
    label2 = torch.ones(128).cuda()

    label = torch.cat([label1,label2],dim=0).long()


    _, predict_c = C(data)

    loss = F.cross_entropy(predict_c,label)

    print(loss)

    optc.zero_grad()
    loss.backward()
    optc.step()
p_M = torch.tensor([[0.8,0.2],[0.2,0.5]]).float().cuda()
for _ in range(40):
    for i in range(1000):

        for _ in range(1):
            data = torch.cat([data1[128 * i:128 * i + 128], data2[128 * i:128 * i + 128]], dim=0).unsqueeze(dim=1)

            label1 = torch.zeros(128).cuda()
            label2 = torch.ones(128).cuda()

            label = torch.cat([label1, label2], dim=0).long()

            ###D
            d_real, _ = D(data)

            z = torch.randn(256, nz).cuda()
            fake_label = torch.LongTensor(256).random_(2).cuda()
            fake_data = G(z, label=fake_label)
            d_fake, _ = D(fake_data)

            D_loss = F.binary_cross_entropy(d_real, torch.ones(256).cuda()) + F.binary_cross_entropy(d_fake,
                                                                                                     torch.zeros(
                                                                                                         256).cuda())

            optd.zero_grad()
            D_loss.backward()
            optd.step()

        ####Dmi
        # z = torch.randn(256, nz).cuda()
        # fake_label = torch.LongTensor(256).random_(2).cuda()
        # fake_data = G(z, label=fake_label)
        # _, mi_c = MI(fake_data)
        #
        # mi_loss = F.cross_entropy(mi_c,fake_label)
        # optmi.zero_grad()
        # mi_loss.backward()
        # optmi.step()

        ####G

        if i % 20 == 0:
            z = torch.randn(256, nz).cuda()
            fake_label = torch.LongTensor(256).random_(2).cuda()
            fake_data = G(z, label=fake_label)
            d_fake, _ = D(fake_data)
            _, fake_cls = C(fake_data)
            _, mi_c = MI(fake_data)

            G_loss = F.binary_cross_entropy(d_fake, torch.ones(256).cuda())  + F.cross_entropy(fake_cls,fake_label) #- F.cross_entropy(mi_c,fake_label) #- continus_cross_entropy(fake_cls,fake_cls)

            optg.zero_grad()
            G_loss.backward()
            optg.step()

        print(D_loss)



z = torch.randn(128000,nz).cuda()
label = torch.zeros(128000).long().cuda()#torch.LongTensor(10000).random_(2).cuda()#
data1 = G(z=z,label=label).squeeze().detach()

# pd.DataFrame(generate_data1.cpu().detach().numpy()).plot(kind='density')
# plt.xlim((-3, 6))
# plt.show()

# plot_density(generate_data1.cpu().detach().numpy())

z = torch.randn(128000,nz).cuda()
label = torch.ones(128000).long().cuda()#torch.LongTensor(10000).random_(2).cuda()#
data2 = G(z=z,label=label).squeeze().detach()

fig = plot_2d_density(data1.cpu().numpy())
fig.savefig(root+'ac_0.png')
fig = plot_2d_density(data2.cpu().numpy())
fig.savefig(root+'ac_1.png')
fig = plot_2d_density(torch.cat([data1,data2],dim=0).cpu().numpy())
fig.savefig(root+'ac_mixture.png')


