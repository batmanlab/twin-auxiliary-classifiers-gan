
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
            nn.Linear(10, 1),
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

            nn.Linear(1, 10),
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
G = G_guassian(nz=nz,num_classes=3).cuda()

D = D_guassian(num_classes=3).cuda()

C = D_guassian(num_classes=3).cuda()

MI = D_guassian(num_classes=3).cuda()

optg = optim.Adam(G.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
optd = optim.Adam(D.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
optc = optim.Adam(C.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
optmi = optim.Adam(MI.parameters(), lr=0.002,
                      betas=(0.5, 0.999))

data1 = torch.randn(128000).cuda()
data2 = torch.randn(128000).cuda()*2 + 3
data3 = torch.randn(128000).cuda()*3 + 4


# pd.DataFrame(data1.cpu().numpy()).plot(kind='density')
# # plt.xlim((-2, 6))
# # plt.show()
# pd.DataFrame(data2.cpu().numpy()).plot(kind='density')
# # plt.xlim((-3, 6))
# # plt.show()
# pd.DataFrame(torch.cat([data1,data2],dim=0).cpu().numpy()).plot(kind='density')
df1 = pd.DataFrame()
df2 = pd.DataFrame()

df1['score_{0}'.format(0)] = data1.cpu().numpy()
df1['score_{0}'.format(1)] = data2.cpu().numpy()
df1['score_{0}'.format(2)] = data3.cpu().numpy()
df2['score_{0}'.format(2)] = torch.cat([data1,data2,data3],dim=0).cpu().numpy()

fig, ax = plt.subplots(1,1)
for s in df1.columns:
    df1[s].plot(kind='kde')

for s in df2.columns:
    df2[s].plot(kind='kde')
plt.xlim((-3, 9))
fig.show()

def continus_cross_entropy(x, target):
    probt = F.softmax(x,dim=1)
    target = F.softmax(target,dim=1)

    out = torch.log(probt)
    loss = ((-target*out).sum(1)).mean()
    return loss

# plot_density(torch.cat([data1,data2],dim=0).cpu().numpy())

for i in range(1000):


    data = torch.cat([data1[128*i:128*i+128],data2[128*i:128*i+128],data3[128*i:128*i+128]],dim=0).unsqueeze(dim=1)

    label1 = torch.zeros(128).cuda()
    label2 = torch.ones(128).cuda()
    label3 = torch.ones(128).cuda() + 1

    label = torch.cat([label1,label2,label3],dim=0).long()


    _, predict_c = C(data)

    loss = F.cross_entropy(predict_c,label)

    print(loss)

    optc.zero_grad()
    loss.backward()
    optc.step()
for _ in range(20):
    for i in range(1000):

        for _ in range(1):
            data = torch.cat([data1[128 * i:128 * i + 128], data2[128 * i:128 * i + 128],data3[128 * i:128 * i + 128]], dim=0).unsqueeze(dim=1)


            ###D
            d_real, _ = D(data)

            z = torch.randn(256, nz).cuda()
            fake_label = torch.LongTensor(256).random_(3).cuda()
            fake_data = G(z, label=fake_label)
            d_fake, _ = D(fake_data)

            D_loss = F.binary_cross_entropy(d_real, torch.ones(384).cuda()) + F.binary_cross_entropy(d_fake,
                                                                                                     torch.zeros(
                                                                                                         256).cuda())

            optd.zero_grad()
            D_loss.backward()
            optd.step()

        ####Dmi
        z = torch.randn(256, nz).cuda()
        fake_label = torch.LongTensor(256).random_(3).cuda()
        fake_data = G(z, label=fake_label)
        _, mi_c = MI(fake_data)

        mi_loss = F.cross_entropy(mi_c,fake_label)
        optmi.zero_grad()
        mi_loss.backward()
        optmi.step()

        ####G

        if i % 10 == 0:
            z = torch.randn(256, nz).cuda()
            fake_label = torch.LongTensor(256).random_(3).cuda()
            fake_data = G(z, label=fake_label)
            d_fake, _ = D(fake_data)
            _, fake_cls = C(fake_data)
            _, mi_c = MI(fake_data)

            G_loss = F.binary_cross_entropy(d_fake, torch.ones(256).cuda())  - F.cross_entropy(mi_c,fake_label) # + F.cross_entropy(fake_cls,fake_label) #- continus_cross_entropy(fake_cls,fake_cls)

            optg.zero_grad()
            G_loss.backward()
            optg.step()

        print(D_loss)



z = torch.randn(10000,nz).cuda()
label = torch.zeros(10000).long().cuda()#torch.LongTensor(10000).random_(2).cuda()#
data1 = G(z=z,label=label).squeeze().cpu().detach()

# pd.DataFrame(generate_data1.cpu().detach().numpy()).plot(kind='density')
# plt.xlim((-3, 6))
# plt.show()

# plot_density(generate_data1.cpu().detach().numpy())

z = torch.randn(10000,nz).cuda()
label = torch.ones(10000).long().cuda()#torch.LongTensor(10000).random_(2).cuda()#
data2 = G(z=z,label=label).squeeze().cpu().detach()

z = torch.randn(10000,nz).cuda()
label = torch.ones(10000).long().cuda() + 1#torch.LongTensor(10000).random_(2).cuda()#
data3 = G(z=z,label=label).squeeze().cpu().detach()

# pd.DataFrame(generate_data2.cpu().detach().numpy()).plot(kind='density')
# plt.xlim((-3, 6))
# plt.show()
#
# pd.DataFrame(torch.cat([generate_data1,generate_data2],dim=0).cpu().detach().numpy()).plot(kind='density')
# plt.xlim((-3, 6))
# plt.show()

df1 = pd.DataFrame()
df2 = pd.DataFrame()

df1['score_{0}'.format(0)] = data1.numpy()
df1['score_{0}'.format(1)] = data2.numpy()
df1['score_{0}'.format(2)] = data3.numpy()
df2['score_{0}'.format(2)] = torch.cat([data1,data2,data3],dim=0).numpy()

fig, ax = plt.subplots(1,1)
for s in df1.columns:
    df1[s].plot(kind='kde')

for s in df2.columns:
    df2[s].plot(kind='kde')
plt.xlim((-3, 9))
fig.show()

# plot_density(generate_data2.cpu().detach().numpy())


