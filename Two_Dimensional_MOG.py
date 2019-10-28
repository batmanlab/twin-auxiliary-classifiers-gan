
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmd_metric import polynomial_mmd
import seaborn as sns
import numpy as np
import scipy.stats as st

def plot_2d_density(data,fig,distance):
    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -4, 9 + distance * 2
    ymin, ymax = -8, 8

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.contourf(xx, yy, f, cmap='Blues')

    ax.contour(xx, yy, f, colors='k')



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

    def __init__(self, num_classes=10,AC=True):
        super(D_guassian, self).__init__()

        self.AC = AC

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
        self.mi_linear = nn.Linear(10, num_classes)

        if not self.AC:
            self.projection = nn.Embedding(num_embeddings=num_classes,embedding_dim=10)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input,y=None):

        x = self.encode(input)
        x = x.view(-1, 10)
        c = self.aux_linear(x)

        mi = self.mi_linear(x)

        s = self.gan_linear(x)
        if not self.AC:
            s += (self.projection(y) * x).sum(dim=1, keepdim=True)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1), mi.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def multi_results(distance):
    time.sleep(distance * 3)
    nz = 2
    G = G_guassian(nz=nz, num_classes=3).cuda()

    D = D_guassian(num_classes=3).cuda()

    optg = optim.Adam(G.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002,
                      betas=(0.5, 0.999))

    distance = (distance + 2) / 2
    if os.path.exists(os.path.join('MOG','2D', str(distance) + '_2D')):
        pass
    else:
        os.makedirs(os.path.join('MOG','2D', str(distance) + '_2D'))
    save_path = os.path.join('MOG','2D', str(distance) + '_2D')

    data1 = torch.randn(128000,2).cuda()
    data2 = torch.randn(128000,2).cuda() * 2
    data2[:,0] += distance
    data3 = torch.randn(128000,2).cuda() * 3
    data3[:,0] += distance * 2

    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data1.cpu().numpy()[:,0], data1.cpu().numpy()[:, 1])
    sns.kdeplot(data2.cpu().numpy()[:,0], data2.cpu().numpy()[:, 1])
    sns.kdeplot(data3.cpu().numpy()[:,0], data3.cpu().numpy()[:, 1])
    fig.legend(["Class_0", "Class_1", "Class_2"])
    plt.xlim((-4, 9 + distance * 2))
    plt.ylim((-8,8))
    plt.title('Original')
    fig.savefig(save_path + '/o_conditional.eps')

    r_data = torch.cat([data1, data2, data3], dim=0).cpu().numpy()


    np.save(save_path + '/o_data', r_data)

    for _ in range(40):
        for i in range(1000):

            #####D step
            for _ in range(1):
                data = torch.cat(
                    [data1[128 * i:128 * i + 128], data2[128 * i:128 * i + 128], data3[128 * i:128 * i + 128]],
                    dim=0).unsqueeze(dim=1)
                label = torch.cat([torch.ones(128).cuda().long()*0, torch.ones(128).cuda().long()*1, torch.ones(128).cuda().long()*2],dim=0)

                ###D
                d_real, c, _ = D(data)

                z = torch.randn(256, nz).cuda()
                fake_label = torch.LongTensor(256).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, _, mi = D(fake_data)

                D_loss = F.binary_cross_entropy(d_real, torch.ones(384).cuda()) \
                         + F.binary_cross_entropy(d_fake, torch.zeros(256).cuda()) \
                         + F.cross_entropy(c, label) \
                         + F.cross_entropy(mi, fake_label)

                optd.zero_grad()
                D_loss.backward()
                optd.step()

            #####G step
            if i % 10 == 0:
                z = torch.randn(256, nz).cuda()
                fake_label = torch.LongTensor(256).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, c, mi = D(fake_data)

                G_loss = F.binary_cross_entropy(d_fake, torch.ones(256).cuda()) + F.cross_entropy(c,fake_label) - F.cross_entropy(mi, fake_label)

                optg.zero_grad()
                G_loss.backward()
                optg.step()


    z = torch.randn(10000, nz).cuda()
    label = torch.zeros(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data1_g = G(z=z, label=label).squeeze().cpu().detach()

    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data2_g = G(z=z, label=label).squeeze().cpu().detach()

    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda() + 1  # torch.LongTensor(10000).random_(2).cuda()#
    data3_g = G(z=z, label=label).squeeze().cpu().detach()

    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data1_g.numpy()[:, 0], data1_g.numpy()[:, 1])
    sns.kdeplot(data2_g.numpy()[:, 0], data2_g.numpy()[:, 1])
    sns.kdeplot(data3_g.numpy()[:, 0], data3_g.numpy()[:, 1])
    fig.legend(["Class_0", "Class_1", "Class_2"])
    plt.xlim((-4, 9 + distance * 2))
    plt.ylim((-8,8))
    plt.title('TAC')
    fig.savefig(save_path + '/twin_ac_conditional.eps')

    g_data = torch.cat([data1_g, data2_g, data3_g], dim=0).cpu().numpy()

    np.save(save_path + '/twin_ac_data', g_data)

    mean0_0, var0_0 = polynomial_mmd(data1_g.numpy(),
                                     data1.cpu().numpy())
    mean0_1, var0_1 = polynomial_mmd(data2_g.numpy(),
                                     data2.cpu().numpy())
    mean0_2, var0_2 = polynomial_mmd(data3_g.numpy(),
                                     data3.cpu().numpy())

    mean0, var0 = polynomial_mmd(g_data, r_data)

    # plot_density(generate_data2.cpu().detach().numpy())
    # ac
    G = G_guassian(nz=nz, num_classes=3).cuda()

    D = D_guassian(num_classes=3).cuda()

    optg = optim.Adam(G.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002,
                      betas=(0.5, 0.999))

    for _ in range(40):
        for i in range(1000):

            #####D step
            for _ in range(1):
                data = torch.cat(
                    [data1[128 * i:128 * i + 128], data2[128 * i:128 * i + 128], data3[128 * i:128 * i + 128]],
                    dim=0).unsqueeze(dim=1)
                label = torch.cat([torch.ones(128).cuda().long() * 0, torch.ones(128).cuda().long() * 1,
                                   torch.ones(128).cuda().long() * 2], dim=0)

                ###D
                d_real, c, _ = D(data)

                z = torch.randn(256, nz).cuda()
                fake_label = torch.LongTensor(256).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, _, mi = D(fake_data)

                D_loss = F.binary_cross_entropy(d_real, torch.ones(384).cuda()) \
                         + F.binary_cross_entropy(d_fake, torch.zeros(256).cuda()) \
                         + F.cross_entropy(c, label)

                optd.zero_grad()
                D_loss.backward()
                optd.step()

            #####G step
            if i % 10 == 0:
                z = torch.randn(256, nz).cuda()
                fake_label = torch.LongTensor(256).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, c, mi = D(fake_data)

                G_loss = F.binary_cross_entropy(d_fake, torch.ones(256).cuda()) + F.cross_entropy(c,fake_label)

                optg.zero_grad()
                G_loss.backward()
                optg.step()


    z = torch.randn(10000, nz).cuda()
    label = torch.zeros(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data1_g = G(z=z, label=label).squeeze().cpu().detach()

    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data2_g = G(z=z, label=label).squeeze().cpu().detach()

    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda() + 1  # torch.LongTensor(10000).random_(2).cuda()#
    data3_g = G(z=z, label=label).squeeze().cpu().detach()

    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data1_g.numpy()[:, 0], data1_g.numpy()[:, 1])
    sns.kdeplot(data2_g.numpy()[:, 0], data2_g.numpy()[:, 1])
    sns.kdeplot(data3_g.numpy()[:, 0], data3_g.numpy()[:, 1])
    fig.legend(["Class_0", "Class_1", "Class_2"])
    plt.xlim((-4, 9 + distance * 2))
    plt.ylim((-8,8))
    plt.title('AC')
    fig.savefig(save_path + '/ac_conditional.eps')

    g_data = torch.cat([data1_g, data2_g, data3_g], dim=0).cpu().numpy()

    np.save(save_path + '/ac_data', g_data)

    mean1_0, var1_0 = polynomial_mmd(data1_g.numpy(),
                                     data1.cpu().numpy())
    mean1_1, var1_1 = polynomial_mmd(data2_g.numpy(),
                                     data2.cpu().numpy())
    mean1_2, var1_2 = polynomial_mmd(data3_g.numpy(),
                                     data3.cpu().numpy())

    mean1, var1 = polynomial_mmd(g_data, r_data)

    ####projection

    G = G_guassian(nz=nz, num_classes=3).cuda()

    D = D_guassian(num_classes=3, AC=False).cuda()

    optg = optim.Adam(G.parameters(), lr=0.002,
                      betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002,
                      betas=(0.5, 0.999))

    for _ in range(40):
        for i in range(1000):

            #####D step
            for _ in range(1):
                data = torch.cat(
                    [data1[128 * i:128 * i + 128], data2[128 * i:128 * i + 128], data3[128 * i:128 * i + 128]],
                    dim=0).unsqueeze(dim=1)
                label = torch.cat([torch.ones(128).cuda().long() * 0, torch.ones(128).cuda().long() * 1,
                                   torch.ones(128).cuda().long() * 2], dim=0)

                ###D
                d_real, c, _ = D(data,label)

                z = torch.randn(256, nz).cuda()
                fake_label = torch.LongTensor(256).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, _, mi = D(fake_data, fake_label)

                D_loss = F.binary_cross_entropy(d_real, torch.ones(384).cuda()) \
                         + F.binary_cross_entropy(d_fake, torch.zeros(256).cuda())

                optd.zero_grad()
                D_loss.backward()
                optd.step()

            #####G step
            if i % 10 == 0:
                z = torch.randn(256, nz).cuda()
                fake_label = torch.LongTensor(256).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, c, mi = D(fake_data, fake_label)

                G_loss = F.binary_cross_entropy(d_fake, torch.ones(256).cuda())

                optg.zero_grad()
                G_loss.backward()
                optg.step()


    z = torch.randn(10000, nz).cuda()
    label = torch.zeros(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data1_g = G(z=z, label=label).squeeze().cpu().detach()

    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data2_g = G(z=z, label=label).squeeze().cpu().detach()

    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda() + 1  # torch.LongTensor(10000).random_(2).cuda()#
    data3_g = G(z=z, label=label).squeeze().cpu().detach()

    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data1_g.numpy()[:, 0], data1_g.numpy()[:, 1])
    sns.kdeplot(data2_g.numpy()[:, 0], data2_g.numpy()[:, 1])
    sns.kdeplot(data3_g.numpy()[:, 0], data3_g.numpy()[:, 1])
    fig.legend(["Class_0", "Class_1", "Class_2"])
    plt.xlim((-4, 9 + distance * 2))
    plt.ylim((-8,8))
    fig.savefig(save_path + '/projection_conditional.eps')

    g_data = torch.cat([data1_g, data2_g, data3_g], dim=0).cpu().numpy()


    np.save(save_path + '/projection_data', g_data)

    mean2_0, var2_0 = polynomial_mmd(data1_g.numpy(),
                                     data1.cpu().numpy())
    mean2_1, var2_1 = polynomial_mmd(data2_g.numpy(),
                                     data2.cpu().numpy())
    mean2_2, var2_2 = polynomial_mmd(data3_g.numpy(),
                                     data3.cpu().numpy())

    mean2, var2 = polynomial_mmd(g_data, r_data)

    result = [str(mean0_0) + ',' + str(var0_0),
              str(mean0_1) + ',' + str(var0_1),
              str(mean0_2) + ',' + str(var0_2),
              str(mean0) + ',' + str(var0),
              str(mean1_0) + ',' + str(var1_0),
              str(mean1_1) + ',' + str(var1_1),
              str(mean1_2) + ',' + str(var1_2),
              str(mean1) + ',' + str(var1),
              str(mean2_0) + ',' + str(var2_0),
              str(mean2_1) + ',' + str(var2_1),
              str(mean2_2) + ',' + str(var2_2),
              str(mean2) + ',' + str(var2)]

    file = open(save_path + '/result.text', 'w')

    for content in result:
        file.write(content + '\n')


if __name__ == '__main__':
    distance = 5
    multi_results(distance)