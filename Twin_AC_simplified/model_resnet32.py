import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torch.optim as optim
import math
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

import functools
from torch.autograd import Variable


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name,rate):
        self.name = name
        self.rate = rate

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / (sigma*self.rate)
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name,rate):
        fn = SpectralNorm(name,rate)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight',rate=1):
    SpectralNorm.apply(module, name,rate)

    return module


def spectral_init(module, gain=1,rate=1):
    init.xavier_uniform_(module.weight, gain)
    # if module.bias is not None:
    #     module.bias.data.zero_()

    return spectral_norm(module,rate=rate)


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=2 ** 0.5,SN=True):
        super().__init__()

        if SN == True:
            self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                       gain=gain)
            self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                     gain=gain)
            self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                       gain=gain)
        else:
            self.query = nn.Conv1d(in_channel, in_channel // 8, 1)
            self.key = nn.Conv1d(in_channel, in_channel // 8, 1)
            self.value = nn.Conv1d(in_channel, in_channel, 1)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out




class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()

        self.bn = SyncBN2d(in_channel, eps=1e-5, momentum=0.5, affine=False)#nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(n_class, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id,emb=None):
        out = self.bn(input)

        if emb is not None:
            gamma = emb(class_id)
            beta = gamma*1.0
            # print(1)
        else:
            embed = self.embed(class_id)
            gamma, beta = embed.chunk(2, 1)
            # print(2)



        gamma = 1 + gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False,SN=True,emb=None,rate=1.0):
        super().__init__()

        gain = 2 ** 0.5

        self.emb = emb

        if SN == True:
            self.conv1 = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True),
                                       gain=gain,rate=rate)
            self.conv2 = spectral_init(nn.Conv2d(out_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True),
                                       gain=gain,rate=rate)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True)
            self.conv2 = nn.Conv2d(out_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            if SN == True:
                self.conv_skip = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                         1, 1, 0),rate=rate)
            else:
                self.conv_skip = nn.Conv2d(in_channel, out_channel,
                                                         1, 1, 0)

            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.norm1 = ConditionalNorm(in_channel, n_class)
            self.norm2 = ConditionalNorm(out_channel, n_class)

    def forward(self, input, class_id=None):
        out = input

        if self.bn:
            out = self.norm1(out, class_id,emb=self.emb)
        out = self.activation(out)
        if self.upsample:
            out = F.upsample(out, scale_factor=2)
        out = self.conv1(out)
        if self.bn:
            out = self.norm2(out, class_id,emb=self.emb)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_skip(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


class SA_Generator(nn.Module):
    def __init__(self, code_dim=100, n_class=100,nc=3, SN=True,Resolution=64,ch=64,SA_Resolution=256):
        super().__init__()

        self.ch = ch
        rate=1.0
        if SN == True:
            self.lin_code = spectral_init(nn.Linear(code_dim, 4 * 4 * ch*4),rate=rate)
            self.embedding = spectral_init(nn.Embedding(n_class, ch*4),rate=rate)
        else:
            self.lin_code = nn.Linear(code_dim, 4 * 4 * ch*4)
            self.embedding = nn.Embedding(n_class, ch*4)

        layer_num = int(math.log(Resolution/4,2))
        SA_layer = int(math.log(SA_Resolution/4,2))
        self.conv = []

        for i in range(layer_num):
            self.conv.append(ConvBlock(ch*4, ch*4, n_class=n_class,SN=SN,emb = self.embedding))
            # if i+1 == SA_layer:
            #     self.conv.append(SelfAttention(ch*4, SN=SN))
            #     print('apply sa G')

        self.conv = nn.ModuleList(self.conv)


        self.bn = SyncBN2d(ch*4, eps=1e-5, momentum=0.5, affine=False)#nn.BatchNorm2d(ch*4)
        if SN == True:
            self.colorize = spectral_init(nn.Conv2d(ch*4, nc, [3, 3], padding=1),rate=rate)
        else:
            self.colorize = nn.Conv2d(ch*4, nc, [3, 3], padding=1)

        self.optim = optim.Adam(params=self.parameters(), lr=1e-4,
                                betas=(0.0, 0.999), eps=1e-8)
        self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                init.orthogonal_(module.weight)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, input, class_id):
        out = self.lin_code(input)
        out = out.view(-1, self.ch*4, 4, 4)

        for conv in self.conv:
            if isinstance(conv, ConvBlock):
                out = conv(out, class_id)

            else:
                out = conv(out)

        out = self.bn(out)
        out = F.relu(out)
        out = self.colorize(out)

        return F.tanh(out)

def orthogonal_(tensor, gain=1):

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor*1.0

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

class SA_Discriminator(nn.Module):
    def __init__(self, n_class=100,nc=3,SN=True,AC=False,Resolution=64,ch=64,SA_Resolution=256):
        super().__init__()

        self.ch = ch
        def conv(in_channel, out_channel, downsample=True,SN=True):
            return ConvBlock(in_channel, out_channel,
                             bn=False,
                             upsample=False, downsample=downsample,SN=SN)

        gain = 2 ** 0.5

        self.AC = AC

        if SN == True:
            self.pre_conv = nn.Sequential(spectral_init(nn.Conv2d(nc, ch*4, 3,
                                                                  padding=1),
                                                        gain=gain),
                                          nn.ReLU(),
                                          spectral_init(nn.Conv2d(ch*4, ch*4, 3,
                                                                  padding=1),
                                                        gain=gain),
                                          nn.AvgPool2d(2))
            self.pre_skip = spectral_init(nn.Conv2d(nc, ch*4, 1))
        else:
            self.pre_conv = nn.Sequential(nn.Conv2d(nc, ch*4, 3,padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(ch*4, ch*4, 3,padding=1),
                                          nn.AvgPool2d(2))
            self.pre_skip = nn.Conv2d(nc, ch*4, 1)

        layer_num = int(math.log(Resolution / 8, 2))
        SA_layer = int(math.log(SA_Resolution / 8, 2))

        self.conv = []
        for i in range(layer_num):
            # if layer_num-i == SA_layer:
                # self.conv.append(SelfAttention(ch * 4, SN=SN))
                # print('apply sa D')
            self.conv.append(conv(ch*4, ch*4, SN=SN))

        self.conv = nn.ModuleList(self.conv)


        if SN == True:


            if not self.AC:
                self.embed = spectral_norm(nn.Embedding(num_embeddings=n_class, embedding_dim=ch*4))

            self.linear = spectral_init(nn.Linear(ch*4, 1))
            self.linear_mi = spectral_init(nn.Linear(ch * 4, n_class))
            self.linear_c = spectral_init(nn.Linear(ch * 4, n_class))


        else:
            if not self.AC:
                self.embed = nn.Embedding(num_embeddings=n_class, embedding_dim=ch*4)
            self.linear = nn.Linear(ch * 4, 1)
            self.linear_mi = nn.Linear(ch * 4, n_class)
            self.linear_c = nn.Linear(ch * 4, n_class)

        self.optim = optim.Adam(params=self.parameters(), lr=4e-4,
                            betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
        self.init_weights()


    def orthogonal_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                orthogonal_(m.weight.data)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                init.orthogonal_(module.weight)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)


    def forward(self, input,class_id=None):
        # self.orthogonal_weights()
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))


        for conv in self.conv:
            out = conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        out_mi = self.linear_mi(out).squeeze(1)
        out_c = self.linear_c(out).squeeze(1)

        if not self.AC:
            embed = self.embed(class_id)
            out_projection = (out * embed).sum(1)
            return out_linear + out_projection,out_mi,out_c
        else:
            return out_linear,out_mi,out_c




class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            G_z = self.G(z, gy)
            # Cast as necessary
        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        D_input = torch.cat([G_z, x], 0) if x is not None else G_z
        D_class = torch.cat([gy, dy], 0) if dy is not None else gy
        # Get Discriminator output
        D_out,mi,c_cls = self.D(D_input, D_class)
        if x is not None:
            return D_out[:G_z.shape[0]], D_out[G_z.shape[0]:],mi,c_cls # D_fake, D_real
        else:
            if return_G_z:
                return D_out, G_z
            else:
                return D_out,mi,c_cls

