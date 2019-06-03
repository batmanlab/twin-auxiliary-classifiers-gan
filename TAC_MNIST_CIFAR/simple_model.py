import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torch.optim as optim

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
    def __init__(self, name):
        self.name = name

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
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

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


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    # if module.bias is not None:
    #     module.bias.data.zero_()

    return spectral_norm(module)


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

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
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
                 activation=F.relu, upsample=True, downsample=False,SN=True,emb=None):
        super().__init__()

        gain = 2 ** 0.5

        self.emb = emb

        if SN == True:
            self.conv1 = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True),
                                       gain=gain)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True)


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

        if self.downsample:
            out = F.avg_pool2d(out, 2)



        return out


class G_MNIST(nn.Module):
    def __init__(self, code_dim=100, n_class=100,nc=3, SN=True,ch=256):
        super().__init__()
        if SN == True:
            self.lin_code = spectral_init(nn.Linear(code_dim, 4 * 4 * ch))
            self.embedding = spectral_init(nn.Embedding(n_class, ch))
        else:
            self.lin_code = nn.Linear(code_dim, 4 * 4 * ch)
            self.embedding = nn.Embedding(n_class, ch)

        self.ch = ch



        self.conv = nn.ModuleList([ConvBlock(ch, ch, n_class=n_class,SN=SN,emb = self.embedding),
                                   # ConvBlock(ch, ch, n_class=n_class,SN=SN),
                                   # SelfAttention(ch, SN=SN),
                                   ConvBlock(ch, ch, n_class=n_class, SN=SN,emb = self.embedding),
                                   ConvBlock(ch, ch, n_class=n_class,SN=SN,emb = self.embedding)])

        self.bn = nn.BatchNorm2d(ch)
        if SN == True:
            self.colorize = spectral_init(nn.Conv2d(ch, nc, [3, 3], padding=1))
        else:
            self.colorize = nn.Conv2d(ch, nc, [3, 3], padding=1)

        self.optim = optim.Adam(params=self.parameters(), lr=2e-4,
                                betas=(0.0, 0.99), weight_decay=0, eps=1e-8)


    def forward(self, input, class_id):
        out = self.lin_code(input)
        out = out.view(-1, self.ch, 4, 4)

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

class D_MNIST(nn.Module):
    def __init__(self, n_class=100,nc=3,SN=True,AC=False,ch=256):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True,SN=True):
            return ConvBlock(in_channel, out_channel,
                             bn=False,
                             upsample=False, downsample=downsample,SN=SN)
        self.AC = AC
        self.ch = ch

        gain = 2 ** 0.5

        if SN == True:
            self.pre_conv = nn.Sequential(spectral_init(nn.Conv2d(nc, ch, 3,
                                                                  padding=1),
                                                        gain=gain),
                                          nn.ReLU(),
                                          spectral_init(nn.Conv2d(ch, ch, 3,
                                                                  padding=1),
                                                        gain=gain),
                                          nn.AvgPool2d(2))
            self.pre_skip = spectral_init(nn.Conv2d(nc, ch, 1))
        else:
            self.pre_conv = nn.Sequential(nn.Conv2d(nc, ch, 3,padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(ch, ch, 3,padding=1),
                                          nn.AvgPool2d(2))
            self.pre_skip = nn.Conv2d(nc, ch, 1)

        self.conv = nn.Sequential(conv(ch, ch,SN=SN),
                                  conv(ch, ch,SN=SN),
                                  conv(ch, ch,SN=SN))
        if SN == True:
            self.linear = spectral_init(nn.Linear(ch, 1))
            self.embed = spectral_norm(nn.Embedding(num_embeddings=n_class, embedding_dim=ch))
            self.linear_mi = spectral_init(nn.Linear(ch, n_class))
            self.linear_c = spectral_init(nn.Linear(ch, n_class))

        else:
            self.linear = nn.Linear(ch, 1)
            self.embed = nn.Embedding(num_embeddings=n_class, embedding_dim=ch)
            self.linear_mi = nn.Linear(ch, n_class)
            self.linear_c = nn.Linear(ch, n_class)

        self.optim = optim.Adam(params=self.parameters(), lr=2e-4,
                            betas=(0.0, 0.99), weight_decay=0, eps=1e-8)


    def orthogonal_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                orthogonal_(m.weight.data)


    def forward(self, input,class_id):
        # self.orthogonal_weights()
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))

        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        out_c = self.linear_c(out)
        out_mi = self.linear_mi(out)

        if self.AC == False:
            embed = self.embed(class_id)
            out_linear += (out * embed).sum(1)

        return out_linear, out_mi, out_c


# class G_D(nn.Module):
#     def __init__(self, G, D):
#         super(G_D, self).__init__()
#         self.G = G
#         self.D = D
#
#     def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False):
#         # If training G, enable grad tape
#         with torch.set_grad_enabled(train_G):
#             # Get Generator output given noise
#             G_z = self.G(z, gy)
#             # Cast as necessary
#         # Split_D means to run D once with real data and once with fake,
#         # rather than concatenating along the batch dimension.
#         D_input = torch.cat([G_z, x], 0) if x is not None else G_z
#         D_class = torch.cat([gy, dy], 0) if dy is not None else gy
#         # Get Discriminator output
#         D_out = self.D(D_input, D_class)
#         if x is not None:
#             return torch.split(D_out, [G_z.shape[0], x.shape[0]])  # D_fake, D_real
#         else:
#             if return_G_z:
#                 return D_out, G_z
#             else:
#                 return D_out

class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D

  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):
    # If training G, enable grad tape
    with torch.set_grad_enabled(train_G):
      # Get Generator output given noise
      G_z = self.G(z, gy)
    D_input = torch.cat([G_z, x], 0) if x is not None else G_z
    D_class = torch.cat([gy, dy], 0) if dy is not None else gy
    # Get Discriminator output
    D_out, mi, cls = self.D(D_input, D_class)
    if x is not None:
        return D_out[:G_z.shape[0]], D_out[G_z.shape[0]:], mi, cls  # D_fake, D_real
    else:
        if return_G_z:
            return D_out, G_z, mi, cls
        else:
            return D_out, mi, cls

