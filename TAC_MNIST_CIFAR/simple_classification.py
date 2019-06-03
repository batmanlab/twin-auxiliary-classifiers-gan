import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

class simple_net(nn.Module):
    def __init__(self, num_classes=10,nc=3):
        super(simple_net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = spectral_norm(nn.Conv2d( 64,128, kernel_size=3, stride=2, padding=1, bias=False))
        self.layer2 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3 = spectral_norm(nn.Conv2d(256,512, kernel_size=3, stride=2, padding=1, bias=False))
        self.layer4 = nn.Conv2d(512,512, kernel_size=3, stride=2, padding=1, bias=False)
        self.aux_linear = nn.Linear(512, num_classes)

        self.optim = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)
        return c.squeeze(1)