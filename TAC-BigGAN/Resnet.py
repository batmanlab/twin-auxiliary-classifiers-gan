import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,nc=3,ch=64):
        super(ResNet, self).__init__()
        self.in_planes = ch

        self.conv1 = nn.Conv2d(nc, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.layer1 = self._make_layer(block, ch, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ch*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ch*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ch*8, num_blocks[3], stride=2)
        self.aux_linear = nn.Linear(ch*8*block.expansion, num_classes)

        self.optim = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)

        return c

class ResNet_64(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,nc=3,ch=64):
        super(ResNet_64, self).__init__()
        self.in_planes = ch

        self.conv1 = nn.Conv2d(nc, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.layer1 = self._make_layer(block, ch, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, ch * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ch * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ch * 8, num_blocks[3], stride=2)
        self.aux_linear = nn.Linear(ch * 8 * block.expansion, num_classes)

        self.optim = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)

        return c

class ResNet_128(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,nc=3,ch=64):
        super(ResNet_128, self).__init__()
        self.in_planes = ch

        self.conv1 = nn.Conv2d(nc, ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.layer1 = self._make_layer(block, ch, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, ch * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ch * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ch * 8, num_blocks[3], stride=2)
        self.aux_linear = nn.Linear(ch * 8 * block.expansion, num_classes)

        self.optim = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)

        return c

class Resnet_mi(nn.Module):
    def __init__(self, classifier,ch=64, num_classes=10):
        super(Resnet_mi, self).__init__()
        self.classifier = classifier

        self.mi = nn.Linear(ch * 8 , num_classes)

        self.optim = optim.Adam(params=self.mi.parameters(), lr=2e-4,betas=(0.0, 0.999))

    def forward(self, x):
        out = F.relu(self.classifier.bn1(self.classifier.conv1(x)))
        out = self.classifier.layer1(out)
        out = self.classifier.layer2(out)
        out = self.classifier.layer3(out)
        out = self.classifier.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.classifier.aux_linear(out)

        mi = self.mi(out)

        return c, mi



def ResNet18(nclass,nc):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=nclass,nc=nc)

def ResNet18_64(nclass,nc):
    return ResNet_64(BasicBlock, [2,2,2,2],num_classes=nclass,nc=nc)

def ResNet34(nclass,nc):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=nclass,nc=nc)

def ResNet34_64(nclass,nc):
    return ResNet_64(BasicBlock, [3,4,6,3],num_classes=nclass,nc=nc)

def ResNet34_128(nclass,nc):
    return ResNet_128(BasicBlock, [3,4,6,3],num_classes=nclass,nc=nc)

def multi_resolution_resnet(resolution,nclass,nc=3,ch=64):
    if resolution == 32:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=nclass,nc=nc,ch=ch)
    elif resolution == 64:
        return ResNet_64(BasicBlock, [3, 4, 6, 3], num_classes=nclass,nc=nc,ch=ch)
    elif resolution == 128:
        return ResNet_128(BasicBlock, [3, 4, 6, 3], num_classes=nclass,nc=nc,ch=ch)
