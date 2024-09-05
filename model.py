import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch.autograd import Variable

class LocalAttention(nn.Module):
    def __init__(self, k_size=3):
        super(LocalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv1(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SpatialReconstruction(nn.Module):
    def __init__(self):
        super(SpatialReconstruction, self).__init__()
        self.depthwise_conv = DepthwiseConv2d(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.depthwise_conv(avg_out)
        x = self.dropout(x)
        return self.sigmoid(x) * x

class DepthwiseConv2d(nn.Module):
    def __init__(self, dim):
        super(DepthwiseConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=6, groups=dim, dilation=2)

    def forward(self, input):
        return self.depth_conv(input)

class SRLABlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=8, stype='normal'):
        super(SRLABlock, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.SR = SpatialReconstruction()
        self.nums = scale - 1 if scale > 1 else 1
        self.stype = stype

        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.convs = nn.ModuleList([nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False) for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.LA = LocalAttention(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.convs[0].in_channels, 1)

        for i in range(self.nums):
            sp = spx[i] if i == 0 or self.stype == 'stage' else sp + self.SR(spx[i])
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = torch.cat([out, sp], 1) if i != 0 else sp

        if self.scale != 1:
            out = torch.cat([out, spx[self.nums] if self.stype == 'normal' else self.pool(spx[self.nums])], 1)

        out = self.LA(self.bn3(self.conv3(out)))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)

class SRLARes2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=8, num_classes=2, loss='softmax'):
        super(SRLARes2Net, self).__init__()
        self.loss = loss
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = AngleLinear(128 * block.expansion, num_classes)

        if self.loss == 'softmax':
            self.cls_layer = nn.Sequential(nn.Linear(128 * block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        x = input
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = w.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(w) / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = torch.acos(cos_theta)
            k = (self.m * theta / 3.14159265).floor()
            phi_theta = (-1) ** k * cos_m_theta - 2 * k
        else:
            phi_theta = cos_theta

        cos_theta *= xlen.view(-1, 1)
        phi_theta *= xlen.view(-1, 1)
        return cos_theta, phi_theta
