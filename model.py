import math

import torch
import torch.nn as nn
import torch.nn.init as init


class DnCNN(nn.Module):
    def __init__(self, depth=17, ic=1, nc=64, bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(ic, nc, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(nc, nc, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(nc, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(nc, ic, kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)
        self._init_weight()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    # TODO 修改初始化方式
    def _init_weight(self):
        classname = self.dncnn.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(self.dncnn.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal(self.dncnn.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            # nn.init.uniform(m.weight.data, 1.0, 0.02)
            self.dncnn.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
            nn.init.constant(self.dncnn.bias.data, 0.0)