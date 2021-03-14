"""
Modifed from the original STM code https://github.com/seoungwugoh/STM
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

from model.propagation import mod_resnet


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class MaskRGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f, m, o):

        f = torch.cat([f, m, o], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        return x
 

class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv1 = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.skip_conv2 = ResBlock(up_c, up_c)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv2(self.skip_conv1(skip_f))
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_proj = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)
 
    def forward(self, x):  
        return self.key_proj(x), self.val_proj(x)
