import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
        )

        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, im, seg1, seg2, attn, time):
        h, w = im.shape[-2:]

        time = time.unsqueeze(2).unsqueeze(2)
        time = time.expand(-1, -1, h, w)

        x = torch.cat([im, seg1, seg2, attn, time], 1)

        x = self.conv1(x)

        r = self.conv2(x)
        x = self.relu(x + r)

        r = self.conv3(x)
        x = self.relu(x + r)

        x = self.final_conv(x)

        return x
