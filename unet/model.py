# Model architecture from https://arxiv.org/abs/2001.04689

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool1d(2),
            ConvBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=8, stride=2, padding=3)
        self.model = ConvBlock(in_channels + in_channels_skip, out_channels)

    def forward(self, x_skip, x):
        x = self.up(x)  
        diff = x_skip.size()[2] - x.size()[2]
        x = F.pad(x, (diff // 2, diff - diff // 2))  
        return self.model(torch.cat([x_skip, x], dim=1))


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Unet, self).__init__()
        n = 4
        self.input = ConvBlock(in_channels, n)
        self.down1 = Down(n, 2*n)
        self.down2 = Down(2*n, 4*n)
        self.down3 = Down(4*n, 8*n)
        self.down4 = Down(8*n, 16*n)
        self.up1 = Up(16*n, 8*n, 8*n)
        self.up2 = Up(8*n, 4*n, 4*n)
        self.up3 = Up(4*n, 2*n, 2*n)
        self.up4 = Up(2*n, n, n)
        self.output = nn.Conv1d(n, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        return self.output(x)
