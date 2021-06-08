# Model architecture from https://www.sciencedirect.com/science/article/pii/S2405959520300989

import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
        )
    def forward(self, x):
        return self.model(x)


class ConvLSTM(nn.Module):
    def __init__(self, in_channels):
        super(ConvLSTM, self).__init__()
        self.layer1 = Conv(in_channels, 64)
        self.layer2 = nn.LSTM(64, hidden_size=64, bidirectional=True, batch_first=True)
        self.layer3 = nn.MultiheadAttention(128, 1)
        self.layer4 = nn.LSTM(128, hidden_size=64, bidirectional=True, batch_first=True)
        self.output = nn.Linear(128, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = x.permute(0, 2, 1)
        x = self.layer2(x)[0]
        x = self.layer3(x, x, x)[0]
        x = self.layer4(x)[0]
        x = self.output(x)
        x = x.permute(0, 2, 1)
        return x
