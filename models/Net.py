import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.relu = F.relu6

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm)
        self.conv2 = ConvBlock(out_channels, out_channels, use_batchnorm=use_batchnorm)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return out


class Encoder(nn.Module):
    def __init__(self, channels_list):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.layers.append(ResidualBlock(channels_list[i], channels_list[i + 1]))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels_list):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i in range(len(channels_list) - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(channels_list[i], channels_list[i - 1], kernel_size=2, stride=2)
            )
            if i > 1:
                self.resblocks.append(
                    ResidualBlock(channels_list[i - 1] * 2, channels_list[i - 1])
                )

    def forward(self, features):
        x = features[-1]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            if i <= len(features) - 2:
                skip_connection = features[-2 - i]
                x = torch.cat([x, skip_connection], dim=1)
                x = self.resblocks[i](x)
        return x


class ResidualUNet(nn.Module):
    def __init__(self, channel_in):
        super(ResidualUNet, self).__init__()
        self.encoder = Encoder([channel_in, 64, 128, 256, 512])
        self.decoder = Decoder([3, 64, 128, 256, 512])

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        return x

