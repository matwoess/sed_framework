# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, Softmax, MaxPool2d, Dropout


class DorferCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 1, n_kernels: int = 64, out_features: int = 18):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyper-parameters"""
        super(DorferCNN, self).__init__()

        layers = []
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=5, stride=2, padding=2))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2))
        layers.append(Dropout(p=0.3))
        self.layers1 = torch.nn.Sequential(*layers)

        layers = []
        n_kernels *= 2  # 128
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2))
        layers.append(Dropout(p=0.3))
        self.layers2 = torch.nn.Sequential(*layers)

        layers = []
        in_channels = n_kernels
        n_kernels *= 2  # 256
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2))
        layers.append(Dropout(p=0.3))
        self.layers2 = torch.nn.Sequential(*layers)

        layers = []
        in_channels = n_kernels
        n_kernels *= 2  # 256
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Dropout(p=0.3))
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Dropout(p=0.3))
        n_kernels = int(n_kernels * 1.5)  # 384
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Dropout(p=0.3))
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2))
        layers.append(Dropout(p=0.3))
        self.layers3 = torch.nn.Sequential(*layers)

        layers = []
        in_channels = n_kernels
        n_kernels = int(n_kernels * 4.0 / 3.0)  # 512
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2))
        layers.append(Dropout(p=0.3))
        self.layers4 = torch.nn.Sequential(*layers)

        layers = []
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2))
        layers.append(Dropout(p=0.3))
        self.layers5 = torch.nn.Sequential(*layers)

        layers = []
        in_channels = n_kernels
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=3, stride=1, padding=0))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Dropout(p=0.5))
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=1, stride=1, padding=0))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(ReLU())
        layers.append(Dropout(p=0.5))
        self.layers6 = torch.nn.Sequential(*layers)

        layers = []
        n_kernels = out_features  # 18
        layers.append(Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=1, padding=0, stride=1))
        layers.append(BatchNorm2d(n_kernels))
        layers.append(torch.nn.AvgPool2d(n_kernels))
        layers.append(Softmax())
        self.output_layer = Sequential(*layers)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.layers5(x)
        x = self.layers6(x)
        predictions = self.output_layer(x)
        return predictions
