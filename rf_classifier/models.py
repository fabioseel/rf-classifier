import math

import torch.nn as nn
import torch
from torchvision.transforms import Resize, InterpolationMode

class CNNClassifier(nn.Module):
    def __init__(self, size=(32,32), conv_layers = 5, conv_channels=32, kernel_size = 7, fc_layers=4, fc_nodes = 1024, n_classes=10):
        super().__init__()
        self.resize = Resize(size, antialias=True)
        self.convs = nn.Sequential(nn.Conv2d(3,conv_channels, kernel_size=kernel_size), nn.GELU())
        for _ in range(conv_layers-1):
            self.convs.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size))
            self.convs.append(nn.GELU())

        self.convs.append(nn.Flatten())

        x=torch.empty((1,3, size[0], size[1]))
        test_out = self.convs(x)

        self.fcs = nn.Sequential(nn.Linear(in_features=test_out.shape[1], out_features=fc_nodes),nn.GELU())
        for _ in range(fc_layers-2):
            self.fcs.append(nn.Linear(in_features=fc_nodes, out_features=fc_nodes))
            self.fcs.append(nn.GELU())

        self.fcs.append(nn.Linear(in_features=fc_nodes, out_features=n_classes))
        
    def forward(self, x):
        x = self.resize(x)
        x = self.fcs(self.convs(x))
        if not self.training:
            x = nn.functional.softmax(x, dim=-1)
        return x