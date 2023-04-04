import torch
import torch.nn as nn

from mmpl.registry import MODELS


@MODELS.register_module()
class LinearProj(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        super(LinearProj, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        layers = [nn.Linear(self.in_channels, self.out_channels)]

        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(self.out_channels, self.out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
