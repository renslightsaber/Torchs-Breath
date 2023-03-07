import numpy as np

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, nodes = [16, 32]):
        super().__init__()

        # x: [bs, 3, 32, 32]
        
        input_channel = 3
        middle, out_channel = nodes[0], nodes[1]

        self.fc1 = nn.Conv2d(in_channels= input_channel, out_channels = middle, kernel_size= 3, stride = 1, padding = 1)
        # x: [bs, 3, 32, 32] -> [bs, 16, 32, 32]
        self.fc2 = nn.Conv2d(in_channels= middle, out_channels = out_channel, kernel_size= 3, stride = 1, padding = 1)
        #   [bs, 16, 32, 32] -> [bs, 32, 32, 32]
        self.pool = nn.MaxPool2d(2, 2)
        #   [bs, 32, 32, 32] -> [bs, 32, 16, 16]

        k = out_channel * 16 * 16

        self.seq = nn.Sequential(       
            nn.Linear(k, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        # x: [bs, 3, 32, 32]
        x =  self.pool(self.fc2(self.fc1(x)))
        x = torch.flatten(x, 1)
        return self.seq(x)