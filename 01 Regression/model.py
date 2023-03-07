import numpy as np

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim = 8, output_dim =1, nodes = [6, 4, 2] ):
        super().__init__()
        
        self.start_linear = nn.Linear(input_dim, nodes[0])
        
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(nodes, nodes[1:])])
        self.af = nn.ReLU()
        
        for index in range(0, len(self.linears)):
            self.linears.insert(index *2, self.af)
    
        self.end_linear = nn.Linear(nodes[-1], output_dim) 
        
    def forward(self, x):
        
        # x: [bs, 8]
        
        x = self.start_linear(x)
        for layer in self.linears:
            x = layer(x)
        
        return self.end_linear(x)