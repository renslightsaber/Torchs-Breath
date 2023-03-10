
import numpy as np

import torch
import torch.nn as nn


############ Positionwise FeedForward Layer ##############
class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 pf_dim,
                 dropout, 
                 device, 
                 ):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        # [bs, sl, hid_dim] ->  [bs, sl, hid_dim]
        
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        # [bs, sl, hid_dim] ->  [bs, sl, hid_dim]
        
    def forward(self, x):
       # x(=src): [bs, sl, hid_dim]
        
        x = self.fc1(x) 
        # [bs, sl, hid_dim] ->  [bs, sl, hid_dim]
        x = self.dropout(torch.relu(x))
        # ReLU()와 Dropout()으로 인해 Shape이 바뀌지 않습니다.
        # [bs, sl, hid_dim] ->  [bs, sl, hid_dim]
        x = self.fc2(x)
        # [bs, sl, hid_dim] ->  [bs, sl, hid_dim]
        
        return x
    
    
