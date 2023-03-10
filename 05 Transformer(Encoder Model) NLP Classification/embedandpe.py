import numpy as np

import torch
import torch.nn as nn


############# Embed ##################
class Embed(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 device,
                 dropout):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
        self.embed = nn.Embedding(input_dim, hid_dim) 
        # [bs, sl] ->[bs, sl, hid_dim]
        self.scale = torch.sqrt(torch.Tensor([hid_dim])).to(device)
        
    def forward(self, x):
        # x: [bs, sl]
        
        return self.scale * self.embed(x) 
        # [bs, sl, hid_dim] shape으로 return
        
        

############### Positional Encoding ########################
from torch.autograd import Variable

class PositionalEncodingLayer(nn.Module):
    def __init__(self, 
                 max_len, # max_length, 
                 hid_dim, 
                 device,
                 dropout):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
        # 1)                                             # Shape
        pe = torch.zeros(max_len, hid_dim)               # [max_len, hid_dim]
        
        # 2) 
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        
        # 3)
        _2i = torch.arange(0, hid_dim, 2) 
        # _2i's Shape: [ int(hid_dim / 2) ]
        
        # 4)
        div_term = torch.exp( _2i * (-torch.log( torch.Tensor([10000.0]) ) / hid_dim ))
        # div_term's Shape: [ int(hid_dim / 2) ]
        
        pe[:, 0::2] = torch.sin(position * div_term) # 5)       
        pe[:, 1::2] = torch.cos(position * div_term) # 6)
        
        # 7)
        pe = pe.unsqueeze(0).to(device) # [max_len, hid_dim] -> [1, max_len, hid_dim]
        
        # 8) 매개 변수로 간주되지 않는 버퍼를 등록하는 데 사용
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        # 9) forward
        # return x + Variable(self.pe[:, :x.shape[1]], requires_grad = False)
        return x + torch.Tensor(self.pe[:, :x.shape[1]]).requires_grad_(False)

