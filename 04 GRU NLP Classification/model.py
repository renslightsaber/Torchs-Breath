import numpy as np
import pandas as pd

import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size, num_layers, sl, dropout, device):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.nl = num_layers
        self.hs = hidden_size
        self.sl = sl
                                 
        self.emb = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(input_size = emb_dim, 
                          hidden_size = self.hs, 
                          num_layers = self.nl, 
                          batch_first = True, 
                          bidirectional = False)
        # input's shape: [bs, sl, emb_dim]
        # input(h_0)'s shape: [nl, bs, hidden_size]
        
        # output's shape: [bs, sl, hidden_size]
        # output(h_out)'s shape: [nl, bs, hidden_size]

        # [bs, sl, hidden_size] -> [bs, k]
        k = sl * hidden_size
        
        # [bs, k] -> [bs, 128] -> [bs, 4]
        self.seq = nn.Sequential(nn.Linear(k, 128), nn.LeakyReLU(), nn.Linear(128, 4), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        # x: [bs, sl]

        x = self.emb(x)
        # [bs, sl] -> [bs, sl, emb_dim]
        
        # h_0 : [nl, bs, hidden_size]
        h_0 = torch.randn(self.nl, x.shape[0], self.hs).to(self.device)
        # h_0 = torch.randn(self.nl, x.shape[0], self.hs).to(self.device)
        # x: [bs, sl, emb_dim]
        output, h_n = self.rnn(x, h_0)
        # output's shape: [bs, sl, hidden_size]

        output = output.reshape(output.shape[0], -1)
        output = self.seq(output)
        # output: [bs, 4]
        
        return output
    
