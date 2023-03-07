import numpy as np

import torch
import torch.nn as nn

from encoder import *

class TransformerEncoderModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 sl,
                 max_len,
                 hid_dim,
                 pf_dim, 
                 n_heads,
                 n_layers, 
                 dropout, 
                 device):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.enocder = Encoder(input_dim, max_len, hid_dim, pf_dim, n_heads, n_layers, device, dropout)
        # [bs, sl] -> [bs, sl, hid_dim]

        # [bs, sl, hid_dim] -> [bs, k]
        k = sl * hid_dim

        # [bs, k] -> [bs, 256] -> [bs, 4]
        self.seq = nn.Sequential(
            nn.Linear(k, 256), 
            nn.ReLU(), 
            nn.Linear(256, 4), 
            nn.LogSoftmax(dim=-1)
        )
    
    def make_mask(self, src):
        sl = src.shape[1]
        pad_mask = (src!=0).unsqueeze(1).unsqueeze(2)

        sub_mask = torch.tril( torch.ones((sl, sl), device = self.device)).bool()
        
        # return pad_mask
        # 원하면 pad_mask로 진행해도 된다.

        return pad_mask & sub_mask
        # x: [bs, sl] --> [bs, 1, 1, sl]

    def forward(self, src):
        # src: [bs, sl] from DataLoader
        bs = src.shape[0]

        src_mask = self.make_mask(src)
        # src_mask: [bs, 1, 1, sl]

        enc_src = self.enocder(src, src_mask)
        # enc_src: [bs, sl, hid_dim]

        enc_src = enc_src.reshape(bs, -1)
        # enc_src: [bs, sl, hid_dim] -> [bs, k]

        y = self.seq(enc_src)
        # y: [bs, 4]

        return y
