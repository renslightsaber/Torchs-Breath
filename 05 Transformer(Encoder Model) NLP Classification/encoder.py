import numpy as np

import torch
import torch.nn as nn

from embedandpe import *
from attentions import *
from positionwiseff import *

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 max_len,
                 hid_dim, 
                 pf_dim,
                 n_heads,
                 n_layers,
                 device,
                 dropout = .1):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
        # input data(=src)'s Shape: [bs, sl]
        # Embed와 PostionalEncodingLayer를 여기에 넣었다.
        
        self.embed = Embed(input_dim, hid_dim, device, dropout)
        # [bs, sl] -> [bs, sl, hid_dim]
        
        self.pe = PositionalEncodingLayer(max_len, hid_dim, device, dropout)
        # 위치 정보 부여
        # [bs, sl, hid_dim] -> [bs, sl, hid_dim]
        

        self.layers = nn.ModuleList([ EncoderLayer(hid_dim, 
                                                   pf_dim, 
                                                   n_heads, 
                                                   dropout, 
                                                   device) for _ in range(n_layers)])
        # self.layers = [EncoderLayer(...), EncoderLayer(...), ... ]
        
        
    def forward(self, src, src_mask):
        # input data(=src)'s Shape: [bs, sl]
        # src_mask: [bs, 1, 1, sl]
        
        src = self.pe(self.embed(src))
        # src: [bs, sl, hid_dim] -> [bs, sl, hid_dim]
        
        # self.layers = [EncoderLayer(...), EncoderLayer(...), ... ]
        for layer in self.layers:
            # layer = EncoderLayer(...)
            # src: [bs, sl, hid_dim] 
            src = layer(src, src_mask)
            # src가 layer 입력으로 들어가고 그에 대한 결과물을 src로 받는다.
            # for 문 안에서 이 src가 다시 layer의 입력으로 들어가게 된다.
            # 이런 방법으로 EncoderLayer 6개를 지나는 것이다. 
            
        return src
        # src: [bs, sl, hid_dim] 

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 pf_dim,
                 n_heads,
                 dropout, 
                 device 
                 ):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.attn_layer_norm = nn.LayerNorm(hid_dim)
        
        self.ff = PositionwiseFeedForwardLayer(hid_dim, pf_dim, dropout, device)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        
    def forward(self, src, src_mask):
    	# src: [bs, sl, hid_dim] 
        # src는 현재, Encoder에서 Positional Encoding Layer까지 지난 상태
        
        # Multi-Head Attention 
        _src, _ = self.self_attn(src, src, src, src_mask)
        # _src, _ = self.self_attn(query, key, value, mask)
        # Multi Head Attention 에서 Query, Key, Value 자리에 동일한 src가 들어간다. 
        # 문장 내의 토큰끼리 관계도를 연산하기 위함이다. 
       
        # _src: [bs, sl, hid_dim] 
        # src와 동일한 Shape
        
        # Layer Normalization after Multi-Head Attention
        src = self.attn_layer_norm(self.dropout(_src) + src)
        # Residual Connection(Skip connection)으로 만든 후 Layer Normalization
        # self.dropout(_src) + src: Residual Connection(Skip connection) 부분
        
        # Input
        # _src: [bs, sl, hid_dim] 
        #  src: [bs, sl, hid_dim] 
        
        # Output: 동일한 Shape
        # src: [bs, sl, hid_dim] 
        
        # PositionwiseFeedForwardLayer 
        _src = self.ff(src)
        
        # Input
        #  src: [bs, sl, hid_dim] 
        
        # Output: 동일한 Shape
        # _src: [bs, sl, hid_dim] 
        
        
        # Layer Normalization after PositionwiseFeedForwardLayer
        src = self.ff_layer_norm(self.dropout(_src) + src)
        # Residual Connection(Skip connection)으로 만든 후 Layer Normalization
        # self.dropout(_src) + src: Residual Connection(Skip connection) 부분
        
        # Input
        # _src: [bs, sl, hid_dim] 
        #  src: [bs, sl, hid_dim] 
        
        # Output: 동일한 Shape
        # src: [bs, sl, hid_dim] 
        
        
        return src
        # src: [bs, sl, hid_dim] 