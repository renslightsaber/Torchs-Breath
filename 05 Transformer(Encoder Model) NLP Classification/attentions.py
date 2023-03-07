import numpy as np

import copy

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads,
                 dropout,
                 device
                ):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
        self.attn = Attention(device, dropout)
        
        # Multi Head Attention 만들기 위한 곳이다. 
        # hid_dim이 n_heads로 나누었을 때, 나머지가 0이 되어아야 Multi Head가 만들어질 수 있다.
        # 그래서 assert로 다음과 같은 조건을 넣은 것이다.
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        # 1)
        self.linear_dim = (hid_dim, hid_dim)
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(*self.linear_dim)) for _ in range(4)])
        self.fc = nn.Linear(*self.linear_dim)
        
        
    def forward(self, q, k, v, mask = None):
    	# 2) q, k, v = src, src, src
        # 3) mask = src_mask
        bs = q.shape[0]
        
        # 4) 
        q, k, v = [l(x).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2) for l, x in zip(self.linears, (q, k, v))]
        
        # 5) Scale Dot Product
        x, attn_weights = self.attn(q, k, v, mask = mask)
        # x: [bs, n_heads, ql, head_dim]
        # attn_weights = [bs, n_heads, ql, kl]
        
        # 6) x Reshape
        x = x.transpose(1, 2).contiguous()
        x = x.view(bs, -1, self.hid_dim)
        # [bs, sl, d_model]

        # 7)
        x = self.fc(x)
        
        return x, attn_weights
    

class Attention:
    def __init__(self,
                 device,
                 dropout):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask = None):
        # q: [bs, n_heads, ql, head_dim]
        head_dim = q.shape[-1]
        
        # attn_score(attention score)
        attn_score = torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(torch.Tensor([head_dim])).to(self.device)
        # attn_score: [bs, n_heads, ql, kl]
        
        # masked fill
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e10)
        
        # attn_weights(attention weights)
        attn_weights = torch.softmax(attn_score, dim=-1)
        # attn_weights: [bs, n_heads, ql, kl]
        
        # x: Scale Dot Product Attention
        x = torch.matmul(self.dropout(attn_weights), v)
        # x: [bs, n_heads, ql, head_dim]
        
        return x, attn_weights
    
    # nn.Module을 상속받지 않았기에 다음과 같이 __call__함수를 만들어준다.
    # 그래서 nn.Module을 상속받은 Class에서 forward와 동일한 효과가 나도록 해주기 위함이다.
    def __call__(self, q, k, v, mask = None):
        return self.forward(q, k, v, mask= mask)