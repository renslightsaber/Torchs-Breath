import numpy as np

import torch
import torch.nn as nn


## 03 RNN
class RNNModel(nn.Module):
    def __init__(self, 
                 input_size = 3,   # open, high, low
                 hidden_size = 100,  
                 num_layers= 3, 
                 sequence_length = 12, # sl = 12
                 device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = device

        # RNN 레이어 
        # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True, 
                          bidirectional = False)
        # input(=x)'s shape: [bs, sl, input_size]
        # input_h0 = [num_layers, bs, hidden_size]

        # output(=y)'s shape: [bs, sl, hidden_size]
        # output_h = [num_layers, bs, hidden_size]

        #  [bs, sl, hidden_size] -> [bs, sl * hidden_size]

        # 1) 3차원 -> 2차원
        k = self.sequence_length * self.hidden_size

        # 2) Use output's Last Sequence Length
        # k = 1 * self.hidden_size

        # Fully Connected Layer
        self.seq = nn.Sequential(
            nn.Linear(k, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 1)
            # [bs, k] -> [bs, 256] -> [bs, 1]
        )

    def forward(self, x):
         # x: [bs, sl, input_size]
        bs = x.shape[0]

        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        # h0: [num_layers, bs, hidden_size]
        output, h_n = self.rnn(x, h0)
        # output's shape: [bs, sl, hidden_size]
        # h_n = [num_layers, bs, hidden_size]
        
        # 1) 3차원 -> 2차원
        output = output.reshape(bs, -1) 
        #  [bs, sl, hidden_size] -> [bs, sl * hidden_size]

        # 2) Use output's Last Sequence Length
        # output = output[:, -1] # [bs, hidden_size]

        # [bs, k] -> [bs, 256] ->[bs, 1]
        y_pred = self.seq(output)
        # y_pred: [bs, 1]

        return y_pred
    

## 04 LSTM Version 01) dim: 3-> 2
class LSTMModelV1(nn.Module):
    def __init__(self, 
                 input_size = 3,   # open, high, low
                 hidden_size = 80, 
                 num_layers= 3,    
                 sequence_length = 7, # sl = 7
                 device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = device


        # LSTM 레이어 
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True, bidirectional = False)
        # input(=x)'s shape: [bs, sl, input_size]
        # input_h0 = [num_layers, bs, hidden_size]
        # input_c0 = [num_layers, bs, hidden_size]

        # output(=y)'s shape: [bs, sl, hidden_size]
        # output_h = [num_layers, bs, hidden_size]
        # output_c = [num_layers, bs, hidden_size]

        # 1) 3차원  -> 2차원
        k = self.sequence_length * self.hidden_size

        # 2) Use output's Last Sequence 
        # k = 1 * self.hidden_size

        # Fully Connected Layer
        self.seq = nn.Sequential(
           #  [bs, sl, hidden_size] -> [bs, sl * hidden_size]
            nn.Linear(k, 256), nn.LeakyReLU(), nn.Linear(256, 1)
            # [bs, k] -> [bs, 256] -> [bs, 1]
        )

    def forward(self, x):
         # x: [bs, sl, input_size]
        bs = x.shape[0]

        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        # h0: [num_layers, bs, hidden_size]
        c0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        # c0: [num_layers, bs, hidden_size]
        output, (h_n, c_n) = self.rnn(x, (h0, c0))
        # output's shape: [bs, sl, hidden_size]
        # h_n = [num_layers, bs, hidden_size]
        # c_n = [num_layers, bs, hidden_size]
        
        # 1) 3차원  -> 2차원
        output = output.reshape(bs, -1)

        # 2) Use output's Last Sequence 
        # output = output[:, -1] 
        # output's shape: [bs, hid_dim]

        # [bs, k] -> [bs, 256] ->[bs, 1]
        y_pred = self.seq(output)
        # y_pred: [bs, 1]

        return y_pred
    

## 05 LSTM Version 02) Use_output's_Last_Sequence
class LSTMModelV2(nn.Module):
    def __init__(self, 
                 input_size = 3,   # open, high, low
                 hidden_size = 80, 
                 num_layers= 3,    
                 sequence_length = 3, 
                 device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = device


        # LSTM 레이어 
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True, bidirectional = False)
        # input(=x)'s shape: [bs, sl, input_size]
        # input_h0 = [num_layers, bs, hidden_size]
        # input_c0 = [num_layers, bs, hidden_size]

        # output(=y)'s shape: [bs, sl, hidden_size]
        # output_h = [num_layers, bs, hidden_size]
        # output_c = [num_layers, bs, hidden_size]

        # 1) 3차원  -> 2차원
        # k = self.sequence_length * self.hidden_size

        # 2) Use output's Last Sequence 
        k = 1 * self.hidden_size

        # Fully Connected Layer
        self.seq = nn.Sequential(
           #  [bs, sl, hidden_size] -> [bs, sl * hidden_size]
            nn.Linear(k, 256), nn.LeakyReLU(), nn.Linear(256, 1)
            # [bs, k] -> [bs, 256] -> [bs, 1]
        )

    def forward(self, x):
         # x: [bs, sl, input_size]
        bs = x.shape[0]

        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        # h0: [num_layers, bs, hidden_size]
        c0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        # c0: [num_layers, bs, hidden_size]
        output, (h_n, c_n) = self.rnn(x, (h0, c0))
        # output's shape: [bs, sl, hidden_size]
        # h_n = [num_layers, bs, hidden_size]
        # c_n = [num_layers, bs, hidden_size]
        
        # 1) 3차원  -> 2차원
        # output = output.reshape(bs, -1)

        # 2) Use output's Last Sequence 
        output = output[:, -1] 
        # output's shape: [bs, hid_dim]

        # [bs, k] -> [bs, 256] ->[bs, 1]
        y_pred = self.seq(output)
        # y_pred: [bs, 1]

        return y_pred
    

## 06 GRU
class GRUModel(nn.Module):
    def __init__(self, 
                 input_size = 3,   # open, high, low
                 hidden_size = 100, 
                 num_layers= 2,    
                 sequence_length = 7, 
                 device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = device

        # GRU 레이어 
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.rnn = nn.GRU(input_size = input_size, hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True, bidirectional = False)
        # input(=x)'s shape: [bs, sl, input_size]
        # input_h0 = [num_layers, bs, hidden_size]

        # output(=y)'s shape: [bs, sl, hidden_size]
        # output_h = [num_layers, bs, hidden_size]

        # 1) 3차원  -> 2차원
        k = self.sequence_length * self.hidden_size

        # 2) Use output's Last Sequence 
        # k = 1 * self.hidden_size

        # Fully Connected Layer
        self.seq = nn.Sequential(
           #  [bs, sl, hidden_size] -> [bs, sl * hidden_size]
            nn.Linear(k, 256), nn.LeakyReLU(), nn.Linear(256, 1)
            # [bs, k] -> [bs, 256] -> [bs, 1]
        )

    def forward(self, x):
         # x: [bs, sl, input_size]
        bs = x.shape[0]

        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        # h0: [num_layers, bs, hidden_size]
        output, h_n = self.rnn(x, h0)
        # output's shape: [bs, sl, hidden_size]
        # h_n = [num_layers, bs, hidden_size]
        
        # 1) 3차원  -> 2차원
        output = output.reshape(bs, -1)

        # 2) Use output's Last Sequence 
        # output = output[:, -1] 
        # output's shape: [bs, hid_dim]

        # [bs, k] -> [bs, 256] ->[bs, 1]
        y_pred = self.seq(output)
        # y_pred: [bs, 1]

        return y_pred




## 07 Bidirectional RNN
class BiRNNModel(nn.Module):
    def __init__(self, 
                 input_size = 3,   # open, high, low
                 hidden_size = 100, 
                 num_layers= 3,   
                 sequence_length = 7,
                 device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = device

        # RNN 레이어 
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True, bidirectional = True)
        # input(=x)'s shape: [bs, sl, input_size]
        # input_h0 = [2*num_layers, bs, hidden_size]

        # output(=y)'s shape: [bs, sl, 2*hidden_size]
        # output_h = [2*num_layers, bs, hidden_size]
 
        # 1) 3차원  -> 2차원
        k = self.sequence_length * 2 * self.hidden_size

        # 2) Use output's Last Sequence 
        # k = 1 * 2 * self.hidden_size

        # Fully Connected Layer
        self.seq = nn.Sequential(
           #  [bs, sl, hidden_size] -> [bs, sl * hidden_size]
            nn.Linear(k, 256), nn.LeakyReLU(), nn.Linear(256, 1)
            # [bs, k] -> [bs, 256] -> [bs, 1]
        )

    def forward(self, x):
         # x: [bs, sl, input_size]
        bs = x.shape[0]

        h0 = torch.zeros(2 * self.num_layers, bs, self.hidden_size).to(self.device)
        # h0: [2*num_layers, bs, hidden_size]
        output, h_n = self.rnn(x, h0)
        # output's shape: [bs, sl, 2*hidden_size]
        # h_n = [2*num_layers, bs, hidden_size]

        # 1) 3차원  -> 2차원
        output = output.reshape(bs, -1)

        # 2) Use output's Last Sequence 
        # output = output[:, -1] 
        # output's shape: [bs, 2*hid_dim]


        # [bs, k] -> [bs, 256] ->[bs, 1]
        y_pred = self.seq(output)
        # y_pred: [bs, 1]

        return y_pred