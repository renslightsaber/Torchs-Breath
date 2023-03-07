import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_california_housing

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, df ):
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1:].values

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype = torch.float)
        y = torch.tensor(self.y[index], dtype = torch.float)
        return x, y
    

def prepare_loaders(df, ratio = .7, bs = 128):

    index_num = int(df.shape[0] * ratio)

    train = df[:index_num].reset_index(drop = True)
    valid = df[index_num:].reset_index(drop = True)

    train_ds = MyDataset(df = train)
    valid_ds = MyDataset(df = valid)

    train_loader = DataLoader(train_ds, batch_size= bs, shuffle = True)
    valid_loader = DataLoader(valid_ds, batch_size= bs, shuffle = False)

    print("DataLoader Completed")
    
    return train_loader, valid_loader
