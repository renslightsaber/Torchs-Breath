import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_california_housing

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, df, days = 3):
        # df: pandas dataframe
        self.days = days
        
        # x
        self.x = df.iloc[:, 1:4].values # Open, High, Low
        self.x = self.x / np.max(self.x)

        # y
        self.y = df.iloc[:, 4:5].values # Close
        self.y = self.y / np.max(self.y)

    def __len__(self):
        # 전체 길이 정보를 반환합니다.
        return self.x.shape[0] - self.days

    def __getitem__(self, index):
        # index를 통해서 row 하나를 특정합니다.

        x = self.x[index: index + self.days] # index ~ index + 3 -> x: [3, 3]
        y = self.y[index + self.days] # index + 3 -> y: [1]

        return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.float)
    


def prepare_loaders(df, sl, ratio = .7, bs = 2*128):

    index_num = int(df.shape[0] * ratio)
    
    # train, valid split
    train = df[:index_num].reset_index(drop = True)
    valid = df[index_num:].reset_index(drop = True)
    
    # train_ds, valid_ds MyDataset(Dataset)
    train_ds = MyDataset(df = train, days = sl)
    valid_ds = MyDataset(df = valid, days = sl)

    # train_loader, valid_loader를 만들어줍니다.
    train_loader = DataLoader(train_ds, shuffle = False, batch_size = bs)
    valid_loader = DataLoader(valid_ds, shuffle = False, batch_size = bs)

    print(" DataLoader Completed ")
    return train_loader, valid_loader
