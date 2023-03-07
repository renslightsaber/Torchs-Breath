import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_california_housing

import torch
from torch.utils.data import Dataset, DataLoader

# import torchvision 
# from torchvision import datasets, transforms

# train_data = datasets.CIFAR10('./data', train = True, download=True, transform= transforms.ToTensor())

class MyDataset(Dataset):
    def __init__(self, 
                 data,  # = train_data.data,    # [50000, 3, 32, 32]
                 label, # = train_data.targets  # [50000]
                ):
        self.x = torch.tensor( data / 255. , dtype = torch.float).permute(0, 3, 1, 2) 
        # self.x: Float + Shape Dimension 수정 + Torch Tensor로
        self.y = torch.tensor(label, dtype = torch.long) 
        # self.y: list + Torch Tensor로

    def __len__(self):
        # 전체 길이 정보 반환
        return self.x.shape[0] # len(self.y)

    def __getitem__(self, index):
        # index 로 row 하나를 특정하고, x, y를 뱉어줍니다.
        return self.x[index], self.y[index]
    


def prepare_loaders(train_data, ratio = .6, bs = 128):
  
  datas = train_data.data
  labels = train_data.targets
  index_num = int(datas.shape[0] * ratio)
  
  # train, valid split
  train = datas[:index_num]
  valid = datas[index_num:]
  
  train_label = labels[:index_num] # 30000 labels
  valid_label = labels[index_num:] # 20000 labels
  
  # train_ds, valid_ds by MyDataset
  train_ds = MyDataset(data = train, label = train_label)
  valid_ds = MyDataset(data = valid, label = valid_label)
  
  # train_loader, valid_loader
  # DataLoader에서 batch단위 크기로 row들을 묶어서 뱉어줌
  train_loader = DataLoader(train_ds, batch_size = bs, shuffle = True)
  valid_loader = DataLoader(valid_ds, batch_size = bs, shuffle = False)
  
  print("DataLoader Completed")
  return train_loader, valid_loader
