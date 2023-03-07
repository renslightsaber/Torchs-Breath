import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_california_housing

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, 
                 df, 
                 korbow, 
                 tokenizer, 
                 sl):
        self.x = df.reviews # str
        self.y = df.new_target.values

        self.tokenizer = tokenizer

        self.korbow = korbow
        self.sl = sl

    def __len__(self):
        return self.x.shape[0]

    def make_sentence(self, sentence):

        # sentence = "나는 학교에 간다"
        
        x = self.tokenizer(sentence) # list
        # x: ["나는", "학교", "에", "간-", "다"]

        x = ['<BOS>'] + x + ['<EOS>'] 
        # x: ['<BOS>', "나는", "학교", "에", "간-", "다", '<EOS>']
        # 14 나머지 50은 PAD 로 채워줘야

        x += ['<PAD>'] * (self.sl - len(x))
        # x: ['<BOS>', "나는", "학교", "에", "간-", "다", '<EOS>',  '<PAD>', '<PAD>', '<PAD>', '<PAD>',  ...]

        x = np.array([self.korbow[word] for word in x])
        # x = [1, 3, 442, 23, 11, 345, 2, 0, 0, 0, 0, ....]
        
        return x

    def __getitem__(self, idx):
        sen = self.x[idx] # 문장하나가 픽이 됨
        x = self.make_sentence(sen)
        y = self.y[idx] # 숫자하나(= label)가 픽이 됨

        return x, y # np.array
    
  
def prepare_loaders(df, korbow, tokenizer, sl, ratio = .6, bs = 2*64):
    
    index_num = int(df.shape[0] * ratio)

    # train, valid split
    train_df = df[:index_num].reset_index(drop = True)
    valid_df = df[index_num:].reset_index(drop = True)

    # train_ds, valid_ds
    train_ds = MyDataset(train_df, korbow, tokenizer, sl)
    valid_ds = MyDataset(valid_df, korbow, tokenizer, sl)

    # train_loader, valid_loader
    train_loader = DataLoader(train_ds, batch_size = bs, shuffle= True)
    valid_loader = DataLoader(valid_ds, batch_size = bs, shuffle= False)
    
    print("DataLoader Completed")
    return train_loader, valid_loader