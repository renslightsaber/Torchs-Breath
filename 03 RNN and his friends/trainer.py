import numpy as np
import pandas as pd

import gc
import random

import copy
from copy import deepcopy

import torch 
import torch.nn as nn

from tqdm.auto import tqdm, trange


# Train One Epoch
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_sum_loss=0
    for data in dataloader:
        x = data[0].to(device)      # [bs, sl, input_size]
        y_true = data[1].to(device) # [bs, 1]
        y_pred = model(x)           # [bs, 1]

        loss = loss_fn(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_sum_loss += loss.item()
    train_loss = train_sum_loss / len(dataloader)
    return train_loss
    


# Valid One Epoch
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device):
# @torch.no_grad() : 데코레이터 형태의 'model을 학습 시키지 않겠다는 필수 의지 표명 2' 
# --> 근데 함수형일 때 쓸 수 있는 것으로 알고 있다. 
# --> model을 학습 시키지 않겠다는 필수 의지 표명 1과 2 중 하나만 써도 무방하다.
# 여기서는 optimizer가 필요없다. 학습을 하지 않기 때문

    # model을 학습 안 시키겠다고 선언
    model.eval()
    
    # 예측값들과 실제값들을 담을 빈 리스트를 선언
    # 예측값과 실제값을 볼 수 있는 시각화 용도
    preds = []
    trues = []
    
    valid_sum_loss = 0
    
    # model을 학습 시키지 않겠다는 필수 의지 표명 1
    with torch.no_grad():
        for data in dataloader:
             
         # valid_loader = [([bs, 12, 3], [bs, 1]), ([bs, 12, 3], [bs, 1]), .... ]
        
            x = data[0].to(device)      # [bs, 12, 3] # GPU로 보내준다.
            y_true = data[1].to(device) # [bs, 1]     # GPU로 보내준다.
            y_pred = model(x)           # [bs, 1]     # model을 통해 예측한 값

            loss = loss_fn(y_pred, y_true) # MSE Loss - 성능평가

            valid_sum_loss += loss.item()
    		
            # 예측값들 줍줍
            preds.append(y_pred)

            # 실제값들 줍줍
            trues.append(y_true)
            
    valid_loss = valid_sum_loss / len(dataloader) # epoch 당 valid loss 평균값

    # 줍줍한 예측값들을 concat
    preds_cat = torch.cat(preds, dim = 0)
    # 줍줍한 실제값들을 concat
    trues_cat = torch.cat(trues, dim = 0)
    
    return valid_loss, trues_cat, preds_cat
    


# Run Train
def run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_epochs = 200, print_iter = 10, early_stop = 30):
    
    result = dict()
    best_model = None
    lowest_loss, lowest_epoch = np.inf, np.inf
    train_hs, valid_hs = [], []


    for epoch in range(n_epochs):
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss, trues_cat, preds_cat = valid_one_epoch(model, valid_loader, loss_fn, device)
        
        # 매 epoch 마다 train_loss, valid_loss를 줍줍
        train_hs.append(train_loss)
        valid_hs.append(valid_loss)
        
        # monitoring: print_iter 주기만큼 print문을 찍어줍니다.
        if (epoch + 1) % print_iter == 0:
            print("Epoch:%d, train_loss=%.3e, valid_loss=%.3e, lowest_loss=%.3e" % (epoch+1,train_loss, valid_loss, lowest_loss))
        
        # lowest_loss 갱신
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = epoch
            # model save
            torch.save(model.state_dict(), "./model_rnn.pth")
        else:
            if early_stop >0 and lowest_epoch + early_stop < epoch + 1:
                print("삽질 중")
                break
    print()
    print("The Best Validation Loss=%.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    
    # model load
    model.load_state_dict(torch.load("./model_rnn.pth"))

    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs

    result["Trues"] = trues_cat.detach().cpu().numpy()
    result["Preds"] = preds_cat.detach().cpu().numpy()

    return result, model

