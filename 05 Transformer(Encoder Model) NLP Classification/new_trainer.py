import gc
import copy

import numpy as np
import pandas as pd

import torch 
import torch.nn as nn

import torchmetrics

from tqdm.auto import tqdm, trange


# Train One Epoch
def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, n_classes, scheduler = None, grad_clipping = False):

    train_sum_loss = 0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total = len(dataloader))

    # initialize metric
    metric1 = torchmetrics.Accuracy(task='multiclass', num_classes= n_classes, top_k=1).to(device)
    metric2 = torchmetrics.F1Score(task="multiclass", num_classes= n_classes, top_k=1).to(device)

    model.train()
    for step, data in bar:
        x = data[0].to(device)      # shape: [bs, 3, 32, 32] 

        batch_size = x.shape[0]

        y_true = data[1].to(device) # shape: [bs]
        y_pred = model(x)           # y_pred: [bs, 10]
    
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient-Clipping | source: https://velog.io/@seven7724/Transformer-계열의-훈련-Tricks
        max_norm = 5
        if grad_clipping:
            # print("Gradient Clipping Turned On | max_norm: ", max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        dataset_size += batch_size
        train_sum_loss += float(loss.item() * batch_size) 
        train_loss = train_sum_loss / dataset_size 

        acc = metric1(y_pred, y_true)
        acc = acc.detach().cpu().item()
        f1 = metric2(y_pred, y_true)
        f1 = f1.detach().cpu().item()

        bar.set_postfix(Epoch = epoch, 
                        Train_loss = train_loss,
                        LR = optimizer.param_groups[0]['lr'],
                        ACC = acc,
                        F1 = f1
                        )
        

    train_acc = metric1.compute()
    train_f1 = metric2.compute()

    print("Train's Accuracy: %.2f | F1_SCORE %.3f" % (train_acc, train_f1))
    print()

    # del train_accuracy, train_f1

    gc.collect()

    return train_loss, train_acc, train_f1


# Valid One Epoch
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device, epoch, n_classes):

    # initialize metric
    metric1 = torchmetrics.Accuracy(task='multiclass', num_classes= n_classes, top_k = 1).to(device)
    metric2 = torchmetrics.F1Score(task="multiclass", num_classes= n_classes, top_k = 1).to(device)
    
    valid_sum_loss = 0
    dataset_size = 0
    
    #tqdm의 경우, for문에서 iterate할 때 실시간으로 보여주는 라이브러리입니다. 보시면 압니다. 
    bar = tqdm(enumerate(dataloader), total = len(dataloader))

    ## 그래도 혹시 모르니 ㅋ
    model.eval()
    with torch.no_grad():
        for step, data in bar:
            x = data[0].to(device)      # shape: [bs, 3, 32, 32] 

            batch_size = x.shape[0]

            y_true = data[1].to(device) # shape: [bs]
            y_pred = model(x)           # y_pred: [bs, 10]
        
            loss = loss_fn(y_pred, y_true)

            dataset_size += batch_size
            valid_sum_loss += float(loss.item() * batch_size)
            valid_loss = valid_sum_loss / dataset_size

            acc = metric1(y_pred, y_true)
            acc = acc.detach().cpu().item()
            f1 = metric2(y_pred, y_true)
            f1 = f1.detach().cpu().item()

            bar.set_postfix(Epoch = epoch, 
                            Valid_loss = valid_loss,
                            # LR = optimizer.param_groups[0]['lr'],
                            ACC = acc,
                            F1 = f1
                            )
    

    valid_acc = metric1.compute()
    valid_f1 = metric2.compute()

    print("Valid's Accuracy: %.2f | F1_SCORE %.3f" % (valid_acc, valid_f1))
    print()

    # del valid_acc, valid_f1

    gc.collect()

    return valid_loss, valid_acc, valid_f1


# Run Train
def run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_classes, scheduler = None, grad_clipping = False, n_epochs=80, print_iter=10, early_stop=20):
    
    result = dict()
    lowest_loss, lowest_epoch = np.inf, np.inf
    train_hs, valid_hs, train_accs, valid_accs, train_f1s, valid_f1s = [], [], [], [], [], []
    
    for epoch in range(n_epochs):
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, n_classes, scheduler, grad_clipping)
        valid_loss, valid_acc, valid_f1 = valid_one_epoch(model, valid_loader, loss_fn, device, epoch, n_classes)

        # 줍줍
        train_hs.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        
        valid_hs.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)
        
        # monitoring
        if (epoch + 1) % print_iter == 0:
            print()
            print(f"Epoch{epoch + 1} | Train Loss:{train_loss:.3e} | Valid Loss:{valid_loss:.3e} | Lowest Loss:{lowest_loss:.3e}|")
            print()
           
        # Lowest Loss 갱신  - Valid Loss 기준
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss 
            lowest_epoch= epoch 
            torch.save(model.state_dict(), './model_nlp_transformer_imple.bin') 
        else:
            if early_stop > 0 and lowest_epoch+ early_stop < epoch +1:
                print("삽질중") 
                print("There is no improvement during %d epochs" % early_stop)
                break
                
    print()            
    print("The Best Validation Loss=%.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    
    model.load_state_dict(torch.load('./model_nlp_transformer_imple.bin'))
    
    result = dict()
    
    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs

    result["Train Acc"] = train_accs
    result["Valid Acc"] = valid_accs

    result["Train F1"] = train_f1s
    result["Valid F1"] = valid_f1s
    
    return result, model
