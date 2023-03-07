import numpy as np

import torch 
import torch.nn as nn

from tqdm import tqdm


# Train One Epoch
def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, n_classes =None, scheduler = None, grad_clipping = False):
    model.train()
    
    train_loss, dataset_size = 0,  0 # train loss, accuracy를 실시간으로 구현
    preds, trues = [], []
    bar = tqdm(dataloader, total= len(dataloader))
    
    for data in bar:
        x = data[0].to(device)      
        bs = x.shape[0]
        y_true = data[1].to(device) 
        y_pred = model(x)         
        
        loss = loss_fn(y_pred, y_true)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 실시간 train_loss
        dataset_size += bs                           # 실시간으로 크기가 update
        train_loss += (loss.item() * bs)             # batch 단위 loss가 누적
        train_epoch_loss = train_loss / dataset_size # 실시간 train_loss
        
        preds.append(y_pred) 
        trues.append(y_true) 
        
        preds_cat = torch.cat(preds, dim = 0) 
        trues_cat = torch.cat(trues, dim = 0)
        
        # torch.argmax(preds_cat, dim=1) # [total_bs]
        
        train_acc = 100 * (trues_cat == torch.argmax(preds_cat, dim=1)).sum().detach().cpu().item() / dataset_size # 전체 맞은 개수
        
        bar.set_description(f"EP:[{epoch:02d}]|TL:[{train_epoch_loss:.3e}]|ACC:[{train_acc:.2f}]")
        
    return train_epoch_loss, train_acc


# Valid One Epoch
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device, epoch, n_classes=None):
    model.eval()
    
    valid_loss, dataset_size = 0,  0 # valid loss, accuracy를 실시간으로 구현
    preds, trues = [], []
    bar = tqdm(dataloader, total= len(dataloader))
    
    with torch.no_grad():
        for data in bar:
            x = data[0].to(device)         
            bs = x.shape[0]
            y_true = data[1].to(device)    
            y_pred = model(x)              

            loss = loss_fn(y_pred, y_true)

            # 실시간 train_loss
            dataset_size += bs                           # 실시간으로 크기가 update
            valid_loss += (loss.item() * bs)             # batch 단위 loss가 누적
            valid_epoch_loss = valid_loss / dataset_size # 실시간 train_loss

            # 실시간 Accuracy
            preds.append(y_pred) 
            trues.append(y_true) 

            preds_cat = torch.cat(preds, dim = 0) 
            trues_cat = torch.cat(trues, dim = 0) 

            # torch.argmax(preds_cat, dim=1)      # [total_bs]

            val_acc = 100 * (trues_cat == torch.argmax(preds_cat, dim=1)).sum().detach().cpu().item() / dataset_size # 전체 맞은 개수

            bar.set_description(f"EP:[{epoch:02d}]|VL:[{valid_epoch_loss:.3e}]|ACC:[{val_acc:.2f}]")
        
    return valid_epoch_loss, val_acc


# Run Train
def run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_classes=None, scheduler = None, grad_clipping = False, n_epochs = 50, print_iter = 10, early_stop = 20):
        
    result = dict()
    lowest_loss, lowest_epoch = np.inf, np.inf
    train_hs, valid_hs, train_accs, valid_accs = [], [], [], []
    
    for epoch in range(n_epochs):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        valid_loss, valid_acc = valid_one_epoch(model, valid_loader, loss_fn, device, epoch)

        # 줍줍 FOR 시각화
        train_hs.append(train_loss)
        train_accs.append(train_acc)
        
        valid_hs.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # 모니터링 
        if (epoch + 1) % print_iter == 0:
            print(f"Ep:[{epoch + 1}]|Train Loss:{train_loss:.3e}|Valid Loss:{valid_loss:.3e}|LL:{lowest_loss:.3e}")
            
        # Lowest Loss 갱신 -> valid loss 기준
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = epoch
            torch.save(model.state_dict(), './model_nlp_transformer_imple.bin')
        else:
            if early_stop > 0 and lowest_epoch + early_stop < epoch +1:
                print("삽질중")
                print("There is no improvement during last %d epochs" % early_stop)
                break
                
    print()
    print("The Best Validation Loss= %.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    
    # model load
    model.load_state_dict(torch.load('./model_nlp_transformer_imple.bin'))
    
    # result
    result = dict()
    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs
    
    result["Train Acc"] = train_accs
    result["Valid Acc"] = valid_accs
    
    return result, model
            