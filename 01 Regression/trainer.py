import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm

## train_one_epoch
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_sum_loss = 0

    for data in dataloader:
        x = data[0].to(device)       # [bs, 8]
        y_true = data[1].to(device)  # [bs, 1]
        y_pred = model(x)            # [bs, 1]

        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_sum_loss += loss.item()
    
    train_loss = train_sum_loss / len(dataloader)
    return train_loss


## valid_one_epoch
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device):
    model.eval()

    valid_sum_loss = 0
    
    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)       # [bs, 8]
            y_true = data[1].to(device)  # [bs, 1]
            y_pred = model(x)            # [bs, 1]

            loss = loss_fn(y_pred, y_true)

            valid_sum_loss += loss.item()
    
    valid_loss = valid_sum_loss / len(dataloader)
    return valid_loss


def run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_epochs = 120, print_iter = 10, early_stop = 20):

    result = dict()
    train_hs, valid_hs = [], []
    lowest_loss, lowest_epoch = np.inf, np.inf

    for epoch in range(n_epochs):

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss = valid_one_epoch(model, valid_loader, loss_fn, device)


        train_hs.append(train_loss)
        valid_hs.append(valid_loss)

        if (epoch + 1) % print_iter == 0:
            print(f"Epoch:[{epoch+1}/{n_epochs}]|Train_Loss:[{train_loss:.3e}]|Lowest_Loss:[{lowest_loss:.3e}]")
            
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = epoch
            torch.save(model.state_dict(), './model_reg.bin')
        else:
            if early_stop > 0 and lowest_epoch + early_stop < epoch +1:
                print("삽질 중")
                print("There is no improvemend during last %d epochs" % early_stop)
                break

    print()
    print("The Best Validation Loss=%.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    
        # model load
    model.load_state_dict(torch.load('./model_reg.bin'))

    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs

    return result, model