
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm 

from model import *
from dataloader import *
# from new_trainer import *
# from original_trainer import *

import new_trainer 
import original_trainer 

from utils import *

import argparse
import ast

def define():
    p = argparse.ArgumentParser()

    p.add_argument('--base_path', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--tokenizer', type = str, default = "text_pre", help="mecab or text_pre")

    p.add_argument('--grad_clipping', type = bool, default = False, help="Gradient Clipping")
    p.add_argument('--bs', type = int, default = 128, help="Batch Size")
    p.add_argument('--ratio', type = float, default = 0.7, help="Ratio of Train, Valid ")

    p.add_argument('--trainer_type', type = str, default = "new", help="Original Trainer? New?")

    p.add_argument('--sl', type = int, default = 90, help="Sequence Length")

    p.add_argument('--emb_dim', type = int, default = 256 * 2, help="emb_dim")
    p.add_argument('--hidden_size', type = int, default = 512 * 2, help="hidden_size")
    p.add_argument('--num_layers', type = int, default = 2, help="Number of GRU's Layers")

    p.add_argument('--device', type = str, default = "mps", help="CUDA or MPS or CPU?")
    p.add_argument('--dropout', type = float, default = 0.1, help="Dropout Probability")

    p.add_argument('--n_epochs', type = int, default = 30, help="Number of Epochs")

    config = p.parse_args()
    return config

def main(config):

    train = dacon_competition_data(base_path = config.base_path)
    n_classes = train.iloc[:, -1].nunique()
    print("n_classes: ", n_classes)
    print("Sequence Length: ", config.sl)
    print()

    ## Target: LabelEncoder()
    label_encoder = LabelEncoder()
    train['new_target'] = label_encoder.fit_transform(train['target'])
    print(train.head())
    print(train.shape)

    if config.tokenizer == "mecab":
        tokenizer = mecab.morphs
        print("Tokenizer: ", config.tokenizer)
    else:
        tokenizer = text_pre
        print("Tokenizer: ", config.tokenizer)
    print()

    korbow = getbow(train.reviews.to_list(), tokenizer)
    input_dim = len(korbow.keys())
    print("Input_dim(=len(korbow)): ", input_dim )
    print()


    train_loader, valid_loader = prepare_loaders(df = train, korbow=korbow, tokenizer = tokenizer, sl = config.sl, ratio = config.ratio, bs = config.bs)
    print("Ratio: ", config.ratio)
    data = next(iter(train_loader))
    print("Sample of Train_Loader: ", data[0].shape, data[1].shape)
    print()

    # Device
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")


    if config.trainer_type == "new":
        train_one_epoch = new_trainer.train_one_epoch
        valid_one_epoch = new_trainer.valid_one_epoch
        run_train = new_trainer.run_train
        # device = torch.device("cpu")
        print("Trainer option: ", config.trainer_type)
        print("Device", device)
        # print("There is no option when using torchmetrics on M1 Mac: Only CPU Right now")
    else:
        train_one_epoch = original_trainer.train_one_epoch
        valid_one_epoch = original_trainer.valid_one_epoch
        run_train = original_trainer.run_train
        print("Trainer option: ", config.trainer_type)
    
    print("Device", device)
    print()

    # Model
    model = Model(input_dim = input_dim, emb_dim = config.emb_dim, hidden_size = config.hidden_size, num_layers = config.num_layers, 
                  sl = config.sl, dropout = config.dropout, device =device)
    model = model.to(device)
    print(model)
    print()

    # Loss Function
    loss_fn = nn.NLLLoss()
    print("Loss Function: ", loss_fn)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Optimizer: ", optimizer)
    
    result, model = run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_classes, grad_clipping = config.grad_clipping, n_epochs= config.n_epochs) 
    # Visualization
    make_plot(result, stage = "Loss")
    print("Loss Visualized")

    make_plot(result, stage = "Acc")
    print("Acc Visualized")

    if config.trainer_type == "new":
        make_plot(result, stage = "F1")
        print("F1 Visualized")

if __name__ == '__main__':
    config = define()
    main(config)
    ## new_trainer (CPU Only)
    # python train.py --device cuda --hidden_size 256 --emb_dim 128 --bs 64 --trainer_type new --n_epochs 3
    # torchmetrics is not friendly with mps. This is why cpu

    ## Original trainer
    # python train.py --device cpu --hidden_size 256 --emb_dim 128  --bs 64 --trainer_type original --n_epochs 3
    # M1: Only CPU Mode Works, while MPS Mode doesn't work 
    # CUDA: Works
