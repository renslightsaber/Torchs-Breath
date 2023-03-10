import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

from tqdm.auto import tqdm, trange

from transformerencoder import *
from dataloader import *

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
    p.add_argument('--bs', type = int, default = 32, help="Batch Size")
    p.add_argument('--ratio', type = float, default = 0.7, help="Ratio of Train, Valid ")

    p.add_argument('--trainer_type', type = str, default = "new", help="Original Trainer? New?")

    p.add_argument('--sl', type = int, default = 90, help="Sequence Length")
    p.add_argument('--max_len', type = int, default = 100, help="Max Length")

    p.add_argument('--hid_dim', type = int, default = 256, help="d_model or hidden_size")
    p.add_argument('--pf_dim', type = int, default = 256*2, help="d_model or hidden_size")

    p.add_argument('--n_heads', type = int, default = 8, help="Number of Attention Heads")
    p.add_argument('--n_layers', type = int, default = 6, help="Number of EncoderLayers")

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
    print(train.loc[:, ["reviews", "new_target"]].head(3))
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


    train_loader, valid_loader = prepare_loaders(train, korbow, tokenizer, sl = config.sl, ratio = config.ratio, bs = config.bs)
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
    model = TransformerEncoderModel(input_dim, config.sl, config.max_len, config.hid_dim, config.pf_dim, config.n_heads, config.n_layers, config.dropout, config.device).to(device)
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

    
    
