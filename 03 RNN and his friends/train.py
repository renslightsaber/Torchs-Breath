# import gc
# import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm, trange

from kaggle_dataset import *
from dataloader import *
from trainer import *
from model import *
from utils import *

import argparse
import ast

def define():
    p = argparse.ArgumentParser()

    p.add_argument('--data_path', type = str, default = "./data/bitcoin-historical-data.csv", help="Data Path")
    p.add_argument('--percent', type = float, default = 1, help="Data Percent for train and inference")

    p.add_argument('--model', type = str, default = "GRU", help="Which RNN Model?")
    p.add_argument('--hidden_size', type = int, default = 100, help="Number of RNN Model's Hidden Size")
    p.add_argument('--num_layers', type = int, default = 2, help="Number of RNN Model's Layers")
    p.add_argument('--sl', type = int, default = 7, help="Sequence Length")

    p.add_argument('--bs', type = int, default = 128, help="Batch Size")
    p.add_argument('--ratio', type = float, default = 0.7, help="Ratio of Train, Valid ")
    p.add_argument('--device', type = str, default = "mps", help="CUDA or MPS")
    p.add_argument('--n_epochs', type = int, default = 120, help="Number of Epochs")

    config = p.parse_args()
    return config

def main(config):

    df = kaggle_data_load(percent = config.percent, base_path =  config.data_path) 
    print("Data Shape: ", df.shape)
    print()
    
    print("Sequence Length: ", config.sl)
    train_loader, valid_loader = prepare_loaders(df, sl = config.sl, ratio = config.ratio, bs = config.bs)
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

    print("Device", device)
    print()

    # Model
    if config.model == "GRU":
        model = GRUModel( hidden_size = config.hidden_size, num_layers= config.num_layers, sequence_length = config.sl, device = config.device).to(device)
    
    elif config.model == "RNN":
        model = RNNModel( hidden_size = config.hidden_size, num_layers= config.num_layers, sequence_length = config.sl, device = config.device).to(device)

    elif config.model == "LSTMV1":
        model = LSTMModelV1(hidden_size = config.hidden_size, num_layers= config.num_layers, sequence_length = config.sl, device = config.device).to(device)
    
    elif config.model == "LSTMV2":
        model = LSTMModelV2(hidden_size = config.hidden_size, num_layers= config.num_layers, sequence_length = config.sl, device = config.device).to(device)

    else:
    # elif config.model == "Bidirectional RNN":
        model = BiRNNModel(hidden_size = config.hidden_size, num_layers= config.num_layers, sequence_length = config.sl, device = config.device).to(device)

    print(model)
    print()

    # Loss Function
    loss_fn = nn.MSELoss()
    print("Loss Function: ", loss_fn)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Optimizer: ", optimizer)
    
    result, model = run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_epochs = config.n_epochs,)
        
    # Visualization
    make_plot(result, stage = "Loss")
    print("Loss Visualized")

    make_plot(result, stage = "Trues vs Preds")
    print("Trues vs Preds Visualized")

if __name__ == '__main__':
    config = define()
    main(config)
