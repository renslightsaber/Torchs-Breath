import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

from model import Model
from dataloader import *
from trainer import *
from utils import *

import argparse
import ast

def define():
    p = argparse.ArgumentParser()

    p.add_argument('--nodes', type = str, default = '[6, 5, 4, 3, 2]', help="List of Nodes")
    p.add_argument('--bs', type = int, default = 128, help="Batch Size")
    p.add_argument('--ratio', type = float, default = 0.7, help="Ratio of Train, Valid ")
    p.add_argument('--device', type = str, default = "mps", help="CUDA or MPS")
    p.add_argument('--n_epochs', type = int, default = 120, help="Number of Epochs")

    config = p.parse_args()
    return config

def main(config):

    ch =  fetch_california_housing()
    df = pd.DataFrame(ch.data, columns = ch.feature_names)
    df['target'] = ch.target
    print("Data Shape: ", df.shape)
    print(df.head())

    train_loader, valid_loader = prepare_loaders(df = df, ratio = config.ratio, bs = config.bs)
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
    model = Model(nodes = ast.literal_eval(config.nodes)).to(device)
    print("Nodes: ", ast.literal_eval(config.nodes))
    print(model)
    print()

    # Loss Function
    loss_fn = nn.MSELoss()
    print("Loss Function: ", loss_fn)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Optimizer: ", optimizer)

    result, model = run_train(model, train_loader, valid_loader, loss_fn, optimizer, device) 

    # Visualization
    make_plot(result)

if __name__ == '__main__':
    config = define()
    main(config)