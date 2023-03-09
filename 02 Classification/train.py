
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# torchvision - CIFAR10 데이터를 사용하기 위해서
import torchvision
from torchvision import transforms, datasets

from model import CNNModel
from dataloader import *
from original_trainer import *
from utils import *

import argparse
import ast

def define():
    p = argparse.ArgumentParser()

    p.add_argument('--nodes', type = str, default = '[16, 32]', help="List of Nodes: List of Two Ints in this Task")
    p.add_argument('--sample', type = bool, default = True, help="Show Sample Image")
    p.add_argument('--bs', type = int, default = 128, help="Batch Size")
    p.add_argument('--ratio', type = float, default = 0.7, help="Ratio of Train, Valid ")
    p.add_argument('--device', type = str, default = "mps", help="CUDA or MPS")
    p.add_argument('--n_epochs', type = int, default = 120, help="Number of Epochs")

    config = p.parse_args()
    return config

def main(config):

    train_data = datasets.CIFAR10('./data', download= True,train = True, transform = transforms.ToTensor())
    print("Data Shape: ", train_data.data.shape)
    print("Label: ", len(train_data.targets))
    n_classes = max(train_data.targets)+1
    print("Number of Classes: ", n_classes)
    print()

    if config.sample:
        plt.imshow(train_data.data[30])

    train_loader, valid_loader = prepare_loaders(train_data = train_data, ratio = config.ratio, bs = config.bs)
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
    model = CNNModel(nodes = ast.literal_eval(config.nodes)).to(device)
    print("Nodes: ", ast.literal_eval(config.nodes))
    print(model)
    print()

    # Loss Function
    loss_fn = nn.NLLLoss()
    print("Loss Function: ", loss_fn)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Optimizer: ", optimizer)
    
    result, model = run_train(model, train_loader, valid_loader, loss_fn, optimizer, device, n_classes, n_epochs= config.n_epochs, print_iter = 10, early_stop = 20) 

    # Visualization
    make_plot(result, stage = "Loss")
    print("Loss Visualized")

    make_plot(result, stage = "Acc")
    print("Acc Visualized")

if __name__ == '__main__':
    config = define()
    main(config)
    
    
    
