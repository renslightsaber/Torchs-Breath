import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


def make_plot(result, stage = "Loss"):
    
    plot_from = 0
    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'
    else:
        stage = "Trues vs Preds"
        trains = "Trues"
        valids = "Preds"
        
    ## Modified for converting Type
    if type(result[trains][0]) == torch.Tensor:
        result[trains] = [num.detach().cpu().item() for num in result[trains]]
        result[valids] = [num.detach().cpu().item() for num in result[valids]]

    plt.figure(figsize=(10, 6))
    
    plt.title(f"Train/Valid {stage} History", fontsize = 20)
    
    plt.plot(
        range(0, len(result[trains][plot_from:])), 
        result[trains][plot_from:], 
        label = trains
        )

    plt.plot(
        range(0, len(result[valids][plot_from:])), 
        result[valids][plot_from:], 
        label = valids
        )

    plt.legend()
    if stage == "Loss":
        plt.yscale('log')
    plt.grid(True)
    plt.show()
    

