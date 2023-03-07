import numpy
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


def make_plot(result, stage = "Loss"):
    ## Train/Valid History
    plot_from = 0

    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'
    else:
        stage = "Trues vs Preds"
        trains = "Trues"
        valids = "Preds"

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

