import numpy
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


def make_plot(result):
    ## Train/Valid History
    plot_from = 0

    plt.figure(figsize=(10, 6))
    
    plt.title("Train/Valid Loss History", fontsize = 20)
    
    plt.plot(
        range(0, len(result['Train Loss'][plot_from:])), 
        result['Train Loss'][plot_from:], 
        label = 'Train Loss'
        )

    plt.plot(
        range(0, len(result['Valid Loss'][plot_from:])), 
        result['Valid Loss'][plot_from:], 
        label = 'Valid Loss'
        )

    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

