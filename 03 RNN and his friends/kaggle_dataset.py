import os
import copy

import numpy as np
import pandas as pd


def kaggle_data_load(percent = 1, base_path =  './data/bitcoin-historical-data.csv'):
    df = pd.read_csv(base_path)
    df = df[df.Open.notnull()].reset_index(drop = True) # 결측치 제거

    num = int(df.shape[0] * percent)
    print(num)

    df_new = copy.deepcopy(df[:num])

    print(df_new.shape)
    print(df_new.head())

    return df_new
