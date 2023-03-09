import numpy
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


################# Data ######################
def dacon_competition_data( base_path = './data/', test_and_ss = False):

    train = pd.read_csv(base_path + 'train.csv')
    print("Data Shape: ", train.shape)
    print("Nunique of Target Categories: ", train.iloc[:, -1].nunique())
    print("Values of Target Categories: ", train.iloc[:, -1].unique())

    if test_and_ss:
        test = pd.read_csv(base_path + 'test.csv')
        ss = pd.read_csv(base_path + 'sample_submission.csv')
        return train, test, ss
    else:
        return train




########## Mecab Tokenizer ################
from konlpy.tag import Mecab
import re

mecab = Mecab() 
predefined_pos = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP",
                  "VV",
                  "VA", "VX", "VCP", "VCN",
                  "MM", "MAG", "MAJ"]

def text_pre(text, tokenizer = 'morphs'):
    # 1. Cleaning
    # 밑에 있는 cleaning 코드는 3개를 다 써도 되고, 일부만 사용해도 됩니다.
    #text = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text) # 한국어 빼고 다 지우기
    text = re.sub("[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]", "", text) # 특수문자 다 지우기
    #text = re.sub(["A-Za-z"], "", text) # 영어 다 지우기

    if tokenizer =='word':
        tokens = text.split()

    elif tokenizer =='nouns':
        tokens = mecab.nouns(text)

    elif tokenizer =='morphs':
        tokens = mecab.morphs(text)

    elif tokenizer =='predefined':

        tokens = []
        temp = mecab.pos(text)
        for token, pos in temp:
            if pos in predefined_pos:
                tokens.append(token)

    ## 3. Stop words
    SW = set()
    SW.add("불용어")

    result = [token for token in tokens if token not in SW]
    return result 

################## GETBOW ########################
def getbow(corpus, tokenizer):
    # corpus: [sentence1, sentence2, ....]
    bow = {'<PAD>': 0, '<BOS>': 1, '<EOS>':2}

    for line in corpus:
        for tok in tokenizer(line):
            if tok not in bow.keys():
                bow[tok] = len(bow.keys())

    return bow

################# Make Plot #######################
def make_plot(result, stage = "Loss"):
    ## Train/Valid History
    plot_from = 0

    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'

    elif stage == "Acc":
        trains = "Train Acc"
        valids = "Valid Acc"

    elif stage == "F1":
        trains = "Train F1"
        valids = "Valid F1"

    plt.figure(figsize=(10, 6))
    
    plt.title(f"Train/Valid {stage} History", fontsize = 20)
    
    ## Modified for converting Type
    if type(result[trains][0]) == torch.Tensor:
        result[trains] = [num.detach().cpu().item() for num in result[trains]]
        result[valids] = [num.detach().cpu().item() for num in result[valids]]
        
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

