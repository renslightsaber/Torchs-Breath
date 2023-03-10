# 03 RNN and his friends
 - Data: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
    - `data` 폴더에는 위 데이터의 10%만 있다. 

## Velog 
  - [[토치의 호흡] 03 RNN and his friends PART 01](https://velog.io/@heiswicked/토치의-호흡-03-RNN-and-his-firends-PART-01)  
    - 03 RNN.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HaqIvhwdPTBXTTuvjGG-IIrynUtJlZjg?usp=sharing)
  - [[토치의 호흡] 03 RNN and his friends PART 02](https://velog.io/@heiswicked/토치의-호흡-03-RNN-and-his-firends-PART-02)  
    - 04 LSTMV1(dim:3->2).ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x-gp2nNJUB-zPRhDP7P0d8CrmqG5Ni1f?usp=sharing)
    - 04 LSTMV2(Output's Last Sequence).ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u9XJJUGLnWUkZKIwmnkqo1ATjmtUyKaT?usp=sharing)
    - 05 GRU.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oxlFu9D0YNfwguRuu1LGBGva03pAIGJc?usp=sharing)
    - 06 Bidirectional-RNN.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YRP-ybAhiNpc-I2WrflGN55m1ejORMzo?usp=sharing)


## About Github Code 
 1. Jupyter Notebook이 아닌 Python Script로 생성
 2. CLI에서 파라미터 값을 다양하게 변화를 주어서 실험 가능
 3. 위 Velog에서와 코드는 조금 다르게 작성 


## How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JExUnNDGo8w-45LX2qQyBqR9CfpNRPov?usp=sharing)

```python
$ python train.py --data_path "./data/bitcoin-historical-data.csv" --percent 1.0 --model "GRU" --device 'cuda' --n_epochs 5
```

- `data_path` : 데이터가 저장된 위치. 다른 위치에 데이터를 다운 받았다면, 그 위치로 지정 가능하다. Default로 `data`폴더로 지정해두었다.
- `percent` : 사용하고자하는 데이터의 양을 비율로 정할 수 있다.
- `model` : `RNNModel`, `LSTMModelV1`, `LSTMModelV2`, `GRUModel`,  `BiRNNModel` 중에서 선택 가능
- `hidden_size` : `nn.RNN()`, `nn.LSTM()`, `nn.GRU()` 의 `hidden_size`
- `num_layers` : `nn.RNN()`, `nn.LSTM()`, `nn.GRU()` 의 `num_layers` 
- `sl` : Sequence Length
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
- `n_epochs` : Epoch

#### Jupyter Notebook Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C4UC5dZELPEO21eBRberglez_8RmJ17_?usp=sharing) 

