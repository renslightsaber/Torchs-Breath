# 04 GRU NLP Classification
 - Data: [Dacon Basic 쇼핑몰 리뷰 평점 분류 경진대회](https://dacon.io/competitions/official/235938/overview/description)
    - 2022년에 Dacon에서 열려서 직계제자와 같이 (각자 개인으로) 참여한 대회인데, 어째서인지 지금은 링크가 막혀있다.   
    - `data` 폴더에는  `train.csv` 파일만 올려두었다. 

## Velog 
  - [[토치의 호흡] 06 NLP Basic Classification with GRU](https://velog.io/@heiswicked/토치의-호흡-05-NLP-Basic-Classification-with-GRU)  
    - NLP_Classification_by_DIY_TRANSFORMER.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RvuYmg7eCr45yBrJrktMxGaOMuVmRCar?usp=sharing)


## About Github Code 
 1. Jupyter Notebook이 아닌 Python Script로 생성
 2. CLI에서 파라미터 값을 다양하게 변화를 주어서 실험 가능
 3. 위 Velog에서와 코드는 조금 다르게 작성 


## [Mecab](https://konlpy.org/ko/v0.4.0/install/)
: Mecab tokenizer를 사용할 예정!
```python

$ pip install konlpy
$ curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -x

```
- 참고: [M1에 Mecab 설치하기](https://velog.io/@heiswicked/M1-Part11-복불복설치-konlpy.tag-MECAB-on-M1-ver.221230)

## [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
: 정확한 Accuracy, F1 Score를 위해서 torchmetrics 라이브러리를 사용했다. 보통 Colab에는 설치가 되어있지 않으니 따로 설치해야 한다.
```python
$ pip install -qqq torchmetrics
```


## How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qM-brx8twYeOO12Lio3F2iLRZ1RrHoQs?usp=sharing)

```python
$ python train.py --tokenizer 'mecab' --grad_clipping True --bs 64 --ratio 0.7 --trainer_type 'new' --sl 90 --dropout 0.14 --device 'cuda' --n_epochs 5
```

- `data_path` : 데이터가 저장된 위치. 다른 위치에 데이터를 다운 받았다면, 그 위치로 지정 가능하다. Default로 `data`폴더로 지정해두었다.
- `tokenizer` : `utils.py` 참고
    - `mecab`으로 하면, `tokenizer = mecab.morphs` 
    - `text_pre`로 하면, `tokenizer = text_pre`
- `trainer_type` 
    - `original_trainer`: Accuracy를 실시간으로(=Batch가 Model을 지날 때마다) 계산하는 코드 ([original_trainer.py](https://github.com/renslightsaber/Torchs-Breath/blob/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification/original_trainer.py))
    - `new_trainer`: [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)로 Accuracy와 F1 Score를 실시간으로(=Batch가 Model을 지날 때마다) 계산하는 코드 ([new_trainer.py](https://github.com/renslightsaber/Torchs-Breath/blob/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification/new_trainer.py))   

- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `bs` : Batch Size (Default: 128) 
- `emb_dim` : `model.py`에서 `Model`에서 `nn.Embedding`에서의 `emb_dim` (Default: 128)
- `num_layers` : `model.py`에서 `Model`에서 `nn.GRU`에서의 `num_layers` (Default: 2)
- `hidden_size` : `model.py`에서 `Model`에서 `nn.GRU`에서의 `hidden_size` (Default: 256)
- `sl` : Sequence Length (Default: 90)
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
- `dropout` : `nn.Dropout()`에 들어가는 Probability (Default: 0.1)
- `n_epochs` : Epoch

#### Jupyter Notebook 'new_trainer.py' Applied Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nFdZ0GgwQ-w16H6p04_Wegz-12jq33u7?usp=sharing) 

#### Jupyter Notebook 'original_trainer.py' Applied Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dJcCsgtWKzr93rRq5RRhbGq9DEd9tacp?usp=sharing) 



