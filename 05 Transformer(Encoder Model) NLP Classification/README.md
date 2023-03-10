Transformer Encoder 구현 및 실행
# 05 Transformer(EncoderModel)NLP Cliassification
 - Data: [Dacon Basic 쇼핑몰 리뷰 평점 분류 경진대회](https://dacon.io/competitions/official/235938/overview/description)
    - 2022년에 Dacon에서 열려서 직계제자와 같이 (각자 개인으로) 참여한 대회인데, 어째서인지 지금은 링크가 막혀있다.   
    - `data` 폴더에는  `train.csv` 파일만 올려두었다. 

## Velog 
  - [[토치의 호흡] 10 About Transformer PART 05 Classification by DIY TRANSFORMER](https://velog.io/@heiswicked/토치의-호흡-11-About-Transformer-PART-05-NLPClassificationbyDIYTRANSFORMER)  
    - NLP_Classification_by_DIY_TRANSFORMER.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OXRdJiXQ3wlfs3U96A9RP4i1fxNPYL17?usp=sharing)

## About Transformer
  - [[토치의 호흡] 07 About Transformer PART 01 간단한 구조 설명](https://velog.io/@heiswicked/토치의-호흡-06-About-Transformer-PART-01-간단한-구조-설명)  
  - [[토치의 호흡] 08 About Transformer PART 02 "Positional Encoding Layer"](https://velog.io/@heiswicked/토치의-호흡-06-About-Transformer-PART-01-PositionalEncodingLayer) 
    -  PositionalEncodingLayer.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FZOVy8oWFBYv90zeJELqp-qil0TNkRER?usp=sharing)
 - [[토치의 호흡] 09 About Transformer PART 03 "Encoder and EncoderLayer"](https://velog.io/@heiswicked/토치의-호흡-09-About-Transformer-PART-03-Encoder-and-EncoderLayer) 
  - [[토치의 호흡] 10 About Transformer PART 05 Classification by DIY TRANSFORMER](https://velog.io/@heiswicked/토치의-호흡-11-About-Transformer-PART-05-NLPClassificationbyDIYTRANSFORMER)  
    - NLP_Classification_by_DIY_TRANSFORMER.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OXRdJiXQ3wlfs3U96A9RP4i1fxNPYL17?usp=sharing)


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


## How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J75wQ6qgWkjHCPgAocbQg49eF7TYhhRo?usp=sharing)

```python
$ python train.py --tokenizer "mecab" --grad_clipping True --bs 64 --ratio 0.8 --trainer_type "new" --device 'cuda' --n_epochs 5
```

- `data_path` : 데이터가 저장된 위치. 다른 위치에 데이터를 다운 받았다면, 그 위치로 지정 가능하다. Default로 `data`폴더로 지정해두었다.
- `tokenizer` : `utils.py` 참고
    - `mecab`으로 하면, `tokenizer = mecab.morphs` 
    - `text_pre`로 하면, `tokenizer = text_pre`
- `trainer_type` 
    - `original_trainer`: Accuracy를 실시간으로(=Batch가 Model을 지날 때마다) 계산하는 코드 ([original_trainer.py](https://github.com/renslightsaber/Torchs-Breath/blob/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification/original_trainer.py))
    - `new_trainer`: [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)로 Accuracy와 F1 Score를 실시간으로(=Batch가 Model을 지날 때마다) 계산하는 코드 ([new_trainer.py](https://github.com/renslightsaber/Torchs-Breath/blob/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification/new_trainer.py))   

- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `hid_dim` : [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)에서는 `d_model`에 해당 (Default: 256)
- `pf_dim` :  [positionwiseff.py](https://github.com/renslightsaber/Torchs-Breath/blob/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification/positionwiseff.py)에서 `PositionwiseFeedForwardLayer`에서 `Head` 수 (Default: 512)
- `n_heads` : `MultiHeadAttention`에서 `Head` 수 (Default: 8) 
- `n_layers` : `EncoderLayer`의 수 (Default: 6)
- `sl` : Sequence Length (Default: 90)
- `max_len` : Max Sequence Length ([embedandpe.py](https://github.com/renslightsaber/Torchs-Breath/blob/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification/embedandpe.py)에서 `PositionalEncodingLayer`에서 필요) (Default: 100)
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
- `dropout` : `nn.Dropout()`에 들어가는 Probability (Default: 0.1)
- `n_epochs` : Epoch

#### Jupyter Notebook 'new_trainer.py' Applied Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_fgbLFHABnQJDHzcDt071T2lZzfljWKx?usp=sharing) 

#### Jupyter Notebook 'original_trainer.py' Applied Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MUPOxD3gyvZNaTcH4NAvFcC8r2U__2zn?usp=sharing) 



