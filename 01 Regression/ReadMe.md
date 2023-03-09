# 01 Regression

## Velog 
  - [[토치의 호흡] 01 REGRESSION](https://velog.io/@heiswicked/토치의-호흡-01-REGRESSION)
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HaqIvhwdPTBXTTuvjGG-IIrynUtJlZjg?usp=sharing) 

## About Github Code 
 1. Jupyter Notebook이 아닌 Python Script로 생성
 2. CLI에서 파라미터 값을 다양하게 변화를 주어서 실험 가능
 3. 위 Velog에서와 코드는 조금 다르게 작성하였다. 

## How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PXKuxM1-XqsuhMO_WJFJ_wjm415-cM5C?usp=sharing) 
```python
python train.py --nodes '[7, 5, 3, 2]' --ratio 0.7 --device 'cpu' --n_epochs 100
```

- `nodes` : Model에 들어가는 노드의 수. `Python List` 형태로 원하는 노드의 수들을 담고 `' ' `로 감싸서 str로 만들어주면, `nn.Linear`들이 자동적으로 만들어진다.
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
- `n_epochs` : Epoch

#### Jupyter Notebook Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qNb7Ce5_iZ80utLqNH6qUdfw9YIlM4Sm?usp=sharing) 

