# Regression

## 01 Velog Links
  - [[토치의 호흡] 01 REGRESSION](https://velog.io/@heiswicked/토치의-호흡-01-REGRESSION)
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HaqIvhwdPTBXTTuvjGG-IIrynUtJlZjg?usp=sharing) 

## 02 About Github Code
 1. Jupyter Notebook이 아닌 Python Script로 생성
 2. CLI에서 파라미터 값을 다양하게 변화를 주어서 실험 가능

## How to train in CLI?
```python
python train.py --nodes [6, 5, 4, 3, 2] --ratio 0.7 --device cpu --n_epochs 120
```

- `nodes` : Model에 들어가는 노드의 수. `Python List` 형태로 원하는 노드의 수들을 담는다면, 이에 따라 자동적으로 `nn.Linear`들이 자동적으로 생성하도록 하였다.
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
- `n_epochs` : Epoch 수. '몇 Epoch을 돌릴 것인가?'
