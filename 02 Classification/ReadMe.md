# 02 Classification
 - Data: [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

## Velog 
  - [[토치의 호흡] 02 CLASSIFICATION](https://velog.io/@heiswicked/토치의-호흡-02-CLASSIFICATION) 
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BCKTek0EGf5lq-mIKQVsj44afvD5pk5V?usp=sharing)


## About Github Code 
 1. Jupyter Notebook이 아닌 Python Script로 생성
 2. CLI에서 파라미터 값을 다양하게 변화를 주어서 실험 가능
 3. 위 Velog에서와 코드는 조금 다르게 작성 


## How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q5A2FfS0zTQdwUQvTv5GoMgnjV4bGlhE?usp=sharing)

```python
$ python train.py --nodes '[8, 32]' --sample False --bs 128 --ratio 0.7 --device 'cuda' --n_epochs 110
```

- `nodes` : Model에 들어가는 노드의 수. `Python List` 형태로 원하는 노드의 2개의 숫자를 지정하고 `' ' `로 감싸서 str로 만들어주면, `nn.Conv2d` 레이어 두 개가 자동적으로 만들어진다.
- `sample` : Sample 이미지를 보겠냐는 여부인데, 딱히 필요없다면 `False`로 지정해도 된다. 
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
- `n_epochs` : Epoch

#### Jupyter Notebook Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yGjPtBH31wSmdkSLDQPKVK-RKz9yDlGc?usp=sharing) 
