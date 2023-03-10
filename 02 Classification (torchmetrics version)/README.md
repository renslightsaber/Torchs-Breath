# 02 Classification (torchmetrics)
 - Data: [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

## Velog 
  - [[토치의 호흡] 02 CLASSIFICATION](https://velog.io/@heiswicked/토치의-호흡-02-CLASSIFICATION) 
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BCKTek0EGf5lq-mIKQVsj44afvD5pk5V?usp=sharing)


## About Github Code 
 1. Jupyter Notebook이 아닌 Python Script로 생성
 2. CLI에서 파라미터 값을 다양하게 변화를 주어서 실험 가능
 3. 위 Velog에서와 코드는 조금 다르게 작성 


## [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
: 정확한 Accuracy, F1 Score를 위해서 torchmetrics 라이브러리를 사용했다. 보통 Colab에는 설치가 되어있지 않으니 따로 설치해야 한다.
```python
$ pip install -qqq torchmetrics
```


## How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_8wJja-KBLhNX3ETz7uwe2veNe7h-8Hv?usp=sharing)

```python
$ python train.py --nodes '[8, 32]' --sample False --grad_clipping False --bs 128 --ratio 0.7 --device 'cuda' --n_epochs 110
```

- `nodes` : Model에 들어가는 노드의 수. `Python List` 형태로 원하는 노드의 2개의 숫자를 지정하고 `' ' `로 감싸서 str로 만들어주면, `nn.Conv2d` 레이어 두 개가 자동적으로 만들어진다.
- `sample` : Sample 이미지를 보겠냐는 여부인데, 딱히 필요없다면 `False`로 지정해도 된다. 
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size를 결정한다.
- `device`: GPU를 통한 학습이 가능하다면, `cuda`, `mps`(M1) 로 설정할 수 있다. 
  - 디바이스의 GPU를 믿고 있었는데, 본의 아니게 배신 당하게 된다면, 자동적으로  `cpu` 로 설정될 것이다. 
  - M1에서 `mps`로 지정할 경우, `torchmetrics`와 호환이 되지 않아 에러가 날 것이다. M1의 경우, 에러가 난다면, `cpu`로 train 해야한다.
- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `n_epochs` : Epoch


#### Jupyter Notebook Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15aiRDgT3e9BBcfwCKae25_oJzaxZTrnU?usp=sharing) 


