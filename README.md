# 토치의 호흡(Torch's Breath)
Finally, Torch's Breath on Github

## INTRO
Pytorch를 처음 입문하는 분들에게는 상당한 어려움을 느끼게 되어있다. 이러한 부담감과 어려움을 덜게 하고자 기획을 하였고 2022년 이어드림스쿨2기 수강생분들에게 관련하여 '토치의 호흡' 이라는 세션을 만들어서 진행하였다. 반응이 매우 좋았고, (당시) 회사('패스트캠퍼스')에서도 만족했다. 그래서 과정이 끝난 이후, '토치의 호흡'에서 강의했던 내용을 내 기술블로그에 ['토치의 호흡'시리즈](https://velog.io/@heiswicked/series/Torchs-Breath)로 연재했다. 각 포스트에는 실습할 수 있는 Colab Link도 추가되어있다. 


## Contents
- [01 Regression](https://github.com/renslightsaber/Torchs-Breath/tree/main/01%20Regression)
- [02 Classification (torchmetrics version)](https://github.com/renslightsaber/Torchs-Breath/tree/main/02%20Classification%20(torchmetrics%20version)) 
- [02 Classification](https://github.com/renslightsaber/Torchs-Breath/tree/main/02%20Classification) 
- [03 RNN and his friends](https://github.com/renslightsaber/Torchs-Breath/tree/main/03%20RNN%20and%20his%20friends) 
- [04 GRU NLP Classification](https://github.com/renslightsaber/Torchs-Breath/tree/main/04%20GRU%20NLP%20Classification) 
- [05 Transformer(EncoderModel) NLP Classification](https://github.com/renslightsaber/Torchs-Breath/tree/main/05%20Transformer(Encoder%20Model)%20NLP%20Classification)

## 참고
 - [Velog 토치의 호흡 시리즈](https://velog.io/@heiswicked/series/Torchs-Breath)
   - 같은 Task이지만, velog에서 안내된 코드보다 조금 더 어렵게 작성해보았다. (저도 성장은 해야죠..?)
   - Github에서는 Python Script(.py)로 작성하는 과정에서 모듈별로 코드가 분리되어있다. 
   - Data Science 공부를 시작한 지 얼마 안 되신 분들의 경우 조금 생소하거나 어려울 수 있기에 Colab 코드도 같이 링크를 첨부해두었다.
   
 - [Velog M1 Settings](https://velog.io/@heiswicked/series/M1Settings)
   - Colab뿐만 아니라, Apple M1, M2에서 실습할 수 있다.
   - `torchmetrics`를 사용할 때는 `mps`를 쓸 수 없고 `cpu`로만 가능하다.
   - `RNN`이 사용된 Task에서는 `mps`보다 `cpu`가 더 빨랐다.



