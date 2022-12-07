---
layout: single
title:  "ML지도학습 정리겸 붓꽃구분해보기"
categories: jupyter
tag: [python, blog, jupyter, TechTem]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 기계 학습 기법


## 개요

앞 단원에서 AI 프로세스의 다음 단계인 모델링에 사용할 수 있도록 데이터를 가져오고 처리하는 방법에 대해 알아보았습니다.



분류, 회귀 및 클러스터링과 같은 다양한 [기계 학습 작업](https://developers.google.com/machine-learning/problem-framing/cases)을 실행하는 데 사용할 수 있는 다양한 유형의 기계 학습 기술이 있습니다. 이 단원에서 우리는 예측, 클러스터 데이터 등을 만들기 위한 자체 모델을 만들 것입니다. 여기서 습득하게 될 기술은 추후 솔루션을 만드는 데 다시 사용됩니다!



## 기계 학습

머신 러닝은 컴퓨터가 데이터 변수 간의 패턴과 관계를 학습하는 능력을 말합니다. 이전의 예나 또는 현재 데이터 세트를 이용하여 학습을 통해 이를 수행할 수 있습니다. 기계 학습에 대해 자세한 내용을 알아보려면 이 [문서](https://hackernoon.com/the-simplest-explanation-of-machine-learning-youll-ever-read-bebc0700047c) 를 읽어 보세요. 워크시트에서 흥미로운 정보를 모두 기록하십시오.



이전에 살펴보았듯이 우리가 탐구하고자 하는 기계 학습 알고리즘에는 지도 학습과 비지도 학습의 두 가지 주요 유형이 있습니다. [둘의 차이점](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d) 을 기억하고 있습니까?


##### your answer here



공통점으로 답지가 없는 학습  

둘 다 주제 즉 질문에 따른 답 없는 학습  



강화학습에서는 보상 중시!  

비지도학습은 보상 상관없이 답을 향해 진행.  



간단 설명으로 괜찮아 보여서..  

https://ebbnflow.tistory.com/165


### 라이브러리 가져오기!

데이터 세트 작업을 돕기 위해 먼저 pandas 라이브러리를 가져와 보겠습니다.


# 1. 지도 학습 기법



```python
# Import pandas here
import pandas as pd
```

동영상 공유 사이트 유튜브를 이용해 본 적이 있습니까? 유튜브는 여러분이 보고 싶어할 것 같은 동영상을 추천한다는 것을 알고 있나요? 나에게 추천한 동영상과 친구에게 추천한 동영상이 다르다는 것을 알고 있으셨나요? 어떻게 그렇게 할 수 있다고 생각하십니까?



유튜브의 추천 알고리즘은 지도 학습 기술로 알려진 것을 사용합니다. 유튜브는 여러분이 동영상에 '좋아요'를 표시할 때마다 그 영상의 이름, 장르, 길이, 업로더 등의 메타 정보를 기록합니다. 동영상을 많이 보고 '좋아요'를 표시할수록 더 많은 정보가 유튜브 시스템에 기록됩니다.



이 데이터 세트는 지도 학습 모델을 훈련하는 데 사용되며, 모델은 사용자가 좋아할 만한 동영상을 예측합니다. 시청한 동영상과 '좋아요'를 표시한 동영상을 분석하고 해당 동영상 정보의 유사점을 확인하면 됩니다. 이 경우 이전에 좋아했던 동영상 정보가 추천 모델이 학습할 특성 또는 데이터가 됩니다. 반면에 여러분이 가장 보고 싶어할 것 같은 동영상(여러분과 같은 동영상를 좋아하는 다른 사람들이 보는 동영상들)은 모델/기술의 라벨 또는 대상이 됩니다. 본질적으로 모델/기술은 특성과 라벨을 최대한 "일치" 시키려고 합니다.


지도 학습 기술에는 라벨링된 훈련 데이터가 필요합니다. 예를 들어, 유튜브 동영상에 좋아요를 누르거나, 평가하지 않거나, 싫어한다는 라벨이 지정될 수 있습니다. 유튜브의 추천 모델을 학습할 때 필요한 훈련 데이터의 라벨입니다. 훈련 데이터가 많을수록 더 정확한 모델을 만들 수 있습니다.



라벨이 적절해야 합니다. 예를 들어 공부한 시간이 주어졌을 때 학생의 시험 점수를 예측하려면 데이터에 각 학생의 시험 점수가 포함되어야 합니다. 이렇게 하면 기계 학습 알고리즘이 주어진 예로부터 학습할 수 있습니다.



지도 학습 기술의 또 다른 중요한 용어는 '특성'입니다. 특성은 대상 또는 라벨을 예측하는 데 사용할 수 있는 데이터를 나타냅니다. 시험 점수 예시에서는 시험 점수가 대상이며, 공부한 시간이 특성입니다. 시험 점수 예에서는 공부한 문제 수, 시험 전 수면 시간 등이 또 다른 특성이 될 수 있습니다.



특성 및 라벨/대상을 사용하여 알고리즘은 특성과 라벨/대상 간의 관계를 "학습"할 수 있습니다.


다음 기사와 동영상을 보고 아래 질문에 답해보세요. (지도 학습에 대하여 이해하십시오. 비지도 학습을 지금 이해하지 않으셔도 됩니다.)<br>

https://towardsdatascience.com/explaining-supervised-learning-to-a-kid-c2236f423e0f  

https://www.geeksforgeeks.org/supervised-unsupervised-learning/  

https://www.youtube.com/watch?v=cfj6yaYE86U  



- 여러분이 이해한 지도 학습에 대해 설명해 보세요.

- 특성, 라벨/대상, 기계 학습과 같은 용어에 대하여 설명하고 예를 들어보세요.


지도 학습 기술은 데이터를 다른 그룹으로 분류하거나 변수 간의 관계를 예측하는 데 사용할 수 있습니다. 데이터가 속한 범주 또는 그룹을 분류하는 기술입니다. 예를 들어, 내일 비가 올지 여부를 예측하려면 비가 오거나 맑음과 같은 범주를 반환하는 알고리즘을 사용할 수 있습니다. 반면 회귀 분석에 사용되는 기술은 수치 데이터를 반환합니다. 예를 들어 내일 비의 양을 예측하려면 범주 대신 숫자 데이터를 반환해야 합니다.



분류와 회귀의 차이점에 대해 자세한 내용을 확인하려면 이 [문서](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/) 와 이 [동영상](https://www.youtube.com/watch?v=f7YB73F0zDo) 을 보십시오. 기사와 영상을 보고 분류를 사용할 시나리오와 회귀를 사용할 시나리오를 생각해 보시고 아래 셀에 시나리오를 작성해 보세요.


##### your answer here

- 분류, 회귀 시나리오





지도 학습이 무엇인지 이해했으면 이제 다양한 지도 학습 기법을 살펴보겠습니다!


# 1.1 K-최근접 이웃 알고리즘(K-Nearest Neighbours)


다음 그래프는 남성과 여성의 키와 몸무게 샘플입니다. 데이터가 어떻게 그룹화 되는지 확인해 보세요.남성은 여성에 비해 키가 크고 무거운 경향이 있으며, 눈에 띄는 집단이 있음을 알 수 있습니다.


녹색 포인트는 확인되지 않은 입력 데이터 입니다. 녹색 포인트를 남성 데이터라고 할 수 있나요, 아니면 여성 데이터라고 할 수 있나요? 그렇게 생각하는 이유는 무엇이며, 최종적으로 어떻게 결정하시겠습니까?



여러분은 녹색 포인트가 남성 데이터에 더 가깝다는 것을 알 수 있으며, 따라서 알려지지 않은 데이터가 남성을 나타낼 가능성이 더 크다고 할 수 있습니다. 이미 알려진 다른 데이터 포인트까지의 거리로 속한 그룹을 결정하는 이 방법을 K-최근접 이웃 알고리즘이라고 합니다.


KNN(K-Nearest Neighbors)은 분류 또는 회귀 문제에 사용할 수 있습니다. 그러나 주로 분류 문제에 사용됩니다. 이름에서 알 수 있듯이 이 알고리즘은 주변 지점이나 이웃에 의존하여 해당 클래스 또는 그룹을 결정합니다. 예를 들어, KNN을 사용하여 알 수 없는 포인트를 클래스 A, 클래스 B로 분류하려고 할 때 알 수 없는 포인트의 가까운 포인트들이 대부분 클래스 A에 속한 경우, 확인되지 않은 포인트는 어느 클래스에 속하였다고 생각하십니까?



클래스 A로 추측했다면 정답입니다. KNN은 대부분의 가장 가까운 포인트의 속성을 활용하여 미지의 포인트의 클래스를 분류하는 방법을 사용하기 때문입니다. 이것은 "유유상종"이라는 사자성어와 유사하게 비슷한 포인트 들이 서로 가까울 것이라고 예상하는 것입니다.



KNN에 대해 자세히 알아보려면 이 [동영상](https://www.youtube.com/watch?v=MDniRwXizWo) 을 보세요. KNN의 주요 장점과 단점은 무엇입니까? 다음 코딩 연습에서는 KNN을 사용할 것입니다. KNN은 사용하기 쉽고 유연한 알고리즘으로 여러 문제에 활용할 수 있습니다.


##### your answer here

- KNN에 관해서..


KNN의 작동 방식을 이해한 후 가상 시나리오에 적용해 보겠습니다. 이 시나리오에서는 가격 및 메모리 양과 같은 특성을 이용하여 장치가 노트북인지 데스크탑인지 예측하려고 합니다. 이와 같이 모든 특성을 이용하여 데이터 포인트에 대한 산점도를 표시했습니다.


x축은 메모리 양(GB)을 나타내고 y축은 가격(USD)을 나타냅니다. 파란색 원은 데스크탑용 데이터 포인트이고 주황색 원은 노트북용 데이터 포인트입니다. 별은 알수 없는 포인트를 나타냅니다.



k=3 인(즉, 인접한 이웃 포인트 3개)인 KNN 알고리즘을 적용한다면, 알수 없는 포인트는 데스크탑으로 분류될까요? 아니면 노트북으로 분류 될까요? k=4 또는 k=5 인 경우 분류 결과가 바뀌나요? K=9인 경우는 어떻습니까?


##### your answer here

- k=3의 경우..  

별도의 데이터로 분류될 가능성이 높음

- k=4,5의 경우  

경우에 따라서 데스크탑, 노트북도 세부적인 성질에 따라 나눠질 수도..

- k=9의 경우..  

알 수 없는 포인트 마저 분류가..


k=9인 경우 알수 없는 포인트의 분류가 변경이 됩니다. 그래프를 보면 알 수 없는 포인트가 데스크탑이 아님을 알 수 있습니다. 그러나 KNN은 과반수 이상의 최근접 이웃의 클래스로 알수 없는 포인트를 분류하기 때문에, 잘못된 수의 k 값을 사용하면 KNN이 알수 없는 포인트를 잘못된 클래스로 분류할 수 있습니다. 따라서 KNN 알고리즘에서는 k(가장 가까운 이웃 수) 값을 파악하는 것이 항상 중요합니다. 모델을 조정하는 방법을 다음의 몇 가지 실습에서 알아보겠습니다! 지금은 KNN에서 k값이 왜 중요한 매개변수인지 이해하는 것으로 충분합니다.


## KNN 및 Iris Flower 데이터 세트를 사용한 꽃 분류



이제 Iris Flower 데이터 세트에 KNN 기법을 적용해 보겠습니다. 먼저 이전에 사용한 Iris Flower 데이터 세트를 pandas 데이터 프레임으로 읽어옵니다. 이전에 데이터 세트를 다운로드하지 않았다면 이 [링크](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/) 에서 파일을 다운로드하십시오. 다운로드할 파일은 iris.data 파일입니다. 자세한 내용은 아래 사진(출처: https://www.researchgate.net/Figure/Trollius-ranunculoide-flower-with-measured-traits_fig6_272514310) 을 참조하세요.



아래 사진에서 꽃받침(sepal)과 꽃잎(petal)이 무엇을 의미하는지 설명할 수 있습니까?


<img src = "./resources/PetalSepal1.PNG">


이제 데이터 세트를 다운로드하고 다음과 같이 탐색을 시작하겠습니다!

1. 제공된 웹 사이트에서 데이터 세트 다운로드

2. csv 파일 열기

3. 데이터 세트를 확인하기 위해 처음 5개 행 출력

4. 열 이름 추가



```python
#your code here
# 이미 다운로드 되어있음.
df = pd.read_csv("./AI_Next_Prj/[Dataset]_Module_18_(iris).data")
df = pd.DataFrame(df.values.tolist(), columns=["sepal_length", "sepal_width","petal_length", "petal_width", "class"])
df.head(5) # 5개 출력 다만 이렇게만 하면.. 열 이름이 없음.. 고로 위에 추가.
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>


## KNN 알고리즘을 위한 데이터 설정



KNN을 사용하여 꽃 또는 클래스 유형을 대상 변수로 데이터 세트를 분류해 보겠습니다. 이를 위해 컴퓨터가 이해할 수 있도록 범주형 클래스를 숫자로 변환해야 합니다. 라벨 인코팅을 사용하여 그렇게 할 수 있습니다.



### 라벨 인코딩

라벨 인코딩은 각 클래스 범주에 번호를 할당하는 것을 의미합니다. 예를 들어, 날씨 예측의 경우 비와 맑음 두 가지 클래스가 있는 경우 비는 0으로, 맑음은 1로 라벨을 지정할 수 있습니다. 이러한 방식으로 범주를 숫자로 변환할 수 있습니다.



iris 데이터 세트에 라벨 인코딩을 어떻게 적용하시겠습니까?


Iris Flower 데이터 세트의 클래스(꽃 유형)를 숫자로 변환할 수 있습니다. 데이터 세트 내의 클래스는 무엇이 있습니까? 총 몇 개의 클래스가 있습니까?



```python
#your answer here
len(df['class'].unique())
```

<pre>
3
</pre>
이제 iris 데이터 세트에 클래스를 라벨 인코딩을 합니다. 'Iris-setosa'는 1, 'Iris-versicolor'는 2, 'Iris-virginica'는 3이 될 수 있습니다. 아래는 클래스 중 하나에 대해 라벨 인코딩한 인코딩한 코드입니다. 모든 클래스에 대해 라벨 인코딩을 진행하는 코드로 편집해보세요.



```python
# 각 클래스의 데이터 포인트 수 출력
print(df['class'].value_counts())

# 다른 클래스에 대해 다른 숫자를 지정하는 딕셔너리
label_encode = {"class": {"Iris-setosa":1}}

# .replace를 사용하여 다른 클래스를 숫자로 변경
df.replace(label_encode,inplace=True)

# 각 클래스의 데이터 포인트 수를 출력하여 클래스가 숫자로 변경되었는지 확인
print(df['class'].value_counts())
```

<pre>
Iris-virginica     50
Iris-versicolor    50
Iris-setosa        49
Name: class, dtype: int64
Iris-virginica     50
Iris-versicolor    50
1                  49
Name: class, dtype: int64
</pre>

```python
#ANSWER:

# 각 클래스의 데이터 포인트 수 출력
print(df['class'].value_counts())

# 다른 클래스에 대해 다른 숫자를 지정하는 딕셔너리
label_encode = {"class": {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}}

# .replace를 사용하여 다른 클래스를 숫자로 변경
df.replace(label_encode,inplace=True)

# 각 클래스의 데이터 포인트 수를 출력하여 클래스가 숫자로 변경되었는지 확인
print(df['class'].value_counts())
```

<pre>
Iris-virginica     50
Iris-versicolor    50
1                  49
Name: class, dtype: int64
3    50
2    50
1    49
Name: class, dtype: int64
</pre>
<font color=blue>보너스: 범주를 숫자로 변환하는 또 다른 방법이 있습니다. 이 방법을 원-핫 인코딩이라고 합니다. 원-핫 인코딩은 범주를 이진(0 또는 1) 범주로 변경합니다. 예를 들어 카테고리로 비가 오거나 맑은 날이 있는 경우 원-핫 인코딩은 데이터 프레임에 2개의 열(비오는 날과 맑은 날)을 추가합니다. 데이터 포인트에 비인 열은 값 1로 변환되고, 맑은 열은 값 0으로 변환됩니다(아래 표 참조). 이 [문서](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) 를 읽고 원-핫 인코딩에 대해 자세히 알아보세요. 이제 이 [문서](http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example) 를 읽고 pandas에서 원-핫 인코딩을 하는 방법을 배워보세요. Iris Flower 데이터 세트를 df2로 다시 가져오고 이에 대한 원-핫 인코딩을 수행합니다.</font>



```python
df2 = pd.read_csv("./AI_Next_Prj/[Dataset]_Module_18_(iris).data",header=None)
names = ["sepal_length", "sepal_width","petal_length", "petal_width", "class"]
df2.columns = names
```


```python
#your code 
df2_1 = pd.get_dummies(df2,prefix=['class'])
df2_1.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class_Iris-setosa</th>
      <th>class_Iris-versicolor</th>
      <th>class_Iris-virginica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


잘하셨습니다! 레이블 인코딩 및 원-핫 인코딩을 수행하는 방법을 배웠습니다. 이제 KNN 알고리즘을 테스트해 볼 수 있습니다. 먼저 scikit learn에서 KNN 알고리즘을 가져와야 합니다. [Scikit Learn](https://scikit-learn.org/stable/) 은 널리 사용되고 있는 많은 수의 기계 학습 알고리즘을 포함하고 있는 오픈 소스 Python 라이브러리입니다. scikit learn에는 어떤 내용이 포함되어 있습니까? 무엇을 할 수 있습니까? 활용 예를 확인하려면 [예제](https://scikit-learn.org/stable/auto_examples/index.html) 사이트를 참조하십시오! scikit learn 만으로도 정말 많은 것을 해볼 수 있습니다.



이제 아래 코드를 실행해 보십시오.



```python
from sklearn.neighbors import KNeighborsClassifier
```

이 [링크](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) 를 읽고 sklearn에서 KNeighborsClassifier를 사용하는 방법을 알아보세요. 이 노트북의 앞부분에서 KNN에 k 값이 중요하다는 것을 확인하였습니다. 링크에서 사용되는 k값은 얼마입니까?


##### your answer here

- 디폴트로 k=5 로 지정


위에서 찾은 k를 사용하여 Iris Flower 데이터 세트를 분류해 보겠습니다.



처음에는 2가지 특성(sepal_length 및 sepal_width)만 사용합니다. 아래 코드에서 이들은 'x' 데이터 프레임에 포함됩니다. 꽃의 '클래스', 즉 꽃 종의 이름은 'y' 데이터 프레임에 포함된 '라벨'이 됩니다.



아래 코드를 실행해 보십시오.



```python
# KNeighborsClassifier 초기화
KNN = KNeighborsClassifier()

# x 값과 y 값을 추출합니다. x는 특성이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 KNN을 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
knn = KNN.fit(x,y)

# sepal_length = 5 및 sepal_width = 3인 경우 학습된 KNN을 사용하여 꽃의 유형을 예측합니다. 
# .predict 메서드를 사용할 수 있습니다.
test = pd.DataFrame()
test['sepal_length'] = [5]
test['sepal_width'] = [3]
predict_flower = KNN.predict(test)

# predict_flower 출력
print(predict_flower)
```

<pre>
   sepal_length  sepal_width
0           4.9          3.0
1           4.7          3.2
2           4.6          3.1
3           5.0          3.6
4           5.4          3.9
0    1
1    1
2    1
3    1
4    1
Name: class, dtype: int64
[1]
</pre>
위의 출력에서 sepal_length = 5, sepal_width = 3인 데이터 포인트에 대해 어떤 유형의 꽃이 예측되었습니까? (이전 라벨 인코딩을 참조하세요)


##### your answer here

- 1 : Iris-setosa 로 예측


sepal_length = 5, sepal_width = 3인 경우에는 어떻게 될까요? 위에 코드를 변경하여 알아보세요!



```python
#your code here 나중에 다시 확인...
```


```python
#your answer here
```

첫 번째 지도 학습 모델을 훈련시켰습니다. 그러나 우리는 2개의 변수만 사용했습니다. 꽃받침 길이(sepal_length)와 꽃받침 너비(sepal_width) 대신 다른 모든 변수('sepal_length','sepal_width','petal_length','petal_width')를 사용하여 KNN2라는 다른 KNN 모델을 훈련시켜 봅시다.



```python
#your code here
# KNeighborsClassifier 초기화
KNN2 = KNeighborsClassifier()

# x 값과 y 값을 추출합니다. x는 특성이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 KNN을 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
knn = KNN.fit(x,y)
```

<pre>
   sepal_length  sepal_width  petal_length  petal_width
0           4.9          3.0           1.4          0.2
1           4.7          3.2           1.3          0.2
2           4.6          3.1           1.5          0.2
3           5.0          3.6           1.4          0.2
4           5.4          3.9           1.7          0.4
0    1
1    1
2    1
3    1
4    1
Name: class, dtype: int64
</pre>
새 모델을 사용하여 sepal_length = 5.8, sepal_width = 2.3, feather_length = 5.0 및 feather_width = 1.3인 꽃 유형을 예측해 봅시다.



```python
#your code here
test = pd.DataFrame()
test['sepal_length'] = [5.8]
test['sepal_width'] = [2.3]
test['feather_length'] = [5.0]
test['feather_width'] = [1.3]
predict_flower = KNN.predict(test)

# predict_flower 출력
print(predict_flower)
```

<pre>
[2]
</pre>
##### your answer here

k=2 >> Iris-versicolor 가 예측


잘하셨습니다! KNN 알고리즘을 사용하여 꽃받침 길이, 꽃받침 너비, 꽃잎 길이 및 꽃잎 너비와 같은 꽃의 특성을 고려하여 꽃의 유형을 분류할 수 있는 분류 모델을 훈련시켰습니다.



이런 종류의 모델은 어떻게 유용할까요? 아래에 답을 작성해 주세요!


##### your answer here

.....


KNN 알고리즘을 어떤 곳에 사용할 수 있다고 생각하십니까? 아래에 답을 작성해 주세요!


##### your answer here

.....


분류에 사용할 수 있는 많은 모델이 있습니다. 다음으로 분류를 잘 할 수 있도록 도와주는 의사 결정 트리를 살펴보겠습니다.


## 1.2 의사 결정 트리(Decision Trees)


또 다른 지도 기계 학습 기술은 의사 결정 트리입니다. 이 [동영상](https://www.youtube.com/watch?v=eKD5gxPPeY00) 을 시청하고 결정 트리에 대해 자세히 알아보세요. 시청한 후 워크시트에 의사 결정 트리이 예를 그려보세요.(아래에 표시된 예를 참조)



우리는 매일 수많은 결정을 내립니다. 여러분은 특정한 결정을 어떻게 내리나요? 어떤 종류의 의사 결정 트리를 작성하실 건가요?


<img src = "./resources/dt1.jpg">


의사 결정 트리를 사용하려면 먼저 데이터로 모델을 훈련해야 합니다. 훈련 과정에서 알고리즘은 특정 매개변수를 기반으로 데이터 세트를 분할합니다. 지금은 알고리즘이 자동으로 매개변수를 계산하므로 이러한 매개변수를 계산하는 방법을 알 필요는 없습니다. 의사 결정 트리가 매개변수를 계산하는 방법에 대해 자세히 알아보려면 온라인에서 검색하여 찾아 볼 수 있습니다.



훈련 과정이 끝나면 의사 결정 트리는 결정 지점을 "기억"하고 새로운 데이터 세트에 적용하여 분류를 예측합니다. 이제 의사 결정 트리를 사용하여 Iris Flower 데이터 세트를 분류해 보겠습니다.


먼저 scikit learn에서 의사 결정 트리를 가져와야 합니다.



```python
from sklearn import tree
```

다음으로 이 노트북의 앞부분에서 사용한 것과 동일한 df 데이터 프레임을 사용합니다. df 데이터 프레임에는 Iris Flower 데이터 세트가 포함되어 있으며 출력은 라벨 인코딩되어 있습니다. 먼저 꽃받침 길이(epal_length)와 꽃받침 너비(sepal_width)를 x 값으로 사용하고 클래스를 대상/결과 또는 y 값으로 사용합니다. 아래 코드를 실행해 보십시오.



```python
# 의사 결정 트리 초기화
dt = tree.DecisionTreeClassifier()

# x 값과 y 값을 추출합니다. x는 sepal_length, sepal_width이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 의사결정 트리를 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
dt = dt.fit(x,y)
```

<pre>
   sepal_length  sepal_width
0           4.9          3.0
1           4.7          3.2
2           4.6          3.1
3           5.0          3.6
4           5.4          3.9
0    1
1    1
2    1
3    1
4    1
Name: class, dtype: int64
</pre>
의사 결정 트리를 초기화 시키는 코드를 보십시오. DecisionTreeRegressor 대신 DecisionTreeClassifier를 사용하는 이유는 무엇입니까?


##### your answer here

- 재귀보다 여기서는 분류이기에..


이제 훈련된 의사 결정 트리를 사용하여 또 다른 예측을 진행 할 수 있습니까? 

sepal_length = 5, sepal_width = 3인 경우에는 어떻게 될까요? 위에 코드를 변경하여 알아보세요!



```python
# Create a dataframe called test2 with sepal_length and sepal_width as its columns. Sepal_length has been done for you in the code below.
test2 = pd.DataFrame()
test2['sepal_length'] = [5]
test2['sepal_width'] = [3]

# Use the .predict method to predict the new flower. You can call the predicted flower as predict_flower.
predict_flower = dt.predict(test2)

# Print predict_flower
print(predict_flower)
```

<pre>
[1]
</pre>
예측된 꽃 유형은 무엇이며 이전에KNN에서 예측한 유형과 동일합니까?


##### your answer here

-  KNN과 동일하게 1: Iris-setosa로 예측


이제 꽃받침 길이, 꽃받침 너비, 꽃잎 길이 및 꽃잎 너비를 기반으로 새로운 의사 결정 트리 dt2를 훈련시키십시오. 



sepal_length = 5.8, sepal_width = 2.3, feather_length = 5.0, feather_width = 1.3 으로 꽃 유형을 예측합니다.



```python
#your code here

# 의사 결정 트리 초기화
dt2 = tree.DecisionTreeClassifier()

# x 값과 y 값을 추출합니다. x는 sepal_length, sepal_width이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width', 'petal_length', 'petal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 의사결정 트리를 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
dt2 = dt2.fit(x,y)

# Create a dataframe called test2 with sepal_length and sepal_width as its columns. Sepal_length has been done for you in the code below.
test3 = pd.DataFrame()
test3['sepal_length'] = [5.8]
test3['sepal_width'] = [2.3]
test3['feather_length'] = [5.0]
test3['feather_width'] = [1.3]

# Use the .predict method to predict the new flower. You can call the predicted flower as predict_flower.
predict_flower = dt2.predict(test3)

# Print predict_flower
print(predict_flower)
```

<pre>
   sepal_length  sepal_width  petal_length  petal_width
0           4.9          3.0           1.4          0.2
1           4.7          3.2           1.3          0.2
2           4.6          3.1           1.5          0.2
3           5.0          3.6           1.4          0.2
4           5.4          3.9           1.7          0.4
0    1
1    1
2    1
3    1
4    1
Name: class, dtype: int64
[3]
</pre>
새로 학습된 의사 결정 트리가 예측하는 종류는 무엇입니까? KNN에서 예측한 것과 동일한가요?


##### your answer here

- 3: Iris-virginica 결과가 나옴,  2 : Iris-versicolor 로 예측된 KNN과 다른 결과가 나옴.


위의 결과를 통해 대부분의 기계 학습 기술이 동일한 문제를 해결하는 데 사용될 수 있지만 결과/예측값은 모델에 따라 다를 수 있음을 알 수 있습니다! 따라서 모델의 정확도에 따라 어떤 모델을 사용할지 선택해야 합니다. 이후 노트북 레슨에서 모델의 품질을 평가하는 방법을 배울 것입니다.


#### 마무리

- KNN, 의사결정 등을 통한 붓꽃 종류분류 지도학습을 예전에도 했으나 정리한다는 느낌으로 진행

- 두 모델다 어떤게 좋을지 안 좋을지 그냥 판단하기 힘든부분.

- 위 탬플릿 설명대로 데이터 상태, 구분대상 등 여러 종합적으로 봐서 어떠한 모델이 나은지 해봐야 될듯함.

- 그렇다 하더라도 해보고 보자 식으로 구별해서 선별할 가능성이...


#### 주의사항



본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.

