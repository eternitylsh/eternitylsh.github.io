---
layout: single
title:  "과소, 과대를 통한 붗꽃데이터를 이용한 ML고찰"
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


# 모델의 결과 출력


앞서 모델을 훈련하는 방법과 예측에 사용하는 방법을 배웠습니다. 이는 모든 데이터 과학 프로젝트에서 필수적이지만, 단순히 모델을 훈련하고 예측에 활용하는 것만으로는 충분하지 않습니다. 이는 모델이 학습되었음에도 성능을 평가하지 않았기 때문입니다.



모델을 훈련하는 데 사용하지 않은 데이터를 이용하여 모델을 평가해야 합니다. 이는 모델이 훈련된 데이터에 대해서는 예측을 잘 수행할 수 있지만, 훈련에 사용하지 않은 데이터를 모델에 사용하였을 때 성능이 좋지 않으면 사용할 수 없기 때문입니다. 따라서 이 노트북에서는 이러한 일이 발생하지 않도록 하는 방법을 배웁니다. 그렇게 하기 전에 먼저 모델이 훈련 후에 훈련되지 않은 데이터로 인해 성능이 저하되는 원인을 먼저 이해해야 합니다. 이것은 과소적합(underfitting) 또는 과대적합(overfitting) 때문일 수 있습니다.



```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import tree
```

## 1. 과소적합 vs 과대적합


과소적합은 모델이 너무 단순화되어 데이터를 제대로 설명할 수 없음을 의미합니다. 예를 들어, 비선형 관계가 있는 데이터가 있지만 선형 모델을 사용하여 학습하는 경우 과소적합이 발생할 수 있습니다. 이는 선형 모델이 데이터에서 관찰되는 비선형 관계나 추세를 설명할 수 없기 때문입니다. 따라서 과소적합된 모델을 사용하여 예측하면 성능이 저하됩니다.



반면에 과대적합은 모델이 너무 잘 적합되어 데이터 세트 내의 모든 노이즈 또는 이상값도 학습했다는 것을 의미합니다. 따라서 훈련된 데이터에 대해 테스트하면 성능이 매우 우수합니다. 그러나 너무 잘 훈련되었기 때문에 일반화할 수 없으며 훈련되지 않은 데이터에 대해 테스트할 때는 성능이 좋지 않습니다.



이 [문서](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76) 와 이 [문서 ](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229) 에서 과소적합 및 과대적합에 대한 자세한 내용을 확인하고, 흥미로운 내용을 기록하십시오. 기사 내에 수학 방정식이 있을 수 있지만 방정식을 이해할 필요는 없습니다. 기본 개념을 이해하는 것이 더 중요합니다. 

- 편향과 분산의 차이점은 무엇입니까? 

- 과소적합을 방지하는 방법은 무엇입니까? 

- 과대적합을 방지하는 방법은 무엇입니까?


##### your answer here

- 편향과 분산의 차이점

  - 편향 : 중심에서 어느정도 데이터들의 집합이 치우쳐져 있는가의 정도

  - 분산 : 데이터가 어느정도 퍼져있는가의 정도



- 과소적합 방지 방법

  - 모델 구조를 좀더 세부적으로 구성



- 과대적합 방지 방법

  - 훈련 반복 횟수 줄여보기

  - 훈련 데이터 양 늘려



- 참고 사이트

  https://heytech.tistory.com/125


아래 그래프를 보십시오. 빨간색 선이 모델이고 파란색 점이 데이터 세트인 경우 모델이 과소적합인가요? 과대적합인가요?


<img src = './resources/model1.jpg'>


##### your answer here

- 과소적합 가능성이 높다.


<img src = './resources/model2.jpg'>


###### your answer here

과대적합 가능성이 높다.


## 2. 과소적합과 과대적합의 균형 맞추기


위에서 보면 과소적합 및 과대적합 사이의 균형을 찾는 것이 중요하다는 것을 알 수 있습니다. 이를 통해 모델은 정확하면서도 훈련되지 않은 데이터에 대해서도 예측을 잘 수행할 수 있도록 일반화할 수 있습니다. 이는 모델이 다소 복잡하지만 너무 복잡하지 않아야 함을 의미합니다. 따라서 균형을 이루기 위해 노력할 수 있는 몇 가지 방법이 있습니다. 우리는 이전에 탐색한 다양한 기계 학습 기술에 대해 이러한 방법 중 일부를 시도할 것입니다.


## 2.1 k-최근접 이웃(K-Nearest Neighbor) 알고리즘


이전 노트북에서 KNN(K-Nearest Neighbor) 알고리즘을 적용하여 데이터를 분류하는 방법을 배웠습니다. KNN은 문제의 지점에 가장 가까운 대부분의 다른 지점을 기반으로 데이터 지점을 분류합니다. 그러나 알고리즘을 사용하기 위해서는 이웃의 수를 매개변수로 입력해야 합니다. 과소적합과 과대적합의 경우 이웃의 수가 중요한 역할을 합니다. 이는 이웃의 수가 모델이 과대, 과소적합될 가능성을 결정하기 때문입니다. 이웃 수가 많을수록 모델이 과소적합될 가능성이 줄어듭니다. 이웃 수가 너무 많으면 모델이 과대적합될 가능성이 높습니다. 따라서 모델이 상대적으로 균형을 이룰 수 있도록 적합한 이웃의 수가 있어야 합니다. 이 숫자는 데이터 세트에 따라 달라질 수 있습니다. 이제 앞서 살펴보았던 Iris Flower 데이터 세트에 대해 이웃의 숫자를 찾아봅시다.


먼저 Iris.data에서 데이터 프레임 df로 Iris Flower 데이터 세트를 읽어야 합니다. 데이터 프레임에 열 이름이 있는지 확인하고 데이터 세트에 대한 표준 검사도 수행해야 합니다(예: 오류 데이터 및 이상값 확인). 아래 그림(출처: https://www.researchgate.net/Figure/Trollius-ranunculoide-flower-with-measured-traits_fig6_272514310) 을 참고하여 변수를 이해할 수 있습니다.


<img src = "./resources/PetalSepal1.png">



```python
#your code here
df = pd.read_csv('./AI_Next_Prj/[Dataset]_Module_18_(iris).data')
names = ["sepal_length", "sepal_width","petal_length", "petal_width","class"]
df.columns = names
```

**일부러 셀 나누어서 출력**



```python
df.head(5)
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



```python
df.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.848322</td>
      <td>3.051007</td>
      <td>3.774497</td>
      <td>1.205369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828594</td>
      <td>0.433499</td>
      <td>1.759651</td>
      <td>0.761292</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.400000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>


이제 기계학습 알고리즘이 처리할 데이터를 준비하는 데 필요한 단계를 수행합니다. 먼저 특성을 x_values로 추출하고 대상 변수를 y_values로 추출합니다. 이 경우 x_values는 "sepal_length", "sepal_width", "petal_length" 및 "petal_width"가 되는 반면 y_values는 클래스가 됩니다. 또한 y_values에 레이블을 지정해야 합니다. "Setosa"는 0, "Versicolor"는 1, "Virginica"는 2로 지정할 수 있습니다. 이전 노트북을 참조하여 필요한 코드를 확인해 보세요.



```python
#your code here
# 기존 라벨 데이터 출력
print(df['class'].value_counts())

# 인코딩 진행!
label_encode = {"class": {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}}
df.replace(label_encode,inplace=True)

# 결과
print(df['class'].value_counts())
```

<pre>
Iris-versicolor    50
Iris-virginica     50
Iris-setosa        49
Name: class, dtype: int64
1    50
2    50
0    49
Name: class, dtype: int64
</pre>
기계 학습 기술로 데이터를 처리할 준비가 되었음을 확인하였다면, 과대적합과 과소적합 문제의 균형을 맞추는 방법에 초점을 맞춰야 합니다.



이 균형을 결정하려면 모델이 훈련되지 않은 데이터에 적용되는 경우 모델의 성능이나 정확성을 평가할 수 있어야 합니다. 미래 데이터가 아직 생성되지 않았기 때문에 현재 데이터를 사용하여 이 평가를 수행할 수 있어야 합니다. 따라서 현재 데이터 세트는 일반적으로 2개의 다른 그룹으로 나누어 사용합니다. 한 그룹에는 데이터를 훈련하는 데 사용할 모든 훈련 데이터가 포함됩니다. 다른 그룹에는 모델 학습 단계에서 사용하지 않는 테스트 데이터가 포함됩니다. 테스트 데이터는 모델을 평가하는 데 사용되는 모델 학습 이후의 "미래" 데이터로 사용됩니다.



데이터를 2개의 그룹으로 분할하기 위해 sklearn.model_selection의 train_test_split 함수를 사용합니다. train_test_split 함수를 가져오려면 아래 코드를 실행하십시오.



```python
from sklearn.model_selection import train_test_split
```

train_test_split 함수를 사용하여 데이터를 훈련 그룹과 테스트 그룹으로 나눕니다. 테스트 그룹은 일반적으로 데이터 세트의 20%에서 30%를 포함합니다. Iris Flower 데이터에 대하여 훈련 그룹 75%와 테스트 그룹 25% 기준으로 데이터를 분할할 수 있습니다. train_test_split 함수를 사용하는 방법을 이해하려면 이 [문서](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6) 를 참고하세요. 학습 및 테스트 데이터를 보유하기 위해 x_train, y_train, x_test 및 y_test라는 변수를 생성할 수 있습니다. 또한 random_state를 추가하여 코드를 실행할 때마다 데이터가 항상 같은 방식으로 분할되도록 할 수도 있습니다.



```python
#your code here
x_values = df[['sepal_length','sepal_width','petal_length','petal_width']]
y_values = df['class']

x_train, x_test, y_train, y_test = train_test_split(x_values,y_values,test_size=0.25,random_state=10)
```

데이터를 분할한 후 이제 데이터를 표준화하거나 정규화해야 합니다. 기계 학습에 사용하는 데이터 세트를 표준화하거나 정규화하는 것은 항상 좋은 습관입니다. 이렇게 하면 모든 변수 또는 특성의 값을 유사한 범위로 확장하는 데 도움이 됩니다. 데이터를 분할한 후에는 항상 표준화 또는 정규화를 수행해야 합니다. 이는 테스트 데이터 세트가 항상 모델에 노출되지 않고, 훈련 데이터의 정규화 또는 표준화 프로세스에 사용되지 않도록 하기 위한 것입니다.



이 경우 sklearn.preprocessing의 StandardScaler를 사용하여 데이터를 표준화하도록 선택합니다. .fit_transform 메서드를 x_train 데이터 값에 적용하지만 x_test 데이터에는 .transform 메서드만 적용 합니다. 아래 셀에서 표준화 프로세스를 구현합니다. 표준화 후 훈련 데이터에 대해 x_train_scale이라는 변수를 생성하고, 표준화 후 테스트 데이터에 대해 x_test scale이라는 또 다른 변수를 생성합니다.



```python
#your code here

scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)
```


```python
# x_train_scale data until 5
x_train_scale[:5,:]
```

<pre>
array([[ 0.61249909,  0.28500697,  0.84017212,  1.40227859],
       [-1.12779269, -1.34330492,  0.39766154,  0.6298075 ],
       [-0.77973433,  0.98285493, -1.26175315, -1.3013702 ],
       [ 0.49647964,  0.51762296,  0.50828918,  0.50106232],
       [-1.70788995,  0.28500697, -1.37238079, -1.3013702 ]])
</pre>

```python
# x_test_scale data until 5
x_test_scale[:5,:]
```

<pre>
array([[-0.08361762, -0.87807295,  0.0657786 , -0.0139184 ],
       [ 0.49647964, -0.64545697,  0.72954447,  0.37231714],
       [-1.24381214,  0.75023894, -1.20643932, -1.3013702 ],
       [-0.31565653, -0.180225  ,  0.17640625,  0.11482678],
       [-0.43167598, -1.8085369 ,  0.12109242,  0.11482678]])
</pre>
데이터를 표준화한 후 이제 KNN 알고리즘에 대한 최적의 이웃 수를 찾는 방법을 구현할 수 있습니다. 이를 위해 서로 다른 수의 이웃으로 KNN 알고리즘을 훈련하고 테스트 데이터와 비교하여 평가합니다. 그렇게 함으로써, 우리는 다른 수의 이웃에 대한 KNN 모델의 정확도를 얻을 수 있을 것입니다. 그런 다음 가장 높은 정확도에 해당하는 이웃의 수를 찾을 수 있습니다. 그 수가 최적의 이웃 수가 됩니다. 아래 코드를 실행해보세요!



```python
# 각 KNN 모델에 대한 정확도와 이웃 수를 저장하기 위해 빈 목록을 만듭니다.
accuracy = []
num_neigh = []

# ii를 사용하여 값 1에서 15까지 반복합니다. 이것은 KNN 분류기의 이웃 수가 됩니다.
for ii in range(1,16):
    # 이웃 수를 ii로 설정
    KNN = KNeighborsClassifier(n_neighbors=ii)
    # 데이터로 모델 훈련 또는 피팅
    KNN.fit(x_train_scale,y_train)
    # .score는 테스트 데이터를 기반으로 모델의 정확도를 제공합니다. 정확도를 목록에 저장합니다.
    accuracy.append(KNN.score(x_test_scale,y_test))
    # 목록에 이웃 수 추가
    num_neigh.append(ii)

print(accuracy);
```

<pre>
[0.9210526315789473, 0.8947368421052632, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9210526315789473, 0.9210526315789473, 0.9473684210526315, 0.8947368421052632, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473]
</pre>

최적의 이웃 수를 결정하는 데 도움이 되도록 그래프에 정확도 값을 표시해 보겠습니다. 아래 코드를 실행해보세요! 

matplotlib.pyplot을 plt로 가져와야 합니다.



```python
plt.scatter(num_neigh,accuracy)
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show();
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkAAAAG2CAYAAACXuTmvAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA25UlEQVR4nO3df1hUdd7/8dcwIjMpkEoCpgK1rmKUChgKpVmJWf7gvmvVdqXMvWptLWV1u9UtL3+0iT/SdlNhRZdWs4K73czctTYqK43dJjAqpbC7VTEFWX+B5i3icL5/+GXuJtAcAw5wno/rOtflfOZzzud9zswwL8+cHzbDMAwBAABYiJ/ZBQAAADQ3AhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAc0wNQRkaGoqKi5HA4FBcXp+3bt1+0/+rVqxUdHS2n06nevXtrw4YNXs//6U9/ks1mqzedOXOmKVcDAAC0Iu3MHDw3N1dpaWnKyMhQUlKS1qxZo5EjR6q4uFg9e/as1z8zM1Nz5szR2rVrNXDgQLlcLj344IPq1KmTRo8e7ekXFBSkkpISr3kdDkeTrw8AAGgdbGbeDDUhIUGxsbHKzMz0tEVHRyslJUXp6en1+icmJiopKUnLli3ztKWlpamgoEA7duyQdH4PUFpamk6cONHk9QMAgNbJtD1AZ8+eVWFhoWbPnu3VnpycrPz8/Abnqa6urrcnx+l0yuVyqaamRv7+/pKkU6dOKSIiQm63W/3799eTTz6pAQMGXLCW6upqVVdXex7X1tbq2LFj6tKli2w22+WuIgAAaEaGYejkyZPq1q2b/Py+5ygfwyQHDx40JBkffPCBV/tTTz1l/PjHP25wnjlz5hhhYWFGQUGBUVtba3z00UdG165dDUnGoUOHDMMwjH/84x/G888/bxQVFRnvv/++cffddxtOp9PYs2fPBWuZN2+eIYmJiYmJiYmpDUwHDhz43hxi6jFAkurtYTEM44J7XebOnavy8nINGjRIhmEoNDRUkyZN0tKlS2W32yVJgwYN0qBBgzzzJCUlKTY2VitXrtSzzz7b4HLnzJmjGTNmeB5XVlaqZ8+eOnDggIKCgn7oKgIAgGZQVVWlHj16KDAw8Hv7mhaAQkJCZLfbVV5e7tVeUVGh0NDQBudxOp3Kzs7WmjVrdPjwYYWHhysrK0uBgYEKCQlpcB4/Pz8NHDhQX3755QVrCQgIUEBAQL32oKAgAhAAAK3MpRy+Ytpp8O3bt1dcXJzy8vK82vPy8pSYmHjRef39/dW9e3fZ7Xbl5ORo1KhRF/ytzzAMFRUVKTw8vNFqBwAArZupP4HNmDFDqampio+P1+DBg5WVlaXS0lJNmTJF0vmfpg4ePOi51s+ePXvkcrmUkJCg48ePa8WKFdq1a5fWr1/vWeaCBQs0aNAg9erVS1VVVXr22WdVVFSk1atXm7KOAACg5TE1AI0fP15Hjx7VwoULVVZWppiYGG3dulURERGSpLKyMpWWlnr6u91uLV++XCUlJfL399ewYcOUn5+vyMhIT58TJ07ooYceUnl5uYKDgzVgwAC9//77uvHGG5t79QAAQAtl6nWAWqqqqioFBwersrKSY4AAAGglfPn+Nv1WGAAAAM2NAAQAACyHAAQAACyHAAQAACyHAAQAACzH9FthwDrctYZce4+p4uQZdQ106MaozrL7Nd/NZs0ev6XUYGVsfwB1CEBoFm/sKtOCLcUqqzzjaQsPdmje6L66I6bpr9Jt9vgtpQYrY/sD+DauA9QArgPUuN7YVaaHN+7Ud99odf/vzpwY26RfQGaP31JqsDK2P2ANXAcILYa71tCCLcX1vngkedoWbCmWu7ZpcrjZ47eUGqyM7Q+gIQQgNCnX3mNePzl8lyGprPKMXHuPtcnxW0oNVsb2B9AQAhCaVMXJC3/xXE6/1jZ+S6nBytj+ABpCAEKT6hroaNR+rW38llKDlbH9ATSEAIQmdWNUZ4UHO3ShE41tOn8mzo1Rndvk+C2lBitj+wNoCAEITcruZ9O80X0lqd4XUN3jeaP7Ntm1WMwev6XUYGVsfwANIQChyd0RE67MibEKC/b+iSEs2NEspx+bPX5LqcHK2P4AvovrADWA6wA1DbOvwmv2+C2lBitj+wNtmy/f3wSgBhCAAABofbgQIgAAwEUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOUQgAAAgOWYHoAyMjIUFRUlh8OhuLg4bd++/aL9V69erejoaDmdTvXu3VsbNmy4YN+cnBzZbDalpKQ0ctUAAKA1a2fm4Lm5uUpLS1NGRoaSkpK0Zs0ajRw5UsXFxerZs2e9/pmZmZozZ47Wrl2rgQMHyuVy6cEHH1SnTp00evRor7779+/Xr3/9a918883NtToAAKCVsBmGYZg1eEJCgmJjY5WZmelpi46OVkpKitLT0+v1T0xMVFJSkpYtW+ZpS0tLU0FBgXbs2OFpc7vdGjp0qB544AFt375dJ06c0KuvvnrJdVVVVSk4OFiVlZUKCgq6vJUDAADNypfvb9N+Ajt79qwKCwuVnJzs1Z6cnKz8/PwG56murpbD4fBqczqdcrlcqqmp8bQtXLhQV111lX7+859fUi3V1dWqqqrymgAAQNtlWgA6cuSI3G63QkNDvdpDQ0NVXl7e4DwjRozQunXrVFhYKMMwVFBQoOzsbNXU1OjIkSOSpA8++EB//OMftXbt2kuuJT09XcHBwZ6pR48el79iAACgxTP9IGibzeb12DCMem115s6dq5EjR2rQoEHy9/fX2LFjNWnSJEmS3W7XyZMnNXHiRK1du1YhISGXXMOcOXNUWVnpmQ4cOHDZ6wMAAFo+0w6CDgkJkd1ur7e3p6Kiot5eoTpOp1PZ2dlas2aNDh8+rPDwcGVlZSkwMFAhISH69NNPtW/fPq8DomtrayVJ7dq1U0lJia699tp6yw0ICFBAQEAjrh0AAGjJTNsD1L59e8XFxSkvL8+rPS8vT4mJiRed19/fX927d5fdbldOTo5GjRolPz8/9enTR5999pmKioo805gxYzRs2DAVFRXx0xYAAJBk8mnwM2bMUGpqquLj4zV48GBlZWWptLRUU6ZMkXT+p6mDBw96rvWzZ88euVwuJSQk6Pjx41qxYoV27dql9evXS5IcDodiYmK8xrjyyislqV47AACwLlMD0Pjx43X06FEtXLhQZWVliomJ0datWxURESFJKisrU2lpqae/2+3W8uXLVVJSIn9/fw0bNkz5+fmKjIw0aQ0AAEBrZOp1gFoqrgMEAEDr0yquAwQAAGAWAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAc0wNQRkaGoqKi5HA4FBcXp+3bt1+0/+rVqxUdHS2n06nevXtrw4YNXs+/8sorio+P15VXXqkOHTqof//+ev7555tyFQAAQCvTzszBc3NzlZaWpoyMDCUlJWnNmjUaOXKkiouL1bNnz3r9MzMzNWfOHK1du1YDBw6Uy+XSgw8+qE6dOmn06NGSpM6dO+vxxx9Xnz591L59e/31r3/VAw88oK5du2rEiBHNvYoAAKAFshmGYZg1eEJCgmJjY5WZmelpi46OVkpKitLT0+v1T0xMVFJSkpYtW+ZpS0tLU0FBgXbs2HHBcWJjY3XXXXfpySefvKS6qqqqFBwcrMrKSgUFBfmwRgAAwCy+fH+b9hPY2bNnVVhYqOTkZK/25ORk5efnNzhPdXW1HA6HV5vT6ZTL5VJNTU29/oZh6O2331ZJSYmGDBlywVqqq6tVVVXlNQEAgLbLtAB05MgRud1uhYaGerWHhoaqvLy8wXlGjBihdevWqbCwUIZhqKCgQNnZ2aqpqdGRI0c8/SorK9WxY0e1b99ed911l1auXKnhw4dfsJb09HQFBwd7ph49ejTOSgIAgBbJ9IOgbTab12PDMOq11Zk7d65GjhypQYMGyd/fX2PHjtWkSZMkSXa73dMvMDBQRUVF+uijj/TUU09pxowZevfddy9Yw5w5c1RZWemZDhw48IPXCwAAtFymBaCQkBDZ7fZ6e3sqKirq7RWq43Q6lZ2drdOnT2vfvn0qLS1VZGSkAgMDFRIS4unn5+enH/3oR+rfv79mzpype+65p8FjiuoEBAQoKCjIawIAAG2XaQGoffv2iouLU15enld7Xl6eEhMTLzqvv7+/unfvLrvdrpycHI0aNUp+fhdeFcMwVF1d3Sh1AwCA1s/U0+BnzJih1NRUxcfHa/DgwcrKylJpaammTJki6fxPUwcPHvRc62fPnj1yuVxKSEjQ8ePHtWLFCu3atUvr16/3LDM9PV3x8fG69tprdfbsWW3dulUbNmzwOtMMAABYm6kBaPz48Tp69KgWLlyosrIyxcTEaOvWrYqIiJAklZWVqbS01NPf7XZr+fLlKikpkb+/v4YNG6b8/HxFRkZ6+nzzzTf65S9/qa+//lpOp1N9+vTRxo0bNX78+OZePQAA0EKZeh2glorrAAEA0Pq0iusAAQAAmIUABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALMfnABQZGamFCxeqtLS0KeoBAABocj4HoJkzZ2rz5s265pprNHz4cOXk5Ki6uropagMAAGgSPgegRx99VIWFhSosLFTfvn01bdo0hYeH65FHHtHOnTubokYAAIBGZTMMw/ghC6ipqVFGRoZmzZqlmpoaxcTEaPr06XrggQdks9kaq85mVVVVpeDgYFVWViooKMjscgAAwCXw5fu73eUOUlNTo02bNum5555TXl6eBg0apJ///Oc6dOiQHn/8cb311lt68cUXL3fxAAAATcbnALRz504999xzeumll2S325WamqpnnnlGffr08fRJTk7WkCFDGrVQAACAxuJzABo4cKCGDx+uzMxMpaSkyN/fv16fvn37asKECY1SIAAAQGPzOQD961//UkRExEX7dOjQQc8999xlFwUAANCUfD4LrKKiQh9++GG99g8//FAFBQWNUhQAAEBT8jkATZ06VQcOHKjXfvDgQU2dOrVRigIAAGhKPgeg4uJixcbG1msfMGCAiouLG6UoAACApuRzAAoICNDhw4frtZeVlaldu8s+qx4AAKDZ+ByAhg8frjlz5qiystLTduLECf3mN7/R8OHDG7U4AACApuDzLpvly5dryJAhioiI0IABAyRJRUVFCg0N1fPPP9/oBQIAADQ2nwPQ1VdfrU8//VQvvPCCPvnkEzmdTj3wwAO69957G7wmEAAAQEtzWQftdOjQQQ899FBj1wIAANAsLvuo5eLiYpWWlurs2bNe7WPGjPnBRQEAADSly7oS9H/8x3/os88+k81mU93N5Ovu/O52uxu3QgAAgEbm81lg06dPV1RUlA4fPqwrrrhCu3fv1vvvv6/4+Hi9++67TVAiAABA4/J5D9A//vEPvfPOO7rqqqvk5+cnPz8/3XTTTUpPT9e0adP08ccfN0WdAAAAjcbnPUBut1sdO3aUJIWEhOjQoUOSpIiICJWUlDRudQAAAE3A5z1AMTEx+vTTT3XNNdcoISFBS5cuVfv27ZWVlaVrrrmmKWoEAABoVD4HoCeeeELffPONJOm3v/2tRo0apZtvvlldunRRbm5uoxcIAADQ2GxG3WlcP8CxY8fUqVMnz5lgrV1VVZWCg4NVWVmpoKAgs8sBAACXwJfvb5/2AJ07d04Oh0NFRUWKiYnxtHfu3PnyKrUYd60h195jqjh5Rl0DHboxqrPsfm0jNKJ1MPs9aPb4MP81YHzzPwNm12D2+HV8CkDt2rVTREQE1/q5DG/sKtOCLcUqqzzjaQsPdmje6L66IybcxMpgFWa/B80eH+a/Boxv/mfA7BrMHv/bfP4J7LnnntPLL7+sjRs3ttk9P439E9gbu8r08Mad+u6Grsu7mRNj+QJAkzL7PWj2+DD/NWB88z8DZtfQHOP78v3t82nwzz77rLZv365u3bqpd+/eio2N9ZrgzV1raMGW4novuCRP24ItxXLX/uBDsYAGmf0eNHt8mP8aML75nwGzazB7/Ib4fBZYSkpKE5TRdrn2HvPa1fddhqSyyjNy7T2mwdd2ab7CYBlmvwfNHh/mvwaMb/5nwOwazB6/IT4HoHnz5jVFHW1WxckLv+CX0w/wldnvQbPHh/mvAeOb/xkwuwazx2+Izz+BwTddAx2N2g/wldnvQbPHh/mvAeOb/xkwuwazx2+IzwHIz89Pdrv9ghO83RjVWeHBDl3oBD+bzh8Bf2NU2zygHOYz+z1o9vgw/zVgfPM/A2bXYPb4DfE5AG3atEmvvPKKZ8rNzdXs2bMVHh6urKwsnwvIyMhQVFSUHA6H4uLitH379ov2X716taKjo+V0OtW7d29t2LDB6/m1a9fq5ptvVqdOndSpUyfdfvvtcrlcPtfVWOx+Ns0b3VeS6r3wdY/nje7LtVDQZMx+D5o9Psx/DRjf/M+A2TWYPX5DGuVK0JL04osvKjc3V5s3b77keXJzc5WamqqMjAwlJSVpzZo1WrdunYqLi9WzZ896/TMzMzVr1iytXbtWAwcOlMvl0oMPPqgXX3xRo0ePliT97Gc/U1JSkhITE+VwOLR06VK98sor2r17t66++upLqqsprgTdkq59AGsy+z1o9vgw/zVgfPM/A2bX0NTj+/L93WgB6KuvvtINN9zguU/YpUhISFBsbKwyMzM9bdHR0UpJSVF6enq9/omJiUpKStKyZcs8bWlpaSooKNCOHTsaHMPtdqtTp05atWqV7rvvvkuqq6luhdFSrn4J6zL7PWj2+DD/NWB88z8DZtfQlOM32a0wLuR///d/tXLlSnXv3v2S5zl79qwKCws1e/Zsr/bk5GTl5+c3OE91dbUcDu8DpJxOp1wul2pqauTv719vntOnT6umpuaiF22srq5WdXW153FVVdUlr4cv7H42TvOFqcx+D5o9Psx/DRjf/M+A2TWYPX4dnwPQd296ahiGTp48qSuuuEIbN2685OUcOXJEbrdboaGhXu2hoaEqLy9vcJ4RI0Zo3bp1SklJUWxsrAoLC5Wdna2amhodOXJE4eH1d5/Nnj1bV199tW6//fYL1pKenq4FCxZccu0AAKB18zkAPfPMM14ByM/PT1dddZUSEhLUqVMnnwv47h3kDcO44F3l586dq/Lycg0aNEiGYSg0NFSTJk3S0qVLGzwDbenSpXrppZf07rvv1ttz9G1z5szRjBkzPI+rqqrUo0cPn9cFAAC0Dj4HoEmTJjXKwCEhIbLb7fX29lRUVNTbK1TH6XQqOztba9as0eHDhz1nngUGBiokJMSr79NPP61Fixbprbfe0g033HDRWgICAhQQEPDDVggAALQaPp8GX3cz1O96+eWXtX79+kteTvv27RUXF6e8vDyv9ry8PCUmJl50Xn9/f3Xv3l12u105OTkaNWqU/Pz+b1WWLVumJ598Um+88Ybi4+MvuSYAAGANPgegxYsX19vbIkldu3bVokWLfFrWjBkztG7dOmVnZ+vzzz/Xr371K5WWlmrKlCmSzv809e0zt/bs2aONGzfqyy+/lMvl0oQJE7Rr1y6vcZcuXaonnnhC2dnZioyMVHl5ucrLy3Xq1ClfVxUAALRRPv8Etn//fkVFRdVrj4iIUGlpqU/LGj9+vI4ePaqFCxeqrKxMMTEx2rp1qyIiIiRJZWVlXst0u91avny5SkpK5O/vr2HDhik/P1+RkZGePhkZGTp79qzuuecer7HmzZun+fPn+1QfAABom3y+DlDPnj21atUqjRkzxqt98+bNmjp1qr7++utGLdAMTXUdIAAA0HR8+f72+SewCRMmaNq0adq2bZvcbrfcbrfeeecdTZ8+XRMmTLjsogEAAJqLzz+B/fa3v9X+/ft12223qV2787PX1tbqvvvu8/kYIAAAADNc9q0wvvzySxUVFcnpdOr666/3HLfTFvATGAAArU+z3AqjV69e6tWr1+XODgAAYBqfjwG65557tHjx4nrty5Yt009+8pNGKQoAAKAp+RyA3nvvPd1111312u+44w69//77jVIUAABAU/I5AJ06dUrt27ev1+7v799kd1EHAABoTD4HoJiYGOXm5tZrz8nJUd++fRulKAAAgKbk80HQc+fO1d13362vvvpKt956qyTp7bff1osvvqg///nPjV4gAABAY/M5AI0ZM0avvvqqFi1apD//+c9yOp3q16+f3nnnHU4ZBwAArcJlXweozokTJ/TCCy/oj3/8oz755BO53e7Gqs00XAcIAIDWp0lvhVHnnXfe0cSJE9WtWzetWrVKd955pwoKCi53cQAAAM3Gp5/Avv76a/3pT39Sdna2vvnmG40bN041NTX6y1/+wgHQAACg1bjkPUB33nmn+vbtq+LiYq1cuVKHDh3SypUrm7I2AACAJnHJe4DefPNNTZs2TQ8//DC3wAAAAK3aJe8B2r59u06ePKn4+HglJCRo1apV+ve//92UtQEAADSJSw5AgwcP1tq1a1VWVqZf/OIXysnJ0dVXX63a2lrl5eXp5MmTTVknAABAo/lBp8GXlJToj3/8o55//nmdOHFCw4cP12uvvdaY9ZmC0+ABAGh9muU0eEnq3bu3li5dqq+//lovvfTSD1kUAABAs/nBF0Jsi9gDBABA69Nse4AAAABaIwIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHAIQAACwHNMDUEZGhqKiouRwOBQXF6ft27dftP/q1asVHR0tp9Op3r17a8OGDV7P7969W3fffbciIyNls9n0u9/9rgmrBwAArZGpASg3N1dpaWl6/PHH9fHHH+vmm2/WyJEjVVpa2mD/zMxMzZkzR/Pnz9fu3bu1YMECTZ06VVu2bPH0OX36tK655hotXrxYYWFhzbUqAACgFbEZhmGYNXhCQoJiY2OVmZnpaYuOjlZKSorS09Pr9U9MTFRSUpKWLVvmaUtLS1NBQYF27NhRr39kZKTS0tKUlpbmU11VVVUKDg5WZWWlgoKCfJoXAACYw5fvb9P2AJ09e1aFhYVKTk72ak9OTlZ+fn6D81RXV8vhcHi1OZ1OuVwu1dTUXHYt1dXVqqqq8poAAEDbZVoAOnLkiNxut0JDQ73aQ0NDVV5e3uA8I0aM0Lp161RYWCjDMFRQUKDs7GzV1NToyJEjl11Lenq6goODPVOPHj0ue1kAAKDlM/0gaJvN5vXYMIx6bXXmzp2rkSNHatCgQfL399fYsWM1adIkSZLdbr/sGubMmaPKykrPdODAgcteFgAAaPlMC0AhISGy2+319vZUVFTU2ytUx+l0Kjs7W6dPn9a+fftUWlqqyMhIBQYGKiQk5LJrCQgIUFBQkNcEAADaLtMCUPv27RUXF6e8vDyv9ry8PCUmJl50Xn9/f3Xv3l12u105OTkaNWqU/PxM35kFAABaiXZmDj5jxgylpqYqPj5egwcPVlZWlkpLSzVlyhRJ53+aOnjwoOdaP3v27JHL5VJCQoKOHz+uFStWaNeuXVq/fr1nmWfPnlVxcbHn3wcPHlRRUZE6duyoH/3oR82/kgAAoMUxNQCNHz9eR48e1cKFC1VWVqaYmBht3bpVERERkqSysjKvawK53W4tX75cJSUl8vf317Bhw5Sfn6/IyEhPn0OHDmnAgAGex08//bSefvppDR06VO+++25zrRoAAGjBTL0OUEvFdYAAAGh9WsV1gAAAAMxCAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZjegDKyMhQVFSUHA6H4uLitH379ov2X716taKjo+V0OtW7d29t2LChXp+//OUv6tu3rwICAtS3b19t2rSpqcoHAACtkKkBKDc3V2lpaXr88cf18ccf6+abb9bIkSNVWlraYP/MzEzNmTNH8+fP1+7du7VgwQJNnTpVW7Zs8fT5xz/+ofHjxys1NVWffPKJUlNTNW7cOH344YfNtVoAAKCFsxmGYZg1eEJCgmJjY5WZmelpi46OVkpKitLT0+v1T0xMVFJSkpYtW+ZpS0tLU0FBgXbs2CFJGj9+vKqqqvT66697+txxxx3q1KmTXnrppUuqq6qqSsHBwaqsrFRQUNDlrh4AAGhGvnx/m7YH6OzZsyosLFRycrJXe3JysvLz8xucp7q6Wg6Hw6vN6XTK5XKppqZG0vk9QN9d5ogRIy64zLrlVlVVeU0AAKDtMi0AHTlyRG63W6GhoV7toaGhKi8vb3CeESNGaN26dSosLJRhGCooKFB2drZqamp05MgRSVJ5eblPy5Sk9PR0BQcHe6YePXr8wLUDAAAtmekHQdtsNq/HhmHUa6szd+5cjRw5UoMGDZK/v7/Gjh2rSZMmSZLsdvtlLVOS5syZo8rKSs904MCBy1wbAADQGpgWgEJCQmS32+vtmamoqKi3B6eO0+lUdna2Tp8+rX379qm0tFSRkZEKDAxUSEiIJCksLMynZUpSQECAgoKCvCYAANB2mRaA2rdvr7i4OOXl5Xm15+XlKTEx8aLz+vv7q3v37rLb7crJydGoUaPk53d+VQYPHlxvmW+++eb3LhMAAFhHOzMHnzFjhlJTUxUfH6/BgwcrKytLpaWlmjJliqTzP00dPHjQc62fPXv2yOVyKSEhQcePH9eKFSu0a9curV+/3rPM6dOna8iQIVqyZInGjh2rzZs366233vKcJQYAAGBqABo/fryOHj2qhQsXqqysTDExMdq6dasiIiIkSWVlZV7XBHK73Vq+fLlKSkrk7++vYcOGKT8/X5GRkZ4+iYmJysnJ0RNPPKG5c+fq2muvVW5urhISEpp79QAAQAtl6nWAWiquAwQAQOvTKq4DBAAAYBYCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsJx2ZheA5uOuNeTae0wVJ8+oa6BDN0Z1lt3PZnZZgGXwGQRaDgKQRbyxq0wLthSrrPKMpy082KF5o/vqjphwEysDrIHPINCy8BOYBbyxq0wPb9zp9YdXksorz+jhjTv1xq4ykyoDrIHPINDyEIDaOHetoQVbimU08Fxd24ItxXLXNtQDwA/FZxBomQhAbZxr77F6/+v8NkNSWeUZufYea76iAAvhMwi0TASgNq7i5IX/8F5OPwC+4TMItEwEoDaua6CjUfsB8A2fQaBlIgC1cTdGdVZ4sEMXOtHWpvNnotwY1bk5ywIsg88g0DIRgNo4u59N80b3laR6f4DrHs8b3ZdrkQBNhM8g0DIRgCzgjphwZU6MVViw9y72sGCHMifGcg0SoInxGQRaHpthGJx7+R1VVVUKDg5WZWWlgoKCzC6n0XAVWsBcfAaBpuXL9zdXgrYQu59Ng6/tYnYZgGXxGQRaDn4CAwAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlsOVoBtQd3eQqqoqkysBAACXqu57+1Lu8kUAasDJkyclST169DC5EgAA4KuTJ08qODj4on24GWoDamtrdejQIQUGBspma1s3KqyqqlKPHj104MCBNnWj10tl9fWX2AZWX3+JbWD19Zfa7jYwDEMnT55Ut27d5Od38aN82APUAD8/P3Xv3t3sMppUUFBQm3rT+8rq6y+xDay+/hLbwOrrL7XNbfB9e37qcBA0AACwHAIQAACwHAKQxQQEBGjevHkKCAgwuxRTWH39JbaB1ddfYhtYff0ltoHEQdAAAMCC2AMEAAAshwAEAAAshwAEAAAshwAEAAAshwBkEenp6Ro4cKACAwPVtWtXpaSkqKSkxOyyTJOeni6bzaa0tDSzS2k2Bw8e1MSJE9WlSxddccUV6t+/vwoLC80uq9mcO3dOTzzxhKKiouR0OnXNNddo4cKFqq2tNbu0JvH+++9r9OjR6tatm2w2m1599VWv5w3D0Pz589WtWzc5nU7dcsst2r17tznFNpGLbYOamhrNmjVL119/vTp06KBu3brpvvvu06FDh8wruJF933vg237xi1/IZrPpd7/7XbPVZzYCkEW89957mjp1qv75z38qLy9P586dU3Jysr755huzS2t2H330kbKysnTDDTeYXUqzOX78uJKSkuTv76/XX39dxcXFWr58ua688kqzS2s2S5Ys0R/+8AetWrVKn3/+uZYuXaply5Zp5cqVZpfWJL755hv169dPq1atavD5pUuXasWKFVq1apU++ugjhYWFafjw4Z57IbYFF9sGp0+f1s6dOzV37lzt3LlTr7zyivbs2aMxY8aYUGnT+L73QJ1XX31VH374obp169ZMlbUQBiypoqLCkGS89957ZpfSrE6ePGn06tXLyMvLM4YOHWpMnz7d7JKaxaxZs4ybbrrJ7DJMdddddxmTJ0/2avvP//xPY+LEiSZV1HwkGZs2bfI8rq2tNcLCwozFixd72s6cOWMEBwcbf/jDH0yosOl9dxs0xOVyGZKM/fv3N09RzehC6//1118bV199tbFr1y4jIiLCeOaZZ5q9NrOwB8iiKisrJUmdO3c2uZLmNXXqVN111126/fbbzS6lWb322muKj4/XT37yE3Xt2lUDBgzQ2rVrzS6rWd100016++23tWfPHknSJ598oh07dujOO+80ubLmt3fvXpWXlys5OdnTFhAQoKFDhyo/P9/EysxVWVkpm81mmT2jtbW1Sk1N1WOPPabrrrvO7HKaHTdDtSDDMDRjxgzddNNNiomJMbucZpOTk6PCwkIVFBSYXUqz+9e//qXMzEzNmDFDv/nNb+RyuTRt2jQFBATovvvuM7u8ZjFr1ixVVlaqT58+stvtcrvdeuqpp3TvvfeaXVqzKy8vlySFhoZ6tYeGhmr//v1mlGS6M2fOaPbs2frpT3/a5m4OeiFLlixRu3btNG3aNLNLMQUByIIeeeQRffrpp9qxY4fZpTSbAwcOaPr06XrzzTflcDjMLqfZ1dbWKj4+XosWLZIkDRgwQLt371ZmZqZlAlBubq42btyoF198Udddd52KioqUlpambt266f777ze7PFPYbDavx4Zh1GuzgpqaGk2YMEG1tbXKyMgwu5xmUVhYqN///vfauXOnJV9ziYOgLefRRx/Va6+9pm3btql79+5ml9NsCgsLVVFRobi4OLVr107t2rXTe++9p2effVbt2rWT2+02u8QmFR4err59+3q1RUdHq7S01KSKmt9jjz2m2bNna8KECbr++uuVmpqqX/3qV0pPTze7tGYXFhYm6f/2BNWpqKiot1eoraupqdG4ceO0d+9e5eXlWWbvz/bt21VRUaGePXt6/ibu379fM2fOVGRkpNnlNQv2AFmEYRh69NFHtWnTJr377ruKiooyu6Rmddttt+mzzz7zanvggQfUp08fzZo1S3a73aTKmkdSUlK9yx7s2bNHERERJlXU/E6fPi0/P+//89nt9jZ7GvzFREVFKSwsTHl5eRowYIAk6ezZs3rvvfe0ZMkSk6trPnXh58svv9S2bdvUpUsXs0tqNqmpqfWOhRwxYoRSU1P1wAMPmFRV8yIAWcTUqVP14osvavPmzQoMDPT8zy84OFhOp9Pk6ppeYGBgveOdOnTooC5duljiOKhf/epXSkxM1KJFizRu3Di5XC5lZWUpKyvL7NKazejRo/XUU0+pZ8+euu666/Txxx9rxYoVmjx5stmlNYlTp07pf/7nfzyP9+7dq6KiInXu3Fk9e/ZUWlqaFi1apF69eqlXr15atGiRrrjiCv30pz81serGdbFt0K1bN91zzz3auXOn/vrXv8rtdnv+Lnbu3Fnt27c3q+xG833vge8GPn9/f4WFhal3797NXao5TD4LDc1EUoPTc889Z3ZpprHSafCGYRhbtmwxYmJijICAAKNPnz5GVlaW2SU1q6qqKmP69OlGz549DYfDYVxzzTXG448/blRXV5tdWpPYtm1bg5/5+++/3zCM86fCz5s3zwgLCzMCAgKMIUOGGJ999pm5RTeyi22DvXv3XvDv4rZt28wuvVF833vgu6x2GrzNMAyjmbIWAABAi8BB0AAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAB+kH379slms6moqMjsUjy++OILDRo0SA6HQ/3792+ycebPn+/z8m+55RalpaVdtI/NZtOrr7562XUB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8ePHNXnyZEnS1KlTdezYMd17771yuVz617/+pTfffFOTJ0+W2+32aZzHHntMS5YsUW5urkpKSjR79mwVFRVp+vTpjbYu0vkz6wYPHqyUlBT9/e9/1759+5Sfn68nnnhCBQUFDc7z6KOPKj09XZs3b1ZJSYmmT5+u48ePX9beqZdfflnZ2dnas2eP5s2bJ5fL5TnI+Wc/+5kcDofuv/9+7dq1S9u2bdOjjz6q1NRUhYaGSpJuvfVW/e1vf9Pf/vY3ffHFF/rlL3+pEydOfO+4zzzzjHJycvTFF19oz549evnllxUWFnbRvV5Aa0QAAtqgJ598st7PH9HR0crIyNDq1avVr18/uVyui54h5avFixdryZIl6tevn7Zv367Nmzd79pR069ZNH3zwgdxut0aMGKGYmBhNnz5dwcHBXscbXYpp06Zp5syZmjlzpq6//nq98cYbeu2119SrV69GWxfp/E9JW7du1ZAhQzR58mT9+Mc/1oQJE7Rv3z5PyPiuWbNm6d5779V9992nwYMHq2PHjhoxYoQcDofP4y9YsEA5OTm64YYbtH79er3wwgvq27evJOmKK67Q3//+dx07dkwDBw7UPffco9tuu02rVq3yzD958mTdf//9uu+++zR06FBFRUVp2LBh3ztux44dtWTJEsXHx2vgwIHat2+ftm7d6vPrBLR0NqOhH6gBAD9YbW2toqOjNW7cOD355JNmlwPgWzgIGgAayf79+/Xmm29q6NChqq6u1qpVq7R371799Kc/Nbs0AN/BPk0AaCR+fn7605/+pIEDByopKUmfffaZ3nrrrUY/QBvAD8dPYAAAwHLYAwQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACzn/wEvRAO2VV1FCwAAAABJRU5ErkJggg=="/>

위의 그래프에서 사용할 최적의 이웃 수는 몇입니까? 그렇게 생각하는 이유를 설명해 보세요.


###### your answer here

- 정확도만 보자면 k = 3, 4, 5, 8 이 최고의 정확도를 보여줌.

- 이전은 최소 적합으로 정확도가 낮은 수치를 보임

- 8 이후로 안오른 걸 보면 과대적합으로 의심


## 2.2 결정 트리(decision tree)


의사 결정 트리가 과소적합되거나 과대적합될 수도 있습니다.



예를 들어, 트리에 1개의 결정 지점만 있는 경우 트리가 과소적합될 가능성이 있습니다. 반대로 트리에 여러 결정 지점이 있는 경우 트리가 과대적합될 수 있습니다. 따라서 트리가 깊을수록 트리가 과대적합될 가능성이 높아집니다. 과대적합 및 과소적합에 대한 이해를 바탕으로 데이터를 과대적합하거나 과소적합할 가능성이 있는 트리(아래 참조)를 찾아서 답변해 주세요.


<img src = "./resources/dt1.jpg">


###### your answer here

- 트리 레이어가 2번째에서 실행되는 경우 적당히 결과가 나올 가능성이 높아보임

- 다만 다음 트리 레이어에서는 과대적합 가능성도..


<img src = './resources/dt2.jpg'>


###### your answer here

- 너무 간단해서 과소적합 가능성이 높음


또한 과대적합 또는 과소적합을 제어하는 또 다른 방법은 분할을 수행하기 전에 결정 지점에 있는 최소 샘플 수를 기반으로 결정하는 것입니다. 예를 들어 트리에 날씨에 기반한 결정 지점이 있고 맑은 날이나 비오는 날로 구성된 50개의 다른 지점이 있는 경우 표본이나 데이터 지점이 꽤 많기 때문에 해당 결정 지점에서 데이터를 분할해야 합니다. 그러나 해당 날씨 결정 지점에 2개의 데이터 포인트만 있는 경우 과대적합으로 이어질 수 있으므로 날씨에 따라 데이터 포인트를 분할할 필요가 없을 수 있습니다. 따라서 결정 지점에서 표본 수를 사용하여 적합성을 제어할 수도 있습니다.



이 [문서](https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3) 를 참고하여 의사결정 트리에서 과대적합 및 과소적합을 제어하는 방법에 대해 자세히 알아볼 수 있습니다. iris flower 데이터 셋트를 이용한 의사 결정 트리에서 적합성을 제어하는 데 사용할 수 있는 다른 변수는 무엇입니까?


###### your answer here

- ... 나중에 찾아서 답을


의사 결정 트리가 Iris Flower 데이터 세트에 잘 맞도록 하는 최적의 매개변수 세트를 찾아보겠습니다. 위에서 사용한 분할 데이터 세트를 그대로 사용하여 아래의 코드를 실행해 보세요! 트리의 적합도를 제어하기 위해 사용하는 변수는 max_depth라고 알려져 있습니다. 이것은 트리의 최대 깊이를 나타냅니다. 트리가 깊을수록 과대적합될 가능성이 높아집니다. sklearn에서 트리를 가져올 수 있습니다.



```python
# 각 의사 결정 트리에 대해 정확도와 가장 잘 테스트된 매개변수를 저장할 빈 목록을 만듭니다.
accuracy = []
depth = []

# ii를 사용하여 값 1에서 9까지 반복합니다. 이것은 의사결정 트리의 max_depth 값이 됩니다.
for ii in range(1,10):
    # max_depth를 ii로 설정
    dt = tree.DecisionTreeClassifier(max_depth=ii)
    # 데이터로 모델 훈련 또는 피팅
    dt.fit(x_train_scale,y_train)
    # .score는 테스트 데이터를 기반으로 모델의 정확도를 제공합니다. 정확도를 목록에 저장합니다.
    accuracy.append(dt.score(x_test_scale,y_test))
    # 목록에 max_depth 값 추가
    depth.append(ii)

print(accuracy)
```

<pre>
[0.5526315789473685, 0.9473684210526315, 0.9473684210526315, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473]
</pre>
그래프의 max_depth 값에 대해 정확도 값을 표시할 수 있습니까? KNN에 사용하였던 그래프를 참조해 보세요.



```python
#your code here

plt.scatter(depth,accuracy)
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy')
plt.show();
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkAAAAGxCAYAAACKvAkXAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA66ElEQVR4nO3de1RVdf7/8dcB5eKF44UEFEQyS5ByFJRbOtNYqKmjNZM4M1HeaixHJadWMeo0Ok6M9c1JTfhmqWiZMuVY9h0vYRfRZeOFxDIa07RAPcSoyUFNMNi/P1ye35xAR/TABvbzsdZea/icz97n/XZa8vKzbzbDMAwBAABYiJfZBQAAADQ0AhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALCcFmYX0BhVV1fr+PHjatu2rWw2m9nlAACAq2AYhsrLy9W5c2d5eV15jYcAVIvjx48rLCzM7DIAAMA1KC4uVmho6BXnEIBq0bZtW0kX/wADAgJMrgYAAFwNp9OpsLAw1+/xKyEA1eLSaa+AgAACEAAATczVXL7CRdAAAMByCEAAAMByCEAAAMByCEAAAMByCEAAAMByCEAAAMByCEAAAMByTA9AmZmZioiIkJ+fn2JiYrRt27Yrzl+8eLEiIyPl7++vW265RStXrnT7PDs7WzabrcZ2/vz5+mwDAAA0IaY+CDEnJ0dpaWnKzMxUUlKSXnrpJQ0dOlSFhYXq2rVrjflZWVlKT0/Xyy+/rH79+mnXrl166KGH1L59e40YMcI1LyAgQAcOHHDb18/Pr977AQAATYPNMAzDrC+Pi4tT3759lZWV5RqLjIzUqFGjlJGRUWN+YmKikpKS9Nxzz7nG0tLStGfPHm3fvl3SxRWgtLQ0nT59+prrcjqdstvtKisr40nQdVBVbWjXkVMqLT+vTm391D+ig7y9mtfLZK3QIwA0VXX5/W3aClBlZaXy8/P11FNPuY0nJydrx44dte5TUVFRYyXH399fu3bt0oULF9SyZUtJ0pkzZxQeHq6qqir96Ec/0p/+9Cf16dOnfhqBJGnTfodmv1MoR9n/P9UYYvfT0yOiNCQ6xMTKPMcKPQKAVZh2DdCJEydUVVWloKAgt/GgoCCVlJTUus/gwYP1yiuvKD8/X4ZhaM+ePVq2bJkuXLigEydOSJJ69uyp7OxsrV+/XqtXr5afn5+SkpJ08ODBy9ZSUVEhp9PptuHqbdrv0COvfewWDCSppOy8HnntY23a7zCpMs+xQo8AYCWmXwT9wxeWGYZx2ZeYzZo1S0OHDlV8fLxatmypkSNHauzYsZIkb29vSVJ8fLzuv/9+9e7dWwMGDNDf/vY33XzzzVq0aNFla8jIyJDdbndtYWFhnmnOAqqqDc1+p1C1nUe9NDb7nUJVVZt2pvW6WaFHALAa0wJQYGCgvL29a6z2lJaW1lgVusTf31/Lli3TuXPn9NVXX6moqEjdunVT27ZtFRgYWOs+Xl5e6tev3xVXgNLT01VWVubaiouLr70xi9l15FSNVZH/ZEhylJ3XriOnGq4oD7NCjwBgNaYFIB8fH8XExCg3N9dtPDc3V4mJiVfct2XLlgoNDZW3t7fWrFmj4cOHy8ur9lYMw1BBQYFCQi5/jYavr68CAgLcNlyd0vKre7zA1c5rjKzQIwBYjam3wU+fPl2pqamKjY1VQkKClixZoqKiIk2aNEnSxZWZY8eOuZ7188UXX2jXrl2Ki4vTt99+q/nz52v//v1asWKF65izZ89WfHy8evToIafTqYULF6qgoECLFy82pcfmrlPbq3u8wNXOa4ys0CMAWI2pASglJUUnT57UnDlz5HA4FB0drQ0bNig8PFyS5HA4VFRU5JpfVVWl559/XgcOHFDLli11xx13aMeOHerWrZtrzunTp/Xwww+rpKREdrtdffr0UV5envr379/Q7VlC/4gOCrH7qaTsfK3XyNgkBdsv3i7eVFmhRwCwGlOfA9RY8Rygurl0h5Qkt4Bw6VL2rPv7NvnbxK3QIwA0dXX5/W36XWBo+oZEhyjr/r4KtrufAgq2+zWbYGCFHgHASlgBqgUrQNfGCk9JtkKPANBUNYknQaP58fayKaF7R7PLqFdW6BEArIBTYAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHJ4FQYAl+b+rrPm3p/U/Hukv6avsfRIAAIgSdq036HZ7xTKUXbeNRZi99PTI6Kaxdvum3t/UvPvkf6avsbUI2+DrwVvg4fVbNrv0COvfawf/mVw6d9kWff3bdJ/ATf3/qTm3yP9Ne3+pIbpsS6/v7kGCLC4qmpDs98prPGXkiTX2Ox3ClVV3TT/rdTc+5Oaf4/017T7kxpnjwQgwOJ2HTnlthz9Q4YkR9l57TpyquGK8qDm3p/U/Hukv6bdn9Q4eyQAARZXWn75v5SuZV5j09z7k5p/j/RXt3mNUWPskQAEWFyntn4endfYNPf+pObfI/3VbV5j1Bh7JAABFtc/ooNC7H663E2oNl28S6N/RIeGLMtjmnt/UvPvkf6adn9S4+yRAARYnLeXTU+PiJKkGn85Xfr56RFRTfZZJM29P6n590h/Tbs/qXH2SAACoCHRIcq6v6+C7e7Lz8F2v2Zx+21z709q/j3SX9PuT2p8PfIcoFrwHCBYVWN5Qmt9ae79Sc2/R/pr+uqzx7r8/iYA1YIABABA08ODEAEAAK6AAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACzH9ACUmZmpiIgI+fn5KSYmRtu2bbvi/MWLFysyMlL+/v665ZZbtHLlyhpz1q5dq6ioKPn6+ioqKkrr1q2rr/IBAEATZGoAysnJUVpammbMmKG9e/dqwIABGjp0qIqKimqdn5WVpfT0dP3xj3/UZ599ptmzZ2vy5Ml65513XHM++ugjpaSkKDU1Vfv27VNqaqpGjx6tnTt3NlRbAACgkTP1SdBxcXHq27evsrKyXGORkZEaNWqUMjIyasxPTExUUlKSnnvuOddYWlqa9uzZo+3bt0uSUlJS5HQ6tXHjRtecIUOGqH379lq9evVV1cWToAEAaHqaxJOgKysrlZ+fr+TkZLfx5ORk7dixo9Z9Kioq5Ofn/hI1f39/7dq1SxcuXJB0cQXoh8ccPHjwZY956bhOp9NtAwAAzZdpAejEiROqqqpSUFCQ23hQUJBKSkpq3Wfw4MF65ZVXlJ+fL8MwtGfPHi1btkwXLlzQiRMnJEklJSV1OqYkZWRkyG63u7awsLDr7A4AADRmpl8EbbO5vwHWMIwaY5fMmjVLQ4cOVXx8vFq2bKmRI0dq7NixkiRvb+9rOqYkpaenq6yszLUVFxdfYzcAAKApMC0ABQYGytvbu8bKTGlpaY0VnEv8/f21bNkynTt3Tl999ZWKiorUrVs3tW3bVoGBgZKk4ODgOh1Tknx9fRUQEOC2AQCA5su0AOTj46OYmBjl5ua6jefm5ioxMfGK+7Zs2VKhoaHy9vbWmjVrNHz4cHl5XWwlISGhxjHffffd/3pMAABgHS3M/PLp06crNTVVsbGxSkhI0JIlS1RUVKRJkyZJunhq6tixY65n/XzxxRfatWuX4uLi9O2332r+/Pnav3+/VqxY4TrmtGnTNHDgQM2bN08jR47U22+/rS1btrjuEgMAADA1AKWkpOjkyZOaM2eOHA6HoqOjtWHDBoWHh0uSHA6H2zOBqqqq9Pzzz+vAgQNq2bKl7rjjDu3YsUPdunVzzUlMTNSaNWs0c+ZMzZo1S927d1dOTo7i4uIauj0AANBImfocoMaK5wABAND0NInnAAEAAJiFAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACyHAAQAACzH9ACUmZmpiIgI+fn5KSYmRtu2bbvi/FWrVql3795q1aqVQkJCNG7cOJ08edL1eXZ2tmw2W43t/Pnz9d0KAABoIkwNQDk5OUpLS9OMGTO0d+9eDRgwQEOHDlVRUVGt87dv364HHnhAEyZM0GeffaY33nhDu3fv1sSJE93mBQQEyOFwuG1+fn4N0RIAAGgCTA1A8+fP14QJEzRx4kRFRkbqhRdeUFhYmLKysmqd/89//lPdunXT1KlTFRERodtvv12/+c1vtGfPHrd5NptNwcHBbhsAAMAlpgWgyspK5efnKzk52W08OTlZO3bsqHWfxMREHT16VBs2bJBhGPrmm2/05ptvatiwYW7zzpw5o/DwcIWGhmr48OHau3dvvfUBAACaHtMC0IkTJ1RVVaWgoCC38aCgIJWUlNS6T2JiolatWqWUlBT5+PgoODhY7dq106JFi1xzevbsqezsbK1fv16rV6+Wn5+fkpKSdPDgwcvWUlFRIafT6bYBAIDmy/SLoG02m9vPhmHUGLuksLBQU6dO1R/+8Afl5+dr06ZNOnLkiCZNmuSaEx8fr/vvv1+9e/fWgAED9Le//U0333yzW0j6oYyMDNntdtcWFhbmmeYAAECjZFoACgwMlLe3d43VntLS0hqrQpdkZGQoKSlJTzzxhG677TYNHjxYmZmZWrZsmRwOR637eHl5qV+/fldcAUpPT1dZWZlrKy4uvvbGAABAo2daAPLx8VFMTIxyc3PdxnNzc5WYmFjrPufOnZOXl3vJ3t7eki6uHNXGMAwVFBQoJCTksrX4+voqICDAbQMAAM1XCzO/fPr06UpNTVVsbKwSEhK0ZMkSFRUVuU5ppaen69ixY1q5cqUkacSIEXrooYeUlZWlwYMHy+FwKC0tTf3791fnzp0lSbNnz1Z8fLx69Oghp9OphQsXqqCgQIsXLzatTwAA0LiYGoBSUlJ08uRJzZkzRw6HQ9HR0dqwYYPCw8MlSQ6Hw+2ZQGPHjlV5eblefPFF/e53v1O7du3005/+VPPmzXPNOX36tB5++GGVlJTIbrerT58+ysvLU//+/Ru8PwAA0DjZjMudO7Iwp9Mpu92usrIyTocBANBE1OX3t+l3gQEAADQ0AhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAc0wNQZmamIiIi5Ofnp5iYGG3btu2K81etWqXevXurVatWCgkJ0bhx43Ty5Em3OWvXrlVUVJR8fX0VFRWldevW1WcLAACgiTE1AOXk5CgtLU0zZszQ3r17NWDAAA0dOlRFRUW1zt++fbseeOABTZgwQZ999pneeOMN7d69WxMnTnTN+eijj5SSkqLU1FTt27dPqampGj16tHbu3NlQbQEAgEbOZhiGYdaXx8XFqW/fvsrKynKNRUZGatSoUcrIyKgx/3/+53+UlZWlL7/80jW2aNEiPfvssyouLpYkpaSkyOl0auPGja45Q4YMUfv27bV69eqrqsvpdMput6usrEwBAQHX2h4AAGhAdfn9bdoKUGVlpfLz85WcnOw2npycrB07dtS6T2Jioo4ePaoNGzbIMAx98803evPNNzVs2DDXnI8++qjGMQcPHnzZY0pSRUWFnE6n2wYAAJov0wLQiRMnVFVVpaCgILfxoKAglZSU1LpPYmKiVq1apZSUFPn4+Cg4OFjt2rXTokWLXHNKSkrqdExJysjIkN1ud21hYWHX0RkAAGjsTL8I2mazuf1sGEaNsUsKCws1depU/eEPf1B+fr42bdqkI0eOaNKkSdd8TElKT09XWVmZa7t0Og0AADRPLcz64sDAQHl7e9dYmSktLa2xgnNJRkaGkpKS9MQTT0iSbrvtNrVu3VoDBgzQ3LlzFRISouDg4DodU5J8fX3l6+t7nR0BAICmwrQVIB8fH8XExCg3N9dtPDc3V4mJibXuc+7cOXl5uZfs7e0t6eIqjyQlJCTUOOa777572WMCAADrMW0FSJKmT5+u1NRUxcbGKiEhQUuWLFFRUZHrlFZ6erqOHTumlStXSpJGjBihhx56SFlZWRo8eLAcDofS0tLUv39/de7cWZI0bdo0DRw4UPPmzdPIkSP19ttva8uWLdq+fbtpfQIAgMbF1ACUkpKikydPas6cOXI4HIqOjtaGDRsUHh4uSXI4HG7PBBo7dqzKy8v14osv6ne/+53atWunn/70p5o3b55rTmJiotasWaOZM2dq1qxZ6t69u3JychQXF9fg/QEAgMbJ1OcANVY8BwgAgKanSTwHCAAAwCwEIAAAYDkEIAAAYDkEIAAAYDkEIAAAYDkEIAAAYDkEIAAAYDl1DkDdunXTnDlz3B5QCAAA0JTUOQD97ne/09tvv60bb7xRd911l9asWaOKior6qA0AAKBe1DkATZkyRfn5+crPz1dUVJSmTp2qkJAQ/fa3v9XHH39cHzUCAAB41HW/CuPChQvKzMzUk08+qQsXLig6OlrTpk3TuHHjZLPZPFVng+JVGAAAND11+f19zS9DvXDhgtatW6fly5crNzdX8fHxmjBhgo4fP64ZM2Zoy5Ytev3116/18AAAAPWmzgHo448/1vLly7V69Wp5e3srNTVVf/3rX9WzZ0/XnOTkZA0cONCjhQIAAHhKnQNQv379dNdddykrK0ujRo1Sy5Yta8yJiorSmDFjPFIgAACAp9U5AB0+fFjh4eFXnNO6dWstX778mosCAACoT3W+C6y0tFQ7d+6sMb5z507t2bPHI0UBAADUpzoHoMmTJ6u4uLjG+LFjxzR58mSPFAUAAFCf6hyACgsL1bdv3xrjffr0UWFhoUeKAgAAqE91DkC+vr765ptvaow7HA61aHHNd9UDAAA0mDoHoLvuukvp6ekqKytzjZ0+fVq///3vddddd3m0OAAAgPpQ5yWb559/XgMHDlR4eLj69OkjSSooKFBQUJBeffVVjxcIAADgaXUOQF26dNEnn3yiVatWad++ffL399e4ceP0y1/+stZnAgEAADQ213TRTuvWrfXwww97uhYAAIAGcc1XLRcWFqqoqEiVlZVu4z/72c+uuygAAID6dE1Pgr7nnnv06aefymaz6dLL5C+9+b2qqsqzFQIAAHhYne8CmzZtmiIiIvTNN9+oVatW+uyzz5SXl6fY2Fh9+OGH9VAiAACAZ9V5Beijjz7S+++/rxtuuEFeXl7y8vLS7bffroyMDE2dOlV79+6tjzoBAAA8ps4rQFVVVWrTpo0kKTAwUMePH5ckhYeH68CBA56tDgAAoB7UeQUoOjpan3zyiW688UbFxcXp2WeflY+Pj5YsWaIbb7yxPmoEAADwqDoHoJkzZ+rs2bOSpLlz52r48OEaMGCAOnbsqJycHI8XCAAA4Gk249JtXNfh1KlTat++vetOsKbO6XTKbrerrKxMAQEBZpcDAACuQl1+f9fpGqDvv/9eLVq00P79+93GO3TocM3hJzMzUxEREfLz81NMTIy2bdt22bljx46VzWarsfXq1cs1Jzs7u9Y558+fv6b6AABA81OnANSiRQuFh4d77Fk/OTk5SktL04wZM7R3714NGDBAQ4cOVVFRUa3zFyxYIIfD4dqKi4vVoUMH3XfffW7zAgIC3OY5HA75+fl5pGYAAND01fkusJkzZyo9PV2nTp267i+fP3++JkyYoIkTJyoyMlIvvPCCwsLClJWVVet8u92u4OBg17Znzx59++23GjdunNs8m83mNi84OPi6awUAAM1HnS+CXrhwoQ4dOqTOnTsrPDxcrVu3dvv8448/vqrjVFZWKj8/X0899ZTbeHJysnbs2HFVx1i6dKnuvPNOhYeHu42fOXPGtVL1ox/9SH/6059cb64HAACocwAaNWqUR774xIkTqqqqUlBQkNt4UFCQSkpK/uv+DodDGzdu1Ouvv+423rNnT2VnZ+vWW2+V0+nUggULlJSUpH379qlHjx61HquiokIVFRWun51O5zV0BAAAmoo6B6Cnn37aowX88OJpwzCu6oLq7OxstWvXrkYgi4+PV3x8vOvnpKQk9e3bV4sWLdLChQtrPVZGRoZmz55d9+IBAECTVOdrgDwlMDBQ3t7eNVZ7SktLa6wK/ZBhGFq2bJlSU1Pl4+NzxbleXl7q16+fDh48eNk56enpKisrc23FxcVX3wgAAGhy6hyAvLy85O3tfdntavn4+CgmJka5ublu47m5uUpMTLzivlu3btWhQ4c0YcKE//o9hmGooKBAISEhl53j6+urgIAAtw0AADRfdT4Ftm7dOrefL1y4oL1792rFihV1Po00ffp0paamKjY2VgkJCVqyZImKioo0adIkSRdXZo4dO6aVK1e67bd06VLFxcUpOjq6xjFnz56t+Ph49ejRQ06nUwsXLlRBQYEWL15cx04BAEBzVecANHLkyBpjv/jFL9SrVy/l5ORc1arMJSkpKTp58qTmzJkjh8Oh6OhobdiwwXVXl8PhqPFMoLKyMq1du1YLFiyo9ZinT5/Www8/rJKSEtntdvXp00d5eXnq379/HboEAADNmUdehSFJX375pW677TbXe8KaMl6FAQBA01Nvr8K4nO+++06LFi1SaGioJw4HAABQr+p8CuyHLz01DEPl5eVq1aqVXnvtNY8WBwAAUB/qHID++te/ugUgLy8v3XDDDYqLi1P79u09WhwAAEB9qHMAGjt2bD2UAQAA0HDqfA3Q8uXL9cYbb9QYf+ONN7RixQqPFAUAAFCf6hyA/vKXvygwMLDGeKdOnfTMM894pCgAAID6VOcA9PXXXysiIqLGeHh4eI1n9gAAADRGdQ5AnTp10ieffFJjfN++ferYsaNHigIAAKhPdQ5AY8aM0dSpU/XBBx+oqqpKVVVVev/99zVt2jSNGTOmPmoEAADwqDrfBTZ37lx9/fXXGjRokFq0uLh7dXW1HnjgAa4BAgAATcI1vwrj4MGDKigokL+/v2699VbX+7uaA16FAQBA01OX3991XgG6pEePHurRo8e17g4AAGCaOl8D9Itf/EJ/+ctfaow/99xzuu+++zxSFAAAQH2qcwDaunWrhg0bVmN8yJAhysvL80hRAAAA9anOAejMmTPy8fGpMd6yZUs5nU6PFAUAAFCf6hyAoqOjlZOTU2N8zZo1ioqK8khRAAAA9anOF0HPmjVLP//5z/Xll1/qpz/9qSTpvffe0+uvv64333zT4wUCAAB4Wp0D0M9+9jO99dZbeuaZZ/Tmm2/K399fvXv31vvvv88t4wAAoEm45ucAXXL69GmtWrVKS5cu1b59+1RVVeWp2kzDc4AAAGh66vL7u87XAF3y/vvv6/7771fnzp314osv6u6779aePXuu9XAAAAANpk6nwI4ePars7GwtW7ZMZ8+e1ejRo3XhwgWtXbuWC6ABAECTcdUrQHfffbeioqJUWFioRYsW6fjx41q0aFF91gYAAFAvrnoF6N1339XUqVP1yCOP8AoMAADQpF31CtC2bdtUXl6u2NhYxcXF6cUXX9S///3v+qwNAACgXlx1AEpISNDLL78sh8Oh3/zmN1qzZo26dOmi6upq5ebmqry8vD7rBAAA8Jjrug3+wIEDWrp0qV599VWdPn1ad911l9avX+/J+kzBbfAAADQ9DXIbvCTdcsstevbZZ3X06FGtXr36eg4FAADQYK77QYjNEStAAAA0PQ22AgQAANAUEYAAAIDlEIAAAIDlEIAAAIDlmB6AMjMzFRERIT8/P8XExGjbtm2XnTt27FjZbLYaW69evdzmXXo3ma+vr6KiorRu3br6bgMAADQhpgagnJwcpaWlacaMGdq7d68GDBigoUOHqqioqNb5CxYskMPhcG3FxcXq0KGD7rvvPtecjz76SCkpKUpNTdW+ffuUmpqq0aNHa+fOnQ3VFgAAaORMvQ0+Li5Offv2VVZWlmssMjJSo0aNUkZGxn/d/6233tK9996rI0eOKDw8XJKUkpIip9OpjRs3uuYNGTJE7du3v+pnFXEbPAAATU+TuA2+srJS+fn5Sk5OdhtPTk7Wjh07ruoYS5cu1Z133ukKP9LFFaAfHnPw4MFXPGZFRYWcTqfbBgAAmi/TAtCJEydUVVWloKAgt/GgoCCVlJT81/0dDoc2btyoiRMnuo2XlJTU+ZgZGRmy2+2uLSwsrA6dAACApsb0i6BtNpvbz4Zh1BirTXZ2ttq1a6dRo0Zd9zHT09NVVlbm2oqLi6+ueAAA0CS1MOuLAwMD5e3tXWNlprS0tMYKzg8ZhqFly5YpNTVVPj4+bp8FBwfX+Zi+vr7y9fWtYwcAAKCpMm0FyMfHRzExMcrNzXUbz83NVWJi4hX33bp1qw4dOqQJEybU+CwhIaHGMd99993/ekwAAGAdpq0ASdL06dOVmpqq2NhYJSQkaMmSJSoqKtKkSZMkXTw1dezYMa1cudJtv6VLlyouLk7R0dE1jjlt2jQNHDhQ8+bN08iRI/X2229ry5Yt2r59e4P0BAAAGj9TA1BKSopOnjypOXPmyOFwKDo6Whs2bHDd1eVwOGo8E6isrExr167VggULaj1mYmKi1qxZo5kzZ2rWrFnq3r27cnJyFBcXV+/9AACApsHU5wA1VjwHCACApqdJPAcIAADALAQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOQQgAABgOaYHoMzMTEVERMjPz08xMTHatm3bFedXVFRoxowZCg8Pl6+vr7p3765ly5a5Ps/OzpbNZquxnT9/vr5bAQAATUQLM788JydHaWlpyszMVFJSkl566SUNHTpUhYWF6tq1a637jB49Wt98842WLl2qm266SaWlpfr+++/d5gQEBOjAgQNuY35+fvXWBwAAaFpMDUDz58/XhAkTNHHiREnSCy+8oM2bNysrK0sZGRk15m/atElbt27V4cOH1aFDB0lSt27dasyz2WwKDg6u19oBAEDTZdopsMrKSuXn5ys5OdltPDk5WTt27Kh1n/Xr1ys2NlbPPvusunTpoptvvlmPP/64vvvuO7d5Z86cUXh4uEJDQzV8+HDt3bu33voAAABNj2krQCdOnFBVVZWCgoLcxoOCglRSUlLrPocPH9b27dvl5+endevW6cSJE3r00Ud16tQp13VAPXv2VHZ2tm699VY5nU4tWLBASUlJ2rdvn3r06FHrcSsqKlRRUeH62el0eqhLAADQGJl6Cky6eLrqPxmGUWPskurqatlsNq1atUp2u13SxdNov/jFL7R48WL5+/srPj5e8fHxrn2SkpLUt29fLVq0SAsXLqz1uBkZGZo9e7aHOgIAAI2daafAAgMD5e3tXWO1p7S0tMaq0CUhISHq0qWLK/xIUmRkpAzD0NGjR2vdx8vLS/369dPBgwcvW0t6errKyspcW3Fx8TV0BAAAmgrTApCPj49iYmKUm5vrNp6bm6vExMRa90lKStLx48d15swZ19gXX3whLy8vhYaG1rqPYRgqKChQSEjIZWvx9fVVQECA2wYAAJovU58DNH36dL3yyitatmyZPv/8cz322GMqKirSpEmTJF1cmXnggQdc83/1q1+pY8eOGjdunAoLC5WXl6cnnnhC48ePl7+/vyRp9uzZ2rx5sw4fPqyCggJNmDBBBQUFrmMCAACYeg1QSkqKTp48qTlz5sjhcCg6OlobNmxQeHi4JMnhcKioqMg1v02bNsrNzdWUKVMUGxurjh07avTo0Zo7d65rzunTp/Xwww+rpKREdrtdffr0UV5envr379/g/QEAgMbJZhiGYXYRjY3T6ZTdbldZWRmnwwAAaCLq8vvb9FdhAAAANDQCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBwCEAAAsBzTA1BmZqYiIiLk5+enmJgYbdu27YrzKyoqNGPGDIWHh8vX11fdu3fXsmXL3OasXbtWUVFR8vX1VVRUlNatW1efLQAAgCbG1ACUk5OjtLQ0zZgxQ3v37tWAAQM0dOhQFRUVXXaf0aNH67333tPSpUt14MABrV69Wj179nR9/tFHHyklJUWpqanat2+fUlNTNXr0aO3cubMhWgIAAE2AzTAMw6wvj4uLU9++fZWVleUai4yM1KhRo5SRkVFj/qZNmzRmzBgdPnxYHTp0qPWYKSkpcjqd2rhxo2tsyJAhat++vVavXn1VdTmdTtntdpWVlSkgIKCOXQEAADPU5fe3aStAlZWVys/PV3Jystt4cnKyduzYUes+69evV2xsrJ599ll16dJFN998sx5//HF99913rjkfffRRjWMOHjz4sseULp5WczqdbhsAAGi+Wpj1xSdOnFBVVZWCgoLcxoOCglRSUlLrPocPH9b27dvl5+endevW6cSJE3r00Ud16tQp13VAJSUldTqmJGVkZGj27NnX2REAAGgqTL8I2mazuf1sGEaNsUuqq6tls9m0atUq9e/fX3fffbfmz5+v7Oxst1WguhxTktLT01VWVubaiouLr6MjAADQ2Jm2AhQYGChvb+8aKzOlpaU1VnAuCQkJUZcuXWS3211jkZGRMgxDR48eVY8ePRQcHFynY0qSr6+vfH19r6MbAADQlJi2AuTj46OYmBjl5ua6jefm5ioxMbHWfZKSknT8+HGdOXPGNfbFF1/Iy8tLoaGhkqSEhIQax3z33Xcve0wAAGA9pp4Cmz59ul555RUtW7ZMn3/+uR577DEVFRVp0qRJki6emnrggQdc83/1q1+pY8eOGjdunAoLC5WXl6cnnnhC48ePl7+/vyRp2rRpevfddzVv3jz961//0rx587RlyxalpaWZ0SIAAGiETDsFJl28Zf3kyZOaM2eOHA6HoqOjtWHDBoWHh0uSHA6H2zOB2rRpo9zcXE2ZMkWxsbHq2LGjRo8erblz57rmJCYmas2aNZo5c6ZmzZql7t27KycnR3FxcQ3eHwAAaJxMfQ5QY8VzgAAAaHqaxHOAAAAAzEIAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAltPC7AKspKra0K4jp1Rafl6d2vqpf0QHeXvZzC4LAADLIQA1kE37HZr9TqEcZeddYyF2Pz09IkpDokNMrAwAAOvhFFgD2LTfoUde+9gt/EhSSdl5PfLax9q032FSZQAAWJPpASgzM1MRERHy8/NTTEyMtm3bdtm5H374oWw2W43tX//6l2tOdnZ2rXPOnz9/2ePWp6pqQ7PfKZRRy2eXxma/U6iq6tpmAACA+mDqKbCcnBylpaUpMzNTSUlJeumllzR06FAVFhaqa9eul93vwIEDCggIcP18ww03uH0eEBCgAwcOuI35+fl5tvirtOvIqRorP//JkOQoO69dR04poXvHhisMAAALMzUAzZ8/XxMmTNDEiRMlSS+88II2b96srKwsZWRkXHa/Tp06qV27dpf93GazKTg42NPlXpPS8qtbebraeQAA4PqZdgqssrJS+fn5Sk5OdhtPTk7Wjh07rrhvnz59FBISokGDBumDDz6o8fmZM2cUHh6u0NBQDR8+XHv37r3i8SoqKuR0Ot02T+nU9upWnq52HgAAuH6mBaATJ06oqqpKQUFBbuNBQUEqKSmpdZ+QkBAtWbJEa9eu1d///nfdcsstGjRokPLy8lxzevbsqezsbK1fv16rV6+Wn5+fkpKSdPDgwcvWkpGRIbvd7trCwsI806Sk/hEdFGL30+Vudrfp4t1g/SM6eOw7AQDAldkMwzDl6tvjx4+rS5cu2rFjhxISElzjf/7zn/Xqq6+6Xdh8JSNGjJDNZtP69etr/by6ulp9+/bVwIEDtXDhwlrnVFRUqKKiwvWz0+lUWFiYysrK3K41ulaX7gKT5HYx9KVQlHV/X26FBwDgOjmdTtnt9qv6/W3aClBgYKC8vb1rrPaUlpbWWBW6kvj4+Cuu7nh5ealfv35XnOPr66uAgAC3zZOGRIco6/6+Cra7n+YKtvsRfgAAMIFpF0H7+PgoJiZGubm5uueee1zjubm5Gjly5FUfZ+/evQoJuXyAMAxDBQUFuvXWW6+r3us1JDpEd0UF8yRoAAAaAVPvAps+fbpSU1MVGxurhIQELVmyREVFRZo0aZIkKT09XceOHdPKlSslXbxLrFu3burVq5cqKyv12muvae3atVq7dq3rmLNnz1Z8fLx69Oghp9OphQsXqqCgQIsXLzalx//k7WXjVncAABoBUwNQSkqKTp48qTlz5sjhcCg6OlobNmxQeHi4JMnhcKioqMg1v7KyUo8//riOHTsmf39/9erVS//4xz909913u+acPn1aDz/8sEpKSmS329WnTx/l5eWpf//+Dd4fAABonEy7CLoxq8tFVAAAoHFoEhdBAwAAmIUABAAALIcABAAALIcABAAALIcABAAALIcABAAALIcABAAALMfUByE2VpcejeR0Ok2uBAAAXK1Lv7ev5hGHBKBalJeXS5LCwsJMrgQAANRVeXm57Hb7FefwJOhaVFdX6/jx42rbtq1sNs++rNTpdCosLEzFxcXN8inTzb0/qfn3SH9NX3Pvkf6avvrq0TAMlZeXq3PnzvLyuvJVPqwA1cLLy0uhoaH1+h0BAQHN9j9sqfn3JzX/Humv6WvuPdJf01cfPf63lZ9LuAgaAABYDgEIAABYDgGogfn6+urpp5+Wr6+v2aXUi+ben9T8e6S/pq+590h/TV9j6JGLoAEAgOWwAgQAACyHAAQAACyHAAQAACyHANRA8vLyNGLECHXu3Fk2m01vvfWW2SV5VEZGhvr166e2bduqU6dOGjVqlA4cOGB2WR6TlZWl2267zfXMioSEBG3cuNHssupNRkaGbDab0tLSzC7FY/74xz/KZrO5bcHBwWaX5VHHjh3T/fffr44dO6pVq1b60Y9+pPz8fLPL8phu3brV+P/QZrNp8uTJZpfmEd9//71mzpypiIgI+fv768Ybb9ScOXNUXV1tdmkeU15errS0NIWHh8vf31+JiYnavXu3KbXwIMQGcvbsWfXu3Vvjxo3Tz3/+c7PL8bitW7dq8uTJ6tevn77//nvNmDFDycnJKiwsVOvWrc0u77qFhobqL3/5i2666SZJ0ooVKzRy5Ejt3btXvXr1Mrk6z9q9e7eWLFmi2267zexSPK5Xr17asmWL62dvb28Tq/Gsb7/9VklJSbrjjju0ceNGderUSV9++aXatWtndmkes3v3blVVVbl+3r9/v+666y7dd999JlblOfPmzdP//u//asWKFerVq5f27NmjcePGyW63a9q0aWaX5xETJ07U/v379eqrr6pz58567bXXdOedd6qwsFBdunRp2GIMNDhJxrp168wuo16VlpYakoytW7eaXUq9ad++vfHKK6+YXYZHlZeXGz169DByc3ONH//4x8a0adPMLsljnn76aaN3795ml1FvnnzySeP22283u4wGNW3aNKN79+5GdXW12aV4xLBhw4zx48e7jd17773G/fffb1JFnnXu3DnD29vb+L//+z+38d69exszZsxo8Ho4BYZ6UVZWJknq0KGDyZV4XlVVldasWaOzZ88qISHB7HI8avLkyRo2bJjuvPNOs0upFwcPHlTnzp0VERGhMWPG6PDhw2aX5DHr169XbGys7rvvPnXq1El9+vTRyy+/bHZZ9aayslKvvfaaxo8f7/F3Nprl9ttv13vvvacvvvhCkrRv3z5t375dd999t8mVecb333+vqqoq+fn5uY37+/tr+/btDV4Pp8DgcYZhaPr06br99tsVHR1tdjke8+mnnyohIUHnz59XmzZttG7dOkVFRZldlsesWbNG+fn52rNnj9ml1Iu4uDitXLlSN998s7755hvNnTtXiYmJ+uyzz9SxY0ezy7tuhw8fVlZWlqZPn67f//732rVrl6ZOnSpfX1898MADZpfncW+99ZZOnz6tsWPHml2Kxzz55JMqKytTz5495e3traqqKv35z3/WL3/5S7NL84i2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NXjwYCUmJuqpp57ySH2GYchms0m6eHrq0KFDatu2rduc8+fPu52S6927t1q1auX6OSEhQWfOnFFxcbHCw8Ov+H233Xab63+HhIRIkkpLSxv8mijAqghAABrM+PHj9dvf/laStHjx4hqfjxgxQmFhYXr55ZfVuXNnVVdXKzo62u20kyTl5eXJ29tbx48f19mzZxUQEHBddVVVVengwYPq16+fJKm6uloxMTFatWpVjbk33HDDfz3epSB1JS1btqwx/1LgA1D/uAYIQIO5dA1NZWWlBg8e7PbZyZMn9fnnn2vmzJkaNGiQIiMj9e2339Y4xo4dO/Tss8/qnXfeUUBAgKZMmXLdda1YsULffvutfv7zn0u6eNH2wYMH1alTJ910001um91ud+23b98+tzvb/vnPf6pNmzYKDQ2VJPn4+Kiqquq66wPgeQQgAA3G29tbn3/+uT7//HN5e3u7fXbpbqslS5bo0KFDev/99zV9+nS3OeXl5UpNTdWUKVM0dOhQvf766/rb3/6mN95446prOHfunEpKSnT06FHt3LlTTz75pCZNmqRHHnlEd9xxhyTp17/+tQIDAzVy5Eht27ZNR44c0datWzVt2jQdPXrUdazKykpNmDBBhYWF2rhxo55++mn99re/lZfXxb9au3Xrpp07d+qrr75yO6UHwHwEIAANKiAgoNZTVl5eXlqzZo3y8/MVHR2txx57TM8995zbnGnTpql169Z65plnJEm9evXSvHnzNGnSJB07duyqvv/ll19WSEiIunfvrnvuuUeFhYXKyclRZmama06rVq2Ul5enrl276t5771VkZKTGjx+v7777zq32QYMGqUePHho4cKBGjx6tESNG6I9//KPr88cff1ze3t6KiorSDTfcoKKiorr8UQGoRzbDMAyziwCApmbs2LE6ffq03nrrLbNLAXANWAECAACWQwAC0Cxs27ZNbdq0uewGAP+JU2AAmoXvvvvuitcB3XTTTQ1YDYDGjgAEAAAsh1NgAADAcghAAADAcghAAADAcghAAADAcghAAADAcghAAADAcghAAADAcghAAADAcv4fwadq6YQ+198AAAAASUVORK5CYII="/>

그래프에서 확인할 수 있는 최적의 max_depth 값은 무엇입니까?


###### your answer here

- max_depth = 2, 3 정도가 적당한 최적의 max_depth 값으로 본다.

- 1은 아직 훈련 및 모델구성이 빈약한 최소적합의심

- 4 이후부터는 더 이상 더 좋은 결과가 안나오는 걸 봐서 과대적합으로 의심.


앞서 배운 내용으로, 결정 지점에서 최소 샘플 수를 사용하여 과대적합 또는 과소적합을 방지할 수도 있습니다. 결정 트리 알고리즘에서 이 값은 min_samples_split에 의해 제어됩니다. 위의 코드를 복사하고 수정하여 최상의 min_samples_split 값을 찾아보세요. 2에서 15 사이의 범위를 사용할 수 있습니다. 쉽게 시각화할 수 있도록 정확도 값도 함께 표시해 보세요.



```python
# 각 의사 결정 트리에 대해 정확도와 가장 잘 테스트된 매개변수를 저장하기 위해 빈 목록을 만듭니다.
accuracy = []
min_samples = []
# ii를 사용하여 값 1에서 9까지 반복합니다. 이것은 의사결정 트리의 max_depth 값이 됩니다.
for ii in range(2,16):
    # max_depth를 ii로 설정
    dt = tree.DecisionTreeClassifier(min_samples_split=ii)
    # 데이터로 모델 훈련 또는 피팅
    dt.fit(x_train_scale,y_train)
    # .score는 테스트 데이터를 기반으로 모델의 정확도를 제공합니다. 정확도를 목록에 저장합니다.
    accuracy.append(dt.score(x_test_scale,y_test))
    # 목록에 max_depth 값 추가
    min_samples.append(ii)
print(accuracy)
plt.scatter(min_samples,accuracy)
plt.xlabel('Min_Samples_Split')
plt.ylabel('Accuracy')
plt.show();
```

<pre>
[0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473]
</pre>
<img src="data:image/png;base64,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+mjw4MFat26dnE6nnE6nPv30UyUmJqpPnz6VkREAAKBClfs2+LFjx+rQoUP6wx/+IC+vi4sXFxfrySef5BogAABwQ7ju2+D37dun7du3y8/PT3feeafr2T03A26DBwDgxlOpT4K+pFGjRmrUqNH1Lg4AAGCbcl8D9Mgjj2jChAmlxidPnqxHH320QkIBAABUpnIXoA0bNqhbt26lxh944AF99tlnFRIKAACgMpW7AP3888+qVq1aqXFvb2/l5+dXSCgAAIDKVO4CFB0drfT09FLjy5cvV9OmTSskFAAAQGUq90XQI0eO1B//+Ed99913+v3vfy9J+uSTT7R06VKtXLmywgMCAABUtHIXoB49eujdd9/V+PHjtXLlSvn5+al58+b69NNPuWUcAADcEK77OUCXnD59WkuWLNFbb72lHTt2yOl0VlQ22/AcIAAAbjzl+f1d7muALvn000/Vt29fhYaGasaMGXrwwQe1ZcuW610dAABAlSnXR2Dff/+9FixYoHnz5uns2bN67LHHVFRUpHfeeYcLoAEAwA3jms8APfjgg2ratKmys7M1ffp0HT16VNOnT6/MbAAAAJXims8AffTRRxo8eLBeeOEFvgIDAADc0K75DNDGjRt15swZtWrVSrGxsZoxY4Z+/PHHyswGAABQKa65ALVt21Zz5sxRbm6unn/+eS1fvlz16tVTcXGxMjIydObMmcrMCQAAUGF+023we/bs0VtvvaW3335bp0+fVufOnbVmzZqKzGcLboMHAODGUyW3wUtS48aNNWnSJH3//fdatmzZb1kVAABAlfnND0K8GXEGCACAG0+VnQECAAC4EVGAAACAcShAAADAOBQgAABgHAoQAAAwDgUIAAAYhwIEAACMQwECAADGoQABAADjUIAAAIBxKEAAAMA4FCAAAGAcChAAADAOBQgAABiHAgQAAIxDAQIAAMahAAEAAONQgAAAgHEoQAAAwDgUIAAAYBwKEAAAMA4FCAAAGIcCBAAAjEMBAgAAxqEAAQAA41CAAACAcShAAADAOBQgAABgHAoQAAAwDgUIAAAYhwIEAACMQwECAADGoQABAADjUIAAAIBxKEAAAMA4FCAAAGAcChAAADAOBQgAABjH9gKUmpqqyMhI+fr6KiYmRhs3brzi/JkzZyoqKkp+fn5q3LixFi1aVGrOO++8o6ZNm8rHx0dNmzbV6tWrKys+AAC4AdlagNLT05WUlKThw4dr27Zt6tChg7p27aqcnJwy56elpSk5OVmvvfaadu/erdGjR2vQoEF6//33XXM+//xz9e7dWwkJCdqxY4cSEhL02GOP6csvv6yq3QIAAG7OYVmWZdfGY2Nj1bJlS6WlpbnGoqKi1KtXL6WkpJSaHxcXp3bt2mny5MmusaSkJG3ZskWbNm2SJPXu3Vv5+fn68MMPXXMeeOAB1axZU8uWLbumXPn5+QoMDFReXp4CAgKud/cAAEAVKs/vb9vOABUWFiorK0vx8fElxuPj47V58+YylykoKJCvr2+JMT8/P2VmZqqoqEjSxTNAv15nly5dLrtOAABgHtsK0IkTJ+R0OhUcHFxiPDg4WMeOHStzmS5dumju3LnKysqSZVnasmWL5s2bp6KiIp04cUKSdOzYsXKtU7pYrPLz80u8AADAzcv2i6AdDkeJ95ZllRq7ZOTIkeratavatGkjb29v9ezZU/369ZMkeXp6Xtc6JSklJUWBgYGuV1hY2HXuDQAAuBHYVoCCgoLk6elZ6szM8ePHS53BucTPz0/z5s3TuXPndPDgQeXk5CgiIkL+/v4KCgqSJNWtW7dc65Sk5ORk5eXluV6HDx/+jXsHAADcmW0FqFq1aoqJiVFGRkaJ8YyMDMXFxV1xWW9vb9WvX1+enp5avny5HnroIXl4XNyVtm3bllrnRx99dMV1+vj4KCAgoMQLAADcvLzs3PjQoUOVkJCgVq1aqW3btpo9e7ZycnI0cOBASRfPzBw5csT1rJ+9e/cqMzNTsbGxOnXqlKZOnapdu3Zp4cKFrnUmJibq3nvv1cSJE9WzZ0+99957+vjjj113iQEAANhagHr37q2TJ09qzJgxys3NVXR0tNauXavw8HBJUm5ubolnAjmdTk2ZMkV79uyRt7e3OnXqpM2bNysiIsI1Jy4uTsuXL9eIESM0cuRI3X777UpPT1dsbGxV7x4AAHBTtj4HyF3xHCAAAG48N8RzgAAAAOxCAQIAAMahAAEAAONQgAAAgHEoQAAAwDgUIAAAYBwKEAAAMA4FCAAAGIcCBAAAjEMBAgAAxqEAAQAA41CAAACAcShAAADAOBQgAABgHAoQAAAwDgUIAAAYhwIEAACMQwECAADGoQABAADjUIAAAIBxKEAAAMA4FCAAAGAcChAAADAOBQgAABiHAgQAAIxDAQIAAMahAAEAAONQgAAAgHEoQAAAwDgUIAAAYBwKEAAAMA4FCAAAGIcCBAAAjEMBAgAAxqEAAQAA41CAAACAcShAAADAOBQgAABgHAoQAAAwDgUIAAAYhwIEAACMQwECAADGoQABAADjUIAAAIBxKEAAAMA4FCAAAGAcChAAADAOBQgAABiHAgQAAIxDAQIAAMahAAEAAONQgAAAgHEoQAAAwDgUIAAAYBwKEAAAMA4FCAAAGIcCBAAAjEMBAgAAxqEAAQAA41CAAACAcShAAADAOBQgAABgHNsLUGpqqiIjI+Xr66uYmBht3LjxivOXLFmi5s2bq3r16goJCVH//v118uTJEnNef/11NW7cWH5+fgoLC9OQIUN0/vz5ytwNAABwA7G1AKWnpyspKUnDhw/Xtm3b1KFDB3Xt2lU5OTllzt+0aZOefPJJPfPMM9q9e7dWrFihr776Ss8++6xrzpIlS/Tqq69q1KhR+uabb/TWW28pPT1dycnJVbVbAADAzdlagKZOnapnnnlGzz77rKKiovT6668rLCxMaWlpZc7/4osvFBERocGDBysyMlLt27fX888/ry1btrjmfP7552rXrp2eeOIJRUREKD4+Xo8//niJOQAAwGy2FaDCwkJlZWUpPj6+xHh8fLw2b95c5jJxcXH6/vvvtXbtWlmWpR9++EErV65Ut27dXHPat2+vrKwsZWZmSpL279+vtWvXlpgDAADM5mXXhk+cOCGn06ng4OAS48HBwTp27FiZy8TFxWnJkiXq3bu3zp8/rwsXLqhHjx6aPn26a06fPn30448/qn379rIsSxcuXNALL7ygV1999bJZCgoKVFBQ4Hqfn5//G/cOAAC4M9svgnY4HCXeW5ZVauyS7OxsDR48WH/961+VlZWlf/3rXzpw4IAGDhzomrN+/XqNGzdOqamp2rp1q1atWqUPPvhAf/vb3y6bISUlRYGBga5XWFhYxewcAABwSw7Lsiw7NlxYWKjq1atrxYoVevjhh13jiYmJ2r59uzZs2FBqmYSEBJ0/f14rVqxwjW3atEkdOnTQ0aNHFRISog4dOqhNmzaaPHmya87ixYv13HPP6eeff5aHR+nOV9YZoLCwMOXl5SkgIKCidhkAAFSi/Px8BQYGXtPvb9vOAFWrVk0xMTHKyMgoMZ6RkaG4uLgylzl37lypAuPp6Snp4pmjK82xLEuX63o+Pj4KCAgo8QIAADcv264BkqShQ4cqISFBrVq1Utu2bTV79mzl5OS4PtJKTk7WkSNHtGjRIklS9+7dNWDAAKWlpalLly7Kzc1VUlKS7rnnHoWGhrrmTJ06VS1atFBsbKy+/fZbjRw5Uj169HCVJQAAYDZbC1Dv3r118uRJjRkzRrm5uYqOjtbatWsVHh4uScrNzS3xTKB+/frpzJkzmjFjhl5++WXdeuut+v3vf6+JEye65owYMUIOh0MjRozQkSNHdNttt6l79+4aN25cle8fAABwT7ZdA+TOyvMZIgAAcA83xDVAAAAAdqEAAQAA41CAAACAcShAAADAOBQgAABgHAoQAAAwDgUIAAAYhwIEAACMQwECAADGoQABAADjUIAAAIBxKEAAAMA4FCAAAGAcChAAADAOBQgAABiHAgQAAIxDAQIAAMahAAEAAONQgAAAgHEoQAAAwDgUIAAAYBwKEAAAMA4FCAAAGIcCBAAAjEMBAgAAxqEAAQAA41CAAACAcShAAADAOF52B3BHlmVJkvLz821OAgAArtWl39uXfo9fCQWoDGfOnJEkhYWF2ZwEAACU15kzZxQYGHjFOQ7rWmqSYYqLi3X06FH5+/vL4XBU6Lrz8/MVFhamw4cPKyAgoELXfaPimJSN41Iax6Q0jknZOC6lmXBMLMvSmTNnFBoaKg+PK1/lwxmgMnh4eKh+/fqVuo2AgICb9i/g9eKYlI3jUhrHpDSOSdk4LqXd7Mfkamd+LuEiaAAAYBwKEAAAMA4FqIr5+Pho1KhR8vHxsTuK2+CYlI3jUhrHpDSOSdk4LqVxTEriImgAAGAczgABAADjUIAAAIBxKEAAAMA4FKAqkJKSotatW8vf31916tRRr169tGfPHrtjuZWUlBQ5HA4lJSXZHcV2R44cUd++fVW7dm1Vr15dd999t7KysuyOZZsLFy5oxIgRioyMlJ+fnxo2bKgxY8aouLjY7mhV6rPPPlP37t0VGhoqh8Ohd999t8TPLcvSa6+9ptDQUPn5+em+++7T7t277QlbRa50TIqKijRs2DDdeeeduuWWWxQaGqonn3xSR48etS9wFbna35X/9vzzz8vhcOj111+vsnzuggJUBTZs2KBBgwbpiy++UEZGhi5cuKD4+HidPXvW7mhu4auvvtLs2bN111132R3FdqdOnVK7du3k7e2tDz/8UNnZ2ZoyZYpuvfVWu6PZZuLEiXrzzTc1Y8YMffPNN5o0aZImT56s6dOn2x2tSp09e1bNmzfXjBkzyvz5pEmTNHXqVM2YMUNfffWV6tatq86dO7u+2udmdKVjcu7cOW3dulUjR47U1q1btWrVKu3du1c9evSwIWnVutrflUveffddffnllwoNDa2iZG7GQpU7fvy4JcnasGGD3VFsd+bMGatRo0ZWRkaG1bFjRysxMdHuSLYaNmyY1b59e7tjuJVu3bpZTz/9dImx//mf/7H69u1rUyL7SbJWr17tel9cXGzVrVvXmjBhgmvs/PnzVmBgoPXmm2/akLDq/fqYlCUzM9OSZB06dKhqQrmByx2X77//3qpXr561a9cuKzw83Jo2bVqVZ7MbZ4BskJeXJ0mqVauWzUnsN2jQIHXr1k3333+/3VHcwpo1a9SqVSs9+uijqlOnjlq0aKE5c+bYHctW7du31yeffKK9e/dKknbs2KFNmzbpwQcftDmZ+zhw4ICOHTum+Ph415iPj486duyozZs325jMveTl5cnhcBh9RlW6+H2XCQkJeuWVV9SsWTO749iG7wKrYpZlaejQoWrfvr2io6PtjmOr5cuXKysrS1u2bLE7itvYv3+/0tLSNHToUP3lL39RZmamBg8eLB8fHz355JN2x7PFsGHDlJeXpyZNmsjT01NOp1Pjxo3T448/bnc0t3Hs2DFJUnBwcInx4OBgHTp0yI5Ibuf8+fN69dVX9cQTT9zU34N1LSZOnCgvLy8NHjzY7ii2ogBVsT/96U/6+uuvtWnTJruj2Orw4cNKTEzURx99JF9fX7vjuI3i4mK1atVK48ePlyS1aNFCu3fvVlpamrEFKD09XYsXL9bSpUvVrFkzbd++XUlJSQoNDdVTTz1ldzy34nA4Sry3LKvUmImKiorUp08fFRcXKzU11e44tsrKytI//vEPbd261fi/G3wEVoVeeuklrVmzRuvWrav0b5t3d1lZWTp+/LhiYmLk5eUlLy8vbdiwQW+88Ya8vLzkdDrtjmiLkJAQNW3atMRYVFSUcnJybEpkv1deeUWvvvqq+vTpozvvvFMJCQkaMmSIUlJS7I7mNurWrSvp/50JuuT48eOlzgqZpqioSI899pgOHDigjIwM48/+bNy4UcePH1eDBg1c//YeOnRIL7/8siIiIuyOV6U4A1QFLMvSSy+9pNWrV2v9+vWKjIy0O5Lt/vCHP2jnzp0lxvr3768mTZpo2LBh8vT0tCmZvdq1a1fqEQl79+5VeHi4TYnsd+7cOXl4lPx/NU9PT+Nug7+SyMhI1a1bVxkZGWrRooUkqbCwUBs2bNDEiRNtTmefS+Vn3759WrdunWrXrm13JNslJCSUuuayS5cuSkhIUP/+/W1KZQ8KUBUYNGiQli5dqvfee0/+/v6u/0sLDAyUn5+fzens4e/vX+oaqFtuuUW1a9c2+tqoIUOGKC4uTuPHj9djjz2mzMxMzZ49W7Nnz7Y7mm26d++ucePGqUGDBmrWrJm2bdumqVOn6umnn7Y7WpX6+eef9e2337reHzhwQNu3b1etWrXUoEEDJSUlafz48WrUqJEaNWqk8ePHq3r16nriiSdsTF25rnRMQkND9cgjj2jr1q364IMP5HQ6Xf/21qpVS9WqVbMrdqW72t+VXxdBb29v1a1bV40bN67qqPay+S40I0gq8zV//ny7o7kVboO/6P3337eio6MtHx8fq0mTJtbs2bPtjmSr/Px8KzEx0WrQoIHl6+trNWzY0Bo+fLhVUFBgd7QqtW7dujL/HXnqqacsy7p4K/yoUaOsunXrWj4+Pta9995r7dy5097QlexKx+TAgQOX/bd33bp1dkevVFf7u/Jrpt4Gz7fBAwAA43ARNAAAMA4FCAAAGIcCBAAAjEMBAgAAxqEAAQAA41CAAACAcShAAADAOBQgAABgHAoQgKu67777lJSUZHeMKrFgwQLdeuutdse4ZuvXr5fD4dDp06cl3Xj5AbtQgABD9evXTw6HQwMHDiz1sxdffFEOh0P9+vWTJK1atUp/+9vfKmzb69atU6dOnVSrVi1Vr15djRo10lNPPaULFy5U2DZuBPv379fjjz+u0NBQ+fr6qn79+urZs6f27t173evs3bt3ieVfe+013X333RWQFri5UIAAg4WFhWn58uX65ZdfXGPnz5/XsmXL1KBBA9dYrVq15O/vXyHb3L17t7p27arWrVvrs88+086dOzV9+nR5e3sb9Q3vhYWF6ty5s/Lz87Vq1Srt2bNH6enpio6OVl5e3nWv18/PT3Xq1KnApMDNiQIEGKxly5Zq0KCBVq1a5RpbtWqVwsLC1KJFC9fYrz8Ci4iI0Pjx4/X000/L399fDRo0uOZvrM/IyFBISIgmTZqk6Oho3X777XrggQc0d+5c1zd0nzx5Uo8//rjq16+v6tWr684779SyZctKrOe+++7TSy+9pKSkJNWsWVPBwcGaPXu2zp49q/79+8vf31+33367PvzwQ9cylz4u+uc//6nmzZvL19dXsbGx2rlz5xUzv//++4qJiZGvr68aNmyo0aNHlzhb9dprr6lBgwby8fFRaGioBg8efNXjkJ2drf379ys1NVVt2rRReHi42rVrp3Hjxql169aSpIMHD8rhcGj58uWKi4uTr6+vmjVrpvXr1192vf/9EdiCBQs0evRo7dixQw6HQw6HQwsWLLhqNsAEFCDAcP3799f8+fNd7+fNm6enn376qstNmTJFrVq10rZt2/Tiiy/qhRde0H/+85+rLle3bl3l5ubqs88+u+yc8+fPKyYmRh988IF27dql5557TgkJCfryyy9LzFu4cKGCgoKUmZmpl156SS+88IIeffRRxcXFaevWrerSpYsSEhJ07ty5Esu98sor+vvf/66vvvpKderUUY8ePVRUVFRmln//+9/q27evBg8erOzsbM2aNUsLFizQuHHjJEkrV67UtGnTNGvWLO3bt0/vvvuu7rzzzqseh9tuu00eHh5auXKlnE7nFee+8sorevnll7Vt2zbFxcWpR48eOnny5FW30bt3b7388stq1qyZcnNzlZubq969e191OcAIdn8dPQB7PPXUU1bPnj2tH3/80fLx8bEOHDhgHTx40PL19bV+/PFHq2fPntZTTz1lWZZldezY0UpMTHQtGx4ebvXt29f1vri42KpTp46VlpZ21e1euHDB6tevnyXJqlu3rtWrVy9r+vTpVl5e3hWXe/DBB62XX37Z9b5jx45W+/btS6z3lltusRISElxjubm5liTr888/tyzLstatW2dJspYvX+6ac/LkScvPz89KT0+3LMuy5s+fbwUGBrp+3qFDB2v8+PElsrz99ttWSEiIZVmWNWXKFOt3v/udVVhYeNV9/7UZM2ZY1atXt/z9/a1OnTpZY8aMsb777jvXzw8cOGBJsiZMmOAaKyoqsurXr29NnDixxD6dOnWqzPyjRo2ymjdvXu5swM2OM0CA4YKCgtStWzctXLhQ8+fPV7du3RQUFHTV5e666y7XfzscDtWtW1fHjx+/6nKenp6aP3++vv/+e02aNEmhoaEaN26c6yyFJDmdTo0bN0533XWXateurRo1auijjz5STk7OZTN4enqqdu3aJc6+BAcHS1KpXG3btnX9d61atdS4cWN98803ZebNysrSmDFjVKNGDddrwIABys3N1blz5/Too4/ql19+UcOGDTVgwACtXr36mi/mHjRokI4dO6bFixerbdu2WrFihZo1a6aMjIzL5vXy8lKrVq0umxfAtaEAAdDTTz+tBQsWaOHChdf08ZckeXt7l3jvcDjKdRFzvXr1lJCQoJkzZyo7O1vnz5/Xm2++Kenix2vTpk3T//7v/+rTTz/V9u3b1aVLFxUWFl41w3+PORwOSbqmXJfm/lpxcbFGjx6t7du3u147d+7Uvn375Ovrq7CwMO3Zs0czZ86Un5+fXnzxRd17772X/Ujt1/z9/dWjRw+NGzdOO3bsUIcOHTR27Njrzgvg2lCAAOiBBx5QYWGhCgsL1aVLlyrffs2aNRUSEqKzZ89KkjZu3KiePXuqb9++at68uRo2bKh9+/ZV2Pa++OIL13+fOnVKe/fuVZMmTcqc27JlS+3Zs0d33HFHqZeHx8V/Qv38/NSjRw+98cYbWr9+vT7//POrXlhdFofDoSZNmriOQ1l5L1y4oKysrMvm/bVq1apd9RojwERedgcAYD9PT0/XRyqenp6Vuq1Zs2Zp+/btevjhh3X77bfr/PnzWrRokXbv3q3p06dLku644w6988472rx5s2rWrKmpU6fq2LFjioqKqpAMY8aMUe3atRUcHKzhw4crKChIvXr1KnPuX//6Vz300EMKCwvTo48+Kg8PD3399dfauXOnxo4dqwULFsjpdCo2NlbVq1fX22+/LT8/P4WHh18xw/bt2zVq1CglJCSoadOmqlatmjZs2KB58+Zp2LBhJebOnDlTjRo1UlRUlKZNm6ZTp05d85m6iIgIHThwQNu3b1f9+vXl7+8vHx+fa1oWuJlRgABIkgICAqpkO/fcc482bdqkgQMH6ujRo6pRo4aaNWumd999Vx07dpQkjRw5UgcOHFCXLl1UvXp1Pffcc+rVq9dvej7Of5swYYISExO1b98+NW/eXGvWrHHdgv9rXbp00QcffKAxY8Zo0qRJ8vb2VpMmTfTss89Kkm699VZNmDBBQ4cOldPp1J133qn3339ftWvXvmKG+vXrKyIiQqNHj3bd7n7p/ZAhQ0rlnThxorZt26bbb79d77333jVdpyVJf/zjH7Vq1Sp16tRJp0+f1vz5810PuARM5rAsy7I7BABUhfXr16tTp046derUDfF1EQcPHlRkZKS2bdvG05yBCsY1QAAAwDgUIAAVavz48SVuGf/vV9euXe2OV2U2btx42eNQo0YNu+MBxuMjMAAV6qefftJPP/1U5s/8/PxUr169Kk5kj19++UVHjhy57M/vuOOOKkwD4NcoQAAAwDh8BAYAAIxDAQIAAMahAAEAAONQgAAAgHEoQAAAwDgUIAAAYBwKEAAAMA4FCAAAGOf/A5fqYDrq3g2mAAAAAElFTkSuQmCC"/>

그래프에서 가장 좋은 min_samples_split 값은 무엇입니까? 가장 높은 정확도를 갖는 가장 낮은 값을 선택하겠습니까?


###### your answer here

- 다 일정한 값이 나오기에 왜 이러는지 원인 파악중;;


#### 참고한 외부사이트



- sklearn preprocessing StandardScaler 메뉴얼  

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html



- ......


#### 마무리

- KNN, DTree등에서 최소적합과 과소적합, 분산, 편향 등으로 훈련을 많이 하면 좋은가? 에 대한 고찰의 기회.

- 맨 마지막 문제에서 Min_Samples_Split에 관해서는 유감, 현재 원인 분석 중

- 최대 깊이 등.. 설정을 범위 내로 순차적으로 반복해서 돌려서 최선의 값을 파악하는 이 방식은 어떠한 설정 값이 나은지  

파악하는데 많은 도움이 됨.


#### 주의사항



본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.

