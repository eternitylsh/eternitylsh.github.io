---
layout: single
title:  "KNN모델로 장바구니 추천해보기"
categories: jupyter
tag: [python, blog, jupyter]
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


# Shopping Basket Recommendation System with Python





# Python을 이용한 장바구니 추천 시스템


## Introduction



인공지능은 프로세스를 자동화하고, 비즈니스에 대한 통찰력을 모으고, 프로세스 속도를 높이기 위해 다양한 산업에서 사용되고 있습니다. 인공지능이 실제로 산업에 어떤 영향을 미치는지 실제 시나리오에서 인공지능의 사용을 연구하기 위해 Python을 사용할 것입니다.



장바구니에는 기본적으로 사람이 구매한 아이템 목록이 포함되어 있습니다. 이러한 데이터는 어떤 제품이 수요가 있는지, 어떤 제품이 계절적 상품인지 등과 같은 정보를 나타낼 것이기 때문에 상점에 매우 유용할 정보가 될수 있습니다. 기업은 어떠한 상품에 초점을 맞춰야 하는지 파악하고 이를 바탕으로 추천할 수 있습니다. 장바구니 추천은 AI를 이용해 한 사람의 쇼핑 리스트를 연구해 그 사람에게 구매할 가능성이 있는 물건을 제안할 수 있습니다.



이 노트북에서는 KNN 모델을 사용한 장바구니 추천 시스템에 중점을 둘 것입니다.



## Context



[Kaggle]에서 가져온 Amazon 제품 리뷰를 사용하여 작업합니다. Kaggle은 데이터 전문가들이 모여 지식을 공유하고 서로 경쟁하여 보상을 받을 수 있는 데이터 공유 플랫폼입니다.



## 고객 리뷰 데이터



Amazon, Walmart와 같은 대형 전자 상거래 회사는 매일 수백만 명의 고객과 거래합니다. 고객은 제품을 검색하고 구매하고 때로는 리뷰를 남깁니다. 이를 감안할 때 고객은 전자 상거래 회사에게 가장 중요한 요소입니다: 그들을 지속적으로 만족시켜야 합니다.



고객의 쇼핑 이력, 즉 고객이 무엇을 구매하고 무엇을 선호하는지 알고 있다고 상상해 보십시오. 당신은 그들이 미래에 무엇을 사고 싶어할지 예측하고 그러한 것들을 제안함으로써 이 정보를 당신에게 유리하게 사용할 수 있습니다.


### Side note: KNN 이란?



KNN(K-Nearest Neighbors)은 분류와 회귀에 모두 사용되는 알고리즘입니다. KNN 알고리즘은 '유유상종'이라는 말처럼 주변에 비슷한 것이 존재한다고 가정한다. KNN 알고리즘은 가장 가까운 이웃의 클래스, 특히 k 수를 기반으로 새 데이터 포인트를 분류합니다. k는 객체의 클래스를 결정하는 데 도움이 되는 가장 가까운 이웃의 수를 나타냅니다. 다음 다이어그램에서 명확하게 확인할 수 있습니다.



![Knn where k = 3](https://cambridgecoding.files.wordpress.com/2016/01/knn2.jpg)


# Types of recommendations



There are mainly 6 types of the recommendations systems :-



1. Popularity based systems :- It works by recommeding items viewed and purchased by most people and are rated high.It is not a personalized recommendation.

2. Classification model based:- It works by understanding the features of the user and applying the classification algorithm to decide whether the user is     interested or not in the prodcut.

3. Content based recommedations:- It is based on the information on the contents of the item rather than on the user opinions.The main idea is if the user likes an item then he or she will like the "other" similar item.

4. Collaberative Filtering:- It is based on assumption that people like things similar to other things they like, and things that are liked by other people with similar taste. it is mainly of two types:

 a) User-User 

 b) Item -Item

 

5. Hybrid Approaches:- This system approach is to combine collaborative filtering, content-based filtering, and other approaches . 

6. Association rule mining :- Association rules capture the relationships between items based on their patterns of co-occurrence across transactions.


## Use Python to open csv files



[scikit-learn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/)를 사용하여 데이터 세트를 작업합니다. Scikit-learn은 예측 데이터 분석을 위한 효율적인 도구를 제공하는 매우 유용한 기계 학습 라이브러리입니다. Pandas는 데이터 과학을 위한 인기 있는 Python 라이브러리입니다. 강력하고 유연한 데이터 구조를 제공하여 데이터 조작 및 분석을 더 쉽게 만듭니다. Matplotlib은 고품질의 데이터 시각화를 위한 Python의 2차원 그래프 라이브러리입니다. 코드 몇 줄만으로 간단하게 복잡한 그래프를 만들 수 있어서 사용성이 매우 높습니다.



## Import Libraries



```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
%matplotlib inline
```

이제 라이브러리를 가져왔으므로 csv 파일을 읽어오겠습니다.



```python
# electronics_data 변수로 [Dataset]_Module11_(Recommendation).csv 파일 읽어오기
# your code here
col = ['userId', 'productId', 'Rating', 'timestamp']
df_dataset = pd.read_csv("./[Dataset]_Module11_(Recommendation).csv",names=col, header=None)
```

데이터가 어떻게 구성되었는지 살펴보겠습니다.



```python
# your code here
df_dataset.head(5)
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
      <th>userId</th>
      <th>productId</th>
      <th>Rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AKM1MP6P0OYPR</td>
      <td>0132793040</td>
      <td>5.0</td>
      <td>1365811200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2CX7LUOHB2NDG</td>
      <td>0321732944</td>
      <td>5.0</td>
      <td>1341100800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2NWSAGRHCP8N5</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1367193600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A2WNBOD3WNDNKT</td>
      <td>0439886341</td>
      <td>3.0</td>
      <td>1374451200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A1GI0U4ZRJA8WN</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1334707200</td>
    </tr>
  </tbody>
</table>
</div>


### Task 1: electronic data의 첫 20 행 표시



```python
#yourcodehere
df_dataset.head(20)
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
      <th>userId</th>
      <th>productId</th>
      <th>Rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AKM1MP6P0OYPR</td>
      <td>0132793040</td>
      <td>5.0</td>
      <td>1365811200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2CX7LUOHB2NDG</td>
      <td>0321732944</td>
      <td>5.0</td>
      <td>1341100800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2NWSAGRHCP8N5</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1367193600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A2WNBOD3WNDNKT</td>
      <td>0439886341</td>
      <td>3.0</td>
      <td>1374451200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A1GI0U4ZRJA8WN</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1334707200</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A1QGNMC6O1VW39</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1397433600</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A3J3BRHTDRFJ2G</td>
      <td>0511189877</td>
      <td>2.0</td>
      <td>1397433600</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A2TY0BTJOTENPG</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1395878400</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A34ATBPOK6HCHY</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1395532800</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A89DO69P0XZ27</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1395446400</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AZYNQZ94U6VDB</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1401321600</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A1DA3W4GTFXP6O</td>
      <td>0528881469</td>
      <td>5.0</td>
      <td>1405641600</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A29LPQQDG7LD5J</td>
      <td>0528881469</td>
      <td>1.0</td>
      <td>1352073600</td>
    </tr>
    <tr>
      <th>13</th>
      <td>AO94DHGC771SJ</td>
      <td>0528881469</td>
      <td>5.0</td>
      <td>1370131200</td>
    </tr>
    <tr>
      <th>14</th>
      <td>AMO214LNFCEI4</td>
      <td>0528881469</td>
      <td>1.0</td>
      <td>1290643200</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A28B1G1MSJ6OO1</td>
      <td>0528881469</td>
      <td>4.0</td>
      <td>1280016000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>A3N7T0DY83Y4IG</td>
      <td>0528881469</td>
      <td>3.0</td>
      <td>1283990400</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A1H8PY3QHMQQA0</td>
      <td>0528881469</td>
      <td>2.0</td>
      <td>1290556800</td>
    </tr>
    <tr>
      <th>18</th>
      <td>A2CPBQ5W4OGBX</td>
      <td>0528881469</td>
      <td>2.0</td>
      <td>1277078400</td>
    </tr>
    <tr>
      <th>19</th>
      <td>A265MKAR2WEH3Y</td>
      <td>0528881469</td>
      <td>4.0</td>
      <td>1294790400</td>
    </tr>
  </tbody>
</table>
</div>


## 데이터셋에 대한 정보 얻기



데이터 세트에 대한 다양한 정보를 수집할 수 있다면 데이터 세트에 대한 명확한 그림을 제공하고, 데이터를 처리하는 데 도움이 될 것입니다.



```python
# 데이터의 형태
# your code here
df_dataset.shape
```

<pre>
(7824482, 4)
</pre>

```python
# 데이터 세트의 하위 집합 1048576 개 데이터 가져오기
# your code here
df_dataset.iloc[:1048576]
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
      <th>userId</th>
      <th>productId</th>
      <th>Rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AKM1MP6P0OYPR</td>
      <td>0132793040</td>
      <td>5.0</td>
      <td>1365811200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2CX7LUOHB2NDG</td>
      <td>0321732944</td>
      <td>5.0</td>
      <td>1341100800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2NWSAGRHCP8N5</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1367193600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A2WNBOD3WNDNKT</td>
      <td>0439886341</td>
      <td>3.0</td>
      <td>1374451200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A1GI0U4ZRJA8WN</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1334707200</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1048571</th>
      <td>A1H16CBCNEL4G3</td>
      <td>B000IF51UQ</td>
      <td>5.0</td>
      <td>1356739200</td>
    </tr>
    <tr>
      <th>1048572</th>
      <td>A1C2OANTC49AQI</td>
      <td>B000IF51UQ</td>
      <td>5.0</td>
      <td>1382486400</td>
    </tr>
    <tr>
      <th>1048573</th>
      <td>A2JL0387FDDFS</td>
      <td>B000IF51UQ</td>
      <td>5.0</td>
      <td>1340409600</td>
    </tr>
    <tr>
      <th>1048574</th>
      <td>A2KIYE5RF0OEMY</td>
      <td>B000IF51UQ</td>
      <td>4.0</td>
      <td>1391212800</td>
    </tr>
    <tr>
      <th>1048575</th>
      <td>A1G9Q5UJ5Y7DES</td>
      <td>B000IF51UQ</td>
      <td>2.0</td>
      <td>1390780800</td>
    </tr>
  </tbody>
</table>
<p>1048576 rows × 4 columns</p>
</div>



```python
# 데이터 타입 확인
# your code here
df_dataset.dtypes
```

<pre>
userId        object
productId     object
Rating       float64
timestamp      int64
dtype: object
</pre>

```python
# 데이터 정보 확인
# your code here
df_dataset.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7824482 entries, 0 to 7824481
Data columns (total 4 columns):
 #   Column     Dtype  
---  ------     -----  
 0   userId     object 
 1   productId  object 
 2   Rating     float64
 3   timestamp  int64  
dtypes: float64(1), int64(1), object(2)
memory usage: 238.8+ MB
</pre>
### Task 2: 데이터 세트의 Rating 열에 대한 정보 표시



```python
# yourcode here
df_dataset['Rating'].describe()
```

<pre>
count    7.824482e+06
mean     4.012337e+00
std      1.380910e+00
min      1.000000e+00
25%      3.000000e+00
50%      5.000000e+00
75%      5.000000e+00
max      5.000000e+00
Name: Rating, dtype: float64
</pre>

```python
# 등급이 1-5 척도인지 1-10 척도인지 알아보기 위해 최소 및 최대 등급을 찾아보겠습니다.
# yourcode here
print( "Minimum rating is: " + str(int(df_dataset['Rating'].min())) )
print( "Maximum rating is: " + str(int(df_dataset['Rating'].max())) )
```

<pre>
Minimum rating is: 1
Maximum rating is: 5
</pre>
### Task 3: 데이터 세트에서 누락된 값 확인




```python
# 데이터세트에서 누락된 값을 확인하겠습니다.
# your code here
df_dataset.isnull().sum()
```

<pre>
userId       0
productId    0
Rating       0
timestamp    0
dtype: int64
</pre>

```python
# 막대 그래프를 이용하여 다양한 등급 분포를 알아보겠습니다.
# 해당 코드는 나중에 진행;;; catplot으로 하는 방법에 관해서 빨리...
# with sns.axes_style('white'):
#     g = sns.catplot(data=df_dataset["Rating"], kind='bar', x='Rating' )

with sns.axes_style('white'):
    g = sns.factorplot("Rating", data=df_dataset, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")
```

<pre>
C:\ProgramData\Anaconda3\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
  warnings.warn(msg)
C:\ProgramData\Anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+AAAAHtCAYAAACZJhSKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAsX0lEQVR4nO3deZBV9Z3//1fbgKzign5BRAWV1sQNcdfYGdTomDIZcCah3BJELY0YQwZNkExGUaZlZlwSDUbLJYMLWUQYFcs9Gk2MoMaoiJq4BFBURBIExMbm/v7IL8wwLvSFvufS18ejqqv6nnvuue+2PlXWk3vOuXWlUqkUAAAAoKI2qvYAAAAA8GkgwAEAAKAAAhwAAAAKIMABAACgAAIcAAAACiDAAQAAoAACHAAAAAogwAEAAKAAAhwAAAAKIMABAACgADUR4LNmzcppp52Wgw8+OA0NDbnvvvvKPkapVMq1116bI444IrvuumsaGxvz4x//uALTAgAA8GnUodoDtIXly5enoaEhw4YNy5lnnrlOx5gwYUIeeeSRnHPOORk4cGCWLl2axYsXt/GkAAAAfFrVRIA3NjamsbHxY59vbm7OZZddlttvvz3vvvtudtppp4wZMyb77bdfkuSll17KlClTcvvtt2fAgAFFjQ0AAMCnSE2cgr42Y8eOzZNPPplLL700t912W4488sicfPLJefXVV5MkDzzwQLbZZps8+OCDGTJkSIYMGZJx48blz3/+c1XnBgAAoHbUfIDPnTs3M2bMyA9+8IPsvffe2XbbbTNy5MgMHjw4t956a5Jk3rx5ef3113PXXXfl3//939PU1JTZs2fnm9/8ZpWnBwAAoFbUxCnon2T27NkplUo58sgj19je3NycTTfdNMlfb8DW3NyciRMnpn///kn+ek34sGHD8vLLLzstHQAAgPVW8wFeKpVSX1+fqVOnpr6+fo3nunbtmiTZcsst06FDh9XxnSQ77LBDkmTBggUCHAAAgPVW8wG+yy67pKWlJe+880723nvvj9xnr732ygcffJC5c+dm2223TZLV14dvvfXWRY0KAABADauJa8CXLVuWOXPmZM6cOUmS+fPnZ86cOXn99dfTv3//HH300TnnnHNyzz33ZN68eXn66adz9dVX56GHHkqSHHjggfnsZz+bc889N88991yeffbZfP/7389BBx20xqfiAAAAsK7qSqVSqdpDrK/HHnssJ5544oe2Dx06NBdddFFWrlyZK6+8MtOnT89bb72VTTfdNHvuuWfOPPPMNDQ0JEnefPPNXHjhhXnkkUfStWvXHHLIIfnOd76z+jpxAAAAWB81EeAAAACwoauJU9ABAABgQyfAAQAAoADtNsBLpVKWLl0aZ9ADAADQHrTbAF+2bFkGDx6cZcuWVXsUAAAAWKt2G+AAAADQnghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAAA2eKtWlao9AjWmGmuqQ+HvCAAAUKaNNqrLPTc9mcVvLq32KNSAzf5f93zhuL0Kf18BDgAAtAuL31yaha/9pdpjwDpzCjoAAAAUQIADAABAAQQ4AAAAFECAAwAAQAEEOAAAABRAgAMAAEABBDgAAAAUQIADAABAAQQ4AAAAFECAAwAAQAEEOAAAABRAgAMAAEABBDgAAAAUQIADAABAAQQ4AAAAFECAAwAAQAEEOAAAABRAgAMAAEABBDgAAAAUQIADAABAAQQ4AAAAFECAAwAAQAEEOAAAABRAgAMAAEABBDgAAAAUYIMI8KuuuioNDQ2ZMGFCtUcBAACAiqh6gD/99NP52c9+loaGhmqPAgAAABVT1QBftmxZzj777Fx44YXp2bNnNUcBAACAiqpqgI8fPz6NjY058MADqzkGAAAAVFyHar3xjBkzMnv27EydOrVaIwAAAEBhqhLgCxYsyIQJE3Lddddl4403rsYIAAAAUKiqBPjs2bOzaNGiDBs2bPW2lpaWzJo1KzfddFOeeeaZ1NfXV2M0AAAAqIiqBPj++++f22+/fY1tY8eOzYABA3LKKaeIbwAAAGpOVQK8e/fuGThw4Brbunbtmk033fRD2wEAAKAWVP17wAEAAODToGp3Qf+/brjhhmqPAAAAABXjE3AAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACiAAAcAAIACCHAAAAAogAAHAACAAghwAAAAKIAABwAAgAIIcAAAACjAegX40qVLc9999+Wll15qq3kAAACgJpUV4GeddVZuvPHGJMmKFStyzDHH5Fvf+la+9KUv5e67767IgAAAAFALygrwxx9/PHvvvXeS5N57702pVMqsWbMybty4XHnllRUZEAAAAGpBWQH+7rvvpmfPnkmShx9+OF/4whfSpUuXfP7zn8+f/vSnigwIAAAAtaCsAO/Tp09+97vfZfny5Xn44Ydz0EEHJUmWLFmSTp06VWRAAAAAqAUdytn5xBNPzNlnn52uXbtm6623zn777ZckmTVrVgYOHFiRAQEAAKAWlBXgxx13XHbfffe88cYbOfDAA7PRRn/9AL1fv3751re+VYn5AAAAoCaUFeBJsttuu2W33XZbY9vnP//5tpoHAAAAalJZAd7U1PSR2+vq6rLxxhtn2223zaGHHppNN920LWYDAACAmlFWgD/33HN57rnnsmrVqvTv3z+lUimvvvpq6uvrM2DAgNx8882ZOHFibr755uy4446VmhkAAADanbLugn7ooYfmwAMPzMMPP5xbb70106ZNy8MPP5wDDzwwX/ziF/OrX/0qe++998d+Uv43N998c44++ujstdde2WuvvfLVr341Dz300Hr9IQAAALAhKyvAr7322px11lnp3r376m3du3fPmWeemWuuuSZdunTJGWeckWefffYTj9O7d++MGTMmU6dOzdSpU7P//vvnjDPOyB/+8Id1+ysAAABgA1dWgC9dujSLFi360PZ33nknS5cuTZJssskmWbly5SceZ8iQIWlsbEz//v3Tv3//jB49Ol27ds1TTz1VzjgAAADQbpQV4EOGDMm5556be++9N2+88UbefPPN3HvvvRk3blwOO+ywJMnTTz+d7bffvtXHbGlpyYwZM7J8+fIMGjSorOEBAACgvSjrJmzjx49PU1NTRo8enZaWliRJfX19hg4dmrFjxyZJBgwYkAkTJqz1WC+88EKGDx+e999/P127ds2PfvQjN24DAACgZtWVSqVSuS9atmxZ5s2blyTp169funXrVvYbNzc3Z8GCBVmyZEnuueee/OIXv8iNN97Y6ghfunRpBg8enCeeeGKNa9IBAIDa9LNLfpWFr/2l2mNQA7bs2zNf/fYhhb9vWZ+A/023bt2y8847r9cbd+rUKdttt12SZLfddsszzzyTyZMnZ/z48et1XAAAANgQlRXgy5cvz9VXX53f/va3WbRoUVatWrXG8/fff/86D1IqldLc3LzOrwcAAIANWVkB/r3vfS8zZ87Ml7/85Wy55Zapq6tbpze95JJLcsghh6R3795ZtmxZ7rzzzsycOTPXXHPNOh0PAAAANnRlBfivfvWrXHXVVRk8ePB6venbb7+dc845J2+99VZ69OiRhoaGXHPNNTnooIPW67gAAACwoSorwDfZZJNsuumm6/2m//Zv/7bexwAAAID2pKzvAT/rrLPygx/8IO+9916l5gEAAICaVNYn4Ndff33mzp2bAw88MNtss006dFjz5dOmTWvT4QAAAKBWlBXghx12WKXmAAAAgJpWVoCPGjWqUnMAAABATSvrGnAAAABg3az1E/B99903d911VzbffPPss88+n/jd3zNnzmzT4QAAAKBWrDXAx44dm+7du6/+/ZMCHAAAAPhoaw3woUOHrv592LBhFR0GAAAAalVZ14DvsssuWbRo0Ye2L168OLvsskubDQUAAAC1pqwAL5VKH7m9ubk5HTt2bJOBAAAAoBa16mvIJk+enCSpq6vLL37xi3Tt2nX1c6tWrcqsWbMyYMCAykwIAAAANaBVAf6Tn/wkyV8/Af/pT3+ajTb6nw/OO3bsmG222Sbnn39+RQYEAACAWtCqAH/ggQeSJCeccEKuuOKK9OzZs6JDAQAAQK1pVYD/zQ033FCpOQAAAKCmlRXgSfLGG2/k/vvvz4IFC7Jy5co1nhs7dmybDQYAAAC1pKwAf/TRR3P66adnm222ySuvvJKddtopr732WkqlUj7zmc9UakYAAABo98r6GrKLL744I0aMyB133JFOnTrl8ssvz4MPPph99tknRx55ZKVmBAAAgHavrAB/6aWXMnTo0CRJhw4dsmLFinTr1i1nnXVWrrnmmooMCAAAALWgrADv2rVrmpubkyRbbbVV5s6du/q5xYsXt+1kAAAAUEPKugZ8jz32yJNPPpkdd9wxjY2NmThxYl588cXce++92WOPPSo1IwAAALR7ZQX42LFjs2zZsiTJmWeemeXLl+fOO+/Mdttt5w7oAAAA8AlaHeAtLS1ZsGBBGhoakiRdunTJeeedV6m5AAAAoKa0+hrw+vr6jBw5MkuWLKnkPAAAAFCTyroJ28CBAzN//vxKzQIAAAA1q6wAHz16dCZOnJhf/vKXeeutt7J06dI1fgAAAICPVtZN2E4++eQkyemnn566urrV20ulUurq6jJnzpy2nQ4AAABqRFkBPnny5ErNAQAAADWtrADfd999KzUHAAAA1LSyrgEHAAAA1o0ABwAAgAIIcAAAACjAWgP8/vvvz8qVK4uYBQAAAGrWWgN81KhReffdd5Mku+yySxYtWlTxoQAAAKDWrDXAN9988zz11FNJ/uf7vgEAAIDyrPVryIYPH55vfOMbqaurS11dXQ466KCP3XfOnDltOhwAAADUirUG+Jlnnpmjjjoqc+fOzemnn56mpqb06NGjiNkAAACgZqw1wJNkhx12yA477JBRo0blyCOPTJcuXSo9FwAAANSUVgX434waNSpJ8s477+Tll19OXV1d+vfvn80337wiwwEAAECtKCvA33vvvYwfPz633XZbWlpakiT19fX58pe/nH/5l3/xyTgAAAB8jLXeBf1/a2pqyqxZszJp0qQ8/vjjefzxxzNp0qTMmjUrF110UaVmBAAAgHavrAC/++67M2HChDQ2NqZ79+7p3r17Ghsbc8EFF+Tuu++u1IwAAADQ7pUV4CtWrEivXr0+tH2LLbbIihUr2mwoAAAAqDVlBfiee+6ZH/7wh3n//fdXb1uxYkWuuOKK7Lnnnm09GwAAANSMsm7CNm7cuJx88sk55JBDsvPOO6euri5z5szJxhtvnGuvvbZSMwIAAEC7V1aADxw4MPfcc09uu+22vPzyyymVSvniF7+Yo48+Op07d67UjAAA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Vh2orq5OgAMAAMDHWGuAP/DAA0XMAQAAADXNNeAAAABQgLV+Av5/vfHGG7n//vuzYMGCrFy5co3nxo4d22aDAQAAQC0pK8AfffTRnH766dlmm23yyiuvZKeddsprr72WUqmUz3zmM5WaEQAAANq9sk5Bv/jiizNixIjccccd6dSpUy6//PI8+OCD2WeffXLkkUdWakYAAABo98oK8JdeeilDhw5NknTo0CErVqxIt27dctZZZ+Waa66pyIAAAABQC8oK8K5du6a5uTlJstVWW2Xu3Lmrn1u8eHHbTgYAAAA1pKxrwPfYY488+eST2XHHHdPY2JiJEyfmxRdfzL333ps99tijUjMCAABAu1dWgI8dOzbLli1Lkpx55plZvnx57rzzzmy33XbugA4AAACfoKwA79ev3+rfu3TpkvPOO6+t5wEAAICaVNY14IceeuhHXuu9ZMmSHHrooW02FAAAANSasgL8tddey6pVqz60vbm5OW+++WabDQUAAAC1plWnoN9///2rf3/44YfTo0eP1Y9XrVqVRx99NH379m376QAAAKBGtCrAzzjjjCRJXV1dvvvd7655gA4d0rdv3w9tBwAAAP5HqwL8+eefT5IMGTIkt9xySzbffPOKDgUAAAC1pqy7oD/wwAOVmgMAAABqWlkBniQzZ87Mddddl5deeil1dXUZMGBATj755Oy9996VmA8AAABqQll3Qf/v//7vjBgxIp07d84JJ5yQ4447Lp07d87Xv/713H777ZWaEQAAANq9sj4B//GPf5yzzz47X//611dv+9rXvpbrr78+kyZNytFHH93W8wEAAEBNKOsT8Hnz5uXv/u7vPrR9yJAhmT9/fpsNBQAAALWmrADv06dPHn300Q9tf/TRR9OnT582GwoAgNYptbRUewRqjDUFldOqU9DHjh2bcePGZcSIEbnwwgszZ86cDBo0KHV1dXniiScybdq0jBs3rtKzAgDwf9TV12fOBRdk+Z/+VO1RqAFdt9suu/zLv1R7DKhZrQrw6dOnZ8yYMTn22GOz5ZZb5rrrrstdd92VJBkwYEAuvfTSHHbYYRUdFACAj7b8T3/K0hf/UO0xAFiLVgV4qVRa/fvhhx+eww8/vGIDAQAAQC1q9TXgdXV1lZwDAAAAalqrv4bsiCOOWGuEz5w5c70HAgAAgFrU6gA/88wz06NHj0rOAgAA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9ImvsdZp76666qo0NDRkwoQJn7iftU57dPnll6ehoWGNn4MOOugTX2Otb1gEOKstX748DQ0N+f73v9+q/efNm5dTTz01gwcPzvTp03PaaadlwoQJufvuuys8Kay7mTNn5rjjjsvPf/7zXH/99WlpacnIkSOzfPnyj32NtU571bt374wZMyZTp07N1KlTs//+++eMM87IH/7wh4/c31qnvXv66afzs5/9LA0NDZ+4n7VOe7bTTjvlkUceWf1z++23f+y+1vqGp2rfA86Gp7GxMY2Nja3e/6c//Wn69OmTcePGJUl22GGHPPPMM7nuuutyxBFHVGpMWC/XXnvtGo+bmppywAEHZPbs2dlnn30+8jXWOu3VkCFD1ng8evToTJkyJU899VR22mmnD+1vrdOeLVu2LGeffXYuvPDCXHnllZ+4r7VOe1ZfX58tt9yyVfta6xsen4Czzp566qkPnfLyuc99Ls8++2xWrlxZpamgPO+++26SpGfPnh+7j7VOLWhpacmMGTOyfPnyDBo06CP3sdZpz8aPH5/GxsYceOCBa93XWqc9+9Of/pSDDz44Q4YMyejRozNv3ryP3dda3/D4BJx19vbbb6dXr15rbNtiiy3ywQcfZPHixdlqq62qNBm0TqlUSlNTUwYPHpyBAwd+7H7WOu3ZCy+8kOHDh+f9999P165d86Mf/Sg77rjjR+5rrdNezZgxI7Nnz87UqVNbtb+1Tnu1++67Z+LEidl+++2zaNGiXHnllRk+fHjuuOOObLbZZh/a31rf8Ahw1ktdXd0aj0ul0kduhw3R+PHj8+KLL+bmm29e677WOu1V//79M3369CxZsiT33HNPvvOd7+TGG2/82Ai31mlvFixYkAkTJuS6667Lxhtv3OrXWeu0R//3ctE999wzhx9+eKZPn54RI0Z85Gus9Q2LAGed9erVKwsXLlxj2zvvvJMOHTpk0003rc5Q0EoXXHBBHnjggdx4443p3bv3J+5rrdOederUKdttt12SZLfddsszzzyTyZMnZ/z48R/a11qnPZo9e3YWLVqUYcOGrd7W0tKSWbNm5aabbsozzzyT+vr6NV5jrVMrunbtmoEDB+bVV1/9yOet9Q2PAGed7bnnnvnlL3+5xrZHHnkku+66azp27FilqeCTlUqlXHDBBbn33ntzww03pF+/fmt9jbVOLSmVSmlubv7I56x12qP999//Q3eBHjt2bAYMGJBTTjnlQ/GdWOvUjubm5rz00ksZPHjwRz5vrW943ISN1ZYtW5Y5c+Zkzpw5SZL58+dnzpw5ef3115MkF198cc4555zV+w8fPjyvv/56mpqa8tJLL+WWW27J1KlTc9JJJ1VlfmiN888/P7fddlsuvvjidOvWLQsXLszChQuzYsWK1ftY69SKSy65JI8//njmz5+fF154IZdeemlmzpyZo48+Oom1Tm3o3r17Bg4cuMZP165ds+mmm66+v4e1Tq2YOHFiZs6cmXnz5uX3v/99vvnNb2bp0qUZOnRoEmu9PfAJOKs9++yzOfHEE1c/bmpqSpIMHTo0F110URYuXJgFCxasfr5fv365+uqr09TUlJtuuilbbbVVxo0b5ysN2KBNmTIlSXLCCSessb2pqWn16YvWOrXi7bffzjnnnJO33norPXr0SENDQ6655prVd8S11vm0sNapFW+88Ua+/e1v589//nM222yz7Lnnnvn5z3+evn37JrHW24O60t+uwgcAAAAqxinoAAAAUAABDgAAAAUQ4AAAAFAAAQ4AAAAFEOAAAABQAAEOAAAABRDgAAAAUAABDgAAAAUQ4ADwKTd//vw0NDRkzpw51R4FAGpaXalUKlV7CABg7b773e9m2rRpSZL6+vpstdVWaWxszLe//e307Nmz1cdYsmRJJk2atHpbS0tL3nnnnWy22Wbp0KFDRWYHABL/lwWAduRzn/tcmpqa0tLSkj/+8Y8599xz8+677+aSSy5Z52PW19dnyy23bMMpAYCP4hR0AGhHOnXqlC233DK9e/fOwQcfnKOOOiq//vWvk/z1k+xzzz03Q4YMye67754jjjgi//Vf/7X6tZdffnmmTZuW+++/Pw0NDWloaMhjjz32oVPQH3vssTQ0NOTRRx/NsGHDsscee2T48OF5+eWX15hl0qRJOeCAAzJo0KCMGzcu//mf/5kvf/nLxf3HAIB2RoADQDs1b968PPzww6tPG1+1alV69+6dyy67LDNmzMgZZ5yRSy+9NHfeeWeS5KSTTsrf//3f53Of+1weeeSRPPLIIxk0aNDHHv/SSy/Nd7/73UydOjX19fU599xzVz9322235cc//nHGjBmTW2+9NX369MmUKVMq+wcDQDvnFHQAaEcefPDBDBo0KC0tLXn//feTJGPHjk2SdOzYMd/85jdX79uvX7/87ne/y1133ZWjjjoq3bp1S+fOndPc3NyqU85Hjx6dfffdN0ly6qmn5tRTT83777+fjTfeODfeeGP+8R//Mcccc0ySZNSoUfn1r3+d5cuXt/WfDAA1Q4ADQDuy33775bzzzst7772XW265Ja+88kqOP/741c9PmTIlv/jFL/L666/n/fffz8qVK7Pzzjuv03s1NDSs/v1vwb5o0aJsvfXWeeWVV3Lssceusf/uu++e3/72t+v0XgDwaeAUdABoR7p06ZLtttsuO++8c773ve+lubk5V1xxRZLkzjvvTFNTU4455phcd911mT59eoYNG5aVK1eu03v97zui19XVJfnrae4fxxerAMAnE+AA0I6NGjUq1113Xd5888088cQTGTRoUI477rh85jOfyXbbbZe5c+eusX/Hjh0/MaJbq3///nnmmWfW2Pbss8+u93EBoJYJcABox/bbb7/suOOOueqqq7Ltttvm2WefzcMPP5xXXnkll1122YciuW/fvnnhhRfy8ssv55133lnnT8ePP/743HLLLZk2bVpeffXVTJo0KS+88MLqT8oBgA8T4ADQzo0YMSI///nPc9hhh+ULX/hCRo8ena985Sv585///KHrtL/yla+kf//+OeaYY3LAAQfkySefXKf3/NKXvpRTTz01EydOzNChQzN//vwMHTo0G2+8cVv8SQBQk+pKLtgCANrAiBEj0qtXr/zHf/xHtUcBgA2Su6ADAGV777338tOf/jQHH3xwNtpoo8yYMSO/+c1vcv3111d7NADYYPkEHAAo24oVK3LaaaflueeeS3Nzc/r375/TTz89X/jCF6o9GgBssAQ4AAAAFMBN2AAAAKAAAhwAAAAKIMABAACgAAIcAAAACiDAAQAAoAACHAAAAArw/wEEw4Pfc+YokQAAAABJRU5ErkJggg=="/>


```python
print("Total data ")
print("-"*50)
print("\nTotal no of ratings :", df_dataset.shape[0] )# your code here
print("Total No of Users   :", len(np.unique(df_dataset.userId)))# your code here
print("Total No of products  :", len(np.unique(df_dataset.productId)))# your code here
```

<pre>
Total data 
--------------------------------------------------

Total no of ratings : 7824482
Total No of Users   : 4201696
Total No of products  : 476002
</pre>
## 관심 있는 데이터 세트만 선택합니다. 



때로는 추정을 위해 모든 데이터 세트가 필요하지는 않습니다. 데이터의 모든 속성이 우리가 구축하는 모델에 유용한 것은 아닙니다. 이 경우 해당 속성을 안전하게 삭제할 수 있습니다. 예를 들어, 사용자가 구매하고 싶어할 수 있는 제품을 추천하는 데 timestamp 열은 어떠한 도움도 주지 않기 때문에 여기에서는 삭제할 수 있습니다.



```python
# timestamp 열은 필요하지 않으므로 삭제합니다.
# your code here
df_modiset = df_dataset.drop('timestamp', axis='columns')
df_modiset.head()
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
      <th>userId</th>
      <th>productId</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AKM1MP6P0OYPR</td>
      <td>0132793040</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2CX7LUOHB2NDG</td>
      <td>0321732944</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2NWSAGRHCP8N5</td>
      <td>0439886341</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A2WNBOD3WNDNKT</td>
      <td>0439886341</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A1GI0U4ZRJA8WN</td>
      <td>0439886341</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 구매 고객이 부여한 평가를 분석해 보겠습니다.
# your code here
no_of_rated_products_per_user = df_modiset.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
no_of_rated_products_per_user.head()
```

<pre>
userId
A5JLAU2ARJ0BO     520
ADLVFFE4VBT8      501
A3OXHLG6DIBRW8    498
A6FIAB28IS79      431
A680RUE1FDO8B     406
Name: Rating, dtype: int64
</pre>
## 분위수 분포 보기



분위수는 샘플을 동일한 크기의 그룹으로 나누는 지점입니다. 정렬된 데이터 집합의 중앙값은 해당 집합의 중간 지점으로 정렬은 오름차순 또는 내림차순으로 정렬됨을 의미합니다. 따라서 중앙값은 데이터 세트를 2개의 동일한 그룹으로 나눌 때 분위수입니다.



![중앙값(분위수 예제)](https://www.statisticshowto.com/wp-content/uploads/2013/09/median.png)



```python
quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')
```


```python
plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# 차이가 0.05인 분위수를 찾습니다.
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")

# 차이가 0.25인 분위수도 구해 보겠습니다.
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()
```

<img src="data:image/png;base64,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+8/7Wpztgw8+MCQZq1evznebu6/7pEmTDD8/P2PDhg0u45yv0+eff24eq1OnjtG/f3/z59WrVxuSjI4dO14wzrPNnj3bkGR89NFH5rGff/7ZkGSMGTOmwPtc6Po695rds2fPeV87ScbYsWPNn7t27WrUqlXLSE9Pdxk3ePBgIzg42Hyem2++2WjZsqXb5wgApYHSPgDwIMMwJMmtsrlzbdq0ST169FC1atXk7++vgIAA9evXT3a7XTt27HAZGxUVpauuusrlWFxcnH7//feiB3+OZcuWqXLlyurevbtOnTplfrVs2VJRUVFKTk6WJLVs2VKBgYH65z//qTlz5mj37t1uP8fu3bvVp08fRUVFmeeckJAgSfrll19cxtpsNnXv3t3l2LnnvHr1alWsWFE9evRwGdenTx8rp35e7rzuy5YtU/PmzdWyZUuX161r166y2Wzm63Yht99+u9sxOcsYz55Veuedd2Sz2XTPPfeYx6xcX0Vx8uRJrVq1SrfeeqtCQ0Ndzr1bt246efKk1q9fL0m66qqr9N///leDBg3Sl19+qYyMjIt+fgC4WCRSAFACqlevrtDQUO3Zs+eC4/bu3auQkBBVq1bN0uPv27dPHTp00B9//KHp06drzZo12rBhg7m2JCsry2V8QY8fFBSUb9zF+PPPP/XXX38pMDBQAQEBLl9paWk6cuSIpLzSt5UrV6pmzZp66KGHdOmll+rSSy8tdL3LiRMn1KFDB3333Xd69tlnlZycrA0bNuijjz6SlP+cQ0NDFRwcnO+cT548af589OhRRUZG5nuuqKioIr0G53Lndf/zzz/1008/5XvNKlasKMMwzNftQqKjo92OKTQ0VHfddZe++OILpaWl6dSpU5o3b54SEhLMskSr11dRHD16VKdOndIrr7yS79y7desmSea5jx49Wi+++KLWr1+vG2+8UdWqVVOnTp30ww8/XHQcAFBUrJECgBLg7++v6667Tv/5z3904MCBAtcaHThwQBs3btQNN9xgHgsODs7XDEFSvjfTS5cu1d9//62PPvpIderUMY8X1Lq8tFSvXl3VqlXTF198UeDtFStWNP/coUMHdejQQXa7XT/88INeeeUVDR06VJGRkbrrrrsKvP9XX32lgwcPKjk52ZyFknRR62SqVaum77//Pt9xd5pNFJfq1asrJCTkvOuOqlevXuhjWJ3RHDhwoGbOnKl3331XjRo10qFDh/TSSy+Zt1/M9eVMXs+9jo8ePeryc5UqVeTv76+7775bDz30UIGPVa9ePUl5zVkeffRRPfroo/rrr7+0cuVKPfHEE+ratav2799/wXV1AFBSSKQAoISMGjVKn3/+uQYNGqQlS5bI39/fvM1ut+tf//qX7Ha7HnnkEfN43bp1dejQIf3555/mTElOTo6+/PJLl8d2vnEOCgoyjxmGoZkzZxY5XudjFXW24eabb9bChQtlt9vVtm1bt+7j7++vtm3bqkmTJpo/f75+/PHH8yZSBZ2zJL355ptFilfK67b3/vvv65NPPnEp71uwYIFb97/Y10zKe90mTpyoatWqmYlDSWvbtq2aN2+u2bNnq1GjRoqIiHApD7yY6ysyMlLBwcH66aefXI5//PHHLj+Hhobq2muv1aZNmxQXF6fAwEC3Yq9cubLuuOMO/fHHHxo6dKj27t2rpk2bunVfAChOJFIAUEKuvvpqTZs2TY888oiuueYaDR48WLVr19a+ffv06quv6ttvv9W4cePUpUsX8z69e/fWU089pbvuukuPPfaYTp48qZdffll2u93lsbt06aLAwED93//9n0aOHKmTJ0/q9ddf17Fjx4oc76WXXqqQkBDNnz9fl112mcLDwxUTE6OYmBi37n/XXXdp/vz56tatmx555BFdddVVCggI0IEDB7R69Wr17NlTt956q9544w199dVXuummm1S7dm2dPHnSnI3p3LnzeR8/Pj5eVapU0YMPPqixY8cqICBA8+fP13//+98in3O/fv00depU9evXTxMmTFDDhg31+eef50tcz8e5F9Nbb72lihUrKjg4WPXq1bNUqjl06FAtXrxYHTt21LBhwxQXFyeHw6F9+/Zp+fLlGj58uNuJqRX33nuvHn30UW3fvl0PPPCAQkJCzNsu5vqy2Wz6xz/+oXfeeUeXXnqpLr/8cn3//fcFJqfTp0/XNddcow4dOuhf//qX6tatq+PHj+u3337Tp59+anZi7N69u5o3b642bdqoRo0a+v333zVt2jTVqVNHDRs2LL4XBQCs8GyvCwDwfevWrTNuv/12IzIy0vDz8zMkGcHBwcZnn31W4PjPP//caNmypRESEmLUr1/fmDFjRoFd8z799FPj8ssvN4KDg41LLrnEeOyxx4z//Oc/+brIJSQkGM2aNcv3PAV1CHzvvfeMJk2aGAEBAS4d1tzp2mcYhpGbm2u8+OKLZlzh4eFGkyZNjAceeMDYuXOnYRiG8e233xq33nqrUadOHSMoKMioVq2akZCQYHzyySduvZbt27c3QkNDjRo1ahj33Xef8eOPP+brEte/f38jLCws3/0LOo8DBw4Yt99+uxEeHm5UrFjRuP32241169a51bXPMAxj2rRpRr169Qx/f3+X+1h53U+cOGH8+9//Nho3bmwEBgYaERERRosWLYxhw4YZaWlp5rjzde374IMPCo3zXIcPHzYCAwMNScb333+f73Z3r6+Czic9Pd247777jMjISCMsLMzo3r27sXfv3nxd+wwjr8vfvffea1xyySVGQECAUaNGDSM+Pt549tlnzTEvvfSSER8fb1SvXt0IDAw0ateubQwcONDYu3ev5fMGgOJiM4zTLaMAAKXi3XffVf/+/TVy5Eg9//zzng4HAAAUAaV9AFDK+vXrp9TUVI0aNUphYWHm5rAAAMB7MCMFAAAAABaxjxQAAAAAWEQiBQAAAAAWkUgBAAAAgEUkUgAAAABgEV37JDkcDh08eFAVK1Y0d3MHAAAAUP4YhqHjx48rJiZGfn7nn3cikZJ08OBBxcbGejoMAAAAAGXE/v37VatWrfPeTiIlqWLFipLyXqxKlSp5OBoAAAAAnpKRkaHY2FgzRzgfEinJLOerVKkSiRQAAACAQpf80GwCAAAAACwikQIAAAAAi0ikAAAAAMAi1ki5yTAMnTp1Sna73dOhACgG/v7+qlChAlseAACAIiGRckNOTo5SU1OVmZnp6VAAFKPQ0FBFR0crMDDQ06EAAAAvQyJVCIfDoT179sjf318xMTEKDAzkE2zAyxmGoZycHB0+fFh79uxRw4YNL7jhHgAAwLlIpAqRk5Mjh8Oh2NhYhYaGejocAMUkJCREAQEB+v3335WTk6Pg4GBPhwQAALwIH8G6iU+rAd/D32sAAFBUvIsAAAAAAItIpAAAAADAIhIpWLZ3717ZbDZt3rzZ06EAAAAAHkEi5cMGDBggm80mm82mChUqqHbt2vrXv/6lY8eOWXqMW265xeVYbGysUlNT1bx582KOGAAAAPAOJFI+7oYbblBqaqr27t2rt99+W59++qkGDRp0UY/p7++vqKgoVahA00cAAACUTyRSFhmGocycUx75MgzDcrxBQUGKiopSrVq1dP3116t3795avny5JMlut2vgwIGqV6+eQkJC1LhxY02fPt2877hx4zRnzhx9/PHH5sxWcnJyvtK+5ORk2Ww2rVq1Sm3atFFoaKji4+O1fft2l1ieffZZ1axZUxUrVtR9992nUaNGqWXLlkX+XQAAAACewpSCRVm5djV96kuPPPe2p7sqNLDov7Ldu3friy++UEBAgKS8zYZr1aql999/X9WrV9e6dev0z3/+U9HR0erVq5dGjBihX375RRkZGZo9e7YkqWrVqjp48GCBjz9mzBi99NJLqlGjhh588EHde++9+uabbyRJ8+fP14QJE/Taa6/p6quv1sKFC/XSSy+pXr16RT4fAAAAwFNIpHzcsmXLFB4eLrvdrpMnT0qSpkyZIkkKCAjQ+PHjzbH16tXTunXr9P7776tXr14KDw9XSEiIsrOzFRUVVehzTZgwQQkJCZKkUaNG6aabbtLJkycVHBysV155RQMHDtQ999wjSXrqqae0fPlynThxorhPGQAAAChxJFIWhQT4a9vTXT323FZde+21ev3115WZmam3335bO3bs0MMPP2ze/sYbb+jtt9/W77//rqysLOXk5BS53C4uLs78c3R0tCTp0KFDql27trZv355vbdZVV12lr776qkjPBQAAAHgSiZRFNpvtosrrSltYWJgaNGggSXr55Zd17bXXavz48XrmmWf0/vvva9iwYXrppZfUvn17VaxYUS+88IK+++67Ij2Xs2RQynudpLzywXOPORVlzRcAAABQFtBsopwZO3asXnzxRR08eFBr1qxRfHy8Bg0apCuuuEINGjTQrl27XMYHBgbKbrdf9PM2btxY33//vcuxH3744aIfFwAAAPAEEqlyJjExUc2aNdPEiRPVoEED/fDDD/ryyy+1Y8cOPfnkk9qwYYPL+Lp16+qnn37S9u3bdeTIEeXm5hbpeR9++GHNmjVLc+bM0c6dO/Xss8/qp59+yjdLBQAAAHgDEqly6NFHH9XMmTN1yy236LbbblPv3r3Vtm1bHT16NN86pvvvv1+NGzdWmzZtVKNGDbMLn1V9+/bV6NGjNWLECLVq1Up79uzRgAEDFBwcXBynBAAAAJQqm8FCFWVkZCgiIkLp6emqVKmSy20nT57Unj17VK9ePd70F7MuXbooKipKc+fO9XQoKKf4+w0AAM51odzgbN7TNQFeLTMzU2+88Ya6du0qf39/vffee1q5cqVWrFjh6dAAAAAAy0ikUCpsNps+//xzPfvss8rOzlbjxo21ePFide7c2dOhAQAAAJaRSKFUhISEaOXKlZ4OAwAAACgWNJsAAAAAAItIpAAAAADAIkr7AAAAAHiMkXtKf326VjkHjimwVhVV7n6NbAFlP00p+xECAAAA8EmH3/pUvz2Ro+yj1SRVkSQFVftYDSYGqsY/u3s2uEJQ2gcAAACg1B1+61NtfSBc2UeruhzPPlpFWx8I1+G3PvVQZO4hkQIAAABQqozcU/rtiRxJ0pFKhjY0OqWdl9hP3+onydBvY3Jk5J7yWIyFIZGCRyQnJ8tms+mvv/6SJCUlJaly5coejclp7969stls2rx58wXHJSYmaujQoaUSU0k493fgrWw2m5YuXerpMAAAgAV/fbr2dDmfTdtj7Xr11mx9HJ971gg/ZR+ppr8+XeupEAtFIoUSV1DCER8fr9TUVEVERHgmqAuIjY1VamqqmjdvLqn4E47FixeradOmCgoKUtOmTbVkyZJC77NlyxYlJCQoJCREl1xyiZ5++mkZhmHe7ozx3K9ff/31vI9ZlN/BgAEDdMstt7g9HgAAoCA5B46Zf3aczkj8jAuPK2toNlFaHHbp8BopK1UKiZZqdJD8/D0dlccEBgYqKirK02EUyN/fv8Ri+/bbb9W7d28988wzuvXWW7VkyRL16tVLa9euVdu2bQu8T0ZGhrp06aJrr71WGzZs0I4dOzRgwACFhYVp+PDhLmO3b9+uSpUqmT/XqFHjvLF48neQk5OjwMBAjzw3AADwvMBaVcw/O2x53/0cFx5X1jAjVRr2fyR9Uldada20rk/e90/q5h0vIX///bf69eun8PBwRUdH66WXXso3M1RQSVTlypWVlJRk/vz444+rUaNGCg0NVf369fXkk08qN/fMtOu4cePUsmVLzZ07V3Xr1lVERITuuusuHT9+XFLeDEZKSoqmT59uzpLs3bvXrVmeTz/9VK1bt1ZwcLDq16+v8ePH69SpM3Wy48aNU+3atRUUFKSYmBgNGTKkwMdJT0+Xv7+/Nm7cKEkyDENVq1bVlVdeaY557733FB0dLcm1tG/v3r269tprJUlVqlSRzWbTgAEDzPs5HA6NHDlSVatWVVRUlMaNG3fe85GkadOmqUuXLho9erSaNGmi0aNHq1OnTpo2bdp57zN//nydPHlSSUlJat68uW677TY98cQTmjJlisuslCTVrFlTUVFR5pe///mT9fOVV3755Ze67LLLFB4erhtuuEGpqamS8l7vOXPm6OOPPzZ/l8nJyZKkP/74Q71791aVKlVUrVo19ezZU3v37jWfyzmTNWnSJMXExKhRo0YaPXq02rVrly+uuLg4jR07VpK0YcMGdenSRdWrV1dERIQSEhL0448/nveccnJyNHjwYEVHRys4OFh169bVpEmTzjseAAB4RuXu1yio2lFJDhmnEymby9sah4KqH1Xl7td4IDr3kEiVtP0fSWvukDIPuB7P/CPveAklU4899phWr16tJUuWaPny5UpOTjYTCSsqVqyopKQkbdu2TdOnT9fMmTM1depUlzG7du3S0qVLtWzZMi1btkwpKSl67rnnJEnTp09X+/btdf/99ys1NVWpqamKjY0t9Hm//PJL/eMf/9CQIUO0bds2vfnmm0pKStKECRMkSR9++KGmTp2qN998Uzt37tTSpUvVokWLAh8rIiJCLVu2NN/0//TTT+b3jIwMSXlJRUJCQr77xsbGavHixZLyZntSU1M1ffp08/Y5c+YoLCxM3333nSZPnqynn35aK1asOO95ffvtt7r++utdjnXt2lXr1q274H0SEhIUFBTkcp+DBw+6JCuSdMUVVyg6OlqdOnXS6tWrz/uY55OZmakXX3xRc+fO1ddff619+/ZpxIgRkqQRI0aoV69eZnKVmpqq+Ph4ZWZm6tprr1V4eLi+/vprrV271kzCcnJyzMdetWqVfvnlF61YsULLli1T37599d1332nXrl3mmK1bt2rLli3q27evJOn48ePq37+/1qxZo/Xr16thw4bq1q2bmaif6+WXX9Ynn3yi999/X9u3b9e8efNUt25dy68DAAAoWbaACmowMVCSTY7TGdSZ0j6HJJsaTAgs0/tJld3IfIHDLm18RFIBBZ8yJNmkjUOlS3oWa5nfiRMnNGvWLL377rvq0qWLpLw3/LVq1bL8WP/+97/NP9etW1fDhw/XokWLNHLkSPO4w+FQUlKSKlasKEm6++67tWrVKk2YMEEREREKDAxUaGiopTKyCRMmaNSoUerfv78kqX79+nrmmWc0cuRIjR07Vvv27VNUVJQ6d+6sgIAA1a5dW1ddddV5Hy8xMVHJyckaPny4kpOT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pKZOnaqHH35Y7du3lyR9++23GjZsmPbu3atnn3222IMEAAAA4DscpzMp/4vqIe5ZlhOp119/XTNnztT//d//mcd69OihuLg4PfzwwyRSAAAAAC7IXh7bn9vtdrVp0ybf8datW+vUqVPFEhQAAAAA3+ULpX2WE6l//OMfev311/Mdf+utt9S3b99iCQoAAACA7zpT2ue9iZRbpX2PPvqo+Webzaa3335by5cvV7t27SRJ69ev1/79+9WvX7+SiRIAAACAzyg37c83bdrk8nPr1q0lSbt27ZKUt29UjRo1tHXr1mIODwAAAICvOdP+3MOBXAS3EqnVq1eXdBwAAAAAygnnjJR/eVojBQAAAAAXw9lswkYiBQAAAADusftAswkSKQAAAAClyizt8+JsxItDBwAAAOCNnIkUpX0AAAAA4Ca7I+97uWo2MWfOHH322WfmzyNHjlTlypUVHx+v33//vViDAwAAAOB7DKMcrpGaOHGiQkJCJEnffvutZsyYocmTJ6t69eoaNmxYsQcIAAAAwLc4m0148YSUe/tInW3//v1q0KCBJGnp0qW644479M9//lNXX321EhMTizs+AAAAAD7GXh73kQoPD9fRo0clScuXL1fnzp0lScHBwcrKyire6AAAAAD4nNN5lFeX9lmekerSpYvuu+8+XXHFFdqxY4duuukmSdLWrVtVt27d4o4PAAAAgI85U9rnvYmU5RmpV199Ve3bt9fhw4e1ePFiVatWTZK0ceNG/d///V+xBwgAAADAt/jChryWZ6QqV66sGTNm5Ds+fvz4YgkIAAAAgG9zdu3z4jzKeiL1008/FXjcZrMpODhYtWvXVlBQ0EUHBgAAAMA32c1EynszKcuJVMuWLS9YyxgQEKDevXvrzTffVHBw8EUFBwAAAMD3mBvyevGUlOU1UkuWLFHDhg311ltvafPmzdq0aZPeeustNW7cWAsWLNCsWbP01Vdf6d///ndJxAsAAADAyxnlcUZqwoQJmj59urp27Woei4uLU61atfTkk0/q+++/V1hYmIYPH64XX3yxWIMFAAAA4P3MfaQsT+uUHZZD37Jli+rUqZPveJ06dbRlyxZJeeV/qampFx8dAAAAAJ/jcHj/jJTlRKpJkyZ67rnnlJOTYx7Lzc3Vc889pyZNmkiS/vjjD0VGRhZflAAAAAB8xuk8yqsTKculfa+++qp69OihWrVqKS4uTjabTT/99JPsdruWLVsmSdq9e7cGDRpU7MECAAAA8H7lch+p+Ph47d27V/PmzdOOHTtkGIbuuOMO9enTRxUrVpQk3X333cUeKAAAAADf4Di9RsqLJ6SsJ1KSFB4ergcffLC4YwEAAABQDjiMcjgjJUk7duxQcnKyDh06JIfD4XLbU089VSyBAQAAAPBNZmmfF09JWU6kZs6cqX/961+qXr26oqKiXDbntdlsJFIAAAAALsjZbMJWnhKpZ599VhMmTNDjjz9eEvEAAAAA8GHO1ueSd5f2WW5/fuzYMd15550lEQsAAAAAH+dcHyV5d2mf5UTqzjvv1PLly0siFgAAAAA+zn5WImWznI2UHZZL+xo0aKAnn3xS69evV4sWLRQQEOBy+5AhQ4otOAAAAAC+5exedd48I2U5kXrrrbcUHh6ulJQUpaSkuNxms9lIpAAAAACcl0tpnxevkbKcSO3Zs6ck4gAAAABQDriU9nlvHmV9jRQAAAAAFJVL1z4vzqTcmpF69NFH9cwzzygsLEyPPvroBcdOmTKlWAIDAAAA4HvOyqPk5+uJ1KZNm5Sbm2v+GQAAAACKwn5WJuXn62ukVq9eXeCfAQAAAMAKZ7MJb240IRVhjdS9996r48eP5zv+999/69577y2WoAAAAAD4Jmci5eV5lPVEas6cOcrKysp3PCsrS++++26xBAUAAADANzlL+7x5fZRkof15RkaGDMOQYRg6fvy4goODzdvsdrs+//xz1axZs0SCBAAAAOAbnN3Pvb20z+1EqnLlyrLZbLLZbGrUqFG+2202m8aPH1+swQEAAADwLeVuRmr16tUyDEPXXXedFi9erKpVq5q3BQYGqk6dOoqJiSmRIAEAAAD4BruPrJFyO5FKSEiQJO3Zs0exsbHy82MvXwAAAADWGM5EysszKbcTKac6depIkjIzM7Vv3z7l5OS43B4XF1c8kQEAAADwOXZH3nf/8lLa53T48GHdc889+s9//lPg7Xa7/aKDAgAAAOCbzDVSXj4jZbk+b+jQoTp27JjWr1+vkJAQffHFF5ozZ44aNmyoTz75pCRiBAAAAOAjfGUfKcszUl999ZU+/vhjXXnllfLz81OdOnXUpUsXVapUSZMmTdJNN91UEnECAAAA8AHORMrbS/ssz0j9/fff5n5RVatW1eHDhyVJLVq00I8//li80QEAAADwKeW2tK9x48bavn27JKlly5Z688039ccff+iNN95QdHR0sQcIAAAAwHeczqPKzz5STkOHDlVqaqokaezYseratavmz5+vwMBAJSUlFXd8AAAAAHyIWdrn5TNSlhOpvn37mn++4oortHfvXv3666+qXbu2qlevXqzBAQAAAPAtDodvNJuwVNqXm5ur+vXra9u2beax0NBQtWrViiQKAAAAQKHsZtc+786kLCVSAQEBys7Ols3LTxoAAACAZzicG/J6+ZSU5WYTDz/8sJ5//nmdOnWqJOIBAAAA4MOca6S8fXLG8hqp7777TqtWrdLy5cvVokULhYWFudz+0UcfFVtwAAAAAHyL3Ww24eFALpLlRKpy5cq6/fbbSyIWAAAAAD7O2WzC2zfktZxIzZ49uyTiAAAAAFAOOPeR8vbSPi+fUAMAAADgTewO39hHikQKAAAAQKkxDN8o7SORAgAAAFBq7GbXPg8HcpFIpAAAAACUmnJV2le1alUdOXJEknTvvffq+PHjJRoUAAAAAN90ekJKfl4+JeVWIpWTk6OMjAxJ0pw5c3Ty5MkSDQoAAACAb3LOSPl5+YyUW+3P27dvr1tuuUWtW7eWYRgaMmSIQkJCChz7zjvvFGuAAAAAAHyHuSGvd+dR7iVS8+bN09SpU7Vr1y7ZbDalp6czKwUAAADAMmfXPm8v7XMrkYqMjNRzzz0nSapXr57mzp2ratWqlWhgAAAAAHyP3ZH3vVyU9p1tz549JREHAAAAgHLAXp73kUpJSVH37t3VoEEDNWzYUD169NCaNWuKOzYAAAAAPsYs7fPyjZgshz9v3jx17txZoaGhGjJkiAYPHqyQkBB16tRJCxYsKIkYAQAAAPgIs2ufl89IWS7tmzBhgiZPnqxhw4aZxx555BFNmTJFzzzzjPr06VOsAQIAAADwHafzqPKxIe/Zdu/ere7du+c73qNHD9ZPAQAAALggh4/MSFlOpGJjY7Vq1ap8x1etWqXY2NhiCQoAAACAb7KXp/bnZxs+fLiGDBmizZs3Kz4+XjabTWvXrlVSUpKmT59eEjECAAAA8BEOM5HycCAXyXIi9a9//UtRUVF66aWX9P7770uSLrvsMi1atEg9e/Ys9gABAAAA+A5naZ+3r5GynEhJ0q233qpbb721uGMBAAAA4ON8ZUNeL+/eDgAAAMCb+EppH4kUAAAAgFLjTKT8vbzZBIkUAAAAgFJjzkh5+ZQUiRQAAACAUmOukSrPM1KGYcg4nVECAAAAQGHM0r7yOCP17rvvqkWLFgoJCVFISIji4uI0d+7c4o4NAAAAgI9xtj/38gkp6+3Pp0yZoieffFKDBw/W1VdfLcMw9M033+jBBx/UkSNHNGzYsJKIEwAAAIAPsPtIswnLidQrr7yi119/Xf369TOP9ezZU82aNdO4ceNIpAAAAACcl69syGu5tC81NVXx8fH5jsfHxys1NbVYggIAAADgm07nUbJ5+YyU5USqQYMGev/99/MdX7RokRo2bFgsQQEAAADwTeW2tG/8+PHq3bu3vv76a1199dWy2Wxau3atVq1aVWCCBQAAAABOZ0r7PBzIRbIc/u23367vvvtO1atX19KlS/XRRx+pevXq+v7773XrrbeWRIwAAAAAfISz/bm3l/ZZnpGSpNatW2vevHnFHQsAAAAAH+fckLfcNZsAAAAAgKIyytsaKT8/v0Kn32w2m06dOnXRQQEAAADwTXajnG3Iu2TJkvPetm7dOr3yyitmdgkAAAAABbH7yD5SbidSPXv2zHfs119/1ejRo/Xpp5+qb9++euaZZ4o1OAAAAAC+xTn34uflU1JFWiN18OBB3X///YqLi9OpU6e0adMmzZkzR7Vr1y7u+AAAAAD4EOeMlJ+Xz0hZSqTS09P1+OOPq0GDBtq6datWrVqlTz/9VC1atCip+AAAAAD4kHK3Ie/kyZP1/PPPKyoqSu+9916BpX4AAAAAcCHOvgpePiHlfiI1atQohYSEqEGDBpozZ47mzJlT4LiPPvqo2IIDAAAA4Ft8pbTP7USqX79+Xr/7MAAAAADPOp1HlZ/SvqSkpBIMAwAAAEB54HCW9hWp7V3Z4eXhAwAAAPAmZmmfl89IkUgBAAAAKDXmjBSJFAAAAAC4x+HI++7v5c0mPJpITZo0SVdeeaUqVqyomjVr6pZbbtH27dtdxhiGoXHjxikmJkYhISFKTEzU1q1bXcZkZ2fr4YcfVvXq1RUWFqYePXrowIEDpXkqAAAAANxgZ0bq4qWkpOihhx7S+vXrtWLFCp06dUrXX3+9/v77b3PM5MmTNWXKFM2YMUMbNmxQVFSUunTpouPHj5tjhg4dqiVLlmjhwoVau3atTpw4oZtvvll2u90TpwUAAADgPBw+so+UzXDuiFUGHD58WDVr1lRKSoo6duwowzAUExOjoUOH6vHHH5eUN/sUGRmp559/Xg888IDS09NVo0YNzZ07V71795YkHTx4ULGxsfr888/VtWvXQp83IyNDERERSk9PV6VKlUr0HAEAAIDyrOeMtfrvgXTN6t9GnS6L9HQ4+bibG5SpNVLp6emSpKpVq0qS9uzZo7S0NF1//fXmmKCgICUkJGjdunWSpI0bNyo3N9dlTExMjJo3b26OOVd2drYyMjJcvgAAAACUPLO0z8unpMpMImUYhh599FFdc801at68uSQpLS1NkhQZ6ZqpRkZGmrelpaUpMDBQVapUOe+Yc02aNEkRERHmV2xsbHGfDgAAAIACOJtNsEaqmAwePFg//fST3nvvvXy32c55kQ3DyHfsXBcaM3r0aKWnp5tf+/fvL3rgAAAAANzmXCPlTyJ18R5++GF98sknWr16tWrVqmUej4qKkqR8M0uHDh0yZ6mioqKUk5OjY8eOnXfMuYKCglSpUiWXLwAAAAAlz2w2USYykaLzaPiGYWjw4MH66KOP9NVXX6levXout9erV09RUVFasWKFeSwnJ0cpKSmKj4+XJLVu3VoBAQEuY1JTU/Xzzz+bYwAAAACUDXaHb7Q/r+DJJ3/ooYe0YMECffzxx6pYsaI58xQREaGQkBDZbDYNHTpUEydOVMOGDdWwYUNNnDhRoaGh6tOnjzl24MCBGj58uKpVq6aqVatqxIgRatGihTp37uzJ0wMAAABwjtN5lNdvyOvRROr111+XJCUmJrocnz17tgYMGCBJGjlypLKysjRo0CAdO3ZMbdu21fLly1WxYkVz/NSpU1WhQgX16tVLWVlZ6tSpk5KSkuTv719apwIAAADADewj5UPYRwoAAAAoHdc8/5UOHMvSkkHxuqJ2lcLvUMq8ch8pAAAAAL7Ncbq2z9tL+0ikAAAAAJQa5xopb282QSIFAAAAoNTYDd/o2kciBQAAAKDUUNoHAAAAABb5Stc+EikAAAAApcbckNfLMykSKQAAAAClxqDZBAAAAABY42w24U8iBQAAAADuOVPa5+FALpKXhw8AAADAm1DaBwAAAAAWmaV9NJsAAAAAAPeYpX3MSAEAAABA4QxnXZ/YRwoAAAAA3OKcjZIo7QMAAAAAt5yVR7EhLwAAAAC4w+FS2kciBQAAAACFcintI5ECAAAAgMKdPSPl5XkUiRQAAACA0uFwnPkzzSYAAAAAwA12g9I+AAAAALCE0j4AAAAAsMhxutmEn02yeXkmRSIFAAAAoFQ4S/u8fX2URCIFAAAAoJQ4u597+2yURCIFAAAAoJQ4S/u8vdGERCIFAAAAoJQ4m034QGUfiRQAAACA0mF3NpvwgUyKRAoAAABAqXDQbAIAAAAArHE2m/BjjRQAAAAAuMcs7SORAgAAAAD3OBMpfx/IQnzgFAAAAAB4A4PSPgAAAACwxm5Q2gcAAAAAltC1DwAAAAAscjjYkBcAAAAALGFDXgAAAACwiH2kAAAAAMAic40UiRQAAAAAuIfSPgAAAACwyGHQbAIAAAAALKH9OQAAAABYZHfkfafZBAAAAAC4idI+AAAAALDIuSEvpX0AAAAA4CbnPlI2SvsAAAAAwD129pECAAAAAGso7QMAAAAAi5zNJnxgQopECgAAAEDpsDMjBQAAAADWOFgjBQAAAADW0LUPAAAAACw6U9rn4UCKgQ+cAgAAAABvYBiskQIAAAAAS5wzUpT2AQAAAICb7KfXSNFsAgAAAADc5Czt84HKPhIpAAAAAKXDWdrn5wOZFIkUAAAAgFJhZx8pAAAAALDmdB4lPxIpAAAAAHAPpX0AAAAAYBEb8gIAAACARWe69jEjBQAAAABusZNIAQAAAIA1DueGvKyRAgAAAAD3OBxsyAsAAAAAltC1DwAAAAAscrCPFAAAAABY4zjdbMKfRAoAAAAA3ENpHwAAAABY5DBoNgEAAAAAllDaBwAAAAAWUdoHAAAAABbRtQ8AAAAALHJuyOvvA1mID5wCAAAAAG9wptkEM1IAAAAA4BY7pX0AAAAAYM2Z0j4SKQAAAABwC/tIAQAAAIBFtD8HAAAAAIvYkBcAAAAALGIfKQAAAACwiNI+AAAAALDILO3zgSzEB04BAAAAgDdgQ14AAAAAsMgs7SORAgAAAAD30GwCAAAAACxyOFgjBQAAAACW2FkjBQAAAADWUNoHAAAAABadKe0jkQIAAAAAt7AhLwAAAABYdGYfKQ8HUgxIpAAAAACUCmci5c8aKQAAAABwj7PZhI1ECgAAAADcQ7MJAAAAALDIuY8UG/ICAAAAgJuca6Qo7QMAAAAANzkced9pNgEAAAAAbrKzRgoAAAAArDlT2ufhQIoBiRQAAACAUmHuI8WMFAAAAAC4x7mPFGukAAAAAMBNzjVSdO0DAAAAADexIS8AAAAAWORcI+UDeRSJFAAAAIDSYTcTKe/PpEikAAAAAJQKc0NeH5iSIpECAAAAUCoczEgBAAAAgDVmaZ8PZCE+cAoAAAAAyjrDMGSwjxQAAAAAuM+5Ga9EaR8AAAAAuMV+ViblR7MJAAAAACics9GExD5SAAAAAOCWsxMp2p8DAAAAgBtcSvtYIwUAAAAAhaPZBAAAAABY5HBQ2gcAAAAAlthpNgEAAAAA1jibTdhsko3SPgAAAAAonMOR993fB5IoiUQKAAAAQClwzkj5wma8EokUAAAAgFLgbH/uI3kUiRQAAACAkueckaK0DwAAAADc5Ox+7gt7SEkeTqS+/vprde/eXTExMbLZbFq6dKnL7YZhaNy4cYqJiVFISIgSExO1detWlzHZ2dl6+OGHVb16dYWFhalHjx46cOBAKZ4FAAAAgMKYpX0+Utvn0UTq77//1uWXX64ZM2YUePvkyZM1ZcoUzZgxQxs2bFBUVJS6dOmi48ePm2OGDh2qJUuWaOHChVq7dq1OnDihm2++WXa7vbROAwAAAEAhzNI+H0mkKnjyyW+88UbdeOONBd5mGIamTZumMWPG6LbbbpMkzZkzR5GRkVqwYIEeeOABpaena9asWZo7d646d+4sSZo3b55iY2O1cuVKde3atdTOBQAAAMD5mV37fCOPKrtrpPbs2aO0tDRdf/315rGgoCAlJCRo3bp1kqSNGzcqNzfXZUxMTIyaN29ujilIdna2MjIyXL4AAAAAlJwzXft8I5Mqs4lUWlqaJCkyMtLleGRkpHlbWlqaAgMDVaVKlfOOKcikSZMUERFhfsXGxhZz9AAAAADOZm7I6yNTUmU2kXKynZOxGoaR79i5ChszevRopaenm1/79+8vllgBAAAAFOxMaR+JVImKioqSpHwzS4cOHTJnqaKiopSTk6Njx46dd0xBgoKCVKlSJZcvAAAAACXH7kykymwGYk2ZPY169eopKipKK1asMI/l5OQoJSVF8fHxkqTWrVsrICDAZUxqaqp+/vlncwwAAAAAzzN8bEbKo137Tpw4od9++838ec+ePdq8ebOqVq2q2rVra+jQoZo4caIaNmyohg0bauLEiQoNDVWfPn0kSRERERo4cKCGDx+uatWqqWrVqhoxYoRatGhhdvEDAAAA4Hl25xopEqmL98MPP+jaa681f3700UclSf3791dSUpJGjhyprKwsDRo0SMeOHVPbtm21fPlyVaxY0bzP1KlTVaFCBfXq1UtZWVnq1KmTkpKS5O/vX+rnAwAAAKBgvrYhr81wzrGVYxkZGYqIiFB6ejrrpQAAAIASsO63I+rz9ndqFBmu5cMSPB3OebmbG5TZNVIAAAAAfIfdx9ZIkUgBAAAAKHHO0j72kQIAAAAANzkXFDEjBQAAAABu8rVmEyRSAAAAAEqcc42Uv2/kUSRSAAAAAEqer23ISyIFAAAAoMQ5N+SltA8AAAAA3OQwZ6Q8HEgxIZECAAAAUOKciRTtzwEAAADATWbXPtZIAQAAAIB7HOwjBQAAAADWOByU9gEAAACAJXbanwMAAACANXTtAwAAAACLKO0DAAAAAItoNgEAAAAAFpntz5mRAgAAAAD3mBvy+kYeRSIFAAAAoOQ56NoHAAAAANbYHXnfKe0DAAAAADedKe0jkQIAAAAAtzjMZhMeDqSY+MhpAAAAACjL7KyRAgAAAABr2JAXAAAAACxiQ14AAAAAsIjSPgAAAACw6Mw+Uh4OpJiQSAEAAAAocayRAgAAAACL2JAXAAAAACyitA8AAAAALHImUv40mwAAAAAA99hPr5GitA8AAAAA3MQ+UgAAAABgEV37AAAAAMAi5xopH5mQIpECAAAAUPLsNJsAAAAAAGso7QMAAAAAi5zNJmzMSAEAAACAe86U9nk4kGJCIgUAAACgxFHaBwAAAAAWnenaRyIFAAAAAG6xO/K+MyMFAAAAAG5y0P4cAAAAAKxhQ14AAAAAsMhOswkAAAAAsOb0hJT8fGRKikQKAAAAQIlzzkj5MSMFAAAAAO6x02wCAAAAAKwxTidSPjIhRSIFAAAAoORR2gcAAAAAFtlPN5ugtA8AAAAA3GSW9vlIBuIjpwEAAACgLDNL+5iRAgAAAAD3ONhHCgAAAACscZzOpPxpNgEAAAAA7rEblPYBAAAAgCUO9pECAAAAAGso7QMAAAAAi8zSPhIpAAAAAHCPw5H3nTVSAAAAAOAm5xopfxIpAAAAAHCPuSGvj2QgPnIaAAAAAMoyNuQFAAAAAIvM0j6aTQAAAACAe9hHCgAAAAAsMtdIUdoHAAAAAO5hQ14AAAAAsIhmEwAAAABgkd25RooZKQAAAABwj1nax4wUAAAAALiHrn0AAAAAYIFhGGfWSPlIJkUiBQAAAKBEOZMoiWYTAAAAAOAWZ1mfxBopAAAAAHCL/awpKT8fyUB85DQAAAAAlFUGpX0AAAAAYI397NI+mk0AAAAAQOFcSvuYkQIAAACAwhnG2YmUBwMpRiRSAAAAAErU2TNSlPYBAAAAgBuca6RsNslGaR8AAAAAFM5Z2ecr66MkEikAAAAAJcxZ2ucrm/FKJFIAAAAASpjjrNI+X0EiBQAAAKBEORx5332l0YREIgUAAACghDmbTVDaBwAAAABuorQPAAAAACxyOJtNUNoHAAAAAO4xS/tIpAAAAADAPc5mE76yGa9EIgUAAACghDloNgEAAAAA1jg35PWhyj4SKQAAAAAlyzkj5edDmRSJFAAAAIAS5aDZBAAAAABYc7qyT36skQIAAAAA97BGCgAAAAAsYkNeAAAAALCI0j4AAAAAsMju7NpHIgUAAAAA7qG0DwAAAAAsMveR8p08ikQKAAAAQMkyu/b5UCZFIgUAAACgRNFsAgAAAAAscpb2+ZNIAQAAAIB7zpT2eTiQYuRDpwIAAACgLHLQ/hwAAAAArDFL+2g2AQAAAADusTvyvjMjBQAAAABuYh8pAAAAALDI4aC0DwAAAAAssdNsAgAAAADcl3PKoe92/0+SbyVSFTwdAAAAAADfdPCvLD204Edt2veXJOmmuGjPBlSMSKQAAAAAFLuUHYc1dOEmHcvMVaXgCprSq6U6N430dFjFhkQKAAAAQLGxOwy9vGqnXv5qpwxDan5JJb3Wp7VqVwv1dGjFikQKAAAAQLE4eiJbQxdt1pqdRyRJfdrW1lM3N1VwgL+HIyt+JFIAAABAeeKwS4fXSFmpUki0VKOD5Fe0RCf7lF07/zyhbQcztPVgur7YmqY/M7IVEuCvCbc2122tahVz8GUHiRQAAABQXuz/SNr4iJR54Myx0FpS6+lS7G0XvOuJ7FNmwrT1YIa2HszQb4eOK9duuIyrXyNMr/dtrcZRFUviDMoMEikAAACgPNj/kbTmDkmuiY8y/8g73uFDM5k6ciL7dLKUlzRtO5ihPUf+LvBhKwVXULOYCDWLqaTml0To+maRCg30/TTD988QAAAA5VMxlrB5fSwOe95M1DlJlGFIB3JramvWpdq6dJW2htfS1tQM/ZmRXeDDREcEq1lMJTWNiVDT6EpqFlNJtaqEyOZD+0O5i0QKAADAKk+/KT6HkXtKf326VjkHjimwVhVV7n6NbAEeeptXVl6biyhh8+ZYcu2OvDVLqXmzSXuP/C2HISnnf9LR+1zGZjmC9evJusqwh5919LAkyWaT6lUPM2eamsVUUtPoSqoWHlSs8Xozm2EYRuHDfFtGRoYiIiKUnp6uSpUqeTocAABQlpWlN+iSDr/1qX57IkfZR6uZx4KqHVWDiYGq8c/upRtMWXltzlfCptOzJmeVsJWFWI7X6K5tBzN04FhWkZ7CXLuUmq4daSeUY3dYun+ALVeNgn5Xs5Ddatasg5q3uE5NoiopLKh8zrm4mxuQSKnsJFJ8muQl8RBLgbh+vSQeYilQmbp+pTL12hDLOcrSG3TlJVFbH3DOJpxdWuWQZFOzN0+UXjJVVl4bh136pK6ZzBmGZJefazyhtaSbd5b89eOwS8sauCSW/ztVSVuzLtW2k/XzyulONtLv2TWL9WkrBlc4XXYXoUaR4Qrw95MyfpG2PecyroJOqUHwfjUM2q9Av1N5BzutliITizUeb0MiZUFZSKT4NMlL4iGWAnH9ekk8xFKgMnX9SmXqtSGWc5zzBj2/02/Qe+wplQTPyD2l9dEfK/toVbkmUU4OBVU/pnYHe5b8BwNl5LXJtTv02y/J2po8QVtPXqqtWfX1S1Z9HXeEldhzFpeYiGDVqxEmfz+/wgefI9DfT42jwtU8JkLNYiIUW7WANUvm7+gP5U92pdK+fsuycpdIvfbaa3rhhReUmpqqZs2aadq0aerQoYNb9/V0IsWnSV4SD7EUiOvXS+IhlgKVqetXKlOvTXmOxe4wtOfICbO984FjmXk3nDwsHUop/AFqJkjBNYotnvPJOXBYf31b+LjK7aXAWiUcj4dfG8OQDhzL0vY/jyvnlLWyttJmk0P1g/5Qs5Bdaha8W81Cdqlpx1GqetldJf/k5t8lyfXvk4f+nyyjylUitWjRIt1999167bXXdPXVV+vNN9/U22+/rW3btql27dqF3t+TidS5nyZtrn9Kp1w+NDIUUPGEGr19lWwVrH9CYYnDIf3woJR99DwDbFJQNanN61IRPi3x6niIpUDGKYd23Pe9co+Hq+BPQ7l+y0Q8xFKgMnX9SmXqtSmPsfzv71xtS81r8/xr6nFl5dqL/FjwrIqB0mUBW/LW+wTvUrOQ3YoOOCLbuYl4wqdSjWtKNpjDa6UU1w9jgvxyFeyX4zquNMvpCpzdjZVaTyOJOq1cJVJt27ZVq1at9Prrr5vHLrvsMt1yyy2aNGlSoff3ZCJ17KNk/ff2Mz8PGfy3Msr+7DMAAD4tNNBfl51u7Vyvepgq+Nmk4zulX6cVfucmQ6WKDUs6RGVu3qkDMwsfV+t+KbRlCcdTBl6bqmFBan5JJcVGBMlvWb2yUcJWVsvpysJ6wzLM3dzA61tx5OTkaOPGjRo1apTL8euvv17r1q0r8D7Z2dnKzj7TGz8jI6NEY7yQnAPHJFUxf66X6q/M4Px/0ULqn1JgdLV8x4tV9mEpY3vh4yo1loJKvmShTMVDLAXKST2qrN2F/zPC9evheIilQGXq+pXK1GtTHmMJCfRX0+hKanp6Q9G61cLk73fuGpNYKfPewt8Ud76udNZItaml9U9+rOyjVSQVNBt3eo3UfaWxRqpsvTZqPf10CZtNBZawtZ5WOnH4+ZedWM6Nq5w3lCgOXp9IHTlyRHa7XZGRkS7HIyMjlZaWVuB9Jk2apPHjx5dGeIUKrFXF5edhi4MLHHf5YqnKbfElG8yfydKqxwsf12m1FFnCsUhlKx5iKdCxj5L132cKH8f1ewHl7JopS7GUqetXKlOvDbGcRxl7U2wLqKAGEwO19QGb8tb1nZ1M5a3zazAhsHQ6UJax10axt+Wt9ymwQcm00i1hK0uxoFiVQtF36Ti3M4lhGOfdYXn06NFKT083v/bv318aIRaocvdrFFTtqPL+wSuIQ0HVj6py9xKu4ZXypnVDa6ngtQLKOx4amzeuNJSleIilQFy/XhIPsRSoTF2/Upl6bYjlApxvikMvcT0eWssjC/Vr/LO7mr15QkHVjrkcD6p+rPSbpZSx10axt0k99uYl2fEL8r732OOZxKUsxYJi4/WJVPXq1eXv759v9unQoUP5ZqmcgoKCVKlSJZcvT3F+mpT3H8S5/5l76NOkvMjOudEDnyaVpXiIpUBcv14SD7EUqExdv1KZem2IpRBl7E1xjX92V7vUnrp8sXTZ9GO6fLHU7mBPz7TvL2OvjVnCVvf/8r57ch1QWYoFxcLrE6nAwEC1bt1aK1ascDm+YsUKxceXQilGMeDTJC+Jh1gKxPXrJfEQS4HK1PUrlanXhlgKUcbeFNsCKqjKbYmKHHKrqtyW6NkNpcvYawOUFJ/o2udsf/7GG2+offv2euuttzRz5kxt3bpVderUKfT+nt5HysnIPaW/Pl2rnAPHFFiriip3v8Zz/xCWtW4uZSkeYikQ16+XxEMsBSpT169Upl4bYgFQ3pSr9udS3oa8kydPVmpqqpo3b66pU6eqY8eObt23rCRSAAAAADyr3CVSF4NECgAAAIDkfm7g9WukAAAAAKC0kUgBAAAAgEUkUgAAAABgEYkUAAAAAFhEIgUAAAAAFpFIAQAAAIBFJFIAAAAAYBGJFAAAAABYRCIFAAAAABaRSAEAAACARSRSAAAAAGARiRQAAAAAWEQiBQAAAAAWkUgBAAAAgEUkUgAAAABgEYkUAAAAAFhEIgUAAAAAFpFIAQAAAIBFJFIAAAAAYBGJFAAAAABYRCIFAAAAABaRSAEAAACARSRSAAAAAGARiRQAAAAAWEQiBQAAAAAWkUgBAAAAgEUVPB1AWWAYhiQpIyPDw5EAAAAA8CRnTuDMEc6HRErS8ePHJUmxsbEejgQAAABAWXD8+HFFRESc93abUViqVQ44HA4dPHhQFStWlM1m82gsGRkZio2N1f79+1WpUiWPxgLvwDUDq7hmYBXXDKzimoFVZemaMQxDx48fV0xMjPz8zr8SihkpSX5+fqpVq5anw3BRqVIlj19E8C5cM7CKawZWcc3AKq4ZWFVWrpkLzUQ50WwCAAAAACwikQIAAAAAi0ikypigoCCNHTtWQUFBng4FXoJrBlZxzcAqrhlYxTUDq7zxmqHZBAAAAABYxIwUAAAAAFhEIgUAAAAAFpFIAQAAAIBFJFIAAAAAYBGJlAe89tprqlevnoKDg9W6dWutWbPmguNTUlLUunVrBQcHq379+nrjjTdKKVKUFVaumY8++khdunRRjRo1VKlSJbVv315ffvllKUaLssDqvzNO33zzjSpUqKCWLVuWbIAoc6xeM9nZ2RozZozq1KmjoKAgXXrppXrnnXdKKVqUBVavmfnz5+vyyy9XaGiooqOjdc899+jo0aOlFC086euvv1b37t0VExMjm82mpUuXFnofb3j/SyJVyhYtWqShQ4dqzJgx2rRpkzp06KAbb7xR+/btK3D8nj171K1bN3Xo0EGbNm3SE088oSFDhmjx4sWlHDk8xeo18/XXX6tLly76/PPPtXHjRl177bXq3r27Nm3aVMqRw1OsXjNO6enp6tevnzp16lRKkaKsKMo106tXL61atUqzZs3S9u3b9d5776lJkyalGDU8yeo1s3btWvXr108DBw7U1q1b9cEHH2jDhg267777SjlyeMLff/+tyy+/XDNmzHBrvNe8/zVQqq666irjwQcfdDnWpEkTY9SoUQWOHzlypNGkSROXYw888IDRrl27EosRZYvVa6YgTZs2NcaPH1/coaGMKuo107t3b+Pf//63MXbsWOPyyy8vwQhR1li9Zv7zn/8YERERxtGjR0sjPJRBVq+ZF154wahfv77LsZdfftmoVatWicWIskmSsWTJkguO8Zb3v8xIlaKcnBxt3LhR119/vcvx66+/XuvWrSvwPt9++22+8V27dtUPP/yg3NzcEosVZUNRrplzORwOHT9+XFWrVi2JEFHGFPWamT17tnbt2qWxY8eWdIgoY4pyzXzyySdq06aNJk+erEsuuUSNGjXSiBEjlJWVVRohw8OKcs3Ex8frwIED+vzzz2UYhv788099+OGHuummm0ojZHgZb3n/W8HTAZQnR44ckd1uV2RkpMvxyMhIpaWlFXiftLS0AsefOnVKR44cUXR0dInFC88ryjVzrpdeekl///23evXqVRIhoowpyjWzc+dOjRo1SmvWrFGFCvy3UN4U5ZrZvXu31q5dq+DgYC1ZskRHjhzRoEGD9L///Y91UuVAUa6Z+Ph4zZ8/X71799bJkyd16tQp9ejRQ6+88kpphAwv4y3vf5mR8gCbzebys2EY+Y4VNr6g4/BdVq8Zp/fee0/jxo3TokWLVLNmzZIKD2WQu9eM3W5Xnz59NH78eDVq1Ki0wkMZZOXfGYfDIZvNpvnz5+uqq65St27dNGXKFCUlJTErVY5YuWa2bdumIUOG6KmnntLGjRv1xRdfaM+ePXrwwQdLI1R4IW94/8tHj6WoevXq8vf3z/dpzaFDh/Jl3U5RUVEFjq9QoYKqVatWYrGibCjKNeO0aNEiDRw4UB988IE6d+5ckmGiDLF6zRw/flw//PCDNm3apMGDB0vKe5NsGIYqVKig5cuX67rrriuV2OEZRfl3Jjo6WpdccokiIiLMY5dddpkMw9CBAwfUsGHDEo0ZnlWUa2bSpEm6+uqr9dhjj0mS4uLiFBYWpg4dOujZZ58tMzMMKBu85f0vM1KlKDAwUK1bt9aKFStcjq9YsULx8fEF3qd9+/b5xi9fvlxt2rRRQEBAicWKsqEo14yUNxM1YMAALViwgPrzcsbqNVOpUiVt2bJFmzdvNr8efPBBNW7cWJs3b1bbtm1LK3R4SFH+nbn66qt18OBBnThxwjy2Y8cO+fn5qVatWiUaLzyvKNdMZmam/Pxc33b6+/tLOjPTADh5zftfDzW5KLcWLlxoBAQEGLNmzTK2bdtmDB061AgLCzP27t1rGIZhjBo1yrj77rvN8bt37zZCQ0ONYcOGGdu2bTNmzZplBAQEGB9++KGnTgGlzOo1s2DBAqNChQrGq6++aqSmpppff/31l6dOAaXM6jVzLrr2lT9Wr5njx48btWrVMu644w5j69atRkpKitGwYUPjvvvu89QpoJRZvWZmz55tVKhQwXjttdeMXbt2GWvXrjXatGljXHXVVZ46BZSi48ePG5s2bTI2bdpkSDKmTJlibNq0yfj9998Nw/De978kUh7w6quvGnXq1DECAwONVq1aGSkpKeZt/fv3NxISElzGJycnG1dccYURGBho1K1b13j99ddLOWJ4mpVrJiEhwZCU76t///6lHzg8xuq/M2cjkSqfrF4zv/zyi9G5c2cjJCTEqFWrlvHoo48amZmZpRw1PMnqNfPyyy8bTZs2NUJCQozo6Gijb9++xoEDB0o5anjC6tWrL/jexFvf/9oMg/lUAAAAALCCNVIAAAAAYBGJFAAAAABYRCIFAAAAABaRSAEAAACARSRSAAAA+P/27iwkyu+NA/h31EInK8usBEdbBDNabDRKoWkFQwzLQGkxhwZLrLRFihbLiMqosWyBgnJJKsYubL2wwtSxRdSKMs2s1KiMaKfGpXHO/yJ6/71uOQbFr/l+bmbO+j7vmZt5OOedISIrMZEiIiIiIiKyEhMpIiIiIiIiKzGRIiIiIiIishITKSIistq0adOwevXqvx3GbykoKIBCocDHjx//dih/VF1dHRQKBe7duwfAdteBiOh3MZEiIrIhc+bMwaxZszpsu3XrFhQKBe7cufOHo+o5rVaLuXPn/rLfv5D49URH66NSqdDQ0IAxY8b8naCIiP4RTKSIiGyITqdDfn4+6uvr27Wlp6fDz88ParX6L0RGf4q9vT2GDh0KBweHvx0KEdF/GhMpIiIbEhoaisGDByMzM1NWbzKZYDAYoNPp8O7dOyxYsAAeHh5QKpUYO3Yszpw50+W8CoUC586dk9W5uLjIrvPy5UtERkZiwIABcHV1RVhYGOrq6jqds7W1FTqdDsOHD4eTkxN8fHyQlpYmtScnJyMrKwvnz5+HQqGAQqFAQUFBu3m0Wi0KCwuRlpYm9fv5uuXl5QgICIBSqURQUBCqq6tl4y9evAh/f384OjpixIgR2L59O8xmc5dxr127Fi4uLnB1dcX69esRHR0t2xkaNmwYDhw4IBvn5+eH5ORkqZyamoqxY8eiT58+UKlUiIuLw5cvX6T2zMxMuLi4IC8vD76+vnB2dsbs2bPR0NDQ5fq0PdrXkZs3b0Kj0cDJyQkqlQrx8fH4+vVrp/2JiGwREykiIhvi4OCAJUuWIDMzE0IIqf7s2bNoaWnBokWL0NTUBH9/f1y6dAkVFRVYtmwZoqKiUFJS0uPrmkwmTJ8+Hc7OzigqKkJxcbH0xb+lpaXDMRaLBR4eHsjJyUFlZSW2bt2KTZs2IScnBwCQmJiIiIgIKXloaGhAUFBQu3nS0tIQGBiImJgYqZ9KpZLaN2/eDL1ej7KyMjg4OGDp0qVSW15eHhYvXoz4+HhUVlbi2LFjyMzMxM6dOzu9V71ej/T0dJw4cQLFxcV4//49cnNzrV4zOzs7HDx4EBUVFcjKykJ+fj7Wr18v62MymbBv3z5kZ2ejqKgIz58/R2JiolXr09aDBw8QHByM8PBw3L9/HwaDAcXFxVi5cqXV90BE9E8TRERkU6qqqgQAkZ+fL9VpNBqxYMGCTseEhISIdevWSeWpU6eKhIQEqQxA5Obmysb0799fZGRkCCGEOHHihPDx8REWi0Vqb25uFk5OTiIvL6/bscfFxYn58+dL5ejoaBEWFvbLcW3jFUKI69evCwDi2rVrUt3ly5cFANHY2CiEEGLKlCli165dsnHZ2dnC3d2902u5u7uLlJQUqfzt2zfh4eEhi9PLy0vs379fNm78+PFi27Ztnc6bk5MjXF1dpXJGRoYAIJ48eSLVHTlyRAwZMkQqd7Q+tbW1AoC4e/euEOL/6/DhwwchhBBRUVFi2bJlsjFGo1HY2dlJ60JERELwgDQRkY0ZNWoUgoKCkJ6ejunTp+Pp06cwGo24cuUKgO9H01JSUmAwGPDy5Us0NzejubkZffr06fE1y8vL8eTJE/Tt21dW39TUhKdPn3Y67ujRozh+/Djq6+vR2NiIlpYW+Pn59TiOjowbN0567+7uDgB48+YNPD09UV5ejtLSUtkOVGtrK5qammAymaBUKmVzffr0CQ0NDQgMDJTqHBwcEBAQINsB7I7r169j165dqKysxOfPn2E2m9HU1ISvX79Kn4VSqcTIkSNl8b9588aq67T147M6deqUVCeEgMViQW1tLXx9fX9rfiKifwUTKSIiG6TT6bBy5UocOXIEGRkZ8PLywsyZMwF8P5q2f/9+HDhwQHpGZ/Xq1Z0ewQO+PyPVNlH49u2b9N5iscDf31/25fwHNze3DufMycnBmjVroNfrERgYiL59+2Lv3r2/dcSwI7169ZLeKxQKKd4fr9u3b0d4eHi7cY6Ojj2+pp2dXZfrVV9fj5CQEMTGxmLHjh0YOHAgiouLodPpZP1+jv1H/NYmbG1ZLBYsX74c8fHx7do8PT1/a24ion8JEykiIhsUERGBhIQEnD59GllZWYiJiZGSCKPRiLCwMCxevBjA9y/WNTU1Xe5EuLm5ST9yAAA1NTUwmUxSWa1Ww2AwYPDgwejXr1+3YjQajQgKCkJcXJxU13b3qnfv3mhtbf3lXN3t15ZarUZ1dTW8vb271b9///5wd3fH7du3odFoAABmsxnl5eWyX0Nsu16fP39GbW2tVC4rK4PZbIZer4ed3ffHmX88G2aNnty3Wq3Gw4cPu33PRES2ij82QURkg5ydnREZGYlNmzbh1atX0Gq1Upu3tzeuXr2KmzdvoqqqCsuXL8fr16+7nG/GjBk4fPgw7ty5g7KyMsTGxsp2SxYtWoRBgwYhLCwMRqMRtbW1KCwsREJCAl68eNHhnN7e3igrK0NeXh4eP36MpKQklJaWyvoMGzYM9+/fR3V1Nd6+fSvbrWnbr6SkBHV1dXj79q204/QrW7duxcmTJ5GcnIyHDx+iqqoKBoMBW7Zs6XRMQkICUlJSkJubi0ePHiEuLq7dn93OmDED2dnZMBqNqKioQHR0NOzt7aX2kSNHwmw249ChQ3j27Bmys7Nx9OjRbsXc9r67sz4/27BhA27duoUVK1bg3r17qKmpwYULF7Bq1Sqrr09E9C9jIkVEZKN0Oh0+fPiAWbNmyY5sJSUlQa1WIzg4GNOmTcPQoUN/+ae3er0eKpUKGo0GCxcuRGJiouz5IaVSiaKiInh6eiI8PBy+vr5YunQpGhsbO92hio2NRXh4OCIjIzFp0iS8e/dOtjsFADExMfDx8UFAQADc3Nxw48aNDudKTEyEvb09Ro8eDTc3Nzx//rxbaxQcHIxLly7h6tWrmDhxIiZPnozU1FR4eXl1OmbdunVYsmQJtFqtdCRx3rx5sj4bN26ERqNBaGgoQkJCMHfuXNmzTn5+fkhNTcWePXswZswYnDp1Crt37+5WzD/r7vr8bNy4cSgsLERNTQ2mTJmCCRMmICkpSXp+jIiIvlOI3z1MTURERF3SarX4+PFju//aIiKi/y7uSBEREREREVmJiRQREREREZGVeLSPiIiIiIjIStyRIiIiIiIishITKSIiIiIiIisxkSIiIiIiIrISEykiIiIiIiIrMZEiIiIiIiKyEhMpIiIiIiIiKzGRIiIiIiIishITKSIiIiIiIiv9D8EuV8STUPW/AAAAAElFTkSuQmCC"/>


```python
print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )
```

<pre>

 No of rated product more than 50 per user : 1540

</pre>
### Task 4: 평가한 사용자 수가 60명 이상인 제품 표시



```python
#your code here

print('\n No of rated product more than 60 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 60)) )
```

<pre>

 No of rated product more than 60 per user : 996

</pre>
## 인기도에 따라 최종 작업 데이터 세트 가져오기



KNN 기능이 있는 Surprise 라이브러리를 import 시킵니다. 인기가 적은 상품은 추천에 크게 영향을 주지 않기 때문에 어떤 상품이 사용자에게 정말 인기가 있는지 확인하고 새로운 상품을 추천할 때 사용합니다.



라이브러리가 설치되어 있지 않은 경우 터미널에서 다음 단계를 수행하십시오. <br>

pip install surprise



conda install -c conda-forge scikit-surprise



```python
%pip install surprise
```

<pre>
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: surprise in c:\users\user\appdata\roaming\python\python39\site-packages (0.1)
Requirement already satisfied: scikit-surprise in c:\users\user\appdata\roaming\python\python39\site-packages (from surprise) (1.1.3)
Requirement already satisfied: scipy>=1.3.2 in c:\programdata\anaconda3\lib\site-packages (from scikit-surprise->surprise) (1.9.1)
Requirement already satisfied: joblib>=1.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-surprise->surprise) (1.1.0)
Requirement already satisfied: numpy>=1.17.3 in c:\programdata\anaconda3\lib\site-packages (from scikit-surprise->surprise) (1.21.5)
Note: you may need to restart the kernel to use updated packages.
</pre>

```python
import surprise
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
```


```python
# 인기 기반
# 평점 50점 이상 받은 고객들을 포함하는 새로운 데이터 프레임을 가져옵니다.
# 새로운 데이터 프레임에 평점 60점 이상 받은 고객들을 포함하는 경우도 테스트해 보세요.
new_df=df_modiset.groupby("productId").filter(lambda x:x['Rating'].count() >=50)
```


```python
no_of_ratings_per_product = new_df.groupby(by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()
```

<img src="data:image/png;base64,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vLy8jO+//77U8TZu3OhUT3H5+flGXl6e+She3+bNm40GDRoYkgxJRlBQkNG3b1/j/fffL/U8SqvbYrEYycnJTu09evQwgoODjezsbMMwDGPq1KmGl5dXifP/17/+ZUgyvvjiC6fzttlsxokTJ8r9fMMwjC5duhgtWrQwz+/w4cPGU089ZUgy7rnnnnLH7dixoxEWFmacPHnSbMvPzzfi4uKM+vXrm9/DoEGDjICAACMtLc2p33XXXWdIMvbu3ev0Wc8++2yp39fQoUPN18OHDzd8fX2NnTt3lnl+W7ZsMSQZc+fOden7AICKwPI5AKjCmjdvrtatW+vw4cO6/vrr1b59e7Vu3Vo//fSTevToodatW6t169Zq0KCBS+MNGjRIvr6+qlWrljp37qzMzEwtW7ZMtWvXdurn2GBh+PDhkoqWSw0bNkz79+/XqlWrPHqOkvTEE08oJCRETz75pNvvrVu3rnx9fc3Hxx9/bB67/vrr9fPPPyspKUl/+ctflJCQoFWrVun+++9X//79ZRhGueO3aNFCrVq1cmobPHiwMjMztW3bNknS559/rri4OLVu3Vr5+fnmo1evXrJYLFqzZo3T+7t16+Y0w1OeHTt2mOcXFRWladOmaciQIXrnnXcuOG52drY2bdqku+++W1dddZXZ7u3trcTERB06dMjczXD16tW65ZZbFB4e7tTPMZN0Mb788kt17dpVzZo1u+gxAOByIBQBQBVVUFBg/nG9du1a/e53v1N+fr6OHj2qXbt2qXPnzsrPz3faba08L774orZs2aK1a9fqr3/9q44cOaIBAwY4bbd98uRJ/fOf/9QNN9ygevXqKSMjQxkZGbrjjjtksVjMwORJwcHB5m5yq1evLnE8Ojpakkq97mjNmjXasmWL/vGPf5Q6tq+vr3r16qUpU6boP//5jw4ePKibb75Zn3/+ub788stya4uIiCiz7fjx45KKrsn64YcfnMKZr6+vgoKCZBiGfvvtN6f3l3WdU1muueYabdmyRd99951SUlKUkZGhBQsWyGazXXDc9PR0GYZR6udFRUU5ncPx48cveK4X49ixYyyFA3BF4JoiAKiibrnlFq1du9Z8vX37dr344ovm6x49ekgquq7o/JmIsjRu3NjcXOGmm25SQECAnn76ab3xxhvm9TEffPCBTp06pc2bN5c6m7FkyRKlp6e7NdPhiocfflivv/66nnzyST388MNOx6KiotSiRQutWLFCZ86ccbquqHXr1pKkrKwslz6nbt26Gj16tNasWaOUlBTdeuutF+yflpZWZlvdunUlFV2/FBAQoDlz5pQ6RmhoqNNrd69r8vf3d2lXv/PHrVOnjry8vJSamlqi7+HDh51qq1u37gXPtTir1eoUpB0cAcuhXr16OnToULl1A0BlY6YIAKqoWbNmacuWLXr++ecVEBCgjRs3asuWLerXr59uuukmbdmyRVu2bNGsWbMu+jOeeOIJNWnSRC+88IJOnjwpqWjpXFBQkFatWqXVq1c7PV5++WXl5ORo4cKFnjpNk5+fnyZPnqwtW7bon//8Z4njf/3rX/Xbb79pzJgxLi17y8vLK/FHusOuXbsknZstuZAdO3bo+++/d2pbtGiRgoKC1LZtW0lS37599csvv6hu3bpq3759iYerN9n1tMDAQHXo0EGffPKJTp8+bbYXFhZqwYIFql+/vrnJRteuXbVq1SqnnQgLCgq0ePHiEuM2bNhQP/zwg1PbV199VSKY9unTR6tXrzaX6JXGarVKklN9AHC5MVMEAFVUbGyspKIbpd5yyy3q0KGDpKIZo6eeesqt+wGVxdfXV88//7wGDhyo119/XQMGDNDmzZv18MMPq1u3biX6d+7cWdOmTdPs2bP16KOPXvLnn+/ee+/VK6+8UuqytnvvvVc7duzQlClT9P3332vYsGFq2rSpCgsLdfDgQc2fP1+SFBQUJEmy2+1q2LCh7rnnHnXv3l3R0dHKysrSmjVr9Prrr6tZs2a68847y60pKipK/fv318SJExUZGakFCxZoxYoVevHFF1WrVi1J0ujRo/Xxxx/rpptu0uOPP66WLVuqsLBQBw4c0PLlyzV27Fjz53e5TZ06VT169FDXrl01btw4+fn5aebMmUpJSdEHH3xgzi49/fTT+vTTT9WtWzc988wzqlWrlt58880SW4pLUmJioiZMmKBnnnlGXbp00c6dOzVjxowSy/mee+45ffnll7rpppv0l7/8RfHx8crIyFBSUpLGjBmj6667Ttdcc40CAgK0cOFCNWvWTFdddZWioqJcCqwA4DGVu88DAOBCCgoKjHr16hmzZs0yDMMwtm3bZkgqd7e58zl2n/vnP/9Z6vEOHToYderUMUaPHm1IKrHbWnGOnc+2bt1qtl3q7nPF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tGzZMvP1E088odq1a6tTp07av3+/R4uD+xzL5wqZKgIAAABc4nYoev755xUQECBJ2rBhg2bMmKGXXnpJoaGhevzxxz1eINzDltwAAACAe9y+T9HBgwfVpEkTSdLSpUt19913a+TIkercubNuvvlmT9cHNzlumksmAgAAAFzj9kzRVVddpePHj0uSli9fru7du0uS/P39dfr0ac9WB7c5ZoqYKgIAAABc4/ZMUY8ePfTggw+qTZs2+umnn3TbbbdJknbs2KGGDRt6uj64iZu3AgAAAO5xe6bozTffVEJCgo4dO6aPP/5YdevWlSRt3bpV9957r8cLhHssbMkNAAAAuMXtmaLatWtrxowZJdonTZrkkYJwac5dU0QqAgAAAFzhdij64YcfSm23WCzy9/dXgwYNZLVaL7kwXBx2nwMAAADc43Yoat26tTkbURpfX18NGjRIs2bNkr+//yUVB/eZM0WEIgAAAMAlbl9TtGTJEjVt2lRvv/22kpOTtX37dr399tuKjY3VokWLNHv2bH311Vd6+umnK6JelMMRV7l5KwAAAOAat2eKpkyZotdff129evUy21q2bKn69etrwoQJ2rx5swIDAzV27Fi98sorHi0W5bvAJB4AAACAUrg9U/Tjjz8qJiamRHtMTIx+/PFHSUVL7FJTUy+9OrjNIpbPAQAAAO5wOxRdd911euGFF5Sbm2u25eXl6YUXXtB1110nSfr1118VHh7uuSrhsnP3KSIVAQAAAK5we/ncm2++qf79+6t+/fpq2bKlLBaLfvjhBxUUFOjzzz+XJP3vf//TI4884vFiUT7uUwQAAAC4x+1Q1KlTJ+3bt08LFizQTz/9JMMwdPfdd2vw4MEKCgqSJCUmJnq8ULjGXD5XyXUAAAAAVwq3Q5EkXXXVVXrooYc8XQs84NxMEbEIAAAAcMVFhaKffvpJa9as0dGjR1VYWOh07JlnnvFIYbg45s1bK7UKAAAA4Mrhdih655139PDDDys0NFQRERFON3K1WCyEokrGzVsBAAAA97gdiiZPnqwpU6boySefrIh6cIlYPgcAAAC4x+0tudPT03XPPfdURC3wAJbPAQAAAO5xOxTdc889Wr58eUXUAg9gS24AAADAPW4vn2vSpIkmTJigjRs3Kj4+Xr6+vk7HR40a5bHi4D7zmiLmigAAAACXuB2K3n77bV111VVau3at1q5d63TMYrEQiiqZuXyOTAQAAAC4xO1QtHfv3oqoAx7C7nMAAACAe9y+pghVm6X8LgAAAACKcWmmaMyYMfrb3/6mwMBAjRkz5oJ9X331VY8UhovDltwAAACAe1yaKdq+fbvy8vLM5xd6eFrDhg1lsVhKPP74xz9KkoYNG1biWMeOHZ3GyMnJ0WOPPabQ0FAFBgaqf//+OnTokFOf9PR0JSYmymazyWazKTExURkZGR4/n4pmOTtXVEgmAgAAAFzi0kzR6tWrS31+OWzZskUFBQXm65SUFPXo0cPpXkm9e/fW3Llzzdd+fn5OY4wePVqfffaZPvzwQ9WtW1djx45V3759tXXrVnl7e0uSBg8erEOHDikpKUmSNHLkSCUmJuqzzz6ryNPzOHOmiN3nAAAAAJe4fU3R8OHDdfLkyRLt2dnZGj58uEeKKq5evXqKiIgwH59//rmuueYadenSxexjtVqd+oSEhJjH7Ha7Zs+erWnTpql79+5q06aNFixYoB9//FErV66UJO3atUtJSUl69913lZCQoISEBL3zzjv6/PPPtXv3bo+f0+XA6jkAAADANW6Honnz5un06dMl2k+fPq3333/fI0WVJTc3VwsWLNDw4cPNXdYkac2aNQoLC9O1116rESNG6OjRo+axrVu3Ki8vTz179jTboqKiFBcXp/Xr10uSNmzYIJvNpg4dOph9OnbsKJvNZvYpTU5OjjIzM50elc3LvE8RAAAAAFe4vCV3ZmamDMOQYRg6efKk/P39zWMFBQX64osvFBYWViFFOixdulQZGRkaNmyY2danTx/dc889iomJ0d69ezVhwgR169ZNW7duldVqVVpamvz8/FSnTh2nscLDw5WWliZJSktLK7X2sLAws09ppk6dqkmTJnnm5Dzk3EYLlVsHAAAAcKVwORTVrl3b3Mjg2muvLXHcYrFUeECYPXu2+vTpo6ioKLNt0KBB5vO4uDi1b99eMTExWrZsme68884yxzIMw2m2qfjzsvqcb/z48U678WVmZio6Otrl86kI58olFQEAAACucDkUrV69WoZhqFu3bvr444+drtvx8/NTTEyMU1jxtP3792vlypX65JNPLtgvMjJSMTEx2rNnjyQpIiJCubm5Sk9Pd5otOnr0qDp16mT2OXLkSImxjh07pvDw8DI/y2q1ymq1XszpVBjH7nPMFAEAAACucTkUOTY22Lt3r6Kjo+XldXnv+zp37lyFhYXptttuu2C/48eP6+DBg4qMjJQktWvXTr6+vlqxYoUGDhwoSUpNTVVKSopeeuklSVJCQoLsdrs2b96sG264QZK0adMm2e12MzhdKc7tPgcAAADAFS6HIoeYmBhJ0qlTp3TgwAHl5uY6HW/ZsqVnKiumsLBQc+fO1dChQ+Xjc67krKwsTZw4UXfddZciIyO1b98+/eUvf1FoaKjuuOMOSZLNZtMDDzygsWPHqm7dugoJCdG4ceMUHx+v7t27S5KaNWum3r17a8SIEZo1a5akoi25+/btq9jYWI+fT0VyrJ7j5q0AAACAa9wORceOHdMf/vAHffnll6UeL35PIU9ZuXKlDhw4UGLLb29vb/344496//33lZGRocjISHXt2lWLFy9WUFCQ2W/69Ony8fHRwIEDdfr0ad1yyy167733zHsUSdLChQs1atQoc5e6/v37a8aMGR4/l4rmmCni5q0AAACAayyGm1MKQ4YM0b59+/Taa6+pa9euWrJkiY4cOaLJkydr2rRp5S5vq84yMzNls9lkt9sVHBxcKTX8cixLt0xbq2B/H/0wsVel1AAAAABUBa7+fe72TNFXX32lf//737r++uvl5eWlmJgY9ejRQ8HBwZo6dWqNDkVVgbl8rlKrAAAAAK4cbu+WkJ2dbd7TJyQkRMeOHZMkxcfHa9u2bZ6tDm6zsNMCAAAA4Ba3Q1FsbKx2794tSWrdurVmzZqlX3/9Vf/4xz/MHd9QeZgpAgAAANzj9vK50aNHKzU1VZL07LPPqlevXlq4cKH8/Pz03nvvebo+uMmcKGL3OQAAAMAlboeiIUOGmM/btGmjffv26b///a8aNGig0NBQjxYH95k3b63kOgAAAIArhVvL5/Ly8tS4cWPt3LnTbKtVq5batm1LIKoizs0UVW4dAAAAwJXCrVDk6+urnJyccxfzo8oymCsCAAAAXOL2RguPPfaYXnzxReXn51dEPbhEXl5FgZWbtwIAAACucfuaok2bNmnVqlVavny54uPjFRgY6HT8k08+8VhxcJ85h0coAgAAAFzidiiqXbu27rrrroqoBR7gWNlYyEVFAAAAgEvcDkVz586tiDrgIb7eRSsi8wsNGYbB9V8AAABAOdy+pghVmyMUSVJeAbNFAAAAQHkIRdWMt9e5mSGW0AEAAADlIxRVM8UyEfcqAgAAAFxAKKpmLGKmCAAAAHCHS6EoJCREv/32myRp+PDhOnnyZIUWhYtXfF8FIhEAAABQPpdCUW5urjIzMyVJ8+bN05kzZyq0KFw8LwszRQAAAIA7XNqSOyEhQQMGDFC7du1kGIZGjRqlgICAUvvOmTPHowXCPU4zRYWVVwcAAABwpXApFC1YsEDTp0/XL7/8IovFIrvdzmxRFVV8pshgAR0AAABQLpdCUXh4uF544QVJUqNGjTR//nzVrVu3QgvDxSm++1whmQgAAAAol0uhqLi9e/dWRB3wEAvXFAEAAABuuagtudeuXat+/fqpSZMmatq0qfr3769vvvnG07XhIjlyEaEIAAAAKJ/boWjBggXq3r27atWqpVGjRunRRx9VQECAbrnlFi1atKgiaoSbzOuKyEQAAABAudxePjdlyhS99NJLevzxx822P/3pT3r11Vf1t7/9TYMHD/ZogXCfl0UqENcUAQAAAK5we6bof//7n/r161eivX///lxvVEVYVDRTxPI5AAAAoHxuh6Lo6GitWrWqRPuqVasUHR3tkaJwabimCAAAAHCd28vnxo4dq1GjRik5OVmdOnWSxWLRunXr9N577+n111+viBrhJsc1RWQiAAAAoHxuh6KHH35YERERmjZtmj766CNJUrNmzbR48WLdfvvtHi8Q7nPcq4hQBAAAAJTP7VAkSXfccYfuuOMOT9cCD3Hcq4jlcwAAAED5Luo+Raja2JEbAAAAcB2hqBryYqYIAAAAcBmhqBo6d00RoQgAAAAoD6GoGjp3TVElFwIAAABcAS4pFBmGwWxEFcTucwAAAIDrLioUvf/++4qPj1dAQIACAgLUsmVLzZ8/39O14SKx+xwAAADgOrdD0auvvqqHH35Yt956qz766CMtXrxYvXv31kMPPaTp06d7tLiJEyfKYrE4PSIiIszjhmFo4sSJioqKUkBAgG6++Wbt2LHDaYycnBw99thjCg0NVWBgoPr3769Dhw459UlPT1diYqJsNptsNpsSExOVkZHh0XO5nM5OFBGKAAAAABe4HYreeOMNvfXWW3rxxRfVv39/3X777XrppZc0c+ZM/f3vf/d4gS1atFBqaqr5+PHHH81jL730kl599VXNmDFDW7ZsUUREhHr06KGTJ0+afUaPHq0lS5boww8/1Lp165SVlaW+ffuq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"/>

#### 위에서 평점60점 이상 받은 고객들 포함시



```python
new_df=df_modiset.groupby("productId").filter(lambda x:x['Rating'].count() >=60)
```


```python
no_of_ratings_per_product = new_df.groupby(by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA0UAAAG3CAYAAAB7Qv6OAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABaSElEQVR4nO3deXgV1f3H8c9kuwkhuZCEbBICCEYgbIJCoBVBVgXEDSo0QlH4qVWKQFWsC7QgbohWikUFRBbFVqFVNGURUGQViBKguBQEJAGEkBCWrPP7I7lDLgnkXrjhhuT9ep77kDlz5sx3kvRpPp6ZM4ZpmqYAAAAAoIby8XYBAAAAAOBNhCIAAAAANRqhCAAAAECNRigCAAAAUKMRigAAAADUaIQiAAAAADUaoQgAAABAjUYoAgAAAFCjEYoAAAAA1GiEIgC4QnzzzTcyDEO7d++WJE2bNk0NGzZ06djVq1fLMAzr4+vrq3r16qlfv376+uuvz3vcv//9bxmGofDwcOXm5lrtN910k9N45/tMmDBBktSwYUP17dvXaWxHn+eff77Med955x0ZhlFubWvXrtU999yjBg0ayGazKTg4WC1atNDYsWP13//+16mvaZp6//339etf/1qRkZEKDAxU/fr11atXL7399tsufe+87dzvdVBQkFq3bq1XX31VRUVFlX5+x89i7969lTL+wYMHNWHCBKWmplbK+ADgCkIRAFwhNm/erDp16uiaa66RJG3YsEE33HCDW2M899xzWr9+vVavXq2nn35a69atU5cuXfT999+X23/WrFmSpGPHjmnJkiVW+4wZM7R+/Xrr89RTT0mS5syZ49R+//33V1jT888/r2PHjrlU/1NPPaVf//rX+umnn/TUU08pJSVFS5Ys0fDhw7V8+XI1a9ZMhYWFVv/x48frnnvuUbNmzfT222/rs88+06RJkxQVFaV//etfLp2zKmjcuLH1PV20aJGuuuoqPfrooxo/fry3S7tkBw8e1MSJEwlFALzKz9sFAABcs3nzZt1www0yDENScSh65JFH3BqjadOm6tixoyTp17/+terUqaOhQ4dq/vz5mjhxolPfjIwMffrpp+rWrZvWrVunWbNmadCgQZKk5s2bO/V1zNAkJiaqffv2LtfTvXt3rV69WpMnT9bUqVMv2Pe9997T5MmT9cADD2jGjBnW90GSevTo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psMVDz74oF577TU9/vjjevDBB532xcbGqkWLFlq+fLnOnDnj9FxRmzZtJEk5OTkunSc8PFyjR4/W6tWrlZaWpltuueWC/TMyMs7bFh4eLqn4+aWgoCDNnj273DEiIiKctt19rikwMNClVf3OHbdu3bry8fFRenp6mb4HDx50qi08PPyC11qazWZzCtIOjoDlUK9ePR04cKDCugHA25gpAoAqaubMmdq8ebOee+45BQUFacOGDdq8ebP69eunG2+8UZs3b9bmzZs1c+bMiz7HY489piZNmuj555/XiRMnJBXfOhcSEqKVK1dq1apVTp+XXnpJubm5WrBggacu0xIQEKBJkyZp8+bN+sc//lFm/5/+9Cf98ssvGjNmjEu3veXn55f5I91h165dks7OllzIjh079M033zi1LVy4UCEhIbruuuskSX379tWPP/6o8PBwtW/fvszH1ZfselpwcLA6dOigjz76SKdPn7bai4qKNH/+fNWvX99aZKNr165auXKl00qEhYWFWrRoUZlxGzZsqG+//dap7fPPPy8TTPv06aNVq1ZZt+iVx2azSZJTfQBwuTFTBABVVEJCgqTiF6XefPPN6tChg6TiGaMnnnjCrfcBnY+/v7+ee+45DRw4UK+99poGDBigTZs26cEHH1S3bt3K9O/cubOmTp2qWbNm6eGHH77k85/rnnvu0csvv1zubW333HOPduzYocmTJ+ubb77RsGHD1LRpUxUVFWn//v2aN2+eJCkkJESSlJWVpYYNG+ruu+9W9+7dFRcXp5ycHK1evVqvvfaamjVrpjvuuKPCmmJjY9W/f39NmDBBMTExmj9/vpYvX64XXnhBtWrVkiSNHj1aH374oW688UY9+uijatWqlYqKirRv3z4tW7ZMY8eOtX5+l9uUKVPUo0cPde3aVePGjVNAQIBmzJihtLQ0vffee9bs0lNPPaV///vf6tatm5555hnVqlVLf/vb38osKS5JycnJevrpp/XMM8+oS5cu2rlzp6ZPn17mdr4///nP+uyzz3TjjTfqySefVMuWLXX8+HGlpKRozJgxuvbaa3X11VcrKChICxYsULNmzVS7dm3Fxsa6FFgBwGO8u84DAOBCCgsLzXr16pkzZ840TdM0t27dakqqcLW5czlWn/vHP/5R7v4OHTqYdevWNUePHm1KKrPaWmmOlc+2bNlitV3q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1r6AgABFRkbK19e3UooEAAAAgMriciiKj4+XJBUVFVVaMbh0hrcLAAAAAK4wbi+0MGXKFM2ePbtM++zZs/XCCy94pChcOu6eAwAAAFzjdiiaOXOmrr322jLtLVq00N///nePFAUAAAAAl4vboSgjI0MxMTFl2uvVq6f09HSPFIWLV7LOgkxWWgAAAABc4nYoiouL01dffVWm/auvvlJsbKxHigIAAACAy8XlhRYc7r//fo0ePVr5+fnW0twrV67UY489prFjx3q8QLiHhRYAAAAA97gdih577DEdO3ZMDz30kPLy8iRJgYGBevzxxzV+/HiPF4iLw81zAAAAgGvcDkWGYeiFF17Q008/rV27dikoKEhNmzaVzWarjPoAAAAAoFK5HYocateureuvv96TtcADDGulBe/WAQAAAFwp3A5FXbt2PfuHdzk+//zzSyoIAAAAAC4nt0NRmzZtnLbz8/OVmpqqtLQ0DR061FN14SJdIK8CAAAAKIfboWjatGnltk+YMEE5OTmXXBA8w+T+OQAAAMAlbr+n6Hx++9vfavbs2Z4aDgAAAAAuC4+FovXr1yswMNBTw+EiOe6eM5koAgAAAFzi9u1zd9xxh9O2aZpKT0/X119/raefftpjhQEAAADA5eD2TJHdbnf6hIWF6aabbtKnn36qZ5991q2xvvjiC/Xr10+xsbEyDENLliyx9uXn5+vxxx9Xy5YtFRwcrNjYWN177706ePCg0xg33XSTDMNw+vzmN79x6pOZmank5GSr5uTkZB0/ftypz759+9SvXz8FBwcrIiJCo0aNsl5Oe0VhpQUAAADALW7PFM2ZM8djJz958qRat26t3/3ud7rzzjud9p06dUpbt27V008/rdatWyszM1OjR49W//799fXXXzv1HTFihP785z9b20FBQU77Bw8erAMHDiglJUWSNHLkSCUnJ+vjjz+WJBUWFurWW29VvXr1tHbtWh09elRDhw6VaZp6/fXXPXa9lxO3zwEAAACuueiXt3pCnz591KdPn3L32e12LV++3Knt9ddf1w033KB9+/apQYMGVnutWrUUHR1d7ji7du1SSkqKNmzYoA4dOkiS3nrrLSUlJWn37t1KSEjQsmXLtHPnTu3fv1+xsbGSpKlTp2rYsGGaPHmyQkNDPXG5AAAAAKogl26fq1u3rsLCwlz6VKasrCwZhqE6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h5hiqggAAABwhVuhyMfHR02bNtXRo0crqx4AAAAAuKzcfqboxRdf1B//+EelpaVVRj0AAAAAcFm5/Z6i3/72tzp16pRat26tgIAABQUFOe0/duyYx4rDxWOhBQAAAMA1boeiV199tRLKgKcYrLQAAAAAuMXtUDR06NDKqAMexkwRAAAA4JqLek/Rjz/+qKeeekr33HOPDh8+LElKSUnRjh07PFoc3Mc8EQAAAOAet0PRmjVr1LJlS23cuFEfffSRcnJyJEnffvutnn32WY8XCAAAAACVye1Q9MQTT2jSpElavny5AgICrPauXbtq/fr1bo31xRdfqF+/foqNjZVhGFqyZInTftM0NWHCBMXGxiooKEg33XRTmdmo3NxcPfLII4qIiFBwcLD69+9f5j1KmZmZSk5Olt1ul91uV3Jyso4fP+7UZ9++ferXr5+Cg4MVERGhUaNGKS8vz63rqUp4TxEAAADgGrdD0fbt23X77beXaa9Xr57b7y86efKkWrdurenTp5e7/8UXX9Qrr7yi6dOna/PmzYqOjlaPHj104sQJq8/o0aO1ePFivf/++1q7dq1ycnLUt29fFRYWWn0GDx6s1NRUpaSkKCUlRampqUpOTrb2FxYW6tZbb9XJkye1du1avf/++/rwww81duxYt66nKmCdBQAAAMA9bi+0UKdOHaWnp6tRo0ZO7du2bdNVV13l1lh9+vRRnz59yt1nmqZeffVV/elPf9Idd9whSZo7d66ioqK0cOFC/d///Z+ysrI0a9YszZs3T927d5ckzZ8/X3FxcVqxYoV69eqlXbt2KSUlRRs2bFCHDh0kSW+99ZaSkpK0e/duJSQkaNmyZdq5c6f279+v2NhYSdLUqVM1bNgwTZ48WaGhoW5dV1XAQgsAAACAa9yeKRo8eLAef/xxZWRkyDAMFRUV6auvvtK4ceN07733eqywPXv2KCMjQz179rTabDabunTponXr1kmStmzZovz8fKc+sbGxSkxMtPqsX79edrvdCkSS1LFjR9ntdqc+iYmJViCSpF69eik3N1dbtmw5b425ubnKzs52+nibwVILAAAAgFvcDkWTJ09WgwYNdNVVVyknJ0fNmzfXjTfeqE6dOumpp57yWGEZGRmSpKioKKf2qKgoa19GRoYCAgJUt27dC/aJjIwsM35kZKRTn3PPU7duXQUEBFh9yjNl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QR79HgAALg6hCABwxdu0aZMWLlyom2++WZKUl5enefPmqV69epKk5cuX69tvv9WePXsUFxcnSZo3b55atGihzZs36/rrr9err76q4cOH6/7775ckTZo0SStWrCgzW3Qh3333nT744AMtX75c3bt3lySnUBUWFiZJioyMVJ06dS75ugEAnsHtcwCAK9Inn3yi2rVrKzAwUElJSbrxxhv1+uuvS5Li4+OtQCRJu3btUlxcnBWIJKl58+aqU6eOdu3aZfVJSkpyOse52xVJTU2Vr6+vunTpcrGXBQDwAmaKAABXpK5du+qNN96Qv7+/YmNjnRZTCA4OduprmqYMwygzxvnaz8fHx0emaTq1OW7Zk8TtcABwhWKmCABwRQoODlaTJk0UHx9f4epyzZs31759+7R//36rbefOncrKylKzZs0kSc2aNdOGDRucjjt3u169ekpPT7e2CwsLlZaWZm23bNlSRUVFWrNmTbl1BAQEWMcBAKoOQhEAoNrr3r27WrVqpSFDhmjr1q3atGmT7r33XnXp0kXt27eXJP3hD3/Q7NmzNXv2bH333Xd69tlntWPHDqdxunXrpqVLl2rp0qX673//q4ceesjpnUMNGzbU0KFDNXz4cC1ZskR79uzR6tWr9cEHH0gqvq3PMAx98sknOnLkiHJyci7b9wAAcH6EIgBAtWcYhpYsWaK6devqxhtvVPfu3dW4cWMtWrTI6jNo0CA988wzevzxx9WuXTv99NNPevDBB53GGT58uIYOHWoFqkaNGlnLbDu88cYbuuuuu/TQQw/p2muv1YgRI6yXyl511VWaOHGinnjiCUVFRenhhx+u/IsHAFTIMM+9ORoAAAAAahBmigAAAADUaIQiAAAAADUaoQgAAABAjUYoAgAAAFCjEYoAAAAA1GiEIgAAAAA1GqEIAAAAQI1GKAIAAABQoxGKAAAAANRohCIAAAAANRqhCAAAAECNRigCAAAAUKP9P9du8w1EeXRbAAAAAElFTkSuQmCC"/>


```python
# Average rating of the product (내림차순 정렬)
# your code here
new_df.groupby("productId")['Rating'].mean().sort_values(ascending=False).head()
```

<pre>
productId
B004I763AW    4.966667
B0043ZLFXE    4.955556
B000TMFYBO    4.953125
B00GMRCAC6    4.951872
B008I6RVZU    4.951456
Name: Rating, dtype: float64
</pre>

```python
# Total no of rating for product
# your code here
new_df.groupby("productId")['Rating'].count().sort_values(ascending=False).head()
```

<pre>
productId
B0074BW614    18244
B00DR0PDNE    16454
B007WTAJTO    14172
B0019EHU8G    12285
B006GWO5WK    12226
Name: Rating, dtype: int64
</pre>

```python
# poductID 별 Rating 평균과 rating_count 로 Pandas Dataframe set 만들기
# your code here
ratings_mc = pd.DataFrame(new_df.groupby("productId")['Rating'].mean())
ratings_mc['rating_counts'] = pd.DataFrame(new_df.groupby("productId")['Rating'].count())

ratings_mc.head()
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
      <th>Rating</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>productId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0972683275</th>
      <td>4.470980</td>
      <td>1051</td>
    </tr>
    <tr>
      <th>1400501466</th>
      <td>3.560000</td>
      <td>250</td>
    </tr>
    <tr>
      <th>1400501520</th>
      <td>4.243902</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1400501776</th>
      <td>3.884892</td>
      <td>139</td>
    </tr>
    <tr>
      <th>1400532620</th>
      <td>3.684211</td>
      <td>171</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 가장 높은 rating_counts 값 출력 
# your code here
ratings_mc['rating_counts'].max()
```

<pre>
18244
</pre>

```python
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mc['rating_counts'].hist(bins=50)
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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Z2wBAACA3kSxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAgUWwAAABiBYgsAAAAjUGwBAABgBIotAAAAjECxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAghF9vt27dr6tSpSklJkcPh0ObNm4PWz5w5Uw6HI+gxduzYoBm/36+5c+cqISFBsbGxmjZtmo4ePRo0U19fL5/PJ7fbLbfbLZ/Pp4aGhqCZw4cPa+rUqYqNjVVCQoLmzZunlpaWUA8JAAAABgi52J45c0YjR47U8uXLLzkzefJk1dTU2I8333wzaP38+fO1adMmFRcXa8eOHTp9+rRyc3PV2tpqz+Tl5amqqkolJSUqKSlRVVWVfD6fvb61tVVTpkzRmTNntGPHDhUXF+u1115TYWFhqIcEAAAAA/QL9QU5OTnKycm57IzL5ZLH47nousbGRq1atUpr167VHXfcIUkqKipSamqqtm7dquzsbB04cEAlJSXauXOnxowZI0lauXKlMjMzdfDgQaWlpam0tFT79+/XkSNHlJKSIkl69tlnNXPmTD3++OMaMGBAqIcGAACAPizkYtsV7777rhITE3Xddddp/Pjxevzxx5WYmChJqqysVCAQkNfrtedTUlKUnp6u8vJyZWdnq6KiQm632y61kjR27Fi53W6Vl5crLS1NFRUVSk9Pt0utJGVnZ8vv96uyslITJkzosF9+v19+v99+3tTUJEkKBAIKBALdnkO79m23tbUpJiZG0f0cioq0On2do59DMTExamtr69H9u1q1H/O1eOzhIrPwkFvoyCw85BY6MguPSbmFcgzdXmxzcnJ09913a8iQIaqurtajjz6q22+/XZWVlXK5XKqtrVVUVJQGDhwY9LqkpCTV1tZKkmpra+0ifL7ExMSgmaSkpKD1AwcOVFRUlD1zoSVLlmjx4sUdlpeWlqp///5hHW8oampqtGHDhn8+a73s7NeGSFM36NixYzp27FhP7tpVrays7ErvQp9DZuEht9CRWXjILXRkFh4Tcjt79myXZ7u92N5zzz321+np6brllls0ZMgQbdmyRdOnT7/k6yzLksPhsJ+f//U3mTnfwoULVVBQYD9vampSamqqvF5vj350IRAIqKysTMnJycrKylJS3pOKSrqh09e11H2iuvUPa/v27Ro5cmSP7d/Vqj23SZMmyel0Xund6RPILDzkFjoyCw+5hY7MwmNSbu2/Ye+KHvkowvmSk5M1ZMgQHTp0SJLk8XjU0tKi+vr6oKu2x48f17hx4+yZurq6Dts6ceKEfZXW4/Fo165dQevr6+sVCAQ6XMlt53K55HK5Oix3Op298k2PiIhQc3OzvjpnyWq9ePk+n/+cpebmZkVERPT5k/Kb6K3vj0nILDzkFjoyCw+5hY7MwmNCbqHsf4/fx/bkyZM6cuSIkpOTJUkZGRlyOp1Bl8Zramq0d+9eu9hmZmaqsbFRu3fvtmd27dqlxsbGoJm9e/eqpqbGniktLZXL5VJGRkZPHxYAAACuMiFfsT19+rQ+/vhj+3l1dbWqqqoUHx+v+Ph4LVq0SHfddZeSk5P16aef6ve//70SEhL005/+VJLkdrs1a9YsFRYWatCgQYqPj9eCBQs0YsQI+y4Jw4cP1+TJk5Wfn68VK1ZIkmbPnq3c3FylpaVJkrxer2666Sb5fD49/fTT+vLLL7VgwQLl5+dzRwQAAIBrUMjF9oMPPgi640D7Z1bvv/9+vfjii/roo4/0yiuvqKGhQcnJyZowYYJeffVVxcXF2a9ZunSp+vXrpxkzZqi5uVkTJ07Uyy+/rMjISHtm3bp1mjdvnn33hGnTpgXdOzcyMlJbtmzRgw8+qFtvvVUxMTHKy8vTM888E3oKAAAA6PNCLrZZWVmyrEvfquqtt97qdBvR0dFatmyZli1bdsmZ+Ph4FRUVXXY7gwcP1htvvNHp+wEAAMB8Pf4ZWwAAAKA3UGwBAABgBIotAAAAjECxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAgUWwAAABiBYgsAAAAjUGwBAABgBIotAAAAjECxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwQsjFdvv27Zo6dapSUlLkcDi0efNme10gENDvfvc7jRgxQrGxsUpJSdEvfvELff7550HbyMrKksPhCHrce++9QTP19fXy+Xxyu91yu93y+XxqaGgImjl8+LCmTp2q2NhYJSQkaN68eWppaQn1kAAAAGCAkIvtmTNnNHLkSC1fvrzDurNnz2rPnj169NFHtWfPHm3cuFF///vfNW3atA6z+fn5qqmpsR8rVqwIWp+Xl6eqqiqVlJSopKREVVVV8vl89vrW1lZNmTJFZ86c0Y4dO1RcXKzXXntNhYWFoR4SAAAADNAv1Bfk5OQoJyfnouvcbrfKysqCli1btkw/+MEPdPjwYQ0ePNhe3r9/f3k8notu58CBAyopKdHOnTs1ZswYSdLKlSuVmZmpgwcPKi0tTaWlpdq/f7+OHDmilJQUSdKzzz6rmTNn6vHHH9eAAQNCPTQAAAD0YSEX21A1NjbK4XDouuuuC1q+bt06FRUVKSkpSTk5OXrssccUFxcnSaqoqJDb7bZLrSSNHTtWbrdb5eXlSktLU0VFhdLT0+1SK0nZ2dny+/2qrKzUhAkTOuyL3++X3++3nzc1NUn6+iMUgUCgOw87SPu229raFBMTo+h+DkVFWp2+ztHPoZiYGLW1tfXo/l2t2o/5Wjz2cJFZeMgtdGQWHnILHZmFx6TcQjmGHi22X331lR5++GHl5eUFXUG97777NHToUHk8Hu3du1cLFy7UX/7yF/tqb21trRITEztsLzExUbW1tfZMUlJS0PqBAwcqKirKnrnQkiVLtHjx4g7LS0tL1b9//7CPs6tqamq0YcOGfz5r7cIrhkhTN+jYsWM6duxYT+7aVe3C3wKgc2QWHnILHZmFh9xCR2bhMSG3s2fPdnm2x4ptIBDQvffeq7a2Nr3wwgtB6/Lz8+2v09PTNWzYMN1yyy3as2ePRo8eLUlyOBwdtmlZVtDyrsycb+HChSooKLCfNzU1KTU1VV6vt0c/uhAIBFRWVqbk5GRlZWUpKe9JRSXd0OnrWuo+Ud36h7V9+3aNHDmyx/bvatWe26RJk+R0Oq/07vQJZBYecgsdmYWH3EJHZuExKbf237B3RY8U20AgoBkzZqi6ulpvv/12p6Vx9OjRcjqdOnTokEaPHi2Px6O6uroOcydOnLCv0no8Hu3atStofX19vQKBQIcrue1cLpdcLleH5U6ns1e+6REREWpubtZX5yxZrRcv3+fzn7PU3NysiIiIPn9SfhO99f0xCZmFh9xCR2bhIbfQkVl4TMgtlP3v9vvYtpfaQ4cOaevWrRo0aFCnr9m3b58CgYCSk5MlSZmZmWpsbNTu3bvtmV27dqmxsVHjxo2zZ/bu3auamhp7prS0VC6XSxkZGd18VAAAALjahXzF9vTp0/r444/t59XV1aqqqlJ8fLxSUlL07//+79qzZ4/eeOMNtba22p93jY+PV1RUlP7xj39o3bp1+vGPf6yEhATt379fhYWFGjVqlG699VZJ0vDhwzV58mTl5+fbtwGbPXu2cnNzlZaWJknyer266aab5PP59PTTT+vLL7/UggULlJ+fzx0RAAAArkEhX7H94IMPNGrUKI0aNUqSVFBQoFGjRukPf/iDjh49qtdff11Hjx7Vv/3bvyk5Odl+lJeXS5KioqL05z//WdnZ2UpLS9O8efPk9Xq1detWRUZG2u+zbt06jRgxQl6vV16vV9/73ve0du1ae31kZKS2bNmi6Oho3XrrrZoxY4buvPNOPfPMM980EwAAAPRBIV+xzcrKkmVd+lZVl1snSampqdq2bVun7xMfH6+ioqLLzgwePFhvvPFGp9sCAACA+br9M7YAAADAlUCxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAgUWwAAABiBYgsAAAAjUGwBAABgBIotAAAAjECxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAghF9vt27dr6tSpSklJkcPh0ObNm4PWW5alRYsWKSUlRTExMcrKytK+ffuCZvx+v+bOnauEhATFxsZq2rRpOnr0aNBMfX29fD6f3G633G63fD6fGhoagmYOHz6sqVOnKjY2VgkJCZo3b55aWlpCPSQAAAAYIORie+bMGY0cOVLLly+/6PqnnnpKzz33nJYvX673339fHo9HkyZN0qlTp+yZ+fPna9OmTSouLtaOHTt0+vRp5ebmqrW11Z7Jy8tTVVWVSkpKVFJSoqqqKvl8Pnt9a2urpkyZojNnzmjHjh0qLi7Wa6+9psLCwlAPCQAAAAboF+oLcnJylJOTc9F1lmXp+eef1yOPPKLp06dLktasWaOkpCStX79eDzzwgBobG7Vq1SqtXbtWd9xxhySpqKhIqamp2rp1q7Kzs3XgwAGVlJRo586dGjNmjCRp5cqVyszM1MGDB5WWlqbS0lLt379fR44cUUpKiiTp2Wef1cyZM/X4449rwIABYQUCAACAvinkYns51dXVqq2tldfrtZe5XC6NHz9e5eXleuCBB1RZWalAIBA0k5KSovT0dJWXlys7O1sVFRVyu912qZWksWPHyu12q7y8XGlpaaqoqFB6erpdaiUpOztbfr9flZWVmjBhQof98/v98vv99vOmpiZJUiAQUCAQ6M4ogrRvu62tTTExMYru51BUpNXp6xz9HIqJiVFbW1uP7t/Vqv2Yr8VjDxeZhYfcQkdm4SG30JFZeEzKLZRj6NZiW1tbK0lKSkoKWp6UlKTPPvvMnomKitLAgQM7zLS/vra2VomJiR22n5iYGDRz4fsMHDhQUVFR9syFlixZosWLF3dYXlpaqv79+3flEL+Rmpoabdiw4Z/PWi87+7Uh0tQNOnbsmI4dO9aTu3ZVKysru9K70OeQWXjILXRkFh5yCx2ZhceE3M6ePdvl2W4ttu0cDkfQc8uyOiy70IUzF5sPZ+Z8CxcuVEFBgf28qalJqamp8nq9PfrRhUAgoLKyMiUnJysrK0tJeU8qKumGTl/XUveJ6tY/rO3bt2vkyJE9tn9Xq/bcJk2aJKfTeaV3p08gs/CQW+jILDzkFjoyC49JubX/hr0rurXYejweSV9fTU1OTraXHz9+3L666vF41NLSovr6+qCrtsePH9e4cePsmbq6ug7bP3HiRNB2du3aFbS+vr5egUCgw5Xcdi6XSy6Xq8Nyp9PZK9/0iIgINTc366tzlqzWyxd9SfKfs9Tc3KyIiIg+f1J+E731/TEJmYWH3EJHZuEht9CRWXhMyC2U/e/W+9gOHTpUHo8n6LJ3S0uLtm3bZpfWjIwMOZ3OoJmamhrt3bvXnsnMzFRjY6N2795tz+zatUuNjY1BM3v37lVNTY09U1paKpfLpYyMjO48LAAAAPQBIV+xPX36tD7++GP7eXV1taqqqhQfH6/Bgwdr/vz5euKJJzRs2DANGzZMTzzxhPr376+8vDxJktvt1qxZs1RYWKhBgwYpPj5eCxYs0IgRI+y7JAwfPlyTJ09Wfn6+VqxYIUmaPXu2cnNzlZaWJknyer266aab5PP59PTTT+vLL7/UggULlJ+fzx0RAAAArkEhF9sPPvgg6I4D7Z9Zvf/++/Xyyy/rt7/9rZqbm/Xggw+qvr5eY8aMUWlpqeLi4uzXLF26VP369dOMGTPU3NysiRMn6uWXX1ZkZKQ9s27dOs2bN8++e8K0adOC7p0bGRmpLVu26MEHH9Stt96qmJgY5eXl6Zlnngk9BQAAAPR5IRfbrKwsWdalb1XlcDi0aNEiLVq06JIz0dHRWrZsmZYtW3bJmfj4eBUVFV12XwYPHqw33nij030GAACA+br1M7YAAADAlUKxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAgUWwAAABiBYgsAAAAjUGwBAABgBIotAAAAjECxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAjdXmy//e1vy+FwdHg89NBDkqSZM2d2WDd27Nigbfj9fs2dO1cJCQmKjY3VtGnTdPTo0aCZ+vp6+Xw+ud1uud1u+Xw+NTQ0dPfhAAAAoI/o9mL7/vvvq6amxn6UlZVJku6++257ZvLkyUEzb775ZtA25s+fr02bNqm4uFg7duzQ6dOnlZubq9bWVnsmLy9PVVVVKikpUUlJiaqqquTz+br7cAAAANBH9OvuDX7rW98Kev7kk0/qxhtv1Pjx4+1lLpdLHo/noq9vbGzUqlWrtHbtWt1xxx2SpKKiIqWmpmrr1q3Kzs7WgQMHVFJSop07d2rMmDGSpJUrVyozM1MHDx5UWlpadx8WAAAArnLdXmzP19LSoqKiIhUUFMjhcNjL3333XSUmJuq6667T+PHj9fjjjysxMVGSVFlZqUAgIK/Xa8+npKQoPT1d5eXlys7OVkVFhdxut11qJWns2LFyu90qLy+/ZLH1+/3y+/3286amJklSIBBQIBDo1mM/X/u229raFBMTo+h+DkVFWp2+ztHPoZiYGLW1tfXo/l2t2o/5Wjz2cJFZeMgtdGQWHnILHZmFx6TcQjmGHi22mzdvVkNDg2bOnGkvy8nJ0d13360hQ4aourpajz76qG6//XZVVlbK5XKptrZWUVFRGjhwYNC2kpKSVFtbK0mqra21i/D5EhMT7ZmLWbJkiRYvXtxheWlpqfr37x/mUXZdTU2NNmzY8M9nrZed/doQaeoGHTt2TMeOHevJXbuqtX+cBV1HZuEht9CRWXjILXRkFh4Tcjt79myXZ3u02K5atUo5OTlKSUmxl91zzz321+np6brllls0ZMgQbdmyRdOnT7/ktizLCrrqe/7Xl5q50MKFC1VQUGA/b2pqUmpqqrxerwYMGNDl4wpVIBBQWVmZkpOTlZWVpaS8JxWVdEOnr2up+0R16x/W9u3bNXLkyB7bv6tVe26TJk2S0+m80rvTJ5BZeMgtdGQWHnILHZmFx6Tc2n/D3hU9Vmw/++wzbd26VRs3brzsXHJysoYMGaJDhw5Jkjwej1paWlRfXx901fb48eMaN26cPVNXV9dhWydOnFBSUtIl38vlcsnlcnVY7nQ6e+WbHhERoebmZn11zpLVeukC3s5/zlJzc7MiIiL6/En5TfTW98ckZBYecgsdmYWH3EJHZuExIbdQ9r/H7mO7evVqJSYmasqUKZedO3nypI4cOaLk5GRJUkZGhpxOZ9Cl85qaGu3du9cutpmZmWpsbNTu3bvtmV27dqmxsdGeAQAAwLWlR67YtrW1afXq1br//vvVr9//vcXp06e1aNEi3XXXXUpOTtann36q3//+90pISNBPf/pTSZLb7dasWbNUWFioQYMGKT4+XgsWLNCIESPsuyQMHz5ckydPVn5+vlasWCFJmj17tnJzc7kjAgAAwDWqR4rt1q1bdfjwYf3yl78MWh4ZGamPPvpIr7zyihoaGpScnKwJEybo1VdfVVxcnD23dOlS9evXTzNmzFBzc7MmTpyol19+WZGRkfbMunXrNG/ePPvuCdOmTdPy5ct74nAAAADQB/RIsfV6vbKsjreziomJ0VtvvdXp66Ojo7Vs2TItW7bskjPx8fEqKir6RvsJAAAAc/TYZ2wBAACA3kSxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAgUWwAAABiBYgsAAAAjUGwBAABgBIotAAAAjECxBQAAgBEotgAAADACxRYAAABGoNgCAADACBRbAAAAGIFiCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIxAsQUAAIARKLYAAAAwAsUWAAAARqDYAgAAwAjdXmwXLVokh8MR9PB4PPZ6y7K0aNEipaSkKCYmRllZWdq3b1/QNvx+v+bOnauEhATFxsZq2rRpOnr0aNBMfX29fD6f3G633G63fD6fGhoauvtwAAAA0Ef0yBXbm2++WTU1Nfbjo48+stc99dRTeu6557R8+XK9//778ng8mjRpkk6dOmXPzJ8/X5s2bVJxcbF27Nih06dPKzc3V62trfZMXl6eqqqqVFJSopKSElVVVcnn8/XE4QAAAKAP6NcjG+3XL+gqbTvLsvT888/rkUce0fTp0yVJa9asUVJSktavX68HHnhAjY2NWrVqldauXas77rhDklRUVKTU1FRt3bpV2dnZOnDggEpKSrRz506NGTNGkrRy5UplZmbq4MGDSktL64nDAgAAwFWsR4rtoUOHlJKSIpfLpTFjxuiJJ57QDTfcoOrqatXW1srr9dqzLpdL48ePV3l5uR54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urpaZWVlQVdrL2b06NFyOp1B59+1ltmFeupn0sTc3nvvPR08eLDTf+eka/Nco9j2oKioKGVkZNi/AmhXVlamcePGXaG96j2WZWnOnDnauHGj3n77bQ0dOrTT15w8eVJHjhxRcnKyJCkjI0NOpzMow5qaGu3du9fOMDMzU42Njdq9e7c9s2vXLjU2NhqRs9/v14EDB5ScnGz/iun8PFpaWrRt2zb7WK/lzFavXq3ExERNmTLlsnOcZx315rmVmZmpvXv3qqamxp4pLS2Vy+VSRkZGjx5nd2svtYcOHdLWrVs1aNCgTl+zb98+BQIB+/y71jK7mJ76mTQxt1WrVikjI0MjR47sdPaaPNd69U/VrkHFxcWW0+m0Vq1aZe3fv9+aP3++FRsba3366adXetd63H/+539abrfbevfdd62amhr7cfbsWcuyLOvUqVNWYWGhVV5eblVXV1vvvPOOlZmZaf3Lv/yL1dTUZG/n17/+tXX99ddbW7dutfbs2WPdfvvt1siRI61z587ZM5MnT7a+973vWRUVFVZFRYU1YsQIKzc3t9ePuTsUFhZa7777rvXJJ59YO3futHJzc624uDj7nHnyySctt9ttbdy40froo4+sn/3sZ1ZycvI1nZllWVZra6s1ePBg63e/+13Qcs6z/3Pq1Cnrww8/tD788ENLkvXcc89ZH374of0X/L11bp07d85KT0+3Jk6caO3Zs8faunWrdf3111tz5szpvTC66HKZBQIBa9q0adb1119vVVVVBf075/f7LcuyrI8//thavHix9f7771vV1dXWli1brO9+97vWqFGjjM3Msi6fW2/+TPal3Dr7+bQsy2psbLT69+9vvfjiix1ef62eaxei2PaC//qv/7KGDBliRUVFWaNHjw663ZXJJF30sXr1asuyLOvs2bOW1+u1vvWtb1lOp9MaPHiwdf/991uHDx8O2k5zc7M1Z84cKz4+3oqJibFyc3M7zJw8edK67777rLi4OCsuLs667777rPr6+l460u51zz33WMnJyZbT6bRSUlKs6dOnW/v27bPXt7W1WY899pjl8Xgsl8tl3XbbbdZHH30UtI1rLTPLsqy33nrLkmQdPHgwaDnn2f955513Lvozef/991uW1bvn1meffWZNmTLFiomJseLj4605c+ZYX331VU8eflgul1l1dfUl/5175513LMuyrMOHD1u33XabFR8fb0VFRVk33nijNW/ePOvkyZNB72NSZpZ1+dx6+2eyr+TW2c+nZVnWihUrrJiYGKuhoaHD66/Vc+1CDsuyrB69JAwAAAD0Aj5jCwAAACNQbAEAAGAEii0AAACMQLEFAACAESi2AAAAMALFFgAAAEag2AIAAMAIFFsAAAAYgWILAAAAI1BsAQAAYASKLQAAAIzw/wPPbIEMjGKX0wAAAABJRU5ErkJggg=="/>


```python
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mc['Rating'].hist(bins=50)
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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goKCZLcFAACAPmR8aAUAAACMPacVAAAA6EBoBQAAgPEIrQAAADAeoRUAAADGI7QOIPv27dOdd94pj8cjm82mF198scf61157TTabrdPl/fff75uGk2Tt2rX64he/qCFDhmj48OG6++67dezYsV7vt3fvXk2ePFkZGRkaPXq0fvazn/VBt8l3LfNK1X3r6aef1uc///nwt8ZMmzZNv/nNb3q8T6ruV1Ls80rV/aora9eulc1mU0lJSY91qbx/dYhmVqm8b5WVlXV63nl5eT3eJ1n7FaF1ALl06ZImTZqkjRs3xnS/Y8eO6ezZs+HLmDFjEtShGfbu3av7779fb7zxhqqrq3X58mUVFhbq0qVL3d7nxIkT+od/+Ad96Utf0rvvvqvvfe97evDBB/XCCy/0YefJcS3z6pBq+9b111+vJ554QrW1taqtrdXf//3f66677tLRo0e7rE/l/UqKfV4dUm2/utrbb7+tZ555Rp///Od7rEv1/UuKflYdUnXfuummmyKe9+9+97tua5O6X1kYkCRZO3fu7LHm1VdftSRZjY2NfdKTqRoaGixJ1t69e7utefjhh61x48ZFbFu+fLl1yy23JLo940QzL/at/zNs2DDr5z//eZe3sV911tO82K8s68KFC9aYMWOs6upq67bbbrO+9a1vdVub6vtXLLNK5X3r+9//vjVp0qSo65O5X3GkFbr55ps1YsQIzZo1S6+++mqy2+lzTU1NkqTs7Oxua15//XUVFhZGbJs7d65qa2sVDAYT2p9poplXh1Tet9rb27Vjxw5dunRJ06ZN67KG/er/RDOvDqm8X91///264447NHv27F5rU33/imVWHVJ13/rDH/4gj8ejUaNG6Wtf+5qOHz/ebW0y9yujv8YViTVixAg988wzmjx5sgKBgLZt26ZZs2bptddeS5lvHbMsSytXrtStt96qCRMmdFtXX1+v3NzciG25ubm6fPmyfD6fRowYkehWjRDtvFJ53/rd736nadOmqbW1VX/zN3+jnTt36nOf+1yXtexXsc0rlfcrSdqxY4cOHjyo2traqOpTef+KdVapvG9NnTpV//Vf/6Ubb7xR586d0+OPP67p06fr6NGjcrvdneqTuV8RWlPY2LFjNXbs2PD1adOm6dSpU/qP//iPAf9L2uGBBx7Qe++9pwMHDvRaa7PZIq5bf/kyuau3D2TRziuV962xY8fq0KFD+vjjj/XCCy9o8eLF2rt3b7dBLNX3q1jmlcr71alTp/Stb31Lu3fvVkZGRtT3S8X961pmlcr71rx588J/njhxoqZNm6bPfvazev7557Vy5cou75Os/YrTAxDhlltu0R/+8Idkt9EnVqxYoZdeekmvvvqqrr/++h5r8/LyVF9fH7GtoaFB6enpXf6f6EAUy7y6kir71qBBg3TDDTdoypQpWrt2rSZNmqSf/vSnXdayX8U2r66kyn518OBBNTQ0aPLkyUpPT1d6err27t2r//zP/1R6erra29s73SdV969rmVVXUmXfutrgwYM1ceLEbp97MvcrjrQiwrvvvjug3zKSrvwf4YoVK7Rz50699tprGjVqVK/3mTZtml5++eWIbbt379aUKVPkcDgS1aoRrmVeXUmFfasrlmUpEAh0eVsq71fd6WleXUmV/WrWrFmdPtG9ZMkSjRs3Tt/97neVlpbW6T6pun9dy6y6kir71tUCgYDq6ur0pS99qcvbk7pfJfyjXugzFy5csN59913r3XfftSRZ69evt959913r5MmTlmVZ1iOPPGItWrQoXL9hwwZr586d1gcffGAdOXLEeuSRRyxJ1gsvvJCsp9An/u3f/s3KysqyXnvtNevs2bPhS0tLS7jm6lkdP37cyszMtL797W9bv//9763NmzdbDofD+u///u9kPIU+dS3zStV9a9WqVda+ffusEydOWO+99571ve99z7Lb7dbu3bsty2K/ulqs80rV/ao7V38inv2re73NKpX3rdLSUuu1116zjh8/br3xxhtWUVGRNWTIEOtPf/qTZVlm7VeE1gGkY8mOqy+LFy+2LMuyFi9ebN12223h+ieffNL67Gc/a2VkZFjDhg2zbr31Vut///d/k9N8H+pqRpKs5557Llxz9awsy7Jee+016+abb7YGDRpkfeYzn7Gefvrpvm08Sa5lXqm6b33zm9+0Pv3pT1uDBg2y/vZv/9aaNWtWOIBZFvvV1WKdV6ruV925Ooixf3Wvt1ml8r71T//0T9aIESMsh8NheTwe65577rGOHj0avt2k/cpmWX85exYAAAAwFB/EAgAAgPEIrQAAADAeoRUAAADGI7QCAADAeIRWAAAAGI/QCgAAAOMRWgEAAGA8QisAAACMR2gFAACA8QitAAAAMB6hFQAAAMb7/5mBQJ1VBn0HAAAAAElFTkSuQmCC"/>


```python
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts', data=ratings_mc, alpha=0.4)
```

<pre>
<seaborn.axisgrid.JointGrid at 0x23196d9a670>
</pre>
<pre>
<Figure size 800x600 with 0 Axes>
</pre>
<img src="data:image/png;base64,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ExEREakyBWQiIiIiVaaATERERKTKFJCJiIiIVJkCMhEREZEqU0AmIiIiUmUKyERERESqTAGZiIiISJUpIBMRERGpMgVkIiIiIlWmgExERESkypzVHoCIiMh8Smbz9EfTJLMWfrdJW9CL363HnywsekeKiMiS1RdJ8UzXKMmsVT4WcJtcsTZEe72viiMTqaQpSxERWZKS2fyUYAwgkbXY2zVKMpuv0shEplJAJiIiS1JpmnI6iaxFfzR9jkckMjMFZCIisiTNFIyVpE5xXuRcUkAmIiJLkt9tnvS87xTnRc4lBWQiIrIktQW9BGYIugLjqy1FFgoFZCIisiT53U6uWBuaEpSVVlmq9YUsJHo3iojIktVe7+Oaza30R9OkshY+9SGTBUrvSBERWdL8bifrmmuqPQyRk9KUpYiIiEiVKSATERERqTIFZCIiIiJVpoBMREREpMoUkImIiIhUmQIyERERkSqrakD2+OOPc9NNN9HR0YFhGPzsZz+rOG8YxrQf3/zmN8vXvOtd75py/sMf/nDF64TDYXbv3k0wGCQYDLJ7924ikUjFNd3d3dx0000EAgGampq48847yWaz83XrIiIiImVVDcgSiQQXX3wx995777Tn+/r6Kj7+5//8nxiGwQc+8IGK626//faK67773e9WnL/11lvZv38/e/bsYc+ePezfv5/du3eXz1uWxQ033EAikeCJJ57goYce4ic/+Ql333333N+0iIiIyCRVbQx7/fXXc/311894vq2treLzf/iHf+Dd734369atqzju9/unXFty8OBB9uzZw9NPP8327dsBuO+++9i5cyevv/46Gzdu5JFHHuHVV1+lp6eHjo4OAL797W9z22238ZWvfIW6urqzuU0RERGRk1o0NWQDAwM8/PDDfOITn5hy7sEHH6SpqYnNmzdzzz33MDY2Vj731FNPEQwGy8EYwI4dOwgGgzz55JPla7Zs2VIOxgCuvfZaMpkM+/btm3FMmUyGWCxW8SEiIrLU6Hk3/xZNQPaDH/yA2tpa3v/+91cc/+hHP8rf/d3f8etf/5rPf/7z/OQnP6m4pr+/n5aWlimv19LSQn9/f/ma1tbWivMNDQ243e7yNdP52te+Vq5LCwaDdHZ2ns0tioiILEh63s2/RbOX5f/8n/+Tj370o3i93orjt99+e/nPW7Zs4bzzzuPyyy/n+eef59JLLwWKiwMms2274vjpXDPZZz/7We66667y57FYTG9SERFZcvS8m3+LIiD7P//n//D666/z4x//+JTXXnrppbhcLg4dOsSll15KW1sbAwMDU64bGhoqZ8Xa2tp45plnKs6Hw2FyudyUzNlEHo8Hj8dzhncjIiKyuOh5N/8WxZTl/fffz2WXXcbFF198ymtfeeUVcrkc7e3tAOzcuZNoNMrevXvL1zzzzDNEo1GuvPLK8jUHDhygr6+vfM0jjzyCx+Phsssum+O7EREREalU1QxZPB7nzTffLH/e1dXF/v37CYVCrFq1CiimRf+f/+f/4dvf/vaUr3/rrbd48MEH+e3f/m2ampp49dVXufvuu9m2bRtXXXUVAJs2beK6667j9ttvL7fDuOOOO7jxxhvZuHEjALt27eLCCy9k9+7dfPOb32R0dJR77rmH22+/XSssRUREZN5VNUP23HPPsW3bNrZt2wbAXXfdxbZt2/jCF75Qvuahhx7Ctm0+8pGPTPl6t9vNr371K6699lo2btzInXfeya5du3jssccwTbN83YMPPsjWrVvZtWsXu3bt4qKLLuKBBx4onzdNk4cffhiv18tVV13FLbfcwvve9z6+9a1vzePdi4iIiBQZtm3b1R7EUhGLxQgGg0SjUWXWRERkydLzbu4tihoyERERkaVMAZmIiIhIlSkgExEREakyBWQiIiIiVaaATERERKTKFJCJiIiIVJkCMhEREZEqU0AmIiIiUmUKyERERESqTAGZiIiISJUpIBMRERGpMgVkIiIiIlXmrPYARESkOpLZPP3RNMmshd9t0hb04nfrsSBSDfqbJyKyDPVFUjzTNUoya5WPBdwmV6wN0V7vq+LIRJYnTVmKiCwzyWx+SjAGkMha7O0aJZnNV2lkIsuXAjIRkWWmNE05nUTWoj+aPscjEhEFZCIiy8xMwVhJ6hTnRWTuKSATEVlm/G7zpOd9pzgvInNPAZmIyDLTFvQSmCHoCoyvthSRc0sBmYjIMuN3O7libWhKUFZaZanWFyLnnv7WiYgsQ+31Pq7Z3Ep/NE0qa+FTHzKRqtLfPBGRZcrvdrKuuabawxARNGUpIiIiUnUKyERERESqTAGZiIiISJUpIBMRERGpMgVkIiIiIlWmgExERESkyhSQiYiIiFSZAjIRERGRKlNAJiIiIlJlCshEREREqkwBmYiIiEiVKSATERERqTIFZCIiIiJVpoBMREREpMoUkImIiIhUmQIyERERkSpTQCYiIiJSZQrIRERERKpMAZmIiIhIlSkgExEREakyBWQiIiIiVVbVgOzxxx/npptuoqOjA8Mw+NnPflZx/rbbbsMwjIqPHTt2VFyTyWT49Kc/TVNTE4FAgJtvvpne3t6Ka8LhMLt37yYYDBIMBtm9ezeRSKTimu7ubm666SYCgQBNTU3ceeedZLPZ+bhtERERkQpVDcgSiQQXX3wx995774zXXHfddfT19ZU/fvGLX1Sc/8xnPsNPf/pTHnroIZ544gni8Tg33ngjlmWVr7n11lvZv38/e/bsYc+ePezfv5/du3eXz1uWxQ033EAikeCJJ57goYce4ic/+Ql333333N+0iIiIyCTOan7z66+/nuuvv/6k13g8Htra2qY9F41Guf/++3nggQe4+uqrAfjRj35EZ2cnjz32GNdeey0HDx5kz549PP3002zfvh2A++67j507d/L666+zceNGHnnkEV599VV6enro6OgA4Nvf/ja33XYbX/nKV6irq5vDuxYRERGptOBryH7961/T0tLC+eefz+23387g4GD53L59+8jlcuzatat8rKOjgy1btvDkk08C8NRTTxEMBsvBGMCOHTsIBoMV12zZsqUcjAFce+21ZDIZ9u3bN9+3KCIiIstcVTNkp3L99dfzwQ9+kNWrV9PV1cXnP/953vOe97Bv3z48Hg/9/f243W4aGhoqvq61tZX+/n4A+vv7aWlpmfLaLS0tFde0trZWnG9oaMDtdpevmU4mkyGTyZQ/j8Vis75XERGRhUrPu/m3oDNkH/rQh7jhhhvYsmULN910E7/85S954403ePjhh0/6dbZtYxhG+fOJfz6bayb72te+Vl4oEAwG6ezsPJ3bEhERWVT0vJt/Czogm6y9vZ3Vq1dz6NAhANra2shms4TD4YrrBgcHyxmvtrY2BgYGprzW0NBQxTWTM2HhcJhcLjclczbRZz/7WaLRaPmjp6fnrO5PRERkIdLzbv4tqoBsZGSEnp4e2tvbAbjssstwuVw8+uij5Wv6+vo4cOAAV155JQA7d+4kGo2yd+/e8jXPPPMM0Wi04poDBw7Q19dXvuaRRx7B4/Fw2WWXzTgej8dDXV1dxYeIiMhSo+fd/KtqDVk8HufNN98sf97V1cX+/fsJ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bdv2mX5Rd3c3K1euxOGojOds26anp4dVq1bN2QAXk1gsRjAYJBqNUldXV+3hiIiIzAs97+berDJka9euZXh4eMrx0dFR1q5de9aDEhEREVlOZhWQzZRUi8fjeL3esxqQiIiIyHJzRkX9d911FwCGYfCFL3wBv99fPmdZFs888wyXXHLJnA5QREREZKk7o4DshRdeAIoZspdffhm3210+53a7ufjii7nnnnvmdoQiIiIiS9wZBWT/8i//AsC/+Tf/hv/yX/6LCvlERERE5sCs+pB973vfm+txiIiIiCxbswrIEokEX//61/nVr37F4OAghUKh4vzhw4fnZHAiIiIiy8GsArJPfvKT/OY3v2H37t20t7djGMZcj0tERERk2ZhVQPbLX/6Shx9+mKuuumquxyMiIrIoJLN5+qNpklkLv9ukLejF757VY1VkdgFZQ0MDoVBorsciIiKyKPRFUjzTNUoya5WPBdwmV6wN0V7vq+LIZLGaVWPY//gf/yNf+MIXSCaTcz0eERGRBS2ZzU8JxgASWYu9XaMks/kqjUwWs1llyL797W/z1ltv0draypo1a3C5XBXnn3/++TkZnIiIyEJTmqacTiJr0R9Ns6655hyPSha7WQVk73vf++Z4GCIiIovDTMFYSeoU50WmM6uA7Itf/OJcj0NERGRR8LvNk573neK8yHRmVUM2Vx5//HFuuukmOjo6MAyDn/3sZxXnbdvmS1/6Eh0dHfh8Pt71rnfxyiuvVFyTyWT49Kc/TVNTE4FAgJtvvpne3t6Ka8LhMLt37yYYDBIMBtm9ezeRSKTimu7ubm666SYCgQBNTU3ceeedZLPZ+bhtERFZxNqCXgIzBF2B8dWWImdqVgGZw+HANM0ZP05XIpHg4osv5t577532/De+8Q3+4i/+gnvvvZdnn32WtrY2rrnmGsbGxsrXfOYzn+GnP/0pDz30EE888QTxeJwbb7wRyzqRMr711lvZv38/e/bsYc+ePezfv5/du3eXz1uWxQ033EAikeCJJ57goYce4ic/+Ql33333LH46IiKylPndTq5YG5oSlJVWWar1hcyGYdu2faZf9A//8A8Vn+dyOV544QV+8IMf8OUvf5lPfOITZz4Qw+CnP/1puT7Ntm06Ojr4zGc+w7/7d/8OKGbDWltb+c//+T/zqU99img0SnNzMw888AAf+tCHADh+/DidnZ384he/4Nprr+XgwYNceOGFPP3002zfvh2Ap59+mp07d/Laa6+xceNGfvnLX3LjjTfS09NDR0cHAA899BC33XYbg4ODp71nZywWIxgMEo1Gtc+niMgSlszm6R5NMhhLYzoMWuu8dNT7lk0wpufd3JvVO+d3fud3phz7vd/7PTZv3syPf/zjWQVkk3V1ddHf38+uXbvKxzweD+985zt58skn+dSnPsW+ffvI5XIV13R0dLBlyxaefPJJrr32Wp566imCwWA5GAPYsWMHwWCQJ598ko0bN/LUU0+xZcuWcjAGcO2115LJZNi3bx/vfve7px1jJpMhk8mUP4/FYmd93yIisrBN14NseCxLwO1csgGZnnfzb05ryLZv385jjz02J6/V398PQGtra8Xx1tbW8rn+/n7cbjcNDQ0nvaalpWXK67e0tFRcM/n7NDQ04Ha7y9dM52tf+1q5Li0YDNLZ2XmGdykiInMhmc1zeCjOgWNRDg/F560X2HLtQabn3fybs4AslUrxne98h5UrV87VSwJM2SfTtu1T7p05+Zrprp/NNZN99rOfJRqNlj96enpOOi4REZl7fZEUj7wywNOHR3mpN8rTh0d59JUB+iKpOf9ep9ODbCnS827+zXrrpImBim3bjI2N4ff7+dGPfjQnA2trawOK2av29vby8cHBwXI2q62tjWw2SzgcrsiSDQ4OcuWVV5avGRgYmPL6Q0NDFa/zzDPPVJwPh8PkcrkpmbOJPB4PHo9nlncoIiJn61QZq2s2t87pNOJy7UGm5938m9W79K/+6q8qPnc4HDQ3N7N9+/Yp04eztXbtWtra2nj00UfZtm0bANlslt/85jf85//8nwG47LLLcLlcPProo9xyyy0A9PX1ceDAAb7xjW8AsHPnTqLRKHv37uWKK64A4JlnniEajZaDtp07d/KVr3yFvr6+cvD3yCOP4PF4uOyyy+bkfkREZO6d66756kEm82VWAdnHP/7xOfnm8XicN998s/x5V1cX+/fvJxQKsWrVKj7zmc/w1a9+lfPOO4/zzjuPr371q/j9fm699VYAgsEgn/jEJ7j77rtpbGwkFApxzz33sHXrVq6++moANm3axHXXXcftt9/Od7/7XQDuuOMObrzxRjZu3AjArl27uPDCC9m9ezff/OY3GR0d5Z577uH222/X6hERkQXsXGesSj3I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YhARWcjagl4CbpPENJmwgNukLeitwqhkvul5N/9mVdSfTCb5kz/5E1paWqipqaGhoaHiYz4cPXqUxx57jE9+8pMnva69vZ3Vq1dz6NAhANra2shms4TD4YrrBgcHaW1tLV8zMDAw5bWGhobK10zns5/9LNFotPzR09NzprclIrKo+N1OrlgbIjApExZwm1yxNoTfPav/58sCp+fd/JtVQPZv/+2/5Z//+Z/5b//tv+HxePibv/kbvvzlL9PR0cEPf/jDuR4jAN/73vdoaWnhhhtuOOl1IyMj9PT00N7eDsBll12Gy+Uqr84E6Ovr48CBA1x55ZUA7Ny5k2g0yt69e8vXPPPMM0Sj0fI10/F4PNTV1VV8iIgsde31Pq7Z3MqOdSEuXhlkx7oQ12xupb3eV+2hyTzR827+nfbWSROtWrWKH/7wh7zrXe+irq6O559/ng0bNvDAAw/wd3/3d1PaTpytQqHA2rVr+chHPsLXv/718vF4PM6XvvQlPvCBD9De3s6RI0f43Oc+R3d3NwcPHqS2thaAP/zDP+R//+//zfe//31CoRD33HMPIyMj7Nu3D9Ms/i/v+uuv5/jx43z3u98F4I477mD16tX8/Oc/P+1xaisJERFZDvS8m3uzypCNjo6W97Osq6tjdHQUgLe//e08/vjjcze6cY899hjd3d38/u//fsVx0zR5+eWX+Z3f+R3OP/98Pv7xj3P++efz1FNPlYMxgL/8y7/kfe97H7fccgtXXXUVfr+fn//85+VgDODBBx9k69at7Nq1i127dnHRRRfxwAMPzPm9iIiIiEw2qwzZRRddxHe+8x3e+c53loOXb33rW/z1X/813/jGN6Z0wV8u9D8GEVlKktl8ue+Yf7xg3+92znhclg897+berP4G/Zt/82948cUXeec738lnP/tZbrjhBr7zne+Qz+f5i7/4i7keo4iInGN9kdSUjvw1bpPz22p5rX+s4nipoF81ZCKzd8YBWS6X4x//8R/LtVbvfve7ee2113juuedYv349F1988ZwPUkREzp2ZtkcC+Mm+XlaEfLgnlHyUtk26ZnOrMmUis3TGf3NcLhcHDhyo6M+1atUqVq1aNacDExGR6phpe6SMVWAonqXW56SpprLthbZNEjk7syrq/9jHPsb9998/12MREZEFYKbtkTK5AgDZ/PSlx+di26RkNs/hoTgH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"/>


```python
popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind = "bar")
```

<pre>
<AxesSubplot:xlabel='productId'>
</pre>
<img src="data:image/png;base64,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"/>


```python
# 데이터 세트를 읽어 옵니다.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df, reader)
```


```python
# 데이터 세트를 훈련과 테스트 데이터로 분할합니다. 
# 이렇게 하면 7:3으로 분리
trainset, testset = train_test_split(data, test_size=0.3, random_state=10) #code here
```


```python
algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)
```

<pre>
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
</pre>
<pre>
<surprise.prediction_algorithms.knns.KNNWithMeans at 0x2319c1097c0>
</pre>

```python
# 테스트 세트에 대해 훈련된 모델을 실행해 보겠습니다.
test_pred = algo.test(testset)
test_pred
```

<pre>
[Prediction(uid='A20PER3PX47JWZ', iid='B00DVHV7TW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3MXC41SY0VS7Q', iid='B0019CSVMW', r_ui=5.0, est=5, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='AFI1OCQJL300N', iid='B00A3YN0Z0', r_ui=5.0, est=4.265865531608766, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AF8HH6GQK5O8G', iid='B000CP4ML6', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AI0YK3KKHLTHN', iid='B005DOK8NW', r_ui=4.0, est=4.353169984653047, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A9919Z9E8A40S', iid='B0017TFVUW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A37M5ZMHCCSTN6', iid='B00172V6XK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A27WH8ZRW8AKBB', iid='B004WNGKF0', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AGTN05CEBVIKZ', iid='B009NB8WRU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1UP6XCLJWKDWF', iid='B001AO1SRE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1RPP2I8VHFWJU', iid='B003LPTAYI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AAALI9W6DSC1E', iid='B002J9G59U', r_ui=5.0, est=3.176470588235294, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1EUHFUW57B6F6', iid='B000TKHBDK', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1XJB1AFKWUXD', iid='B0050D1XMG', r_ui=5.0, est=3.75, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ARTUDDIJCO47', iid='B007PY3ZPG', r_ui=1.0, est=4.09433962264151, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A261DDG96HBQKN', iid='B00B0O1BWG', r_ui=5.0, est=4.737180815329655, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2VDFP4S3S63FF', iid='B001TK3D4K', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1R9BBBW5MJH6V', iid='B000OEV88K', r_ui=1.0, est=3.5643835616438357, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AVU5HD4JYPUUY', iid='B007FELOZO', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3CQ6CPZ9M96IH', iid='B001XHBNN2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1O1UE8SWIUH6U', iid='B002ZKTCUM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1R0D48X7AB3HT', iid='B004J6PI8K', r_ui=3.0, est=4.0, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2YEIHOA570JOO', iid='B0054JE64I', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A9WDTZG61ZOM9', iid='B006ZZ2V9M', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ANQS63GCRU1TR', iid='B0098F5W0Q', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1EVDWA956XRJW', iid='B0050SPZMK', r_ui=3.0, est=3.5641039869045312, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1CTKUEW3VZ0TV', iid='B0064L8Q1E', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2RLJ6LIH51I4T', iid='B000BPD330', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3AL3H4M74IGT3', iid='B002NEGTTW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3TZPXW8WR0UWN', iid='B005K7192G', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1QI6HTLI49CP7', iid='B008JJLW4M', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2EWM90Y3T2SUX', iid='B004I5BUSO', r_ui=5.0, est=4.41199684293607, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ATV56YMBUY58H', iid='B001GTT0VO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2AGZKL5N1MPR1', iid='B00D6PTMHI', r_ui=4.0, est=3.5345528455284554, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1RL1OTV8IJMVD', iid='B000Z80ICM', r_ui=4.0, est=4.370974871253287, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='A2M9GIR4CKZPV7', iid='B000EF3D4Q', r_ui=4.0, est=3.533980582524272, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2R2HS6F0JX5O7', iid='B001498LIO', r_ui=5.0, est=3.591743119266055, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1CR1XSR99YO0R', iid='B008HK3Y5S', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1D45EID9JU8J', iid='B0040720NY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1Q0HJFI9YY485', iid='B00E055H5O', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ALWHRHVQ8EI9C', iid='B006ZW4IVE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A7O9GWTHHGS94', iid='B0000UV2AW', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A26EF0FFHL68HP', iid='B00BGA9WK2', r_ui=5.0, est=4.075403608736942, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A22R6CXMBF617A', iid='B004T1YA5W', r_ui=5.0, est=3.2, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2EMNYT7SFKOOO', iid='B001MSU1FS', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2AAAUV0L6KVMA', iid='B001FA1NZU', r_ui=5.0, est=4.29, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3U0ZXAEVWRHWW', iid='B000W9DJ1Q', r_ui=5.0, est=4.258780034849597, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3VN9ZGCFSIWAD', iid='B00834SJNA', r_ui=5.0, est=4.0602409638554215, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1F98J95UB42Z9', iid='B000WOVD1Y', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1434F8KAR1W0V', iid='B00CAMCCLQ', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3EO4NL9SI2BK2', iid='B00DR0PDNE', r_ui=5.0, est=3.935356547671257, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3UEV6YFLSWA5W', iid='B0075W8Y1S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A6YNST64DV16U', iid='B000NUBY0C', r_ui=3.0, est=3.7162162162162162, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3NW0K0A5J664', iid='B005UA3I72', r_ui=5.0, est=4.694312796208531, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2S6XCFQCN27GB', iid='B004GTCA2C', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AYPF6B7H5XBWR', iid='B000U5TUWE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JKXNX3H2SW7J', iid='B006OBGEHW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1GXRUK2GVNHZ7', iid='B003S5SOLG', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A13AO2VJEATZYT', iid='B004LRPXAU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A61M4Q3JMVZGB', iid='B004616OIQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ASCU0TZ1WAAYP', iid='B0071BTJPI', r_ui=4.0, est=4.077433628318584, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3053LOLVONNW8', iid='B004KDVNZO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1Z2IC0CAI4U8S', iid='B003ZK5NZY', r_ui=5.0, est=4.461538461538462, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3UMINY2XEKTV9', iid='B0000BYDKO', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1UPG37DTOZFP0', iid='B004RRU1B0', r_ui=4.0, est=3.7887589214908806, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='ACI149LLMUJ4B', iid='B001RB24S2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AK4DEV5MJOYY1', iid='B004AD7UJC', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A7FXQ4HQM6JQ0', iid='B000V0DY8Y', r_ui=5.0, est=4.149253731343284, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1TZ0VDFSS1Z4C', iid='B0002IYOKM', r_ui=5.0, est=4.253012048192771, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1P518VRFAYY0R', iid='B004X8EODY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A19M2M6JQBOA1H', iid='B003TW77KC', r_ui=1.0, est=3.90363482671175, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A268GUARLO5TCE', iid='B00AW90T0U', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3TYYEG3B1GQ55', iid='B000WQ21SQ', r_ui=4.0, est=4.1678832116788325, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AYUYJTVD93CL4', iid='B002HWRJBM', r_ui=4.0, est=3.6445199309574763, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='A22PN8Z6FVNJSZ', iid='B00CFIDQZG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AUCMDESPGUXSH', iid='B007YKUWN4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A12LVYSQN3DLNQ', iid='B001PIBE8I', r_ui=5.0, est=4.175461741424802, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AK933TSWSLPMK', iid='B0022TSC5C', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AJTJ0DUWPI5IB', iid='B007Y8N19S', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1LI3TJQ4AUXOY', iid='B008AST7R6', r_ui=5.0, est=4.312622886354699, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A18P92GAMTFZ1E', iid='B002SFDJMQ', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WZNZVKRL7Z9E', iid='B004HHAE9Y', r_ui=3.0, est=4.276209677419355, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3595SIT4104FD', iid='B00004THCZ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AGYTMH5OBK530', iid='B0002U1TJY', r_ui=5.0, est=4.190804597701149, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2D37S73MP28G7', iid='B004GF8TIK', r_ui=5.0, est=4.418014589280051, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1XQRENRB92LAM', iid='B002V92X9Y', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JF0AXWKECBUP', iid='B005HP77RM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ARBKYIVNYWK3C', iid='B004WIUDGM', r_ui=4.0, est=3.9893934935990636, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='A2OJ6XZFKF65T9', iid='B000NVEG8S', r_ui=1.0, est=4.26875, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1JOQ0R3E2GFHZ', iid='B0074BW614', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3AY8UNEVOEJLI', iid='B007PV0LAQ', r_ui=5.0, est=3.931924882629108, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3C2GTCQYLW2LW', iid='B005JALQGI', r_ui=4.0, est=3.6285714285714286, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A11GTHJ4FY14L4', iid='B000FKQ8LA', r_ui=5.0, est=4.747474747474747, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A12FFX9GZJREMR', iid='B00752R4PK', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2H6M68M9YZHA0', iid='B003ZHV70M', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1CZ17TQRK6JGJ', iid='B0014S5FVQ', r_ui=5.0, est=3.735294117647059, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AE5O7S2H8X7HI', iid='B003L62T7W', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1F6G5D85FUVFD', iid='B003TFEHMU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AC6U04WHU0464', iid='B003EB0AXY', r_ui=4.0, est=4.2153846153846155, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3AE539R6FY38', iid='B0090Z3QG6', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2CXPQ1DZ6QGCP', iid='B0030LVHM6', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1189D1Y8FFXB4', iid='B003LZA95W', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2UMFF2OTRUUX1', iid='B000F8LQ0A', r_ui=4.0, est=4.301369863013699, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1JW7PDVDOTVQ9', iid='B0088PUEPK', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2RH0IB10OT4OV', iid='B008Z2661W', r_ui=4.0, est=3.1550094517958414, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AY3D7DG5L5WCK', iid='B00DQZOD8Q', r_ui=5.0, est=4.456623423631869, details={'actual_k': 5, 'was_impossible': False}),
 Prediction(uid='A2O55JT007Q6O0', iid='B00EL93M3S', r_ui=5.0, est=4.205438066465256, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3J6VTUHPAR9HL', iid='B00AGABISW', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2R6PJ570KC3MY', iid='B001O5CCQK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3F9I5273VCE9Z', iid='B00140DBRY', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3LDHIY79W6GWV', iid='B0037WNONS', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1TB1VVM0LTZM2', iid='B0007DDK7A', r_ui=5.0, est=4.218487394957983, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1SWMHJLEVHD6R', iid='B005UBNGY6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3DRKSHKZPZTGE', iid='B008CS5QTW', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AAGAW6J507ZK1', iid='B0013RTHEO', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3LQPV5D0T8XV1', iid='B000VM60I8', r_ui=2.0, est=3.3417721518987342, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A23YGKUGWVMU3U', iid='B000ER5G58', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A17P2DBVJKD196', iid='B000SMVQK8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ACSKNGOYVQNL8', iid='B00005Y1Z7', r_ui=5.0, est=3.7160493827160495, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2QTC05M62OAES', iid='B001FBM0OW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1AUEMVG6E42E6', iid='B00B1928FE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A34J0OYNSQG2C4', iid='B0000513O4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1HPVKJ2J7F4HD', iid='B000EVSLRO', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A5U59HQLVKCGI', iid='B006GDTTM0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A34TWLUP4XK8CC', iid='B001N2789K', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2WV816Z0MR2ZL', iid='B00IDG3IDO', r_ui=5.0, est=4.731543624161074, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1A0XELKJOWOQ3', iid='B001F7AJKI', r_ui=3.0, est=4.541733547351525, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2K27OFTPORP52', iid='B0041MY32Y', r_ui=1.0, est=4.247191011235955, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1F4DFQDVWMXR', iid='B001GCUTE8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A19B9W5QL6YBIX', iid='B006GWO5WK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3RWBRE6PYJYKB', iid='B003D78O1Y', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2X11CHNOPES2V', iid='B003ULJU2A', r_ui=5.0, est=3.4296296296296296, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1JVEATNF4GP94', iid='B004O0TRCO', r_ui=5.0, est=3.4393305439330546, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1UDEWPLNHGE88', iid='B009T5FY44', r_ui=5.0, est=4.029850746268656, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1WLNRTMDJ75RU', iid='B001L6LJJS', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2SEJ4OH1DTHMJ', iid='B004UZVDTI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A4FVAF2S3DAIB', iid='B007TAGX0U', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2GMJ29EIS3TZ0', iid='B002U1ZBG0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1SA8NY0QMVJ54', iid='B000F6SR0O', r_ui=5.0, est=4.59349593495935, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1589HG3K8U6H4', iid='B004R0RQ8S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3E86MBVVUX6SD', iid='B003NR57BY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3R5CKC1C57JHP', iid='B004HW73S4', r_ui=1.0, est=3.939622641509434, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ATI6S1R4HF4LL', iid='B009A5204K', r_ui=3.0, est=4.379393908079654, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3IBMW56ZISN9F', iid='B0012Q72IY', r_ui=1.0, est=4.688172043010753, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1OULT3UQ5HYNS', iid='B00006HCJI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3IEJ0JKID83HX', iid='B003FGWF04', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ASHHOQF9FQZ2V', iid='B000MWAKVU', r_ui=4.0, est=3.810126582278481, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3P8P49CSG79ZN', iid='B00A7EQQ3O', r_ui=4.0, est=4.365079365079365, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1LHP24N5UAMYX', iid='B007CZNS0U', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A31HGDJ8YBJJ45', iid='B001EBE1LI', r_ui=5.0, est=1.91283141138614, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1CPELS9BKO931', iid='B0013MWTB2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3VCU42YMAOUBM', iid='B00D601UC8', r_ui=4.0, est=4.32962962962963, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3EDYB5FJ2OLM2', iid='B007G9GT8U', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A18XGFYKRPZ5YY', iid='B005CPGHAA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2INYA0LKF4455', iid='B002LITT3S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ARIDN94LOCQFD', iid='B001FWYXD2', r_ui=4.0, est=4.244938004594164, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AVF4M4GATYI47', iid='B00023NDLS', r_ui=5.0, est=3.4598930481283423, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2B3CIEV3SW6EG', iid='B004N625AK', r_ui=4.0, est=3.7505945171247204, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='A2MCP6BSKOHM92', iid='B002G1YPIE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1VJN7640G9T59', iid='B004S4R5CK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2NPRWROOCNBP7', iid='B004CLYOHI', r_ui=5.0, est=4.978867623604465, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1DU5ZQKMSZKDW', iid='B0088LYCZC', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2D261TA4EJW1Y', iid='B0091UJRRM', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3EJJR9VA9VIV6', iid='B00EEBS9O0', r_ui=5.0, est=4.7995406910229175, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1SE5DK28LV8R1', iid='B004289ZW0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A11UX5C0TCCYFH', iid='B00EB7812C', r_ui=4.0, est=4.330097087378641, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2LZ8F6TNVQWAX', iid='B00841AGCO', r_ui=5.0, est=4.939918946301925, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3PK4PV5F31K8N', iid='B003U4VIXQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A38TT29T79FXIN', iid='B00A17I8LA', r_ui=4.0, est=3.298902070228021, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1NRBZMZS5QF8M', iid='B00192KF12', r_ui=5.0, est=4.7125, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2V9BYT0B0SNRP', iid='B00BS4KUCK', r_ui=5.0, est=4.006944444444445, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A75RK82U28B8T', iid='B0052YFYFK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='APTXQX6AZ50BB', iid='B0074FGLUM', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2KSMCH46TL9QF', iid='B002M3SOBU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AP4S0LFN71R2G', iid='B0009RKLMG', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2YH6HB8QMS5MD', iid='B00001OWYM', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JEY8FRBTI8OL', iid='B000JV9LUK', r_ui=2.0, est=4.35935397039031, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A12B3A3BWJK4F4', iid='B0076HMDQO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2VKTVEXXBSSCQ', iid='B006B7R9PU', r_ui=5.0, est=4.379310344827586, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1R5LZN1SEGPJG', iid='B007FUDKB4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1PE8Q6S9TKB93', iid='B00030CHRQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2Y5WX7S1TX2X0', iid='B0009O6IXA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AN8SHEH3M8CS5', iid='B007TAGX0U', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3BBNT1BVREC6Q', iid='B0081XAXXM', r_ui=3.0, est=4.075949367088608, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A22F2UXOKEOXWQ', iid='B00005NVBT', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A8NV3O97OY0M4', iid='B003OC6LWM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1VELOT636K6GI', iid='B00746W3HG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2S9HFO4K1IQ0K', iid='B005CLPP84', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ACS3H8PLLX7IP', iid='B000068NYF', r_ui=4.0, est=4.553846153846154, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2G225KBMRK2BJ', iid='B001P3PSSU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3EZ1975PL40OD', iid='B006CZ0C3W', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2JB5WQWNUXBAP', iid='B00IT1WJZQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A318IDAWJ9ZYAY', iid='B005GJC01C', r_ui=5.0, est=4.6521739130434785, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3ESYJAOF8SH1B', iid='B003TFEHMU', r_ui=5.0, est=4.201077199281867, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2NLNRWB1OBHCP', iid='B00BGGDVOO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A88Y4CZT54D40', iid='B00B7QC108', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AG15IN96V6T2V', iid='B0013G8PTS', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2900L97B91N3Y', iid='B006ZBWV0K', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AEJ8B5CGQRTMY', iid='B007KI8IMW', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1HZ1A9ATRQMCA', iid='B0089ZV1WY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3C8HXZKHAMO7N', iid='B007R5YGO2', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A31S0ELIIQCC3Y', iid='B004GCJEZU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AJ6K2U4OU5YX6', iid='B00DVFLJDS', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ANC6NHMNVVUWV', iid='B004XC6GJ0', r_ui=5.0, est=4.7716483848894375, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A27KXK28IGMKN0', iid='B00029U1DK', r_ui=5.0, est=4.171830985915493, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AY8CPUENCNT1O', iid='B008LTBITY', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A16BSOOMVMXGR9', iid='B00AJHCJ2Q', r_ui=5.0, est=4.213114754098361, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A7LNUIJYXEZPH', iid='B001DF2CQQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2KZKHZLVTPOLW', iid='B002M3SOBU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ACO22RBMUABV7', iid='B0002L5R78', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1YAV0278V7IN6', iid='B002RYYZZS', r_ui=4.0, est=3.5949367088607596, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AYCBJJUWHZKSN', iid='B001EYU3L2', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A15PWPL7EYJXDP', iid='B008E0VFZC', r_ui=5.0, est=3.723404255319149, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2I7BUP8HPJW6O', iid='B000MSS5YS', r_ui=5.0, est=4.418032786885246, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1SU2KPHEHQ1CZ', iid='B003ELVLKU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1N010N0NFT9J3', iid='B003DZ167K', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3KQCO30W766OL', iid='B00B9DQ2QI', r_ui=5.0, est=3.971014492753623, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ZGJTC34AJPMW', iid='B006OBGEHW', r_ui=5.0, est=4.479882955376738, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2HJ6OYB0ETCI', iid='B003JD6LVW', r_ui=5.0, est=3.190661478599222, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A92PGEAW9KGT8', iid='B004A7ZEI2', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2BFOGHCKVQWUJ', iid='B0043WJRRS', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A349LVCIYR9TOJ', iid='B002L6HE7S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2IIL50DM336RP', iid='B000KHPIO6', r_ui=4.0, est=4.492094861660079, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1RVKF8QG2BTH2', iid='B005LFT3GG', r_ui=5.0, est=4.71957671957672, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2OB0IK83HTU18', iid='B009PK9SB8', r_ui=4.0, est=4.417207554892846, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2VUJ79X4Y1D2Y', iid='B008GVL9YQ', r_ui=4.0, est=3.2145816072908038, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1IB14VRXM9NT6', iid='B002JSDHCY', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A100NTN4X2G3J6', iid='B0000AZK0D', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AUDY17PSQT28', iid='B00622AG6S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2NO5IEDTZZS4U', iid='B007RTACDM', r_ui=5.0, est=4.573033707865169, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1AC49B7I62O9G', iid='B004NYB68E', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AKT02NIXNM1RN', iid='B0015L0TBI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2VGJH6FVRGBGF', iid='B009GERY14', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1OMOAUSDQUAM7', iid='B001LNO722', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ZPKKWT79C6HW', iid='B0015HS1HQ', r_ui=4.0, est=4.143344709897611, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ZAHTMAQOJ8TZ', iid='B005HARR2W', r_ui=5.0, est=4.267605633802817, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AH3TM3ZL4XQQ0', iid='B006OS71TA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1FWLB7FG829RF', iid='B000067SMH', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATJ67Q3SSZ9RT', iid='B005HY4UPK', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A281JW8EH3JLE', iid='B008GGH5HQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A19UH9SLOO04BA', iid='B000VM60I8', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WX4CIGZ6ZHNG', iid='B00422KZQG', r_ui=5.0, est=4.7360406091370555, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3HKTJKJ3FT120', iid='B005FIFDSQ', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2400MOO10FSPW', iid='B0023RRNJY', r_ui=5.0, est=3.925233644859813, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1ED6D9APTDM1M', iid='B001MQA6K0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3FNCP5BUKC2WW', iid='B0007VTUB2', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AC28UWKJRSEHR', iid='B005PCOKEK', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3H02IH9AAGACB', iid='B00005B8M3', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A39IQR42BQYFG1', iid='B002VKVZ1A', r_ui=5.0, est=4.740259740259741, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3PR6YQUJ1P02G', iid='B008U5ZM6E', r_ui=5.0, est=4.222222222222222, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2IC2CQA7H061S', iid='B004N3XC7I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A14T3IEJX18CNZ', iid='B005UBNGY6', r_ui=2.0, est=4.23336853220697, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1VXEI16N2GC02', iid='B00068U44I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3VXL65WF0VGQ4', iid='B00264GYMG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1PV9IWCOLPBKX', iid='B00AQUXN6C', r_ui=2.0, est=4.56043956043956, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1UEOYHM62IFZN', iid='B003ES5ZUU', r_ui=5.0, est=4.702642867026428, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A38IZV2AO2KN68', iid='B009XN8NKO', r_ui=1.0, est=3.935483870967742, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A344U32NWZMLWD', iid='B0076HMDQO', r_ui=5.0, est=4.2592592592592595, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A11JRX5ZVB4U80', iid='B0041OSQB6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2A1EYZZSCSEDQ', iid='B000M4KXF6', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3QWN5AE4J3GVQ', iid='B002HMWQE2', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3BOYIYEPWX77S', iid='B008JGR9MO', r_ui=5.0, est=4.305681818181818, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3CM3GMHM3QJSP', iid='B0051D3KMG', r_ui=5.0, est=3.7083333333333335, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3SWUNO9FYZ70U', iid='B000G1D8HU', r_ui=4.0, est=3.4523809523809526, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4YY568BUFDX0', iid='B007TAMHRI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3SJW7KR70L33S', iid='B0074BW614', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3N7N8PTWEAP65', iid='B007VB2KIG', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A127GB0CJKS5T2', iid='B000JLK5PK', r_ui=5.0, est=4.033747412008282, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AFKF45OJML03S', iid='B00005T3N3', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='APN8BCFXDI59E', iid='B003CFATMY', r_ui=1.0, est=3.6705882352941175, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ANV9H9I9ERYYI', iid='B005KG44V0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ASMY60JR9EFI2', iid='B0010T8X9A', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2TYHDS8VWAJHE', iid='B0000AAAPF', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2U3CL0RHSLFGS', iid='B002KPGMXW', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1YAZY3KRGUL8V', iid='B000EZV3T8', r_ui=4.0, est=4.450331125827814, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2X3X96HVBCHZE', iid='B00547IVXM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3RKSU78ME5SO5', iid='B007OAFLOY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ZPU1PW7ZOMNK', iid='B00434OWDA', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3TJ2K4LP6V71U', iid='B000Z80ICM', r_ui=4.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2T1QPB4C3O3HE', iid='B000XHS4SK', r_ui=4.0, est=3.951646493199005, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A349XKWYAW7I84', iid='B004DDI0IE', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2Q0UGKMVOO7T3', iid='B00F2CWRLQ', r_ui=4.0, est=4.621119771402329, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A13P92RXIJKBAY', iid='B001UGMTKC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A269FH9V0X3593', iid='B004ZMG55I', r_ui=5.0, est=4.156521739130435, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3ILTXS25QE2G4', iid='B004CETK8S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2LZO0UPHD81DF', iid='B007TYUTY2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3T2WEQ9LXQYAV', iid='B00BMR7UPS', r_ui=5.0, est=4.9361702127659575, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1O2416LTJ81TI', iid='B00B9996LA', r_ui=5.0, est=3.799431009957326, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AM9BBAMJDHXL6', iid='B009VXH3UW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2JJLWP0GNGSL9', iid='B007GC4L7S', r_ui=5.0, est=4.723684210526316, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1AVFBYO3STGBM', iid='B006MRAVFE', r_ui=5.0, est=4.259740259740259, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ATYABHSTWNASF', iid='B000Z80ICM', r_ui=5.0, est=4.822501291464516, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='AQCO0PB0AQRXO', iid='B0041Q38NU', r_ui=5.0, est=4.391069823194486, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A24A47KIW2JYBC', iid='B003WQ2T5S', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3L1Q19L5QPN6Z', iid='B005QX7KYU', r_ui=5.0, est=3.989430894308943, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A365L8A48RNH80', iid='B0072B5E4M', r_ui=3.0, est=4.0625, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A194W7DHXJY03X', iid='B005972X3Q', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2I0KVAGC6YIYZ', iid='B000VX6XL6', r_ui=4.0, est=5, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='ATDPXZR9VNTME', iid='B00ETAU00C', r_ui=5.0, est=4.24, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A21JA9B4M56JQP', iid='B003YH9EZ8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2AKTBRML05X69', iid='B003YL3KUO', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WAT1BNSVC4JB', iid='B002GQRROS', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1YZP0JU46SM61', iid='B003ELVLKU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3GDVQVLORH26Z', iid='B004WIUDGM', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1KEYFTFOLXYWH', iid='B000NLSGA2', r_ui=4.0, est=4.426666666666667, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A28BMWP4M0Y95H', iid='B005GTR0R6', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2OX0L1P49SJRD', iid='B0052Z9HZ2', r_ui=4.0, est=3.924731182795699, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A29471XNU58PXD', iid='B00264TQQM', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2EVBZU30NJYS9', iid='B003QH2MY6', r_ui=5.0, est=3.5698924731182795, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1BC62X6HZJGZR', iid='B0056TYRMW', r_ui=5.0, est=3.6925467933789626, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A2FLFTRLWM4P3U', iid='B002MAPRYU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A18LEN5RY23IHS', iid='B009SYZ8OC', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2FQDTR1ZN5I3E', iid='B00FQ1NHA8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1183442JPOZSP', iid='B008DWH00K', r_ui=5.0, est=4.378531073446328, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A34AZ7U9E1IJS0', iid='B003U8K0N6', r_ui=5.0, est=3.5813953488372094, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1LKTQ1KL91IO6', iid='B0092KRAVQ', r_ui=5.0, est=4.503831417624521, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3P8CLDO9QU3WT', iid='B00020S7XK', r_ui=4.0, est=4.398666666666666, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4OH7YGNBVQ3X', iid='B003GSLE2Q', r_ui=5.0, est=3.65, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1LFT6ZAWDBZ29', iid='B0024UEVUO', r_ui=1.0, est=3.272727272727273, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1F0PVUT71338V', iid='B0031MJ70I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AO8GSL7E23ESA', iid='B00DIOALPE', r_ui=1.0, est=3.603795966785291, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2LOC0VXYAB4CT', iid='B004DI7DFU', r_ui=5.0, est=4.104347826086957, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ZT9UP11DGS0O', iid='B00426C57O', r_ui=5.0, est=4.662703917450569, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2N1E23FNQQRFR', iid='B004Z4FBE2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A25QJBK33C4O0R', iid='B005J31BCO', r_ui=5.0, est=3.995347394540943, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3A79K6Y4AIUEO', iid='B0038JEDAI', r_ui=5.0, est=4.22, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ZLHOJ5ZXTGSM', iid='B005LJQOPK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3HYMJ0SB3RN8T', iid='B00DQGIHZ0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A8KT10C7S433N', iid='B00B46XL50', r_ui=5.0, est=4.333333333333333, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2C2YAJQWO8BQ1', iid='B000V1VG5G', r_ui=1.0, est=4.1692307692307695, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1MJ0311NRIHAY', iid='B00395WIXA', r_ui=5.0, est=4.250513347022587, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4MG0X5RJZD7H', iid='B003Z80IF6', r_ui=1.0, est=4.299212598425197, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2B5DJ0IVKTBIW', iid='B000VOE466', r_ui=3.0, est=3.635869565217391, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A12DGP4ZVB5HVA', iid='B005KDYA44', r_ui=5.0, est=4.378962536023055, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AN55119FPY8FO', iid='B003CH77YK', r_ui=5.0, est=4.356643356643357, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AA39RQTNBFVBS', iid='B004QK7HI8', r_ui=3.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1L80UX2MUH5BQ', iid='B00BFDHVAS', r_ui=5.0, est=3.977777777777778, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3VP06HPCA7X68', iid='B001413D94', r_ui=5.0, est=4.717293233082707, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A16D0KD3THSFJT', iid='B0035KDK72', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A15YEWABO9ZMVL', iid='B005QBK5V2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AHH3QF60X2LJD', iid='B001P5GKBM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3LOALVHMA8GNW', iid='B00908BMVE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1D59XP7UEV264', iid='B003QCOKGO', r_ui=5.0, est=4.2727272727272725, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2TWJOS3X4A5BG', iid='B004ZIMU7Y', r_ui=3.0, est=4.4411764705882355, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AQI9AYI4AFK32', iid='B000065UDU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WJ6S32KCXL3C', iid='B00B5TELRI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A59WH7QIMCGT6', iid='B00B8KGTWY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1DMW823PFV79Y', iid='B001CEYYFK', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A12XB1L10WG270', iid='B000WYVBR0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A8C81CH94HIZK', iid='B0091PEC3Q', r_ui=5.0, est=4.098468271334792, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3D9MBWRXM81VK', iid='B009A6PJKQ', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1N2Z2UVPHNXKO', iid='B008LURQ76', r_ui=5.0, est=4.063408190224571, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A58ONS9WNBT9F', iid='B002G1YPHA', r_ui=5.0, est=4.496046027665565, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='ADHAMKKKMVU05', iid='B008R77ZCO', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AUPOA18DB1CTL', iid='B000FVDD30', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2QA31XZAIW6S', iid='B001DFX2OC', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2Q6TV13Y518D3', iid='B004LTEUDO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A4QCYNKAT2UN', iid='B0088LYCZC', r_ui=1.0, est=4.504412666565197, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AEG5XA48DX393', iid='B007136EBI', r_ui=5.0, est=3.9397590361445785, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A19HM61K6FOP0S', iid='B002SQK2F2', r_ui=4.0, est=3.0502793296089385, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AWHI1SO0226S7', iid='B004JOQSEA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2HPGHR6XJTWBW', iid='B00005ATMI', r_ui=4.0, est=4.3175, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AR0ID8UQUR3FI', iid='B004O0TRD8', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2DTM2TFJJLGRV', iid='B000FBK3QK', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A29CHHNLIZCV41', iid='B000RN1RXE', r_ui=5.0, est=3.3728813559322033, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A37D1ZP1LTQVV0', iid='B007YWMCA8', r_ui=4.0, est=4.040404040404041, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1VT986AIY7AMV', iid='B000UH8I66', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1HPA9TK89PW9B', iid='B007RFYEQW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A17UI5581FD4YC', iid='B003LR7ME6', r_ui=5.0, est=4.505329457364341, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2PPL1ZY0G5HRW', iid='B002JLJNV0', r_ui=1.0, est=3.932806324110672, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AQ85ZGQGC6XPZ', iid='B000B9RI14', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2WG310V8BKUEW', iid='B004J5BYTS', r_ui=5.0, est=4.132743362831858, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A34DQMQ8U9PUHQ', iid='B009LL9VDG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3E0N35AXMJ8GW', iid='B004UBU3SY', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A20FDWFCDWK5UW', iid='B004PYD950', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATTT5VLT7U426', iid='B000069K98', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AK8HJE65M4JRN', iid='B005C31H34', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A37QV6IU7JX7S8', iid='B00065L5TE', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3EAS6MRV6YHNZ', iid='B001CBLN7K', r_ui=1.0, est=4.161016949152542, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1Z17XGPRLZH6J', iid='B000UO6C5S', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2XPQ4ZYCE0QE0', iid='B003CJTR82', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3AQYACF4AM2KO', iid='B00535CD5C', r_ui=5.0, est=4.470588235294118, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='APHGZE8N8TCD6', iid='B003GCLGYS', r_ui=5.0, est=4.8125, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A290WQ5GOBL3WH', iid='B007R5YDYA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A267KS6E4QC1N6', iid='B0007XJSQC', r_ui=5.0, est=4.536000589416456, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A18EA7MNLXP9H5', iid='B0031ANZPS', r_ui=3.0, est=3.220779220779221, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2RHV42BTJSVON', iid='B007Y4TTWU', r_ui=4.0, est=2.7844155844155845, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1FJZOOVP3CI3I', iid='B000U5TUWE', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2UQ3YZHZ6A650', iid='B002RT8LJO', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A26Q9QH8LX5XY', iid='B0019EHU8G', r_ui=4.0, est=4.7555555555555555, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AF79D51F13FFZ', iid='B004XIT4NO', r_ui=5.0, est=3.991892089219874, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1C47ROOK7BSVL', iid='B000UXDHOI', r_ui=3.0, est=4.391304347826087, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2IQ7NYVGYQSCS', iid='B001GS8FZM', r_ui=4.0, est=3.592814371257485, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1KGM5CYLOZI6Y', iid='B0011TS8LM', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A312LE5GYERBD7', iid='B00AAIPT76', r_ui=5.0, est=4.61129207383279, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='APV2WCHC3ON99', iid='B00GXSEG4O', r_ui=2.0, est=3.8875, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3Q8CEYOG0CREQ', iid='B0024G48VA', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AOR53OOPXNUHU', iid='B002BH3I9U', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3R1Y8U9NJ1A22', iid='B00426FEL8', r_ui=2.0, est=3.636015325670498, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4HLDK32K7YCE', iid='B004BQTSKC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A22FAZDOVWCWQO', iid='B005QF2NCW', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2W9GX82SLKROQ', iid='B004J4VYEY', r_ui=3.0, est=2.124718665389902, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='A1K1Q6XI98C7QF', iid='B007KEZMX4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AHYP33WGESPWK', iid='B002PHM0XQ', r_ui=3.0, est=3.894230769230769, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1WB033X9T96AI', iid='B00AAKHCOM', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A130LX4ZZHZGMW', iid='B00004VX39', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A8GCDCKNYAGUD', iid='B008THTWIW', r_ui=5.0, est=4.237288135593221, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ZSCGZYSL51UJ', iid='B005DLDO4U', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A28NQM6FEFI3L4', iid='B00007E7JU', r_ui=3.0, est=4.813983265040059, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1DSFNSUA0ATRH', iid='B00ATQF0DC', r_ui=5.0, est=4.337142857142857, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AWJ52IAL6HWOV', iid='B002MYQTEI', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1B318YWMZACY1', iid='B000SNOT4C', r_ui=4.0, est=2.9844961240310077, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A6EYTU1TZ1IGH', iid='B004XY65WQ', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2L6ISR9UUO6GA', iid='B000NLSGA2', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2CZLOR0X70JCK', iid='B0063K4NN6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2A3BLPBHISRZ4', iid='B000U5TUWE', r_ui=5.0, est=4.37020316027088, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AT8ZGUWLJPEWO', iid='B00E0GNWMS', r_ui=5.0, est=3.758169934640523, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A10CONTO8EF57Q', iid='B00066EK3G', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1Z6QVIP10DCYT', iid='B00BQ4SBSM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3UIUKJN4EOHT', iid='B00005QFZF', r_ui=5.0, est=4.395061728395062, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1D33LH2SO6UZ2', iid='B005LS2J14', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2C87PW728A81', iid='B002R9CQYK', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A363BSK2SNBI82', iid='B0035B4LJM', r_ui=3.0, est=4.10632911392405, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3AB1F25L392LO', iid='B002VX0GJY', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2CFIISATEC56O', iid='B002MUGUFK', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A9DFQ943Z37WY', iid='B001963NZI', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1NZ2P2U3L4VG', iid='B00BBFL2X2', r_ui=5.0, est=4.372881355932203, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1YY7BIGN24C7B', iid='B00393THEK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3N1Z5WQUVODGC', iid='B0054X8C1M', r_ui=1.0, est=3.6310679611650487, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1R0NFYQUCIHM', iid='B001U3Y8Q8', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A28YCH2NGZSP5Y', iid='B004X49TAG', r_ui=1.0, est=3.7986577181208054, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2U8XSU8ZCVPX9', iid='B008R7EVE4', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='ALST0FE91XUYY', iid='B002JCSV8A', r_ui=5.0, est=4.529411764705882, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AL1J1D50X0DQA', iid='B0052DYWU4', r_ui=5.0, est=4.105263157894737, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A25XKO3B7W6U45', iid='B00DULMXTW', r_ui=1.0, est=4.0, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A18CVRV9BOBV7X', iid='B009WSCW4S', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATLP1FM568THE', iid='B000TQPTTM', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2HNXE3GS0N3BX', iid='B005S6XUXA', r_ui=4.0, est=4.1268758526603, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2H5NDVV3WL1HT', iid='B006GWO5WK', r_ui=5.0, est=4.989778270680145, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1POL0R3RNLZ53', iid='B002QEBMAK', r_ui=4.0, est=2.082002603938082, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1016Q5UDME15Z', iid='B002K8A75I', r_ui=5.0, est=4.52054794520548, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AAQBHHPDKMEER', iid='B00B7N9CWG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A14XULA608M2V8', iid='B0002GMDQG', r_ui=1.0, est=2.6451612903225805, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AM6SEQ609F0X8', iid='B006U1VH2S', r_ui=5.0, est=4.580260864568769, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A36QYO0OAN52AV', iid='B004G6002M', r_ui=5.0, est=4.186629526462395, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2H9DLPZY8AJNY', iid='B0028LK6IU', r_ui=5.0, est=3.8078431372549018, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2XY2RQBZOKFYS', iid='B000GAUZFO', r_ui=2.0, est=4.086956521739131, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A9M69KQAD1PYS', iid='B00149PA42', r_ui=2.0, est=3.817142857142857, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AY7OJZBZOAN6C', iid='B0034CL2ZI', r_ui=5.0, est=4.233269598470363, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2V8RH4X4TMMTY', iid='B005FYNSPK', r_ui=2.0, est=4.4568720379146916, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1Y6B7YYZ5K7SS', iid='B003G2Z1M6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1P4H9IB2GUTRL', iid='B001A4HAFS', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A226WYZCHGNE0T', iid='B00542PJTQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AZ2MUNC5MVZ1R', iid='B003NREDC8', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A30J7WQV0ZNRXG', iid='B0072B5E4M', r_ui=1.0, est=4.0625, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ATGQ1PP3EZ6J6', iid='B0043JDU56', r_ui=3.0, est=3.246376811594203, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AOO2ZBQXO0SWH', iid='B00713AA5E', r_ui=4.0, est=4.074074074074074, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1MC6BFHWY6WC3', iid='B001W1TZTS', r_ui=5.0, est=3.8177083333333335, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1MC9CG1VCUOWH', iid='B00007KDVK', r_ui=5.0, est=3.736625514403292, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AK3GKIV8DEY8B', iid='B00065ANYC', r_ui=4.0, est=4.3798076923076925, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2RAYCUTJT088B', iid='B000E6G9RI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ABY421XP1UYCX', iid='B0039RW9WS', r_ui=5.0, est=3.9523809523809526, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A23VBIAH8URA51', iid='B000SOQ6KQ', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1UMTYJSDJ6FR9', iid='B001HSOFI2', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A35SNSDRNE9P8Q', iid='B00004ZC9V', r_ui=5.0, est=4.395973154362416, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2NJMSMQ1Z0UOC', iid='B006ZBWV0K', r_ui=5.0, est=4.545243619489559, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3CU8A5WNQBKYF', iid='B009MAKWC0', r_ui=5.0, est=4.167603344959377, details={'actual_k': 4, 'was_impossible': False}),
 Prediction(uid='A3JYSPSG07OW4E', iid='B00DMS0GTC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A14G6XVOYKESV8', iid='B001GGL7Z4', r_ui=5.0, est=4.30635838150289, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AIB1TXN9L5JA2', iid='B00EZPCWWA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ABPOYED44WCP', iid='B000066CCU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2U8WES162T1A5', iid='B0000XOB7U', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A10DAX20MEBMLK', iid='B005DOK8NW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ACP0NVQ4TV38E', iid='B006ZP8UOW', r_ui=5.0, est=4.08889536578257, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A365S0NLE69831', iid='B0086UXQES', r_ui=4.0, est=4.453333333333333, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2AIPUXFQOBU2H', iid='B00752R4PK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A5FQ7DOUC6PY3', iid='B000V1VG2E', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2GQFW1HVDH9PW', iid='B008OO41P4', r_ui=4.0, est=2.8, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3BBRVJHAOW6SE', iid='B003CGMQ38', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3IH9LSCTYPBUH', iid='B004QBUL1C', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A19IJ6MIJHTKL4', iid='B004P15HD0', r_ui=4.0, est=4.2936507936507935, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AJLMTRCMUXKWR', iid='B0015F1L7A', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1V5LNNI5116KO', iid='B003K1EYM6', r_ui=5.0, est=4.198511166253102, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2KTMPLK5NNBDF', iid='B005GI2VMG', r_ui=3.0, est=4.0436507936507935, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1L5F0YLUTTS85', iid='B003ZUIHY8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AXGMJ32BB8YCX', iid='B004ZP756S', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A29EUR7UXCZMV1', iid='B0007LCLPE', r_ui=5.0, est=3.7131782945736433, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2EMCVIANJ46A1', iid='B00F6E8OUS', r_ui=5.0, est=4.444827586206896, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2QXRID3Z2Y7PD', iid='B008HY8XTG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1BNZ1YYVZOCKH', iid='B003I4FHNA', r_ui=5.0, est=4.397872340425532, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2BRJ1JRNRMLTV', iid='B005FN5DJA', r_ui=5.0, est=4.391304347826087, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2AH1N9QB595D5', iid='B005NF5NTK', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2KCA9V8NV7EMN', iid='B003JUN9YW', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WVMBKLDH2XJ2', iid='B001D60LG8', r_ui=5.0, est=4.184965380811078, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AAZMZS2ZJEVI5', iid='B000B9O83A', r_ui=5.0, est=4.373056994818653, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4LC2671SUPT1', iid='B002D41HKS', r_ui=5.0, est=3.8974358974358974, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A30GYWED56V17J', iid='B0012GQZZU', r_ui=4.0, est=3.662113748320645, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AX4A4234XKOUW', iid='B000XQRAI6', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A53OG3Q99WAVL', iid='B0012WXFPM', r_ui=4.0, est=4.197530864197531, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3F99LRE32DG5X', iid='B0002WPREU', r_ui=5.0, est=4.953027879907012, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AFNBZ6L517NOB', iid='B008BWL4MW', r_ui=3.0, est=3.5961538461538463, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3KR8MYPMPZOCX', iid='B0010TEOLQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A61O8S2173O5S', iid='B00AGABISW', r_ui=5.0, est=4.678082191780822, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2NTFH1NIPI9ZN', iid='B000NVVDKC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1G8MVPVWLOG6M', iid='B002WE0QN8', r_ui=5.0, est=4.831397174254318, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='AF4ZXRVQD8JDM', iid='B00EAY7MBM', r_ui=5.0, est=4.421232876712328, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A17JAB82HTXVZ0', iid='B000HAOVGM', r_ui=5.0, est=4.491606714628297, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2XQP0O6P7JB8V', iid='B005ONMDYE', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2U9LQBSLXY2KM', iid='B00001P4XA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3V9EIKTZ5BGKI', iid='B000EWJYYW', r_ui=4.0, est=3.6049382716049383, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AD000UNQRH2CA', iid='B001KLEUOA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A22RY6AVGS4WHK', iid='B006U5W49O', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2H9T7F8JPT32O', iid='B0038W0K2K', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1N7LFD6NRTRUS', iid='B00AZCGF7K', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3VW8VT32AZF2W', iid='B003VWZFRW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2U5Y74LP0A7X8', iid='B0045EFZUM', r_ui=5.0, est=3.8666666666666667, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AW2QDBFE3GXFF', iid='B00CMM1PI0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3IM7JOR2QWV6W', iid='B0088LYCZC', r_ui=5.0, est=3.9345114345114345, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ABXZHZK97SGKY', iid='B00FNPD1OY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3EGDOLB1P9TDX', iid='B003DZ167A', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1JNOBM32R74D1', iid='B005PXMKI2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A182WJO263DAJD', iid='B001DUQU0A', r_ui=3.0, est=4.148514851485149, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ARM9JQHVXNMN7', iid='B00DR0PDNE', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3DKXGIDV1B514', iid='B0016D1I0G', r_ui=4.0, est=4.301724137931035, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3V8QNJI2EG8HW', iid='B002K42W4Q', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2VH270KTWRTVW', iid='B003V42O6K', r_ui=5.0, est=4.297709923664122, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A35TZ81LT65ICQ', iid='B003XM1WE0', r_ui=5.0, est=4.804878048780488, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A34KEK65UQKSWE', iid='B003DZ165W', r_ui=5.0, est=4.306242274412855, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A6ZEL2ECVHNWD', iid='B007UNULT0', r_ui=5.0, est=4.382513661202186, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AI7CXVDW8BS59', iid='B0014Z29OU', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1PW7IQ6X5SJPL', iid='B001L6LG5K', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A48WD3S5X21MU', iid='B009VL9YGU', r_ui=5.0, est=4.073170731707317, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A17BGWK8T9GTLS', iid='B00BWLL9N8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A32OL90H37H76N', iid='B002Q887BS', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A22R4823SR8211', iid='B00D5Q75RC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1PAZNH4HEGSAG', iid='B009OX22B4', r_ui=5.0, est=3.5902777777777777, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A191MLR7SNDQU', iid='B00FJ8JC8Y', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ALIEILGEE5ZDT', iid='B005Q311OK', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1P631KG2W2A8', iid='B00DR0PDNE', r_ui=5.0, est=3.935356547671257, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AFDOI8OG6C3FE', iid='B0056HNTAU', r_ui=5.0, est=4.601246105919003, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A15MGYDDG3F8MG', iid='B004CVSTVU', r_ui=5.0, est=3.2954545454545454, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A18NOHFD1NRXHB', iid='B009NXFLWW', r_ui=5.0, est=2.994430325347359, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A3ATDZESH9V4SL', iid='B002K40R6G', r_ui=3.0, est=4.57772815331253, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3SDDMEUMCN858', iid='B0041RSDXE', r_ui=3.0, est=4.312632057216628, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A3LEDV9VQYWLTW', iid='B0015DYMVO', r_ui=5.0, est=4.107532210109019, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3Q5NFSOUT4UKA', iid='B003GSLE1M', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A153F3QOGIUSRD', iid='B001G5ZTPY', r_ui=3.0, est=4.47191011235955, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1HIO67M6NGCRJ', iid='B000R9J5OG', r_ui=5.0, est=4.404958677685951, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1X9TO2AXELLKU', iid='B002J9HBSE', r_ui=5.0, est=4.27065527065527, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3B1DE4AGQA4WF', iid='B00B588HY2', r_ui=5.0, est=4.562805872756933, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AXFUMZYEVI4Z0', iid='B006MRAVFE', r_ui=5.0, est=4.259740259740259, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1R6WC4MYJZF6K', iid='B005CG2AL4', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JB1EUKXGJL0', iid='B005HMO6A6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1KY9WUM96VKA9', iid='B0055D66V4', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1G9B1RNQIOIKC', iid='B003LSTD38', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A22LU96KC709NH', iid='B003E2TQI8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A31WBY15IZTN7B', iid='B001EZRJZE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AYNTHGBSRGEG', iid='B009BEXSNW', r_ui=5.0, est=4.348555452003728, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AV6TPBFK8B8OS', iid='B009FU8BTI', r_ui=5.0, est=3.9460580912863072, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A33RS0UBTINWGX', iid='B005PXMKI2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AO6LAZJ2V1IU9', iid='B005P99KWU', r_ui=1.0, est=2.30188679245283, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1MP58OMXBH52K', iid='B00CU2K35I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ZAM559YD2O88', iid='B00BOHNYTW', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A114HR79TYVHR6', iid='B001C219C8', r_ui=5.0, est=4.206030150753769, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2JPGNKUBZB29V', iid='B00021XIJW', r_ui=2.0, est=2.923679060665362, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A16PHHMO2WWYEB', iid='B004HHICKC', r_ui=5.0, est=4.391810517864312, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2SN2R1PBOG46Q', iid='B009CQOXTC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A7JGEKN140F4S', iid='B003VANO7C', r_ui=5.0, est=4.320406278855033, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2LIH5ZPOC8JO9', iid='B004FVMKV2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATGO2APOGM78Q', iid='B008R7EVE4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ULAF163Z7PM7', iid='B000097O5F', r_ui=5.0, est=4.576744186046511, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A9O20WH04SMBS', iid='B009NB8WR0', r_ui=5.0, est=4.844488188976378, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A212K97UBVIOHD', iid='B007PJ4Q4A', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ANKBSC1OU36UY', iid='B00BBHN0RQ', r_ui=5.0, est=3.7549019607843137, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AE3V6RUQT1GTO', iid='B000FJJASO', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1N1NC5OXK2PCD', iid='B009A13IB8', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3RFFRT86GUSIF', iid='B004E10KFG', r_ui=5.0, est=3.878000979911808, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1JPVLJ3Z4K0F1', iid='B00009KH63', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A4DMHUNXXGJ2T', iid='B001DFZ5HO', r_ui=4.0, est=3.771604938271605, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2I1JZ1XHVIYDT', iid='B000CRFOMK', r_ui=5.0, est=4.451086956521739, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AFQPQ8XGD3FKY', iid='B0074BW614', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1DU4G2VLXPJDS', iid='B00D02AHEO', r_ui=5.0, est=4.5060975609756095, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AQZYNJ5W7UDLK', iid='B007PRHNHO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AZ1Q1P7RLWKS', iid='B0063Q3G3I', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1Y51RANSRYQHP', iid='B0000C73CQ', r_ui=5.0, est=4.941860206070732, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1072SLLQ4GYUU', iid='B007B5ZR4G', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AQKTCGNSOSSRL', iid='B0040JHMIU', r_ui=5.0, est=4.557377049180328, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2NUME7NEYG9JG', iid='B001EYU3JO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A19YT1QY9673ZX', iid='B005BUDSGW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2A8IY4GTR7GWQ', iid='B004CLYEE6', r_ui=4.0, est=4.039562091187403, details={'actual_k': 4, 'was_impossible': False}),
 Prediction(uid='ANUV35Y318CJV', iid='B009SYZ8OC', r_ui=5.0, est=4.227865244625021, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A320GY0YFX8VHW', iid='B0030MIU16', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2UIVMG14PC6QD', iid='B003FVVMS0', r_ui=5.0, est=4.847432024169184, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A37IBUI8R3AHF7', iid='B000065BP9', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A30JX5UMHGHPY', iid='B0047XRVWQ', r_ui=4.0, est=4.520370370370371, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3UFYWINH5G5KG', iid='B007M50PTM', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3IAU6ZPVF2R5K', iid='B000EVM5DK', r_ui=1.0, est=3.899267399267399, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AVIH70O2D1J5R', iid='B0031RGKVC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3BBA3IA36YJUH', iid='B006JSR4QU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A25EHXGAXBJBUE', iid='B007FL7GGS', r_ui=5.0, est=4.088050314465409, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ARQXKR5RB00T3', iid='B004OBZ088', r_ui=4.0, est=4.051724137931035, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AQIP2Q5JORD8R', iid='B000CKVOOY', r_ui=5.0, est=4.700152207001522, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3U2T6T0GGE2JP', iid='B0030UL7IG', r_ui=5.0, est=3.780821917808219, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2I1I7DGJHMKK2', iid='B00008VF5W', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AZ6N6C3TQZL3Y', iid='B002W7U3E2', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A38BZNIKZORS0A', iid='B008D4X4GW', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3L30T0VFGDECI', iid='B000SMVQK8', r_ui=5.0, est=4.377427184466019, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1A5N8KGKLAI8O', iid='B0062IPIPQ', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1QPFB16EIUKIT', iid='B004PEIG12', r_ui=5.0, est=4.152626362735382, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ALBAWGOE1KXKR', iid='B005RFOJT6', r_ui=4.0, est=3.070588235294118, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AK2HY2VWM77IB', iid='B004H9C4JK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A289D4W6XK6NGX', iid='B00B7E1D7W', r_ui=5.0, est=4.253549695740365, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2IAGYPSHXU9TQ', iid='B000NNFS4C', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3L67QUVQJOXCV', iid='B00017LSPI', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A351OZ9ZZKRU4K', iid='B003ES5ZUU', r_ui=5.0, est=4.702642867026428, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A19123D9G66E0O', iid='B0000A2QBP', r_ui=5.0, est=4.595918367346939, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AL41JU3AWH5TD', iid='B0099XGZXA', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A141FO19SUYFUF', iid='B008JGR9MO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A157HQWB17JVWG', iid='B0053VZUW4', r_ui=1.0, est=4.3893129770992365, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1PVJOH1KB4H60', iid='B002ZIMEMW', r_ui=5.0, est=4.667013527575442, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1GJZZWBYC0CT1', iid='B0021L9C0A', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A4PW78Q5BN306', iid='B0047XUFH4', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A5OJ1QGW2MMKB', iid='B009YC3Y08', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AEA1O1YK6R8CL', iid='B008OHRJ32', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1Y0H8SEE8076B', iid='B0039BPG1A', r_ui=5.0, est=4.415271265907569, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3AKXQCLUB1D7D', iid='B00DJE33AI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A66BGXSWMDMZS', iid='B000UEZ36W', r_ui=5.0, est=5, details={'actual_k': 4, 'was_impossible': False}),
 Prediction(uid='APZRFQTXKYVLL', iid='B00D5Q75RC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3TAQVX4HF41A4', iid='B009VN9F0S', r_ui=4.0, est=4.135702746365105, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ADW5Z87Z64H3P', iid='B00748IJ2M', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3L3XO8MRHMA89', iid='B004GF8TIK', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3HFA497LX7BW3', iid='B0045JHJSS', r_ui=3.0, est=4.1638795986622075, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AXP6A2Y1IM5JJ', iid='B0092KRAVQ', r_ui=4.0, est=4.503831417624521, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AXBBVFBI1YG62', iid='B00DBX371C', r_ui=4.0, est=3.9019607843137254, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AZGC4124KZEWZ', iid='B00AFUKXCU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1RII5VTF8QDBW', iid='B0089DZNS4', r_ui=3.0, est=4.490445859872612, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1XNIVLZSS3NIN', iid='B005ARQV6U', r_ui=5.0, est=4.280130293159609, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A26M1R1JQTTPS1', iid='B001M5BIX0', r_ui=5.0, est=3.9689119170984455, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3J0EA7CJBO1NP', iid='B0093HMKVI', r_ui=5.0, est=4.448160535117057, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A25MB82PRJIJNB', iid='B009WU5XUG', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1CGSPL74SHZG5', iid='B00803WNOK', r_ui=5.0, est=4.015037593984962, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2H1UN6JMVJ4NO', iid='B007F85R30', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AM6ZG5ORQK5SA', iid='B000CSQRYS', r_ui=4.0, est=4.276190476190476, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AQ11X453S4C8S', iid='B003IE49T8', r_ui=1.0, est=4.391304347826087, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2UU6A74JUJODA', iid='B001QWQDPC', r_ui=1.0, est=3.6625, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3PORXJM4BRG86', iid='B000TMI17I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3NRDH1UE681T0', iid='B003LWXJ2A', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1W4500HT7GGD6', iid='B0000B006W', r_ui=5.0, est=4.6092715231788075, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A11ZPB1C7YORKJ', iid='B004YW79F4', r_ui=5.0, est=4.548223350253807, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3HT0ZCE30LGG2', iid='B004HIN7SI', r_ui=5.0, est=3.685082872928177, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2ZKJHIX3L0AS', iid='B004I5BUSO', r_ui=3.0, est=4.41199684293607, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2D0LH54S7B9G8', iid='B000AYJDD6', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3HZBKRLSWX24A', iid='B008CS5QTW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3KR42X6V874C5', iid='B004COCMRO', r_ui=5.0, est=3.808333333333333, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1J3UZ0LR406C5', iid='B008JJLW4M', r_ui=5.0, est=2.726621611124058, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2VEU7Y09941IH', iid='B0098PRKA6', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3OW6KUQYQIGDV', iid='B007PPYXOC', r_ui=5.0, est=3.6762402088772848, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A234U1ZU1Q937F', iid='B005GSRKT0', r_ui=5.0, est=3.6666666666666665, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1Z9B9UCG0MG1D', iid='B000AMPXN2', r_ui=2.0, est=3.127659574468085, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ABX7SEE84DMLU', iid='B0001DYXOU', r_ui=4.0, est=2.9156626506024095, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A14QIFMKA9756G', iid='B005UG3KS8', r_ui=5.0, est=4.194915254237288, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1O4GF8JXLNEWH', iid='B00005T39Y', r_ui=5.0, est=3.9927884615384617, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AK6UKXIVNYXP0', iid='B009HISC3I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1AXO8VSNXH3B5', iid='B001U0O7SA', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A5BWLE1UZB9AO', iid='B000LRMS66', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2LRGO2BWN6DGG', iid='B000F7857S', r_ui=5.0, est=4.1675042297961955, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2I3BQIA3U672U', iid='B004J3V90Y', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2MI2KIIFAVO5K', iid='B0049VVQ9U', r_ui=3.0, est=4.073529411764706, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A14QPGZPT2AUQH', iid='B004HNCRNO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A17GDDTTARVM0N', iid='B009X5BBT2', r_ui=3.0, est=3.6904761904761907, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2C92ROUSUJHLB', iid='B00198BY48', r_ui=4.0, est=4.4576271186440675, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3DCBAOXOL0DA7', iid='B0067G6PKA', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A9840X4IJFZ7Y', iid='B003CH77YK', r_ui=4.0, est=4.356643356643357, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3J7T3P0WSD97L', iid='B003GIJTR8', r_ui=5.0, est=3.7774740295243303, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3H8MAJCMR86CO', iid='B004XZHY34', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A33GZWCRJJSOK3', iid='B000N4WRFY', r_ui=3.0, est=3.7304347826086954, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1VFSEBVYD0IYM', iid='B0095ONNTC', r_ui=5.0, est=4.212371134020619, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3LI2KFMXR68XK', iid='B001FA1NZK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AYYP0FF77Y13C', iid='B000BQ7GW8', r_ui=5.0, est=5, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A2TTM4086B05EN', iid='B0015AARJI', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WRYBM2ZYB0PW', iid='B00012EYNG', r_ui=5.0, est=4.519607843137255, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1HXBZW8J3R5DY', iid='B00DR0PDNE', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A23M9849DE5L3N', iid='B000HDONV2', r_ui=4.0, est=4.478260869565218, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2FS3VL9BD587V', iid='B0051PGX2I', r_ui=4.0, est=4.101239669421488, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A312TVY5P8489R', iid='B0026ZPFCK', r_ui=5.0, est=4.88710121603828, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A777CAX59NW2A', iid='B00186YU4W', r_ui=3.0, est=4.108695652173913, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A25IMOJF5EWTAB', iid='B001QBG614', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3UOSNL1HUUCUT', iid='B0093XTHHM', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATJR32HT32JIQ', iid='B003ZUIHY8', r_ui=5.0, est=4.641006931776724, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ASL98T4B8CFLT', iid='B000EDK8V4', r_ui=5.0, est=4.4148148148148145, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1QM9B8VEX9G5G', iid='B004V4K4SO', r_ui=5.0, est=4.168269230769231, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AOSSGWETCTP6W', iid='B00AWX6EYQ', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A29FR400FEXXGZ', iid='B008RNQEUW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3LCO0KUI9YC1D', iid='B001TH7GSW', r_ui=5.0, est=4.625899280575539, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AZEZ7XDV5LB5V', iid='B00C94GTJQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2M5OV2SUEUCH2', iid='B00DE0EPCM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AWZRQ3MSIUY34', iid='B00A1A4KHS', r_ui=5.0, est=4.570422535211268, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3NZ5M3T45U18F', iid='B007PRHNHO', r_ui=4.0, est=4.418287937743191, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2H2ZZEC8D0Q', iid='B007B5WHTE', r_ui=1.0, est=3.936046511627907, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4FPN5LGPPOXZ', iid='B005HSG3TC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1THVRRI9ZYCCZ', iid='B0038A9HSK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2F1TJJTGBPGG6', iid='B003DSCU72', r_ui=3.0, est=4.264297337581533, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A12S8UQEIPZJKE', iid='B000ZJZ7OA', r_ui=5.0, est=4.401639344262295, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A37W6P18V841AJ', iid='B002RWJD7A', r_ui=5.0, est=4.930663818691226, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3UNBK7XUJF6R2', iid='B000P9CEV4', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3DPB9LS55U0SS', iid='B0034XIL60', r_ui=5.0, est=3.5127551020408165, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1S716DLZH3DF3', iid='B003LR7ME6', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2IFQR1IVYRCJK', iid='B000BBAKSA', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AD9ZL63H0VKCZ', iid='B0059AK8HQ', r_ui=4.0, est=3.906474820143885, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AN7ESNCB95861', iid='B0009Q4PH4', r_ui=3.0, est=4.063106796116505, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A4WABIZNRFGTJ', iid='B0063KGRBW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AQIJ3ZEEVCIKU', iid='B002JM1V6O', r_ui=4.0, est=3.326500817424996, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A38SB6QSRT1DZ1', iid='B006ZTMEZ4', r_ui=5.0, est=4.823661735549794, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A14N9IXFFCUP3E', iid='B00AASPQLU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2B1KH2I8FDYQ7', iid='B007FSRSZU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1BHW7WCBYWV7D', iid='B00BX2YLVI', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A31DOHQ16EM4SF', iid='B001D60LG8', r_ui=5.0, est=4.184965380811078, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2HSGHTVZNPRG6', iid='B00BIFNTMC', r_ui=5.0, est=4.294007490636704, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2VUQZECNWGF1', iid='B0086V5TVU', r_ui=5.0, est=4.707792207792208, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A228OBD59D3W3D', iid='B001AVIQOK', r_ui=2.0, est=4.087912087912088, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A73TWVE9DJF6K', iid='B004G6002M', r_ui=5.0, est=4.186629526462395, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A26CCLHNEF9O05', iid='B00C5AW7ZO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A20L461CTZ2M5H', iid='B0015AARJI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2GIIXIDSZDK4V', iid='B0011X5I7U', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JDHC4S8CXCKF', iid='B00FFJ0HUE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WM5SUPVBNW4J', iid='B008CBQSKU', r_ui=5.0, est=4.365, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2NPD3OLATX6QI', iid='B000O1FTYC', r_ui=4.0, est=4.715686274509804, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2IA925FJXP26E', iid='B001NIEK3Q', r_ui=5.0, est=4.548022598870056, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AD1IBZH2GNGY7', iid='B007WTAJTO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3O5QHVZI9X1D7', iid='B003MVMTDA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1EWUBNPGTJTGA', iid='B00DVFLJDS', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A14A4YYKPLYY26', iid='B00005T3XH', r_ui=4.0, est=4.037433155080214, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AOVCMUDN30YE', iid='B008P8FDEW', r_ui=3.0, est=3.217391304347826, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AEF39TS5TY3RX', iid='B001UHMCT4', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3ID0ELJFTXJAX', iid='B000V0IE66', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1V4664XHDNM6R', iid='B002CTV060', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2T5ZD2HC9O9K3', iid='B0096T9ACU', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A311H90T52PTUW', iid='B003VANOI6', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AWTMK5Y15U3OT', iid='B004YU6TFC', r_ui=5.0, est=4.005235602094241, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A129CI0G6F0TV9', iid='B0052YFYFK', r_ui=4.0, est=4.305793712652637, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A178W7LO01MOFS', iid='B005KDY8AU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2UALR606NA2BS', iid='B009YN8998', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3QGFY7KEKPI1T', iid='B000095SB6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AZRWLXDPXPBKX', iid='B0049II7W2', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A31V5E1J0T5201', iid='B0014DURIW', r_ui=4.0, est=4.1891891891891895, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A399UYHSKC11HF', iid='B001QGT1CA', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1SHBW6QORA3S9', iid='B00BJH1DRW', r_ui=4.0, est=3.7325581395348837, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='APOUV52VYRBIE', iid='B00AU0HMGA', r_ui=5.0, est=4.283018867924528, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2LD7X71EI6M9L', iid='B000W2MW7U', r_ui=3.0, est=4.4222222222222225, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A26X1DKMFM0NY1', iid='B0000632H7', r_ui=5.0, est=4.645161290322581, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3P9F0LIBMKX0F', iid='B0055QZ216', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2NWDFUODEQJWU', iid='B0097BEFYA', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3R35MFRXP228R', iid='B007IV7KRU', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3QQKFY49G1GXS', iid='B0017Q2W6G', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A214TSUEXJ2Y7E', iid='B0015AE4CE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3GCY69BCZW4VO', iid='B0080YBH8M', r_ui=4.0, est=4.614601018675722, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A20X9NDSET1JAV', iid='B000I68BD4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2M7ZC9SL5CWZR', iid='B001SEQN3U', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1FAS12PEI6XC8', iid='B00119T6NQ', r_ui=4.0, est=4.309579439252336, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3HNYHB17EA931', iid='B001LYX3MQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AOU9MWIUJBQXC', iid='B0061JPXLU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3FX0YLQR245Y2', iid='B000062VUO', r_ui=4.0, est=4.316005471956224, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AK88BSXW84UP1', iid='B0011FQURK', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A33JOJI27299ZM', iid='B004HCKRKA', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AOCAPVEDHC9ZD', iid='B00BGA9WK2', r_ui=1.0, est=4.075403608736942, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1IENYK5ITZ6V5', iid='B0028IKXLS', r_ui=4.0, est=3.9814814814814814, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1VI8P5LEXX6TJ', iid='B0016LFN2C', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2HU3Q6EPCD3RH', iid='B00A7MFRHC', r_ui=5.0, est=4.359281437125748, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ACEE1SLRCAUFO', iid='B005Q314NS', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AFY8QLS51NVI3', iid='B0015TJNEY', r_ui=3.0, est=3.2446958981612446, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1WCK3QM89XB2D', iid='B00ASLSQHK', r_ui=4.0, est=3.4535104364326377, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1VVQF3ID4CH3C', iid='B0099Z70ZK', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ALOUPMQ38Q7ZT', iid='B00B588HY2', r_ui=5.0, est=4.562805872756933, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2EAQ61037OD3Y', iid='B00908BMVE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2C124A7MUL99T', iid='B001M56DI0', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ANM6S7MTGGNSB', iid='B000JLG5ZY', r_ui=1.0, est=3.616580310880829, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AXJ5ZS6TC5FLY', iid='B00BGGDVOO', r_ui=5.0, est=4.684533796383573, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1WUOIEAT0M3Y4', iid='B007ADFKAK', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A18FNQBU4RLF1K', iid='B0012S7GRY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A5YLDNJZ16H59', iid='B004QK7HI8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3UUNL0WMA5A4B', iid='B000B9RI14', r_ui=5.0, est=4.775317946302402, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1HAJHQH2COWNK', iid='B0050SPZMK', r_ui=2.0, est=3.774436090225564, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2OOX3ZIHEJZ7L', iid='B00EL93M3S', r_ui=5.0, est=4.170776021404598, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A259ASOF1K4O7Q', iid='B003ISWI24', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A66IMQ0RK72MG', iid='B009OUFP1Q', r_ui=5.0, est=3.4651162790697674, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1QJX1NAOEEXM2', iid='B00065L5SU', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1QM7NEPNR6J3N', iid='B00009ZWC8', r_ui=5.0, est=3.9855072463768115, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3PX6SIYNRJ0CY', iid='B008V9959O', r_ui=5.0, est=4.125984251968504, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2Y8ZPXERU1GT0', iid='B000A6PPOK', r_ui=1.0, est=3.951244286439817, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2W7XO7Q1UUP0R', iid='B0062QPKAQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2BNO04PJFIVH9', iid='B004GCJEZU', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A36A79N4YP9NZ4', iid='B00328HR44', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1J9V8FPJDES4P', iid='B0096YOQQA', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AOBV10QFFUBEC', iid='B001EYU3L2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3GO9M4ED4A6K8', iid='B000E8X5ZU', r_ui=4.0, est=4.661538461538462, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A167FGHSNHYA62', iid='B0038JE07O', r_ui=3.0, est=3.3576437587657786, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A34U6WKGEO82P2', iid='B00003WGP5', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AXGVX02ZB0YCZ', iid='B00109Y2DQ', r_ui=5.0, est=4.014469929749797, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1Q6WLQMN5BRU6', iid='B004G6002M', r_ui=3.0, est=4.99591544826321, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A3UICOKLNG5PD8', iid='B0055D66V4', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2FN9RRDALVDAB', iid='B000MSDL6K', r_ui=5.0, est=4.47945205479452, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AVAG2D00E6ZDZ', iid='B008D4XBII', r_ui=4.0, est=4.518105849582173, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1Y5YF2AUXK5XT', iid='B00FU83YWS', r_ui=5.0, est=4.122448979591836, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A24DOH1BT6ETNU', iid='B00829THH8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1SW155A81HKLJ', iid='B002XVBAKI', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1U554UZF2PS93', iid='B004U78J1G', r_ui=4.0, est=3.894875164257556, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AN9Q1NAME38FL', iid='B005CT56F8', r_ui=5.0, est=4.517569759896171, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A17UHZF9Z6DKY4', iid='B003U8HTMG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A4S2WAQVQR6PI', iid='B0000AI0N1', r_ui=5.0, est=4.6521739130434785, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3V94T9Z8HB36O', iid='B00B9DQ2QI', r_ui=3.0, est=3.971014492753623, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1JRJV57XJKXPO', iid='B001EZUQ5E', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2M5COJ7NFWOF1', iid='B007DNG9DY', r_ui=5.0, est=3.607843137254902, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AHWM9IM9E2OAO', iid='B000U5TUWE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2OTW9WRG83OLE', iid='B0043M9AU2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3K8CX0C92VN7B', iid='B0058XGN7I', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A12C0UNX4KGNFC', iid='B0035ITWGC', r_ui=5.0, est=4.45, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A13V387HYFGXVG', iid='B0035WSQEC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A38OB30ESYFNC9', iid='B00368CDH6', r_ui=5.0, est=3.869565217391304, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2GSCL8JG3NNKE', iid='B0088O7C7Y', r_ui=3.0, est=3.7534246575342465, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3FG49HTR233KE', iid='B004HO58KW', r_ui=2.0, est=3.6792452830188678, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AEAL8JBU9Y5VM', iid='B0018SHJPM', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2W80WBIE41N8E', iid='B0073FE1F0', r_ui=4.0, est=4.391572456320658, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AGAXK0DNJNMPF', iid='B0011ULQNI', r_ui=4.0, est=4.215323645970938, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3S2ZNBOVW4MW1', iid='B007PRHNHO', r_ui=5.0, est=4.418287937743191, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AHTC44C4YQJ9N', iid='B007JBN6AO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3UNRKDI2UOB6M', iid='B0083XTPH0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3BZ9PF4SV6NUF', iid='B003DZ168E', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A4V3MZPLN3XUU', iid='B005DSPLDA', r_ui=5.0, est=4.475915221579961, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AMEVQ3KBB1VXM', iid='B0042TW3J6', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A27RVF7CVMK9X1', iid='B00004Z6XS', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3FC8Y4E6668W0', iid='B00004SABB', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1FGE7IUZTW5B9', iid='B008YFB4FS', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3SRU584GQ26T7', iid='B003DZ165W', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A28IAAH3KT5L8X', iid='B007VL90AW', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ADY0MS46QQ0G1', iid='B003VAGXWK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AHVXOL1RUWYHR', iid='B000BNY64C', r_ui=2.0, est=4.623563218390805, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3V7D0LH8L7BG0', iid='B0015BYKGI', r_ui=5.0, est=3.7628865979381443, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A28VKA0ZY1EPFE', iid='B008MIQGTQ', r_ui=4.0, est=4.543520309477756, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1RADUDRJZB5QQ', iid='B009X3UW2G', r_ui=5.0, est=4.082377476538061, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A7657U6RAV1BX', iid='B004UAKCS6', r_ui=5.0, est=4.140909090909091, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2TYZUQHR8RM9I', iid='B0044CWG0M', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AG71XMQGQ2UTJ', iid='B00CWBABP4', r_ui=5.0, est=4.076923076923077, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1NKYV39GX42VM', iid='B004SO876S', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A32ATMOQQ156A6', iid='B00CSMYBFS', r_ui=3.0, est=4.303030303030303, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AWP1WU0NXZAEA', iid='B007PJ4PKK', r_ui=2.0, est=4.329866270430906, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3225U2HB84Q37', iid='B004B8GF7Y', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3K36QGTS3NMYI', iid='B00EZ9XLC6', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ENTFDCMLGHWD', iid='B008UHK3KM', r_ui=4.0, est=4.115289765721331, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A26MX4QK3LGO82', iid='B00008ZOYE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A22BZD9BW8YPZF', iid='B0042VLFIE', r_ui=5.0, est=5, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='AHDC88AH4330', iid='B002K40R6G', r_ui=5.0, est=4.036842105263158, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3LGV5JXFSBFTL', iid='B0081F2Z40', r_ui=4.0, est=4.823984351908483, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A3W37O3M32N8J7', iid='B000MUXVZE', r_ui=5.0, est=4.406779661016949, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2F38OFV637M31', iid='B00AQFFSAG', r_ui=4.0, est=2.964349254286092, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1NXUQL5T6N87M', iid='B008D2POAS', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1G7Q74TEVBT6X', iid='B000652M6Y', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3DB5HA693NI5A', iid='B000HARTYI', r_ui=4.0, est=4.186915887850467, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1GVH3AKIDVEAW', iid='B004NBL9WK', r_ui=5.0, est=4.208917197452229, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2GEDIMLZZB49F', iid='B000QY9KIS', r_ui=5.0, est=4.181184668989547, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1K91B9LG7ZIVN', iid='B004W2JKWG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ABNP40K6DDP3N', iid='B0000CE1UO', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3V44AS0DBPFAQ', iid='B0094R4POC', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2JWDEMXQ8SQTI', iid='B00008AWL2', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A37MUF9HEYM5XE', iid='B005EK3OF4', r_ui=5.0, est=4.105263157894737, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2QV0WRTN4T9NK', iid='B0001GZ87I', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1OWSIBZN3BCVS', iid='B00166F8YU', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A26YF89Q2JN162', iid='B001UI2FPE', r_ui=5.0, est=4.487268419186925, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A23PY09MZ3GR4T', iid='B008THTRVO', r_ui=5.0, est=3.9503722084367245, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1SVVLL2YZD6U9', iid='B003EM0RM2', r_ui=5.0, est=4.144736842105263, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1898DG18TDBJJ', iid='B00IBR189Q', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2N4II7G6J86EC', iid='B0002SQ2P2', r_ui=5.0, est=4.530108588351432, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2XA8CW5DF4MNZ', iid='B005EJH6RW', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A30EQDRL4VV5BV', iid='B00CAFPF26', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A34H1TFXR2MZ7U', iid='B0062K951C', r_ui=4.0, est=3.763157894736842, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A28J6S1MGRQ4OC', iid='B004A9NKIG', r_ui=3.0, est=4.095238095238095, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2VVIQ0RSM89HG', iid='B002XJN5B2', r_ui=4.0, est=4.058490566037736, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1W1P440OYSNTA', iid='B0045TYDX2', r_ui=5.0, est=4.324074074074074, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ANUGPFWYN38AW', iid='B00AQRUW4Q', r_ui=5.0, est=3.6606060606060606, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AJ6N8FZERATCQ', iid='B0036RH93K', r_ui=5.0, est=3.7124600638977636, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A329MPRZBOZQBR', iid='B007OY5V68', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATQP8C6703GKN', iid='B004RKQM8I', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3GFYTEL5J6KSE', iid='B009OAZW28', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A324CX5OL0LK64', iid='B00CD1PTF0', r_ui=5.0, est=4.371794871794871, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2PPWRINZBD8KS', iid='B00698ZUHK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A35FRMY5U4SOC3', iid='B0098F5W0Q', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JU16JTNLVK1J', iid='B0019RGQVU', r_ui=4.0, est=3.4680535652517666, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='ADOQEB6QHONTY', iid='B005HJWWW8', r_ui=5.0, est=3.916326530612245, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ALL555YU03H6C', iid='B005EJH6RW', r_ui=3.0, est=4.476543209876543, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A30CP8DCG46RP1', iid='B0009S5HQA', r_ui=5.0, est=4.436285097192225, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3UHY0M4YLBRXX', iid='B0096YOQRY', r_ui=3.0, est=3.6379067550218993, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A2ST04A5UG7K77', iid='B001CJOLBW', r_ui=5.0, est=3.455421686746988, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3QDRRJ2D3SHGO', iid='B00F96PUNW', r_ui=5.0, est=4.107142857142857, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AS3X5RJYR7JBP', iid='B004URBZ4O', r_ui=1.0, est=3.991228070175439, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1RP57MYTPKM97', iid='B000B525DY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3BPBYATV7WH32', iid='B00BIP6G9K', r_ui=5.0, est=4.044943820224719, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AIS0JS2ETHYC7', iid='B000IVDTSG', r_ui=5.0, est=4.321428571428571, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1ANCOCQLRU1U3', iid='B00DR0BQ0I', r_ui=5.0, est=3.98297213622291, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3M2TMPK88UCSR', iid='B0076BNK30', r_ui=5.0, est=3.9868232093780938, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='A202DC58JWUAOS', iid='B004M8SWBU', r_ui=5.0, est=4.426470588235294, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A18B0NCSR7BUM1', iid='B009AIBW1E', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2RMU3IMO7GUE2', iid='B0089ZV1WY', r_ui=2.0, est=4.220481927710844, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A54903TT5GQZD', iid='B000N5T0UI', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A34LEE6XEYR09N', iid='B001MYASTG', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1MMX605RCN4DX', iid='B003VYEYE0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1MQ8RYOAHLEBK', iid='B002U1N95K', r_ui=5.0, est=3.9647058823529413, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3GD5IZ6AQND40', iid='B00JTI4X3E', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3C08BZRVV500V', iid='B0014F9U6U', r_ui=1.0, est=3.0854700854700856, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1UL29HKRBI7W9', iid='B0077QHF1C', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3R2ABK5DW86KM', iid='B00BCGRRWA', r_ui=5.0, est=4.134920634920635, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1SKGR6XGK3VGM', iid='B00139W0XM', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AFZI3G887G93K', iid='B006K5536K', r_ui=4.0, est=3.984126984126984, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2VKI81PBWS274', iid='B004OOODPG', r_ui=5.0, est=4.219298245614035, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1FCFE54EN3NEY', iid='B00478O0JI', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AMS3QJZN6QS47', iid='B00CO9L1F8', r_ui=3.0, est=4.413793103448276, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AILY9J2PT81A7', iid='B00142JKSG', r_ui=5.0, est=4.439024390243903, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2QDM2AWMHP3NT', iid='B00AANMVNQ', r_ui=5.0, est=4.60916179337232, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A13IA7HIQPZ04C', iid='B00EF1OGOG', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AO5NZ5ZX639A9', iid='B0000A0AEM', r_ui=5.0, est=4.044585987261146, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1F0VHM05XHI3H', iid='B005H3AU1Y', r_ui=5.0, est=3.9060402684563758, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AXYDWFIF3RW41', iid='B00DR0PDNE', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2XO6F6MNBPM0C', iid='B00007E7C8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A23HEPFHMLY7SV', iid='B0062EUE54', r_ui=4.0, est=4.384083044982699, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3LYDMSDTL5XIG', iid='B0012X5766', r_ui=5.0, est=4.43646408839779, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2BSTYYYU4V0TF', iid='B009U7WZCA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A15BFEJX8W63CX', iid='B004QK7HI8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A5PJD66R9MSDH', iid='B004ASY5ZY', r_ui=5.0, est=4.523809523809524, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1RBK07YQ3Z6DA', iid='B0038KLCQ0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2P77IVJWZNWMM', iid='B004GJ6FI2', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1ZOIYE8WZT3AJ', iid='B00E87YPNE', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AW5MWFNAAMENP', iid='B0071NWYP8', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2Q9LGNO6OQ23G', iid='B001SH2AVQ', r_ui=5.0, est=3.017897091722595, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3MFUM5ENEX74R', iid='B008SFPMRK', r_ui=5.0, est=4.574324324324325, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3URPD69CU0NOI', iid='B009PK9S90', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1FG7UI419TYKI', iid='B001M56DI0', r_ui=5.0, est=3.385665529010239, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AOFQRV7JN8H3U', iid='B00DQGIHNW', r_ui=5.0, est=4.110320284697509, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1FPKUSM0MP4PH', iid='B00502ZG3O', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A55HFBT6GTF9M', iid='B0041OSQ9I', r_ui=5.0, est=3.9765130984643178, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AQGOVP4F7053Y', iid='B00395WIXA', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AKPJ0G297KLBV', iid='B0054JJ0QW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2D32UFZAR37GH', iid='B00291NGXQ', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3CPALUJ4FKGX0', iid='B000652SOK', r_ui=5.0, est=4.077844311377246, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A122R2NTS8VO3G', iid='B00834SJNA', r_ui=5.0, est=2.4820870984714514, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A4MP0OVTUI0OU', iid='B000NOSUAU', r_ui=5.0, est=3.5606060606060606, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1XOO8GZ4CCL75', iid='B00DIFIM36', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AG86VEKC01IWK', iid='B0036MDUO2', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A314APICCL70GO', iid='B001LJIQ32', r_ui=5.0, est=3.3404255319148937, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A18WTKFTVDFJFZ', iid='B002P8LZ36', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A246I5D7XQ14UR', iid='B005KDYBIO', r_ui=5.0, est=4.370852582948341, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AAFL4M05W5EFE', iid='B0094NXBZ0', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2YGF29E5KEB77', iid='B007STRVTY', r_ui=5.0, est=4.12621359223301, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A27FJAUS15GTVT', iid='B0081JOH1K', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A16UISL4S22SUL', iid='B0079M711S', r_ui=5.0, est=4.183235626343064, details={'actual_k': 3, 'was_impossible': False}),
 Prediction(uid='ASW1VAL9CUMJE', iid='B007QXLIWI', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A29F1V51WQF28V', iid='B00A9SX5WS', r_ui=2.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AIQCLYHZ1C8FE', iid='B00AR1JYN6', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2XLQGIJPFY4XX', iid='B00BQH8UEY', r_ui=5.0, est=4.435729847494553, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2P5A7NPCVNNLS', iid='B002LBQWMG', r_ui=4.0, est=3.590643274853801, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2G25TYPQLWSVB', iid='B0025VKUQQ', r_ui=2.0, est=4.048523206751055, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3NB4QIJ5BVY39', iid='B00EF1OGOG', r_ui=5.0, est=4.294478527607362, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='ABRAS5RI13GA7', iid='B0080AO68E', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3GDOOZ4O4YAC8', iid='B005FNH9RE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A16PMKNN0RI2', iid='B0015M4G5C', r_ui=5.0, est=3.851063829787234, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A157Q8UEW6Z7BK', iid='B006BUN6ZE', r_ui=2.0, est=4.176165803108808, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3VZH3GF6X8AOM', iid='B002LSDKSI', r_ui=5.0, est=3.485294117647059, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A29EI0LHAWPHW', iid='B00BP5MB56', r_ui=5.0, est=4.981242905179374, details={'actual_k': 2, 'was_impossible': False}),
 Prediction(uid='A1SWIA8RZPXOAT', iid='B00BTMEKNQ', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AGXH8X0V9JSZO', iid='B006XGCQ2U', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1BBBVR0ZG6MO4', iid='B008XM630S', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2TE13BTGQTDG9', iid='B002WUVAVE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1OMDVTLBB6BFD', iid='B00DR0B31U', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AF9ATPTTXVDE', iid='B0001FTVEK', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1WATFEN2DCDX1', iid='B0076S5022', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A3JQMG67H45VZF', iid='B000VQ7ZK6', r_ui=5.0, est=4.164383561643835, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2UYM081VA8IGV', iid='B005Y1CYSQ', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A6YZY6ZWPW68U', iid='B0096FT902', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1D9I5RJSPM2WH', iid='B007F9XHCM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2KSJ10GE635YE', iid='B0049S6ZUS', r_ui=5.0, est=4.654118698809871, details={'actual_k': 1, 'was_impossible': False}),
 Prediction(uid='A1CSMXD6S4KA44', iid='B003TVWNAM', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1GPTA5PB5DUXL', iid='B005J7EOIS', r_ui=4.0, est=3.8260869565217392, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A152Z9WE4NDA6C', iid='B00BUCLVZU', r_ui=4.0, est=4.420062695924765, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3DV9XKPRGKNF', iid='B007RNCLBY', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ARZYUYZQ5P0T1', iid='B003SQEAY0', r_ui=5.0, est=4.360091743119266, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2XCKW6UGRT9HX', iid='B00FNPD1VW', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A247704ZT5V4A3', iid='B002TA7VO2', r_ui=5.0, est=3.914772727272727, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A2RXZGI78AO2DO', iid='B006ALR3OE', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='ATH2LYED5CGI5', iid='B009PAEE58', r_ui=3.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2K24MHWQN8Q77', iid='B004J3ZV62', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A1CY7C48DGZ2U8', iid='B0088RIV1W', r_ui=5.0, est=4.786259541984733, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='AZTXO0D8DILTL', iid='B003X7TRWE', r_ui=1.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A639SSAX07GJK', iid='B0032ANC00', r_ui=4.0, est=4.0, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A3MQI7C2ARSG', iid='B000067VB7', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2PU281XKBF8AL', iid='B009NHWVIA', r_ui=5.0, est=4.523188405797101, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A1F74I8YML7SSA', iid='B004QK7HI8', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='AD0OHTWWBGJC7', iid='B003DZ168E', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A7C45E7ZN46SH', iid='B003YKG2XM', r_ui=4.0, est=3.8754208754208754, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A38TOAR2INC1K5', iid='B005HM0SNA', r_ui=5.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2VIGSOQZ6R77I', iid='B00GP4CW24', r_ui=4.0, est=4.068380608555389, details={'was_impossible': True, 'reason': 'User and/or item is unknown.'}),
 Prediction(uid='A2TWTP6HD3CFT8', iid='B0048O0WKW', r_ui=4.0, est=3.9454545454545453, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A276OI0NHBYORX', iid='B002AZ3D3E', r_ui=5.0, est=3.9357429718875503, details={'actual_k': 0, 'was_impossible': False}),
 Prediction(uid='A13U1HCI5MXRS0', iid='B003O1UYHG', r_ui=5.0, est=5, details={'actual_k': 1, 'was_impossible': False}),
 ...]
</pre>
## Getting accuracy



정확도



기계 학습 알고리즘의 정확도는 알고리즘이 얼마나 잘 수행되고 있는지, 즉 알고리즘이 데이터 포인트를 올바르게 분류하는 빈도를 측정하는 것입니다. 정확도는 다음과 같이 주어집니다:



![정확도](https://miro.medium.com/max/1050/1*O5eXoV-SePhZ30AbCikXHw.png)



상관 행렬



상관 행렬은 변수 간의 관계, 즉 다른 변수가 변경될 때 한 변수가 어떻게 변경되는지를 보여주는 테이블입니다. 5개의 변수가 있는 경우 상관 행렬에는 5 곱하기 5 또는 25개의 항목이 있으며 각 항목은 두 변수 간의 상관 관계를 보여줍니다.



RMSE



RMSE는 평균 제곱근 오차를 나타냅니다. 기계 학습 모델을 사용하여 예측을 수행할 때 예측이 정확한지 확인해야 합니다. RMSE는 예측 오류를 측정하는 방법입니다. RMSE가 높으면 예측이 나쁘고, 낮으면 예측이 좋은 것입니다.



```python
# RMSE를 구합니다.
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)
```

<pre>
Item-based Model : Test Set
RMSE: 1.3317
</pre>
<pre>
1.3316598350472562
</pre>
우리의 최종 목표는 예측 모델을 얻는 것입니다. 기존 사용자-항목 상호 작용을 이용하여 사용자에게 가장 적합할 수 있는 상위 5개 항목을 예측하도록 모델을 훈련할 수 있습니다. 상위 10000개의 추천 항목과 SVD 알고리즘을 사용하는 모델을 사용하겠습니다.



```python
new_df1 = new_df.head(10000)
ratings_matrix = new_df1.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
ratings_matrix.head()
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
      <th>productId</th>
      <th>0972683275</th>
      <th>1400501466</th>
      <th>1400501520</th>
      <th>1400501776</th>
      <th>1400532620</th>
      <th>1400532655</th>
      <th>140053271X</th>
      <th>1400532736</th>
      <th>1400599997</th>
      <th>1400698987</th>
      <th>...</th>
      <th>B00000JHWX</th>
      <th>B00000JI4F</th>
      <th>B00000JII6</th>
      <th>B00000JSGF</th>
      <th>B00000JYLO</th>
      <th>B00000JYWQ</th>
      <th>B00000K135</th>
      <th>B00000K13A</th>
      <th>B00000K13L</th>
      <th>B00000K2YR</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A01852072Z7B68UHLI5UG</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>A0266076X6KPZ6CCHGVS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>A0293130VTX2ZXA70JQS</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>A030530627MK66BD8V4LN</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>A0571176384K8RBNKGF8O</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 64 columns</p>
</div>



```python
X = ratings_matrix.T
X.head()
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
      <th>userId</th>
      <th>A01852072Z7B68UHLI5UG</th>
      <th>A0266076X6KPZ6CCHGVS</th>
      <th>A0293130VTX2ZXA70JQS</th>
      <th>A030530627MK66BD8V4LN</th>
      <th>A0571176384K8RBNKGF8O</th>
      <th>A0590501PZ7HOWJKBGQ4</th>
      <th>A0641581307AKT5MAOU0Q</th>
      <th>A076219533YHEV2LJO988</th>
      <th>A0821988FXKFYX53V4QG</th>
      <th>A099626739FNCRNHIKBCG</th>
      <th>...</th>
      <th>AZVL57D2NG3T1</th>
      <th>AZWOPBY75SGAM</th>
      <th>AZX0ZDVAFMN78</th>
      <th>AZX5LAN9JEAFF</th>
      <th>AZX7I110AF0W2</th>
      <th>AZXKUK895VGSM</th>
      <th>AZXP46IB63PU8</th>
      <th>AZYTSU42BZ7TP</th>
      <th>AZZGJ2KMWB7R</th>
      <th>AZZMV5VT9W7Y8</th>
    </tr>
    <tr>
      <th>productId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0972683275</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1400501466</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1400501520</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1400501776</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1400532620</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9841 columns</p>
</div>



```python
X1 = X
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape
```

<pre>
(64, 10)
</pre>
상관 행렬을 찾습니다.



```python
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape
```

<pre>
(64, 64)
</pre>
ID가 "B00000K135"인 책을 고려하고 있다고 가정합니다. 우리는 이 책을 사는 고객을 찾아서 다른 책을 추천해 줄  것입니다.



```python
i = "B00000K135"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID
```

<pre>
60
</pre>
"B00000K135" 항목을 구매하는 고객에게 추천해야 할 주요 품목입니다.



```python
correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape
Recommend = list(X.index[correlation_product_ID > 0.65])
# 고객이 이미 구매한 항목을 제거합니다.
Recommend.remove(i) 
# 추천 항목을 출력합니다.
Recommend[0:24]
```

<pre>
['B00000J1V3', 'B00000K13L']
</pre>
### Task 5: 아이템 'B00000JSGF' 구매 고객을 위한 추천 항목 표시하기



```python
#yourcodehere
i = "B00000JSGF"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID
```

<pre>
57
</pre>


### Task 6: 아이템 'B00000JDF6' 구매 고객을 위한 추천 항목 표시하기



```python
#yourcodehere
correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape
Recommend = list(X.index[correlation_product_ID > 0.65])
# 고객이 이미 구매한 항목을 제거합니다.
Recommend.remove(i) 
# 추천 항목을 출력합니다.
Recommend[0:24]
```

<pre>
['9888002198',
 '9984984354',
 'B000001OM4',
 'B00000J1QR',
 'B00000J434',
 'B00000J4FS',
 'B00000JDF5',
 'B00000JDF6',
 'B00000JHWX']
</pre>
### Conclusion



인공 지능은 다양한 현대 산업에서 문제를 해결하는 데 널리 사용됩니다. 여기, 이 노트북에서 우리는 쇼핑 습관을 기반으로 고객에게 상품을 추천함으로써 전자 상거래 산업에서 인공 지능이 어떻게 사용될 수 있는지에 대한 예를 보았습니다.


#### 그 외 수행하기 위해 참조한 링크들..



https://zzinnam.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D%EC%9D%84-%EC%9C%84%ED%95%9C-20%EA%B0%9C%EC%9D%98-%EA%B0%95%EB%A0%A5%ED%95%9C-Pandas-%ED%95%A8%EC%88%98<br>

**데이터 분석을 위한 20개의 강력한 pandas 함수**







#### 마무리



- 각 설치 문제건등으로 인해서 어쩔 수 없이 이후에 정리해서 마무리;;

- 그래도 결과는 나왔으나 어디서 뭔가 빼먹었는지 수치가 높게 나오는 문제가 있음.

- 겉으로는 정상작동되어보임.. 다만, 제대로 작동하는지는..(만일 문제가 있다면 수정이 필요..)

- 이번 과제를 보면서 오히려 KMM보단 어떻게 아나콘다에서 라이브러리를 관리할까의 과제를 고민하게 되었달까.


#### 주의사항



본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.

