---
layout: single
title:  "Intel AI 캡스톤2 판매예측을 통한 이익실현"
categories: potpolio
tag: [python, blog, jupyter, potpolio]
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


인공지능 판매 예측을 통해 이익실현 프로젝트
===================================



1. 어느(ros..) 매장에서 판매예측을 통해 이익을 실현하고자 함.

2. 이를 위해서 이때까지의 판매데이터를 바탕으로 훈련, 예측이 목적

3. 예측하기 위한 데이터 준비와 이번엔 예측중심의 모델로 해결모색.


## 진행 순서



1. 필요한 파일들 읽어오기

    - Store, Train, Test csv 파일들 읽어오기

2. 파일 상태 확인 

    - 각 Data Frame 의 크기, head, 결측값, info 확인하기

3. 결측값을 0으로 채워 넣기

4. Train Data에서 각 변수가 Sales 에 미치는 영향 파악하기: 그래프 그려보기

5. 매장별 통계 데이터 확인

    - 5.1 spc = sales/customer 로 새로운 열 생성 à train.csv df 

    - 5.2 Groupby 이용 store로 sales, customers, spc 평균값 만들어 보기 : store별 평균값 data 

6. Store df에 4번에서 생성된 컬럼 merge 시키기 à store df columns : 

7. Train df에 store df merge 시키기(기준은 Store 명 기준으로) 

8. Train df에서 Date를 Year, Month, Day, week로 변환하여 각각 컬럼을 생성

9. Train DF에서 Label, Features 컬럼 나누기 X, y 

10. X, y 데이터셋을 Train, test로 나누기

    - 8:2로..

11. 회귀(예측) AI 모델 선택하기

    - 예로.. Linear regression, Ridge regression, Lasso regression, Polynomial regression ... 

12. 학습(훈련), 예측, 성능평가

    - Fit 학습, predict(예측), Score(성능 평가) 

13. 결과 보고

    - test.csv 파일로 판매량 예측하기 à Jupyter File, Submission_OOO.csv 예측 판매량 


#### 0. 필요한 라이브러리



```python

# 기본적인 부분 우선...
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# 훈련 및 필요한 여러모로의 모델들
from sklearn.model_selection import train_test_split # 분리
from sklearn.preprocessing import StandardScaler # 정규화.
from sklearn.metrics import classification_report # 성능 요약

# 이후는 모델 탐색에 필요한 import...
from sklearn.linear_model import Lasso # Lasso
from sklearn.linear_model import Ridge # Ridge
from sklearn.linear_model import LinearRegression # LR
from sklearn.preprocessing import PolynomialFeatures # PF
from sklearn.ensemble import RandomForestRegressor #RFR
```

#### 1. 필요한 파일들 읽어오기

- Store, Train, Test csv 파일들 읽어오기

<pre>
:1: DtypeWarning: Columns (7) have mixed types. Specify dtype option on import or set low_memory=False.
  df_train = pd.read_csv('./train.csv')
</pre>


#### 2. 파일 상태 확인

- 각 Data Frame 의 크기, head, 결측값, info 확인하기


df_train의 크기, 결측값, 정보 확인

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

해당 데이터의 train 크기

<pre>
9154881
</pre>

해당 데이터의 train 정보

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1017209 entries, 0 to 1017208
Data columns (total 9 columns):
 #   Column         Non-Null Count    Dtype 
---  ------         --------------    ----- 
 0   Store          1017209 non-null  int64 
 1   DayOfWeek      1017209 non-null  int64 
 2   Date           1017209 non-null  object
 3   Sales          1017209 non-null  int64 
 4   Customers      1017209 non-null  int64 
 5   Open           1017209 non-null  int64 
 6   Promo          1017209 non-null  int64 
 7   StateHoliday   1017209 non-null  object
 8   SchoolHoliday  1017209 non-null  int64 
dtypes: int64(7), object(2)
memory usage: 69.8+ MB
</pre>

해당 데이터의 train 요약

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.584297e+02</td>
      <td>3.998341e+00</td>
      <td>5.773819e+03</td>
      <td>6.331459e+02</td>
      <td>8.301067e-01</td>
      <td>3.815145e-01</td>
      <td>1.786467e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.219087e+02</td>
      <td>1.997391e+00</td>
      <td>3.849926e+03</td>
      <td>4.644117e+02</td>
      <td>3.755392e-01</td>
      <td>4.857586e-01</td>
      <td>3.830564e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.800000e+02</td>
      <td>2.000000e+00</td>
      <td>3.727000e+03</td>
      <td>4.050000e+02</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.580000e+02</td>
      <td>4.000000e+00</td>
      <td>5.744000e+03</td>
      <td>6.090000e+02</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.380000e+02</td>
      <td>6.000000e+00</td>
      <td>7.856000e+03</td>
      <td>8.370000e+02</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.115000e+03</td>
      <td>7.000000e+00</td>
      <td>4.155100e+04</td>
      <td>7.388000e+03</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



해당 데이터의 train 결측값 확인

<pre>
Store            0
DayOfWeek        0
Date             0
Sales            0
Customers        0
Open             0
Promo            0
StateHoliday     0
SchoolHoliday    0
dtype: int64
</pre>
일단.. Train쪽 결측값은 없어보임  

다음은 test 데이터도 이와 같이 진행

머리부분 확인

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
      <th>Id</th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



해당 데이터의 test 크기

<pre>
328704
</pre>

해당 데이터의 test 정보

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41088 entries, 0 to 41087
Data columns (total 8 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             41088 non-null  int64  
 1   Store          41088 non-null  int64  
 2   DayOfWeek      41088 non-null  int64  
 3   Date           41088 non-null  object 
 4   Open           41077 non-null  float64
 5   Promo          41088 non-null  int64  
 6   StateHoliday   41088 non-null  object 
 7   SchoolHoliday  41088 non-null  int64  
dtypes: float64(1), int64(5), object(2)
memory usage: 2.5+ MB
</pre>

해당 데이터의 test 요약

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
      <th>Id</th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Open</th>
      <th>Promo</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>41088.000000</td>
      <td>41088.000000</td>
      <td>41088.000000</td>
      <td>41077.000000</td>
      <td>41088.000000</td>
      <td>41088.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20544.500000</td>
      <td>555.899533</td>
      <td>3.979167</td>
      <td>0.854322</td>
      <td>0.395833</td>
      <td>0.443487</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11861.228267</td>
      <td>320.274496</td>
      <td>2.015481</td>
      <td>0.352787</td>
      <td>0.489035</td>
      <td>0.496802</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10272.750000</td>
      <td>279.750000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20544.500000</td>
      <td>553.500000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30816.250000</td>
      <td>832.250000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41088.000000</td>
      <td>1115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


- 위 데이터 보아하니 customers가 없음.. 이 customers를 예측해서 test와 합쳐 결과를 내는것...



해당 데이터의 test 결측값 확인

<pre>
Id                0
Store             0
DayOfWeek         0
Date              0
Open             11
Promo             0
StateHoliday      0
SchoolHoliday     0
dtype: int64
</pre>

- open에 11개의 결측값 존재. 처리해야함.

- 다음 프로세스에서...

- store 데이터도 확인

해당 데이터의 store 머리부분  
  
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
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>a</td>
      <td>a</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>c</td>
      <td>c</td>
      <td>620.0</td>
      <td>9.0</td>
      <td>2009.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>a</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



해당 데이터의 store 분량  

<pre>
11150
</pre>

해당 데이터의 store 정보  

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1115 entries, 0 to 1114
Data columns (total 10 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Store                      1115 non-null   int64  
 1   StoreType                  1115 non-null   object 
 2   Assortment                 1115 non-null   object 
 3   CompetitionDistance        1112 non-null   float64
 4   CompetitionOpenSinceMonth  761 non-null    float64
 5   CompetitionOpenSinceYear   761 non-null    float64
 6   Promo2                     1115 non-null   int64  
 7   Promo2SinceWeek            571 non-null    float64
 8   Promo2SinceYear            571 non-null    float64
 9   PromoInterval              571 non-null    object 
dtypes: float64(5), int64(2), object(3)
memory usage: 87.2+ KB
</pre>

해당 데이터의 store 요약  

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
      <th>Store</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1115.00000</td>
      <td>1112.000000</td>
      <td>761.000000</td>
      <td>761.000000</td>
      <td>1115.000000</td>
      <td>571.000000</td>
      <td>571.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>558.00000</td>
      <td>5404.901079</td>
      <td>7.224704</td>
      <td>2008.668857</td>
      <td>0.512108</td>
      <td>23.595447</td>
      <td>2011.763573</td>
    </tr>
    <tr>
      <th>std</th>
      <td>322.01708</td>
      <td>7663.174720</td>
      <td>3.212348</td>
      <td>6.195983</td>
      <td>0.500078</td>
      <td>14.141984</td>
      <td>1.674935</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>279.50000</td>
      <td>717.500000</td>
      <td>4.000000</td>
      <td>2006.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
      <td>2011.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>558.00000</td>
      <td>2325.000000</td>
      <td>8.000000</td>
      <td>2010.000000</td>
      <td>1.000000</td>
      <td>22.000000</td>
      <td>2012.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>836.50000</td>
      <td>6882.500000</td>
      <td>10.000000</td>
      <td>2013.000000</td>
      <td>1.000000</td>
      <td>37.000000</td>
      <td>2013.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1115.00000</td>
      <td>75860.000000</td>
      <td>12.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>50.000000</td>
      <td>2015.000000</td>
    </tr>
  </tbody>
</table>
</div>



해당 데이터의 store 결측값 확인  

<pre>
Store                          0
StoreType                      0
Assortment                     0
CompetitionDistance            3
CompetitionOpenSinceMonth    354
CompetitionOpenSinceYear     354
Promo2                         0
Promo2SinceWeek              544
Promo2SinceYear              544
PromoInterval                544
dtype: int64
</pre>
  
#### 3. 결측값을 0으로 채워 넣기

- test와 store에 결측값이 감지

- 0로 채워넣는 프로세스 시작


<pre>
Id               0
Store            0
DayOfWeek        0
Date             0
Open             0
Promo            0
StateHoliday     0
SchoolHoliday    0
dtype: int64
</pre>
  
store 데이터도 이와같이 진행  
  
<pre>
Store                        0
StoreType                    0
Assortment                   0
CompetitionDistance          0
CompetitionOpenSinceMonth    0
CompetitionOpenSinceYear     0
Promo2                       0
Promo2SinceWeek              0
Promo2SinceYear              0
PromoInterval                0
dtype: int64
</pre>
  
  
#### 4. 위 데이터로 그래프 그려보기

- Train Data에서 각 변수가 Sales 에 미치는 영향 파악 목적

- 일단 특정 한달 2015.7 기준으로 하라고.. 양이 많길래..



train 데이터 정보부터  

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1017209 entries, 0 to 1017208
Data columns (total 9 columns):
 #   Column         Non-Null Count    Dtype 
---  ------         --------------    ----- 
 0   Store          1017209 non-null  int64 
 1   DayOfWeek      1017209 non-null  int64 
 2   Date           1017209 non-null  object
 3   Sales          1017209 non-null  int64 
 4   Customers      1017209 non-null  int64 
 5   Open           1017209 non-null  int64 
 6   Promo          1017209 non-null  int64 
 7   StateHoliday   1017209 non-null  object
 8   SchoolHoliday  1017209 non-null  int64 
dtypes: int64(7), object(2)
memory usage: 69.8+ MB
</pre>

train 행기준 확인

<pre>
Store  DayOfWeek  Date        Sales  Customers  Open  Promo  StateHoliday  SchoolHoliday
1      1          2013-01-07  7176   785        1     1      0             1                1
745    5          2015-06-05  7622   711        1     1      0             0                1
                  2015-03-06  7667   738        1     1      0             0                1
                  2015-03-13  6268   668        1     0      0             0                1
                  2015-03-20  7857   725        1     1      0             0                1
                                                                                           ..
372    7          2013-03-03  0      0          0     0      0             0                1
                  2013-03-10  0      0          0     0      0             0                1
                  2013-03-17  0      0          0     0      0             0                1
                  2013-03-24  0      0          0     0      0             0                1
1115   7          2015-07-26  0      0          0     0      0             0                1
Length: 1017209, dtype: int64
</pre>

train 앞부분 체크

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- 여기서 7월 데이터만 보여주기

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34560</th>
      <td>1111</td>
      <td>3</td>
      <td>2015-07-01</td>
      <td>3701</td>
      <td>351</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34561</th>
      <td>1112</td>
      <td>3</td>
      <td>2015-07-01</td>
      <td>10620</td>
      <td>716</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34562</th>
      <td>1113</td>
      <td>3</td>
      <td>2015-07-01</td>
      <td>8222</td>
      <td>770</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34563</th>
      <td>1114</td>
      <td>3</td>
      <td>2015-07-01</td>
      <td>27071</td>
      <td>3788</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34564</th>
      <td>1115</td>
      <td>3</td>
      <td>2015-07-01</td>
      <td>7701</td>
      <td>447</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>34565 rows × 9 columns</p>
</div>



- 7월 일별당 매출 그래프  
  
<img src="data:image/png;base64,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"/>


소비자와 판매자간의 그래프  
  

<img src="data:image/png;base64,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"/>


해당 그래프는 7월만으로 비교하기 어려워서 전체데이터를 넣어봄

<img src="data:image/png;base64,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"/>


방학기준 소비자와 판매자간의 그래프

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZwAAAEGCAYAAABRvCMcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAAsTAAALEwEAmpwYAABVI0lEQVR4nO29eXxcddn3/75mMkknLTRtU4QkIFURHtYCLaKNqKCURUpFLAgKioj3DYqId0vx5kdLXSityuItSlkeQcESFkPYrLXgUu4HaWtKoWilgEhSELqk2iZtJpnr98c5ZzrLObMkM8kkud6vV16Z+c7ZMznXuZbv5xJVxTAMwzBKTWiwD8AwDMMYGZjBMQzDMAYEMziGYRjGgGAGxzAMwxgQzOAYhmEYA0LFYB/AQFNbW6sHHnjgYB+GYRjGkGLNmjWbVXVif7Yx4gzOgQceyOrVqwf7MAzDMIYUIvJ6f7dhITXDMAxjQDCDYxiGYQwIZnAMwzCMAWHE5XD8iMVitLW1sWvXrsE+lEBGjRpFQ0MDkUhksA/FMAyjT5jBAdra2thrr7048MADEZHBPpwMVJUtW7bQ1tbGpEmTBvtwDMMw+oQZHGDXrl1la2wARIQJEybwzjvvDPahGIYxxGhubWfxsg1s6uiiribK7OkHM/Po+kE5FjM4LuVqbDzK/fgMwyg/mlvbufrhF+iK9QLQ3tHF1Q+/ADAoRscMjmEYxhAkH89l8bINCWPj0RXrZfGyDYNicKxKrUz49a9/zcEHH8z73vc+Fi5cONiHYxhGGeN5Lu0dXSh7PJfm1vaU5TZ1dPmuHzReaszglAG9vb1cdtllPPnkk7z00kv88pe/5KWXXhrswzIMo0zJ5rkkU1cT9V0/aLzUmMHpA82t7Uxb+BST5j7OtIVPZTxVFMpzzz3H+973Pt7znvdQWVnJueeeyyOPPFKkozUMY7iRr+cye/rBRCPhlLFoJMzs6QeX7NiyYTmcAilFEq69vZ39998/8b6hoYE//elP/T9YwzCGJXU1Udp9jE5IhElzH8/I6ViV2hCl3JJwhmGMPGZPPzjlwdejVxXIfBAul3uThdQKpBRJuPr6et54443E+7a2Nurry+MLYhhG+THz6HquP+sI6muiCBD2mTbhl9MZbMzgFEgpknBTp07l5Zdf5rXXXqO7u5ulS5cyY8aMPm/PMIzhz8yj63lm7om8tvB04q5nk85gVaMFYQanQEqRhKuoqOB//ud/mD59Ov/n//wfZs2axWGHHdbfQzUMY4RQbtVoQVgOp0BKlYQ77bTTOO2004pxiIZhjDD8cjqDWY0WhBmcPlBOSTjDMIxyq0YLwgyOYRjGMGAoPAibwTEMwximlJNSNJjBMQzDGJaUm1I0lLBKTURGichzIvK8iKwXkevc8Uki8icR2Sgi94tIpTte5b7f6H5+YNK2rnbHN4jI9KTxU9yxjSIyt1TnYhiGMdTIV29tICllWfRu4ERVPQqYDJwiIscDNwA3qur7gG3Al9zlvwRsc8dvdJdDRA4FzgUOA04BbhWRsIiEgR8DpwKHAp91lzUMwxjxlJtSNJTQ4KjDDvdtxP1R4ETgQXf8bmCm+/pM9z3u5yeJ03XsTGCpqu5W1deAjcBx7s9GVX1VVbuBpe6yQ5KLLrqIffbZh8MPP3ywD8UwjGFAOc7NKenET9cTWQu8DSwHXgE6VLXHXaQN8IKJ9cAbAO7n24EJyeNp6wSN+x3HJSKyWkRWl2ub5i984Qv8+te/HuzDMAxjmFBuStFQYoOjqr2qOhlowPFIDinl/rIcxxJVnaKqUyZOnNj/Da5rghsPh/k1zu91Tf3e5AknnMD48eP7f2yGYRhk6q3V10S5/qwjhn+Vmqp2iMjTwAeBGhGpcL2YBsBrJtMO7A+0iUgFMBbYkjTukbxO0HjpWNcEj14OMTcOuv0N5z3AkbNKvnvDMIx8Kbe5OaWsUpsoIjXu6yjwCeAvwNPA2e5iFwJep7EW9z3u50+pqrrj57pVbJOAg4DngFXAQW7VWyVOYUFLqc4nwYoFe4yNR6zLGTcMwzACKaWHsx9wt1tNFgKaVPUxEXkJWCoi3wFagTvd5e8Efi4iG4GtOAYEVV0vIk3AS0APcJmq9gKIyFeBZUAYuEtV15fwfBy2txU2bhiGYQAlNDiqug442mf8VZx8Tvr4LuAzAdv6LvBdn/EngCf6fbCFMLbBCaP5jRuGYRiBWHuCQjnpWoiklRVGos54P/jsZz/LBz/4QTZs2EBDQwN33nln7pUMwzCGECZtUyheYcCKBU4YbWyDY2z6WTDwy1/+sggHZxiGUb6YwekLR86yijTDGCTKTZDSyB8zOIZhDBnKUZDSyB8zOC6qiqOkU55oQM/y4Y49zRrJZBOktO9F+WNFA8CoUaPYsmVL2d7UVZUtW7YwatSowT6UAcV7mm3v6ELZ8zTb3Fr6+b1GeVKOgpRG/piHAzQ0NNDW1ka56qyBYxQbGkZW6bU9zRrp1NVEafcxLr6ClOuail7cY/QPMzhAJBJh0qRJg30YRhqFPM1a6G1kMHv6wSk5HAgQpDQJqrLEDI5RtuT7NGuJ5OFDrgcH73XOh4tsElRmcAYNMzhG2ZLv06yF3oYH+T445CVIGSA1Fe9o48MLnzIPeJCwogGjbMlXXt0SycODorZEDpCa2qQTrPhkEDEPxyhr8nmaLSiRbJQtyQ8IM0IrmVPRRJ1sZlNnLay7vrBQ2EnXpuZwgE6tZFGPsw3zgAcH83CMIU85djY0Csd7QJgRWsnCyB00hDYTEmgIbXaMRyGNDo+cBWfcAmP3J65CW7yWubGLaYk3JhYxD3jgMQ/HKFvyrTzLO5FslDVezm6ONFEt3akf9iXh70pQfXjhU+YBlwlmcIyypNDKs3LrbGgUjvf3q3tki/8Cfew5lXcptVFyLKRmlCVFTSAbQ4aZR9cTCuot1ceeU/kWnxilxzwcoyyxyrMRjE/Cn0gUDjoZbjzcXzkgh6qAecDlgRkcoyyxyrMRjF/PqYNOhufvS1EO6HzoMha1rOf0I+uY+sI8UxUYAlhIzShLrPJs5LKq5TbeevhbxDve4C1qWfXer8HLv8lQDqiWbi7u/gV1axYFqwoYZYV5OEZZYpVnA0yZCF2uarmNw9dcQ1S6QWBf3mHsmmtQ6caveUidbAECVN77WGRglA4zOEbZYnH3AaKMhC73//Nix9gkEZVueghRQTxj+U06AYAG2Zy5sT4WGRilo2QhNRHZX0SeFpGXRGS9iHzdHZ8vIu0istb9OS1pnatFZKOIbBCR6Unjp7hjG0VkbtL4JBH5kzt+v4hUlup8DGPYkk3ocoDZR/1bhIQ1ThdVKWOecsCinlkZnxGJOl6aUVaUMofTA3xTVQ8FjgcuE5FD3c9uVNXJ7s8TAO5n5wKHAacAt4pIWETCwI+BU4FDgc8mbecGd1vvA7YBXyrh+RjG8CQo9DQIIam3ZaLveLvWclX3l2jX2gzlgOXhj/DiMd+GsfsD4vw+45YU76y5tZ1pC59i0tzHmbbwKdNRGyRKFlJT1TeBN93X/xaRvwDZ4iNnAktVdTfwmohsBI5zP9uoqq8CiMhS4Ex3eycC57nL3A3MB35S7HMxjGHN2AYnjOY3PsC8ccxsxno5HBfPk2mJN9Kyu5Fx1RGqKyvY1NFFvZvbm3r0KcBXfLdp7SvKhwHJ4YjIgcDRwJ+AacBXReQCYDWOF7QNxxg9m7RaG3sM1Btp4x8AJgAdqtrjs3z6/i8BLgE44IADinBGhjGMCJr3MgghqakzvsIqnFzOPrqZTTohYWw8tnXGaL325Ly3ae0ryoeSl0WLyBjgIeAKVf0XjgfyXmAyjgf0g1Ifg6ouUdUpqjpl4kR/l90wRixJQpdBIak+s67Jmaw5v8b5nUuAc10TU1/5EfuymbelNsPYeBy94Dd5h8VsEnH5UFIPR0QiOMbmXlV9GEBV/5n0+e3AY+7bdmD/pNUb3DECxrcANSJS4Xo5ycsbhlEIrtBlUfGpfut6+KvMXdrK6r0/kVnmnrb8vrzDwsgdECPD6GzrjOUdFrNJxOVDKavUBLgT+Iuq/jBpfL+kxT4FvOi+bgHOFZEqEZkEHAQ8B6wCDnIr0ipxCgtaVFWBp4Gz3fUvBB4p1fkYpcGSucMYn+q3KLuZXdHk3wTNZ/lq6WZOhb9XlK+2nk0iLh9K6eFMAz4PvCAia92xb+FUmU3Gma31d9xMn6quF5Em4CWcCrfLVLUXQES+CiwDwsBdqrre3d5VwFIR+Q7QimPgjCFCKZK5+bY0MIpArsmiAVVu9bKZV6vOY5PWcsfjn2Pm0ddlXd6Z3OlPPmExm0RcPojjKIwcpkyZoqtXrx7swxjReEbBL8wBjprvM3NP7NN2/WToTRm4BKSHy8ApNEjO/dx4uH/1WxKdWkn1p3/srBOwfLvWMm33LUBaJ1Ct5Y7KzzH/Gsdg2cNGaRGRNao6pT/bMC01oyQEhco8oxBkbKDvyVxraTCA5DNZ9KRrHSOUhWrp3rOO3/KRKJuOnUNNNOLbCXRez03w2JUp3ysF/5CdMeiYtI1RdLKFyvyMQjp9TeZaNdIAks9k0STVZ93eBqqInyCat46fSvRJ1zL1yFmsnQGdN1xOdVeq7I0ArL6LtWv3oit2XMpnVvpcfpjBMYpONk8j182/P8lcq0YqAUF5mnwni7rVbwJ03nAI1V1vZl8nS7VcdddbAQepXNz9C37GcRmfFPNhw0J2/ccMjlF0snkaNdURtnXGfD+v7+c/cVAr4Y8dMpFpC5/q042irG4ypVB0zrbNbKKefZgsWn3qgoLWSb/2y6P7+hssoC60JSO/s6hnFmv2/kTelyIbplZQHKxowCg60xY+5etpjKuOsGNXD7F46ncuEhYWn31UUf5x029SHztkIg+tae9TIUFZFSHkk6Qv9jaDkv5j94dvvNg3A5i0Tmd0XxbFzuHuHccljDmQKCgRUhsPnF35vywO/Y9vm4LdkRri3Z0pkjhdWsmLx36HqTP8JW8KIeg73dcCl6FIMYoGzOAYRSfoRl1VEaKjK9O7qYlGWDsvf6mSQujPjaKsbjK5bv6l2Ob8Gvx7zQjM7+jbPl38viORsICS8UCSzA+i9/BpXZZ6XJEoVESha2vmCv25PklMmvt40JXgtYWn93v7QwGrUjPKkplH13P9WUdQXxNFcG7Q1591BNt9jA0QOF4M+lNIELRMtgq7Qiho0mspFJ1zbTNIvLMIop5+eb5Yr2Y1NgD/1XUBnLUkU4ana5v/CkVSvA7KA1p+sDAsh2OUBL/maUFzb0r5T9ufQoKgdQXHWPQnrFZwTqAUis65tplvnqYPobW+JvPraqJw5OmZ21+xoKSK10H5QVMrKAzzcIyS4z3Je3H5ZEr9T9sfWZPZ0w/2zRco9HtuT0FzhtY1QffOzPH0m3+hQpl+815CEWdf82ucm/hR52UX9XzsSnj4y+7NXp3fD38FbpiUchzp3lxNdST7sfmQ9e8WMIenWIrXQV67FQwUhuVwjJLR3NrOdY+uz6hK85LB/a1KK+Q4+lppduDcx33H+xS7dz0B3d5GezxTdt93u36JfYDoeDj1huCKMsivqCDZO4mOg+4d0Nud3zbWNTnGJgc94VHMjV3Mg90fyrlsEGERelWzf2dKUcVnJLCigT5gBmdgaG5tZ/aDzxPr9f9+lbJQoJgUrXDAxyB0amWia2XgdvMtFihGUUGh28hDusajLV5LY/ctgZ+Py1Iun47JFQ0OxTA4lsMxikayJ4FAtmeZjq5Yv/MgA0HRYvdZlJBbuhuDtxuQ9NbtbTQmzS1auavNN/xXUNK80MKEAradTYAToLqygurKirwKMkxBYOhiORyjKKRrWeXjOA8FjbOixe7zUEL23W5A0nuTTkjRDdukE/z3mytpnpz3kYDbQRGq1bzjmxFaycrKy3m16jxWVl7OjNBK5/OOLt98W+D2XMNk7S2GFubhGEUhH420dMpB42xVy21uO+N3eFsm8sYxszMmCvpV3BVMQEXYJp3AjNBK5lY2UffIFvhdWu7Bp1KsiypuiKXmJm6IzeKGyjuJsnvPYK6keXqYT33+ftm2cdK10HwpxLOHwjq1kkU9sxLim9Xu5MwG2ZxosLZm708w8+h6Vr++lV/+6Q16VQmLUFkhdMXiGdusq4na7P8hiBkcoyj0xXgM9ByG9OKBr+/TyhmvL3Rmp4vTYXLsmmtYBUWZnZ6Cj+Ho1EpWxCc7N2HcRH2yfAzsCcVJ2DEIY/dn7jtnZBQbtMQbkW64eeKj+SfN/RSfk5GwU6WWtI3Ua1jLTZO/y9S/LPSfdInj6Xp5qpWVlyeMjUe1dHNVpIlV079Kc2s7D61p53T5I3MijkTNm9Ty/Ypz+FXPtMQ6XugxW6WfGZzyxAyOURSC5qx4pOtc3cS5NE6/dMCOz+9p+EOdtxINpd4Ao9LN/n9eDH00OIEVce5Nu+3Bq6mTLWxSp0ptTkVTxk2YWBc8fAkps+m1N+FtrH6iFnyu9eq9PwHfuD7/g82Vg9FeeP4+OOB4OHKW7zW877nXqa8IsZ/iqwTdrrUJ41gnm313UydbmHl0PdMWPsUnen+f4gXVs5nvVdzO2FGRFAmcmUfX84371/purxw8Z8MfMzhGUZg9/WCuCLgB+IZSwndQET4KGJiyVb+n4aAb4D7qP56rvPqa5he499l/JMxEe0cX37h/LVfcv9Yt553G4urbUwzzTXJrwBH7JMFiXfDkVcyevrI4hQxBEz/T97liARw5K+Mazgit5Lth1zvzMTZeKM1jk9bS4HPNxc0Fbero4v7KTAMcpZv5ox9KNFrzMHXwoYcVDRhFYebR9dRE/Sfz+T3FV/TuSm3WVWL8nno3aa3vsm9L5niuBl/Nre0pxsYj2fhc/fALfOyQiSmJ8aBjCKRrKzPDzxSnkCGPBmlAwhNKv4a+3hlOGK0tXptR8r2oZxadWpl6OlQlckR1NdHAhwA/b6w/k3qNwcEMjlE0PnnUfr7j9aGAktgi6Vzlg99Tr+8NUCt545jZGcvmUgZYvGxDwrgEVWJ1xXp5+q/vcP1ZRzDOnWnvdww5WbGAmUfX88zcE3lt4ek8M/fEvuUsjpzlTOr0lAQkoELM9UDSr2GQcVAkMecm+TqAk89pi9cSV6Fda3n0gKuY9kQtk+Y+zs7dPbxJgAH2qYiz2f9DDwupGUXj6b++4zv+JhOow+fmVCSdK8gd7vKbT7M8/BEa313LCf/4CfvoZt6WWt44NrNKDXKLgHq/s1VitcQb2dTRxcyj61m8bENiomOXVhL1igbENzqVSjENdXLDsyC1AtcD+dghE1O8uKAQmVd5l34dbq50woft8Vqu7LmU5t5pyN9AcfbX0RXj+xXn8L2K2/dcj7RjSKcoFYTGgGEejlE0gm7KC7tn0RMelTpYRJ2rfPvZj4rs+brXRCNcf9YRzLrom+w7fyOh6zrYd/7GwOq0XGrB3m+/MJM3wTN5uU0dXYmb8oTQDkScpHtOYwNFNdQeza3tTHuilq/v/CJvMRFN007zKsiSQ4Z+3pmXt/G7DuL+NIScQoAZoZUoqR7hN0P384ScmF2/zRiylMzgiMj+IvK0iLwkIutF5Ovu+HgRWS4iL7u/x7njIiK3iMhGEVknIsckbetCd/mXReTCpPFjReQFd51bRHw7phslxpt8FzTXsyXeyHfkP0p2E8kV7vIMUrJ0yvauGN2tS/MWu5w9/WCnX0sSkbAk8gVePiFbJVZyfqGuJhqYA8mOOIn+fMQ5k8ki7JlssB+JN3L8rps5tHcpzR9dlvgb+V3jlngj8/SSRIgsOW8TmItx8YywZ3QbQpsJiWOMTo0/5TyMzO9wJHXM2AwbShlS6wG+qap/FpG9gDUishz4ArBCVReKyFxgLnAVcCpwkPvzAeAnwAdEZDwwD5iCk4NdIyItqrrNXebLwJ+AJ4BTgCdLeE5GEs2t7cxvWe/bVC2du3ccl1FlVCxyhbvWPr6E5fIL6qr2tB4G+OTrd4Dsmf/S9fBXmbu0ldV7f8JfIDKoIgAntHPE2gXI6/7H+LbUOvmF8DNw4wJW7moD6YuOoSaONzFfJ58um0Gton2qzyBzPkvQNX6w+0NMO+fSjO9BULgtmTrZEugRepVxxvCiZAZHVd8E3nRf/1tE/gLUA2cCH3UXuxv4HY7BORO4Rx010WdFpEZE9nOXXa6qWwFco3WKiPwO2FtVn3XH7wFmYgZnQPDr2JiNfEtV+6LsnLU8dl0Tc2K3Uh1Kzal0aaVP+e1uZlc00djRmDFjffGyDYnmYMlziv71yF7sfiJMZayD9wTMRQFh37O+x0yeSdz4xRnOjoSckq+xDU7LgPTJlUkly1nxm+CZtG4+TeqyXeP6Nx7j8fgi9ksy6It6ZnFz5NaA6+FuXycUVJVmDH0GJIcjIgcCR+N4Iu9yjRHAW8C73Nf1QPKkgDZ3LNt4m8+43/4vEZHVIrL6nXf8E9tGYRQqZfOxQybmXCZnLiYgLJS1PHbFAt8n6PGyw/cYPG2z9N406UUBXgiohn9TFetw8hOBN1d1jEKumf3pjKrZE1bqT0fLHOKbdTVR38q65IeEoGt806Evc/if/z/qZU9IbGHkDgC26pjAQ+rWChb1zAouCy9BnsoYfEpucERkDPAQcIWq/iv5M9ebKXl/BFVdoqpTVHXKxIm5b3xGbgqdzR1UwZZM1lyMFxZKbvT16OWwril7eWyBT8rJIpjtHV0JYciQa036knfpjO5Hc2s78QKPRZM9mv4IaOZY96ZDX+aGtDzKDZE7uOnQlxOLzgw/w5oxV/DqqPNZWXk5XxjzHNefdQRTX/lRqn4be/Iz1/VcEFjyre6//aKeWXSRtkwRC0qM8qKkBkdEIjjG5l5Vfdgd/qcbKsP9/bY73g7sn7R6gzuWbbzBZ9wYAHKFyNKfmKf8a3nObWYN7WQLC0HmvJTwM44XFPA8s40xgRVWHgIJb6vXlb/OlQxPJ67Q0nUkU5tPQArtPaXQdNcPmLbwKb7+zhnOJMlk8r0xn3St08kzmVAkse7UV37k6MklEZVupr7yI+eNa+yru94khNIQ2sz8npuY+ZvGQKWCOtlCS7yRubGL6dHM20yV9DKnoonl4Y/w4jHf8S0oMSXo4Ucpq9QEuBP4i6r+MOmjFsCrNLsQeCRp/AK3Wu14YLsbelsGnCwi49yKtpOBZe5n/xKR4919XZC0LaPEZJOS96s8Wlh5Z86qqqylx0ESLH7jKd5QJp1ayfzYBSmTENvitTzQewJzKpoSRvIMd8JmMoUoA6jCBq1nhj5NvWz2D7mFIoEuvgh86O+3JqrHrur+Eu1am1GynBfpO09+n6vfTVAosGsrQYkoz1NsiTcSIlPtGaAutMXxkmZ8heaPLmPaqIeZ9M8bmPZELdc0v5BXqbsxtCilhzMN+DxwooisdX9OAxYCnxCRl4GPu+/BqTJ7FdgI3A5cCuAWC3wbWOX+LPAKCNxl7nDXeQUrGCg53lPnN+5fS1VFKDFjPhm/sFOU3TmlbLLmYoJmwfuNP3mV7w3Sk1zxDMtNEWci4hWx/+RmzmVWxR9TjWTkjoRKgEchygAicJC8GSj/siU+hlWTv5vViCX3y2mJNzJt9y00jnq4sHLhFQtS20aD8977e+QK12UNBSrxNIuZ7il24J/LCY1tYObR9b65u3uf/UfWUndjaFLKKrWVBNfhnOSzvAKXBWzrLuAun/HVwOH9OEyjANIr0zq6Yr5eTsGVR24v+pnb2zh5zL4sip2ToQysj/T6f5m0N7WXfXRcsFQ+jsHwUwJQCQXmIryOnODc9InhVqltYZuORgTGscPXgwkHPN2LwF7s4ufP/h2YxU2RWwn5rO/XWK1gNeRcHoxP64SUcF0OkU/BMeTJKtiehtqM0EpG43O84crE9v1yd0FenylBD21M2sbIm6CkfjqBczD8nqTT5ohUd73J/MhtzD/vMDjydMAxdFO1lnq/bUbHp94sA4yNx7yKe3yr1lTxfTzya43cEm9MMULgaIb5nXMvISoCjE6l9PC9yJ106F6A4/UkG610TwGcG/i3Kh+A+ef797xJNr7e50EGw/t7eOunrwdOHmz7GzgXx98MtGttQjstnTkVTVRJ5neko7eK3/VOYyaFGRFTgh7amLSNkTf53hgW9cyiKz3sFJTgzlEMAI6huyEWoDTsLZ8HISGwHDqopDlXa2QPv1CbquPhZKsVGM3uRBhPxFlHFXo0xAO9J6SoLc8IreSGyB3syzukV+oBwZV8B52coQrdEx7F/J2f3pOQ753mhOnmdzh/pyevgoe/nGSo/E/CzygmE+Tt7q07EjmZICOS/icxJeihjxkcI2/yfbp8XD/Mi8f6Vx5lkCvcg2PovIqntngtqtCrwijdndOjyaAA8SNViLKLV6vO4+bIrSn5nZsit3JdxZ4or3d8W+JjEgYmoY+WZZ9+uXwRqJA4nwn/IcWwXRVpyqgmSzHOQcb75d+kqEJ3RvdjbuxifrbjuMyEvGe0gq5rdDyM3Z844tuCIJ2g/NQmnZDIyQTl7s4//gBTgh5miBZaqjnEmTJliq5evXqwD2NIUoi6wN8Xnp7fRhMhmzTG7u88cQPTFj6VmOU+I7SS70eWUCk9eR93CtHxdO/aSaXuydekh7JyjSd/vo0xzI9dkLjprqy8nIZQYaXT2WiLO+GqSFj4W+Q8JLimjaxT2uZvT7xMvp7J1NdEeaYquLovsZ/5HUya+3jWIwEIiXC6/DElZwaOV+QZKgFeW3h6nxQmjIFFRNao6pT+bMM8HCNvkidYZiNciIaqXxOwtPBb8hPwnIqm/IxNuMp3eHdPnPt7Tkgphw4i12mIG6L7fmRJwhPxy/n0h3rZzMrKy9lQ8Vk06wFle3CUlJL0rPOdck1ODeiNk8z5xx/AawtPJ66a4pmmC3yCY5QmzX084en0q7+PUfaYwTEKwptgmc3o9BbiNR85C446b095s4Sd90k9Wk594oO8FD6H16rO8y8c8KGzR3lITmF3ZGzKeFWsg0+Hfs+inlm8Z/e9NHbfQnuhXTfTqJQe5kfuob4myq7qfbMuqwpxlax5nZTlIRHKC6l/8UFeW0nKifkZixmhlfy/UV8np/BH905WtdzGzt3BRt9TlfD20xJvpLH7lsT1Tg7B9aomwnqzH3ze5tkMcwo2OCISEpG9S3EwxtAhWwFBLg8ohXVN8Px9TnkzOL+fv88ZX9dE76/+k6rY9oRWWb7OU7V084He1WzuzpwnVC3dzI/cw5rKSxJGrL+R5XGyg00dXSyKnUO3+HtXHj/vPYl/Sm6JpbjiWyrdJ5I8l9nTD+bsyv9NFEGsqbyE70eWuMUIOejayuFrruGE3U8HLuJ9N/xyM3vCbZnrxXqV6x5dn/sYjCFLXgZHRO4Tkb1FZDTwIvCSiGT24TVGDEGCjwA7d/dkPKk2t7Yz/zvzaLv2vcTn19B5wyF7SniDqtRWLCCsfczV4IS39vPrNIozbyal8Vl/b+wKf6y8nK2d3Xyr98uJ4oZ0RODz4d/yt959MoulQxGIjk8k5LPsqmB6EL7+rauZtvAp6t94LEUJYkJoR0E5sWhSQzk/PM/GT+PuxnMm8/eFp2dMFvVI7llkDD/ynYdzqKr+S0TOx5nNPxdYAywu2ZEZg4/fnA431HXToS9z+Jo7ElVTKa2Uu1Ll/Ztb21n5q1tZIEsSbQKqu96k55GvUdG7y3/fRZCn90qa/ebHFLtVn8ieazA3djGN8Vt4teo836K4kEBjaH3a057AMRfAAcez68lrqet8izghX1mYbHN7urUCRTPmvlQQ5/rIHcz9F9StaaJCAq57ngTlqtJLl/vbAtqKCYYX+YbUIq4Q50ygRVVjDIDKszGIZFFnBn/Bx+RWyskyJIuXbeAKlmZMuKzo3UVcAr6CYxv6JVGvCivik1kRn5ytb1rRSb4G2SRrMkNKyu7nH9wjkilKhfjM4YlEua/nRN85Pzt0FP8Vu4TZsa/4CmZ6xxbk9aUTl1DwjH8fBQSRPX/3XLmYaMT/714T3RMCzbd1+ICRpWuqkR/5GpzbgL8Do4E/iMi7gX9lXcMY2gSEutoedMIyGuCBJD/5tnd00dza7jydBiT7JR7PvHmCY+C6tvbZOIjASaG1nBRam+FllLoPuXcNFvXMCgwd+VHZvT3jmqd4Yu58ptvGXMYDvSekbFuEhDeUVTBTtuQlQKrAnN5L+Xr3pTlVtRPruMeTyzA0t7bT43NhQsD8GYcl3udqHV5S0o3LY1dmfQDzXccMUgZ5GRxVvUVV61X1NHV4HfhYiY/NGESyGZT2ji7fJ1zIfPK9+uEXGBuNsC2gGVccIUp3ypN84h7bvbNfxqFeNgcaulJOPxOUP1ddAsDPez+eYXT6tG+vVPzIWcyefjAfD6/N8JLy8a48rbOMVgdpbGcvHuz+UM6y5iCyGYbFyzYQ6828CGOrIynhsnw6kZYEP+9+9V3ZFTFyRAQMh3yLBt4lIneKyJPu+0PZ02LAGGY0t7YHGhRBeaHyi+zH5gzvw+/JtyvWmzUpHxYtOGmf7w27kBn+xSR5fs6a+Pu5InZp/+f9xLocuZkbD2fmI4cFGlLPu7qJc+kJj0r5zPv7OD1ovp31HOZ1fz7xOltZczYKNRgdaQUDWdtVJFNsz8K3HUPAly5bC4c0iSYj/5Daz3D60tS57/8GXFGC4zHKgCDtMnBVjkO7He0vd0xx8gfp/WS8qrVtnTFq8NcwKzUhoaCwVjGplB5HbTrtht3neT9dWxNP0EG2aZNOoL4mSuOnLqXizB/B2P1RhLeYyNWxi1mz9ycSPWgcqRsfIqO5uvKBQN24fAkyDK+4XUPTt5u+fNZ2FcnbLLZnUUjBSq4WDkUofhlO5CVtIyKrVHWqiLSq6tHu2FpVnVzqAyw2Jm2TG0+2ZEZoJXMqmoKbhyURV9hFZUphQLdW8G8dxTjZQZwQFdLXiYt76NRKonQX7BEpRZzTUgBxhW06JkU0dIdWUSW9fZfnCSISLawpW5pSN+CUZouk9M9J/jtu0tqU9gNBRCPhVO0zn30lS9xkLO+Ss0otD2mkggnaZrp8UPL1LsVxlBkDKW2zU0Qm4F5tryNnf3ZslC/pM8TzcRBEyKhCq5QeJoR2EHLFKPubN1GFubGLs34exGAYG3BuUenzffYK7UboZauOIa7iW1GWL6pOHqwzul9hxgacZZNEPRm7P1TtldGsLfnvmN6YLiyCAOOqI9REIxlCm17DvrYHr84IOXk5p7AInz7Wv3w6o3V4+jKl8CyC5JamXBQsSJuHRJOR/zycK3FaQL9XRJ4BJgJnl+yojEFl9vSDU0Q6A/vbJJHP/VzEUXkOoX3KobRrLS3xRr7HnYxJa5bmbT9ov4NF0L4jonTGR3FM9xKuq7iLC8K/TVlWgZ5QlEg8e4K8XWtp3H0L0d4w1/cewcwCjs3xHmrZ1HGD4z189GBmPnJYzvU8Q7FcP5JVwXlVy21MXbOIPxLsIdfJFnpVeWhNO1PePb7wOTa5ev30haD+QNmMeV/WGYHkrRYtIhXAwTj3lg3uXJwhh4XU8mNVy23st3oRdbKZDsYwmi7fRlqFkkuBOQgv/CICP6y8nfDQ/PqlEFc4qPuX/D7yNV+F6XZ1umgGKUQnh6TAVXuee2Je+/ZT/o5GwvzvqMsZF/tn7mNHaDlzfbCBWNdE18Nfzeiimo6nhl3o8Sfvx7dbaaHenpGTYoTUsno4InJWwEfvFxFU9eH+7NwoP5pb2+ltuZJPxX9NyI30jGeHM4O9j8YimULXV3VuvF7e4Jmqy4eFsQGIE+IHs46i4RH/Wfv7sYV/UuurcdajoYzy5ELKhYPmuFwXP5vvVdyeER5NJzS2Ibs3smJBTmMTV1KqGvtU7myexZAiV0jtjCyfKWAGZxjhSdAskl9n5DwqpYceDZZUyYe+GizvCXhGaCV1ec6SHwqEJe7ctH/nHxbapBNYHPsMN4/+vylP8F1UcVXsSxmJ+0LaLwfd3Jt7pxFXZU5FE3WyhQ5GZ3i3PeFRVOTKTeSZQ0k+h7HRCNMWPlW4jM2Rs8zADBGyGhxV/eJAHYgx+Kx9fAkL5dbABHuYeHEVjPNAgesq7uKToWcZLzsGLR9TDO8uHfHKkk+6NiP85HUbHVdd6YSHkp7gX3zv11i+6t0QTw2HFdJ++cIxz3Fx9y+ok80plWdhEVrijbR0p7a29gzQJp3AjzmP63Pd4INyK0kkT06NhISd3T10dDneq6dWAJh22jCikBzO6cBhQGI2maoOuVlNlsPxZ1XLbRy25pqsoZT0m25yK+VSUoqbfT77SUi1aC0r4pP5TPgPvtenT0Y4Lc/QdNcP+PjrP2QcqUa1JzzKmU+TdoPvl6jluqYM4dROreRavYRRx5zLQ2vac3Z1vemcydn355dbSaJTK/m2/AdLdx1PXU2Uzu4eX6XoPuV1jJJQ8hxO0o5+ClTjyNncgVOh9lyOde4CPgm8raqHu2PzgS9DIij9LVV9wv3sauBLQC9wuaouc8dPAW4GwsAdqrrQHZ8ELAUm4ChXf15VsweeDV+aW9uZumZRQcYGBq76ayD3483Z8Ztvsib+fuZV3OPjaQlx1RSjk9MYJxmb5tZ25r12GB+SUYwPpU6Qrejd5Xg3aQanXyrMKxZkqHRXSzcLqh+ieuZ3mfLu8SxetsG3DbXH4mUbsu/fPd7OJ69lVOdbdDAaVRgnO9mkE/jfAy/l+ou+yfXu4pPmPu67mZLL2PiQbMwvHPMccyL3U931VmnzQ1mU2YcT+U4A+JCqXgBsU9XrgA8C78+xzs+AU3zGb1TVye6PZ2wOBc7F8aBOAW4VkbCIhIEfA6cChwKfdZcFuMHd1vuAbTjGyugDi5dtyKkgPJilxQOJiJPMTzY2Xt+fmyK3UiM7M65FSFKNTVzhnt6PZ99RUkfT4x/5COtD5wR3My3ybPUgnbzqrreAPXNfbjpncuA28jIER86i+qq/0jJzPWdEf86U7iV8OPowqz/1B2Zd9M2URfOWsSkxyQrVZ4RWMid2K9Vdb1JSfbQRpMOW7zwc79vVKSJ1wFZgv2wrqOofROTAPLd/JrBUVXcDr4nIRuA497ONqvoqgIgsBc4Ukb8AJwLnucvcDcwHfpLn/oYFxeoVsqmji22VY5ggO3IvPAKokHiit8+xob/x+fBvkwxK7hB0yFWq3qoB1zQ63vnt3mj2pSv7RKb+zClJo7m1nak6wd+4pe1n5tH1zG9Zn8irJFOIIcjHG0uf+wVpeakB8gCSq/fmVDRlev2ePlox951Nhy19P0PcE8rXw3lMRGqARTjhq9eAX/Zxn18VkXUicpeIjHPH6oHkDGObOxY0PgHoUE20g/TGfRGRS0RktYisfuedPNroDgGK0ivE1bZ6ddR5KdIrxp421KnGJn/qZAvX9VzAbk3VAuuVCjj1BueNr0hkGkWerR6kk9eplax679cylp8/47DcemaPXQnXjYf5Y53fj11Z8HH5dQdNTCodQA8g2XMLEkjts8cZJDKar1rCMPCEcs3DmQq8oarfdt+PAV4A/grc2If9/QT4Ns5j4reBHwAX9WE7BaGqS4Al4BQNlHp/A0G2XiF5eTlJSV2B0jeJGYKkJ/ALYZNOcEJyMZgfuYdxrnhpOLr3noUCbjROHknYVb0v1acuoLl3Gov7Ui7sd1wdXbTjHFdy5dminlksX/Vurt+/PWXb3utAT/qxK2H1nUkH37vn/QHHF/Q0HugJFeIB9JO6mmgidxWssKGOwSjEu0gvovCMBeSvljCA16FU5Aqp3QZ8HEBETgAWAl8DJuPcwAuSt1HVxBRmEbkdeMx92w4kS9c2uGMEjG8BakSkwvVykpcfEfS7V0g+T9dlRqmq1Yq93Tjwg/g5ifejksVGu7bmvNH8Uyby7Jm/T2nPfT9LqavazKbOWm761bnApXue/gu4qXs31PTSZ+fA/R9YggxBc2s7n1z9f/1vIqvvdHrIeCHI5BtsoTfHLB5AsVtQJ4f2FvXMYmHkDv9imkLPJ5uxOOlaf7WEdM92GChS5wqphVV1q/v6HGCJqj6kqv8f8L5CdyYiyXmfTwGejGoLcK6IVLnVZwfhVMGtAg4SkUkiUolTWNCiTi330+wxeBcCjxR6PEOZfidZc8yRKEdKVbig4NuyOdv+lOBsjgCzK+7n1arz+GHkp755gLYHr2b+zk9n9KwhEmXfs76XuGmufXwJC2QJDaHNCfHMBbKEtY8v6VOIxU/yP5l8H1i8kG5Ys00ETrtCfe0PE5DD2h0ZW/QW1MmhvUfjjSyKXOoIo/pRyPlkMxZ+Iqp+0jxBubwi5vhKTS4PJ5zkRZwEXJLvuiLyS+CjQK2ItAHzgI+KyGScb+Lfga8AqOp6EWkCXgJ6gMtUtdfdzldxevGEgbtUdb27i6uApSLyHaAVSPLrhz85k6zZGEIx34HAK4F2QkybEfIzbkGLqLrx/6S2z+nUy2aujd1EB2MQ2YuxuoO3pZY3jpjN1KQbzcXdv6A6lGqwqqWbi7t/ASuiBYdYPEP2zabn6fWZg5fvA8vax5ewXH6R17Ip+D3o5PLSTroWHrksQ8VaYv9mrt7OSZVrUyawLl5W2S8vJ9WjOx24zsm7+D1i5Otd5Aqb5aOWkK8nVMZknfgpIv8NnAZsBg4AjlFVFZH3AXer6rSBOcziMZwmfvYlnNDc2s5HHvkA4/h3xmcDNcGynIgrdDCGGnawTcewl+zKq09N0LXqyzXM1hcmPr+GkM+NLo644Qm//1+B+R1Z9+kn3ul1e6lP/y6lG4SDTqZz1c9z6q0FMnb/PUYlX/HNGyY54cg00ifddmolV8cu5ubvXZ+xbL/ob7+bYomMDmKVWjEmfuZUGnB73+wH/EZVd7pj7wfGqOqf+7PzwWA4GZxCaW5t598Pf53PhZaPOMMSRF+NbDENDsCW+Bi6GEWdbOZtmci+Z30PjpxF5w2HuPNAUumM7kd1ZUW/boLeA0t7R1d6azHA6XGzZPJrTH1hXpon5bd0gefu3mw7n7zW9/wS5+HdUIM8DB/eYiL7zt+Y54HkSTEMxhAvaR4QgzPcGMkG56HrzuGs+K/N2JSIuEInVYyRTJXkXpxGZaL+vYAybtbezQwyZGgUkOh4OOxT8Px9GTfBVUdcxxUvHZS35ztt4VOBqgLPVF0ePCHV5xy60rq+ZqMzuh+jOt8iJFnuQd51WLEg77yjIkgOD69PDHGD0V8GsuOnMZRx6//N2JSWkECEHt8OnuFQBWuOuYFN1Pqs6eMZJOViWo9aQAd77ZHLASe89Px9cNR5Kcnmpv1mM+t/GwpKpGcrFMilQJGMAg/0nkBbvDav7q6jut5ik07IvlByJVd6R82ALJqUKol+5CzHc5zf4fweQcamWJjBGe48diU8fAlsf8OMzQBQJb2IX6FAPMb+f17sO+ky8Obslv1esOrd7IhX+Rull3+TuAk2f3QZV/3tkIzAkzc/y8Nr+zxp7uNMW/gUY6ORwPNJVnTOhaew0Nh9C+15rBdXoU42E89lnIIquaZcZG2dhxj5StsYQ5F1TalzIYycFKNwIugp7l36DjdHbmUbY9illYyTnXRF96W7awc1PkUcjG3gukfX0xXrpa4q96z3xcs2BP6lPS8mvVggm0An4DsXRRUQf/+iXjazsvJyVoaO5ezw7zNEQpOpkD2GOa4EVwdmq+QqcHKpMbiYhzOcWbEAMzb506Mh7un9eP4eSAAa8v+3EnF+xssORtHNt/ga1Vf9lZqzfuj7pL7qvV9LSPYHehrRcQmPJZvx8Mqd/RQqstESb2Ru7GK2JYf0AoyN91lDaDNn6tO83jBzj0cSHe9qyAlI5jygkMBWHUOXj+TO/J2fDg4JWphrSGEGZzgzhGYglwMhlHk9FzE3djFt8VriKrTFa/lj/DB6NORIzuQwPnGFVXo4ubSCqqWby+L3OW8CJv5d8dJBCaXqOtnsu+/eXf9i5a9uzWpsBBLzs/oi998Sb2SnX0gvC1HpZvTrK/YYg6tec37md0DAZNFxoZ28eOx36Izul7j2c2MX87Mdx/V7QmepSA9PluMxlhMWUhuGNLe2c92j63k0PoGG0PBpyVxqvAR2suzLjNBKFkbuSAn/5OII/Stkq7xyqZMtSS2Va5k9fRng6pbd18UZoeXB0iouYe1hodxKdyie0XIaHGNz/vEHJKrUkrXCCiFIyDJb3599NOC7FzAJMjS2gakzvsK0lw6ifXfqMRakEzhA+IUnrUtpdszDGWY0t7Yz+8Hn2dYZY1HPrAy1YsOf3RpmUU9mOMZXoj4LiuS9/DYdnVJNNvuB55n94POJsfmRe/LaltdOYUZoZcp4fU2UG8+ZzHdmHpEYyyVtA5m+WTQS5i3xD+n1SijQ83nbWyddJfmgk7Mm+/utEzhAZBPQNfwxgzPMWLxsA7HePU/XYcvh5MVOor4eQqBEfQB+qgBBpN+oY3FN/O2uq7groTCdD9XSzZyKPZJFXmtmPyFOTyvMj2gkzPnHH5DRJqD9mDkZ+ZW4QjhAukcV3jhmtr/em085d/IEypI0YwtqDdAPhophLCcspDacWNfE/Z1XU1e1mW06hhrZSTiP0I4BNezkUxUr+WaoKUWXK1iiPoAC8hw17PQdnxFayefDvy24Wq5OtiRef+yQiVmX7ez2l+/pivXy9F/fyZgs2swnaVr9D76uS6mXzShk7RPUXVnD1BlfcW7ufnpvXjm3Dzcd+jJ1axaxH3v+DsvDH8lPJ9CPbK0B+lFkEBSeHOgupUMJMzjDBfefqiHk/ANY905/gsqet+lovhveky9pkM0sjNzBA70nMCv0B6LkFyaT6Hi0a2tedido0uOciqbAm7mqo6Pm9yCRvL2n/+o0GmxubQ/s2hlEei5iT67iQzzAh1hZeXnW3GBPeBRVZyx23hQqqb+uyZHSEed73CCbuaHyTr7+7q2893f/BY/0ofy5RH1k+iWgO0KxkNpwYQj2tykW+VSPgVNi61f23KmViJCRL6mWbk4KreWq7ot5i4nEVdiho+hV/332hEfBqTewnb3yOmZvzkpy7iUSlqxSMlt1DN+I/afvOSTnoNo7urim+QVmP/B8QcbGIzkXkZ6ryFZA4FWWNfe6ur6FSur7fI+j7Oa9r9/f906XJeojk7VLqeGLeTjDgXVN6PY3RmzTzmyhpx4NEUITXS1b4o2sib8/o9vlTZFbfdevky20xBt5dFdjIjvjVa6lTIYEKo4+H46cxbylrVyfo7rMO2bPkyLmVMfd94E30D/7R+biCtf1XEBLvBGJwey0c0jPQd377D/6lcHzchHpOYmgMGO71tLY7ei//T+vouygk1M7gnocdLL/TgONgE9vnYe/vEf2Jpunkm9HzT4Q2KXU8MUMzlBnXZMj7jjYx1GGdGsF/xW7JONG7Nft8nvcxRgyZ8V3MJpoJMT40VWJeL1f5ZqAk5cAxlVX0tVdmQjD5crFeAn/P1Z8jKmv/Mh3GVX4ee/HE+fySLyRR9I7dqavk323OfFyEem5ikU9s/h+ZElKG4durWBRzyxmhFY6xrxrM9y4P3T756m8a5VBkHEIIp98zDDoIzNcsJDaUGfFgqzyISOZ3Rr2rTxLZ0ZoJdU+xgZcBeRYnJ27e4iEHcsRFFKKd7Qx/zvz+FbvT5gQ2pFQFsgn3Ocl/DXgCV+BeT0X5d5QkUjORfiVUmuaOVOUY0N/Y2HkjkR3Ura/4dvDBgj2ZA46mUz/LofFztV5M9+OmkbJMYMzhGlubSduagKBjJHdGTmSZMZVR4hGwlmT9OPEeULv6IqBwujKMB2M8V22g9Fc3P0LKjW1PYFIbm8jjvDhXU8HFhIUIqJZKJ87/gBuOmcy46r3iHhWVey5NaTnKr5V+QBVkjr/pEp6OT/8VP5zlqLjMsfWNTkl0ylXS2DSCT5K0Wnk+j8wCZyywAzOEKW5tZ3ZDzzPtvjowT6UssXT9fp+ZAlrKi/h1arzEgYoEhbmnXEY1591BHWhLYHbSDYAsbiys7s30GNJtJbuA97kzd/2Ts5ZEJAv0UiYaCT7v/jnjj+AihcfZMqvTmBN72cS16ejK5YiJzPz6HqemXsiry08nX0DWhYEzcnJG9/CF4WtryZ5KAGUqiWBUVTM4AwxPO2mK+5fy6n8kb3Ewmm5qJQeJoR2EHIN0MLIHZwd+X/MDD/DzN9ND5ysGVd8b/TjAkrOx8nOQE+k16dHTjpeVVy6lpvXfroQvIqpXbHsRmDXn5cyJ3ZrIgzmXZ8ZoZXBs+YDbu4SKkDVomtb5li2ajLPQznrdmtJMIQxg1NG5BIC9OZDJCevkxO3Rn5USzf/Fb8raQZ8JqrQySjfz4KMilct5ueh3Nt7Ysa4H15VXGP3Lbxn9700dt9SsLGpiUYSEzezTUIU4AqW+paD/zDyU16tOo/7O7+cWX7s1wwtEoVjv5B3kzRfo5VPCbXlY4Y0ZnDKhGRjEtSpMd/5EEZuxod2+M5bShajHCO7fDXKgoyKV5p8rV6S4aEkq1Bn400mJIoT+kpHV4xv3L/WmYcz/WDf7YVwMiVB36EKiSc8now5L0E3/U/+MGP8lXefkyGJ06WVrHrv1zJ3GmTI0r0Xy8cMWUpmcETkLhF5W0ReTBobLyLLReRl9/c4d1xE5BYR2Sgi60TkmKR1LnSXf1lELkwaP1ZEXnDXuUVkaPezzEcIMHk+RFAi3Ci8f00y6d+iaulmfuSelDGvR4xnVLZF3sWiyKU8Gm+kviZK46cu5Zzq2309lNGhXYEFBHGFZ959KYvPPiolgd8XFGceDsDis49idGVquCvqvs+rGMGvCiz9pg+OjM3Dlzivz1oC33iRC/55DlelhQivil3MFS8dlLkf816GPaL9+e/MtmGRE4AdwD2qerg7tgjYqqoLRWQuME5VrxKR04CvAacBHwBuVtUPiMh4YDUwBed/aA1wrKpuE5HngMuBPwFPALeo6pO5jmvKlCm6evXqop9vf5k093HfG5EAr523E1YsIL69jU3xCayIT+Zz4RWmkxZAXKGDMdSwkw5Gsxc7ifTjWqnC12OX0hJvJCQwNhqhozNGXU00Q3PMI1263m+yqN9xfzj6K1/JlP4wrjrCjl09xHx6OedzXA7iGJd01jXBk1dllj9HonDGLUy6b3Tw93rh6XmegVEOiMgaVZ3Sn22UbL6gqv5BRA5MGz4T+Kj7+m7gd8BV7vg96li/Z0WkRkT2c5ddrqpbAURkOXCKiPwO2FtVn3XH7wFmAjkNTrkSJAR44Zjn4NHbINZFCCfEcYEULuw4XMinBXRIYJR2c0XsPwH4fmQJkDvXFbRtESdf1tLdSFic6rag2eXNre1OP5uOLsZGI4yKhOjojPGtygeozqHHtklrae/o4or71+Y81kLwuob60RJvhBgJ5YW4CBV+1WZ++ZV0UcxkXK+oruYWE7g0Egz0BPV3qeqb7uu3gHe5r+uB5OxtmzuWbbzNZ9wXEbkEuATggAMO6Mfhl44gIcA5kfuhK/UfdqQaG4CdjCKk8ZxP5Mly/fkWVniTNP2urzcxMxZX5resTxiVC8c8x5zI/VR3vUVndF9W7vw07d0fApxcitcAbd+12fNtQRVxA0Gy8sKZoZXcPPr/5jcrP5d+3/Y2Zp9pApfGHgataMD1ZgYkJqSqS1R1iqpOmTgxu2z7YBEkBBjtemuwD61siCs81Lsnf5IrGlwX2kJ9ljk2hZA8H6ejK0Z7RxdnhFYyJ3Yr1V1vAkp115sskCUp+TUvl9IZ3Tdw2/E02ZrBZPXen8g/j5JrsuXYBhO4NFIYaA/nnyKyn6q+6YbM3nbH24HkWV0N7lg7e0Jw3vjv3PEGn+WHNOlCgNc0v8AJOprx1moAcEJlnw//FoDG7lty5h9CYxvo7O5xDUJ+bGMMo7Q7ZZtBEy/9NNU8zypZq02BRbFzuCb80xQZIlVnf/NjF/TJ2ERC4puX6SsJz+PIE/NL1GfTPUvyikzg0vAYaA+nBfAqzS4EHkkav8CtVjse2O6G3pYBJ4vIOLei7WRgmfvZv0TkeLc67YKkbQ0bfvmnN/pVcTUcCQlcEP4tr1Sdx82RW+nSSnZoVeZ1cm94i2LnZJQwq/q71qrwaO/xeU+8DCopTm6E5vGzHcfxHfkP/ikTE9v9euxSjtm9hOXhj1ATza8qzYv21UQjWSXGCo269snz8CtjBoiOt+oyw5eSeTgi8ksc76RWRNqAecBCoElEvgS8DnjfyCdwKtQ2Ap3AFwFUdauIfBtY5S63wCsgAC4FfgZEcYoFhmzBQDpe4rlXNXBW+0hGBLwi3wmyg06t5Oe9H+fsvdYT7XqLf1LL9Ts/w+onamnfcRxbQ90Z7QjmVDRlNBETgZNCa5nXc1GGmrQfQTL9QXpoP9txHD/juNR9Ap8+1rnJ/8ItY07moH1G09kdZ1NHV0pV3LSFTwX2uYlGwnz62Hqe/us7bHLndQVx0zmT++59eAZlxQInvFZoYzRjxFGysuhypVzLoj3Sy2lzdVc0HNritXwy/BN298RTEtRCqjfjyefXhzYH9JwR3rP73rz26RfS69TKgqVo6t2KLb9qLgFu9DEKQWX0kGlEpi18ynfb46ojtF4b0JfGMNIoRlm0KQ2UGekTQBf1zGK3FqBRNUKpky10dMUy5q4oe8JLnoFoCDA2EOyd+JE+AbSvumebOrp8DYJ3/H56ZkFlxfU10Qzj5NdeIBoJM++Mwwo6TsPoL9a3q8xI767YEm/k7Pjv+XBo/Yguh85FNkOhODfiOZ2ZSf5kcpUmJ5qLyWY2aW1Cyiaf8Fs26mqivLV9F70B0Yb07wQEl9H7lRt7Bsgr5c42YdUwSokZnDIjfQLodRV3mbEhVeMsnWzy/TNCK/lW5QPsu2szGsoePhYI9E7Sw2fpraH7SiQkzJ5+cNbJnjU+MjeFGhGrFDPKATM4ZURzazs7d6dOUvxceMWINjaqsJMquokwjh0Z1WjtSZ5GOgkj4bV6zrGv9iy6YvmWQBfKmFEVzDy6nsXLNgSG1Xbs6qG5tT3DYJgRMYYalsMpE7xigeTKoxmhlYG9WkYSIZTxsqdls/fTRWWgsQF/IxFEXGFFfHLg54WUQBeCJzvjl2fxiMXVvy+NYQwxzMMpE5KLBRKVVLJ5RHs34LReDjIauTyMICOR7CV51zck8JnwH1gTf7+vASu0BLoQJl/3G7Z3xRgbjQQKdvrlcQxjqGEeTpng3VBSKqlGuLFRBcnh4WXzMIKk99+UWtq11rcVgae/lk62Hjj9paMrhrJHe80PE7s0hgNmcMqEUW7v+ULCQMOdrTomZ7+WbB5GkJH4YfzcQI21IANWrBLoXCSXcXuY2KUxXLCQWhnQ3NpOVyzOjNBK6q2LJ+AYhut6LgAI1EvL5WGkS+97KgMt8Q/xX6PuZ1/eyVgnmwErRgl0Pnhl3FbCbAw3zOCUAYuXbWBGaCU/jPzUwmjqtCD4VuyiPd6DazTqZTO9hAgRT5kHk40gI/G97s/4qgTkGyL73PFOmws/OZr+Ul8T5Zm5JxZ9u4Yx2JjBKQPaO7p4tOoeKsSn8dUIIDmJ/8f4YVwQ+++Uz0vhWQR7P7n387njD+A7M48A4LHn3wzUNEunJhrhk0ftl9A4q/HpxGnhM2M4YwZnkLmm+QUAxjFyRTqTvbopoZeZEVo5IL1h+mLIxlVHEsYGYP6Mw3K2g67P0YraFACMkYIZnEGkubWde5/9R0rDrpFOXydTBsnOFJuOtHbNnnH4ZtPzvtI0ucJjNnnTGEmYwRlEFi/bwBluGfRwzN0EtWvORaGTKUslO+N7bD7lyZ7BsFbKhpEdK4seRDZ1dA3rMmgR+tRArtDJlNlkZ/JF2FMIEEQ2A2KtlA0jN+bhDCJ1NVHquoZ3GbQC23QM49jh6+2ke0F9mUzZX9mZ+pooHztkIk//NbNMOplcBsTCY4aRHfNwBpGPHTKRDsYM9mGUlDgh5scu4J7ej2d4O91awT29H+9/P5mAyaH5eEqCo2P20Jr2QPFMcCrMzJgYRv8wD2cQ6V3bxGiGt0ZWhcRZGLmDubGLWRN/v28Z8rx+7mNRz6w+z6mpq4lmNL1LJxIS5s+wZmWG0V/M4AwSza3tXBa/j6pQ8I1uuODlUxq7bynJTP2+zqmJhJ1eNN/I0osmW0mzYRiFYQZnEGhubefK+9eysWp452+S6a+Mfy76MqdmdGX2XjQ2498wiovlcAaBtY8v4Q+Vl+dsCDacKIaMf7HZ3hXci8ZKmg2j+AyKwRGRv4vICyKyVkRWu2PjRWS5iLzs/h7njouI3CIiG0VknYgck7SdC93lXxaRCwfjXApmXRPXxG4eFu0H8i15LpaMf7Hx5tRYSbNhDAyDGVL7mKomx5TmAitUdaGIzHXfXwWcChzk/nwA+AnwAREZD8wDpuBU364RkRZV3TaQJ1Ewj15BhQzdLp6ekWnXWqLsYoJkl+Tp0VBJZPz7S7oHYyXNhlF6yimkdiZwt/v6bmBm0vg96vAsUCMi+wHTgeWqutU1MsuBUwb4mAtGYzsH+xD6jKojrjlp9300dt/CdT0XEM9iOzu1kitj/9EvYyPATedM5u8LT88agqyJRvjc8QdkhMYiIWFcdQSAsOtSmgdjGIPDYHk4CvxGRBS4TVWXAO9S1Tfdz98C3uW+rgfeSFq3zR0LGi9brml+gW/7ddgqc1SdVs+/6D2JeT0XJcZb4o0c2/s3Ph/+LSFJXX6rjuG6ngv67dmcf/wBCcNQVxPNmdyf8u7xJoZpGGXKYBmcRlVtF5F9gOUi8tfkD1VVXWNUFETkEuASgAMOyC5fUkruffYffLtq0HbfJ+IKV8QuDTQc83ouCpxf01/SlZlnTz84p16ZhcYMo3wZFIOjqu3u77dF5FfAccA/RWQ/VX3TDZm97S7eDuyftHqDO9YOfDRt/HcB+1sCLAGYMmXKoCRQmlvbmV9x12Dsus/EFZpkep+bnAURAnJ1/omEhHlnpE629AyJeTCGMTQZ8ByOiIwWkb2818DJwItAC+BVml0IPOK+bgEucKvVjge2u6G3ZcDJIjLOrWg72R0rO5pb29n+0OVcEP7tkKlMi0uI0Kdv59z5TdT7KCT3lfqaKD88Z3KiIiwccEHGjKrwNSQzj67nmbkn8trC03lm7olmbAxjCDEYHs67gF+Jc6OpAO5T1V+LyCqgSUS+BLwOeHW0TwCnARuBTuCLAKq6VUS+Daxyl1ugqlsH7jTyZ37LelaHnxoyxqYnPIqKM38ERzp/Ar9QVl/wwl/JYa9Jcx/3XTa974xhGEOfATc4qvoqcJTP+BbgJJ9xBS4L2NZdQNnHqTq6YoSrhkj76LH7U3HStQljA7mbjIVFfMdrohFGV1VkDX8FFQL49Z0xDGNoY9I2A0A5dvTMaI4WicIZt6QYmmSyNRn79LH1PLSmPWN8/ozDcoa88ikEMAxjeGAGp4Q0t7Zz9cPreCZyT9mF07Yxhs74KOpDW5CxDZDm1fiRLWnf13JkKwQwjJGDaF9aMg5hpkyZoqtXry75fppb25n9wPPE4sprVeeVlcHp1Ermxi5m7+POSyk7NgzDCEJE1qjqlP5swzycErF42QZi2abhDzDJkjSLembxx1Efo9WMjWEYA4gZnBKRrXvkQNGrgoDvZEyxKjDDMAYYMzjDGAHes/te38+sCswwjIGmnMQ7jSIT1IPGqsAMwxgMzOAMA1Qze9Mk96CJhBxdMuv1YhjGYGIhtRJwTfMLidelmoPjGZidVPGt2JcAEgKabzKBRbFZrNn7E9xkJcaGYZQJZnCKTHNrO7949h+J93MqmkpWEj1p930p71u6GxlXHaH12pO5uTS7NAzD6DMWUisyV96/NuV9nWz2X7CftGttxlg0Es5QWDYMwygXzOAUiebWdqYtfCpFdv+6irv63WstKD/z5pQ53JSkumy5GcMwyh0LqRWB5tZ2rrx/bYaxKUY7gt1UMCd2SUp+ZtOxc5g64ytMBTMwhmEMGczgFIHZD6zNaCh2fhHaESjwwrHfY81LB/HhjkbTGTMMY0hjBqefNLe2E/PpPBDO2dMyOwrIlC8x9ZNf4ZkZ/dqUYRhGWWAGp598s2mt73gvISqyGJ2M9gDJRMcjp96QU73ZMAxjKGFFA/2gubWd3gB9znt7T8xI9nsFAFviY7in9+O0xWtRhR4NEVfojO4HZ90OV71mxsYwjGGHeTj94KqH1gV+Nq/nIsDJ5YSJ00uIe3tPTIwDzHN/11tuxjCMEYAZnH6wuyd7nmZez0UpBgYgJPDq9aeX8rAMwzDKEgup9ZHm1vaC1xHgh7MmF/1YDMMwhgJmcPrI4mUbClq+qiLEjedMtrCZYRgjFgup9ZFNeTZYE+D84w+wVs6GYYx4hrzBEZFTgJuBMHCHqi4s9j7i145NKWFWhbqaX2Xt6hkSOO8DZmgMwzA8hrTBEZEw8GPgE0AbsEpEWlT1pWLtwzM26XNm/tj5KQ6L3E9XrDcxFo2ETc/MMAwjgKGewzkO2Kiqr6pqN7AUOLOYO/AzNt7Y9WcdYeKZhmEYeTKkPRygHngj6X0b8IH0hUTkEuASgAMOOKBoO595dL0ZGMMwjDwZ6h5OXqjqElWdoqpTJk6cONiHYxiGMSIZ6ganHdg/6X2DO1Y0/PrR+I0ZhmEY2RnqBmcVcJCITBKRSuBcoKWYOwgt2J4wMMk/oQXbi7kbwzCMYc+QzuGoao+IfBVYhlMWfZeqri/2ftKNS3+7eBqGYYxEhrTBAVDVJ4AnBvs4DMMwjOwM9ZCaYRiGMUQwg2MYhmEMCGZwDMMwjAHBDI5hGIYxIIiOsAklIvIO8HofV68FNhfxcIYaI/n8R/K5w8g+/5F87rDn/N+tqv2aOT/iDE5/EJHVqjplsI9jsBjJ5z+Szx1G9vmP5HOH4p6/hdQMwzCMAcEMjmEYhjEgmMEpjCWDfQCDzEg+/5F87jCyz38knzsU8fwth2MYhmEMCObhGIZhGAOCGRzDMAxjQDCDkwcicoqIbBCRjSIyd7CPp1iIyF0i8raIvJg0Nl5ElovIy+7vce64iMgt7jVYJyLHJK1zobv8yyJy4WCcS6GIyP4i8rSIvCQi60Xk6+74SDn/USLynIg8757/de74JBH5k3ue97ttPxCRKvf9RvfzA5O2dbU7vkFEpg/SKRWMiIRFpFVEHnPfj6Rz/7uIvCAia0VktTtW+u++qtpPlh+ctgevAO8BKoHngUMH+7iKdG4nAMcALyaNLQLmuq/nAje4r08DnsTpznA88Cd3fDzwqvt7nPt63GCfWx7nvh9wjPt6L+BvwKEj6PwFGOO+jgB/cs+rCTjXHf8p8J/u60uBn7qvzwXud18f6v5PVAGT3P+V8GCfX57X4ErgPuAx9/1IOve/A7VpYyX/7puHk5vjgI2q+qqqdgNLgTMH+ZiKgqr+AdiaNnwmcLf7+m5gZtL4PerwLFAjIvsB04HlqrpVVbcBy4FTSn7w/URV31TVP7uv/w38Bahn5Jy/quoO923E/VHgROBBdzz9/L3r8iBwkoiIO75UVXer6mvARpz/mbJGRBqA04E73PfCCDn3LJT8u28GJzf1wBtJ79vcseHKu1T1Tff1W8C73NdB12HIXx83RHI0zlP+iDl/N6S0Fngb52bxCtChqj3uIsnnkjhP9/PtwASG7vnfBMwB4u77CYyccwfn4eI3IrJGRC5xx0r+3R/yDdiM0qGqKiLDum5eRMYADwFXqOq/nAdXh+F+/qraC0wWkRrgV8Ahg3tEA4OIfBJ4W1XXiMhHB/lwBotGVW0XkX2A5SLy1+QPS/XdNw8nN+3A/knvG9yx4co/XXcZ9/fb7njQdRiy10dEIjjG5l5VfdgdHjHn76GqHcDTwAdxwiXeg2jyuSTO0/18LLCFoXn+04AZIvJ3nBD5icDNjIxzB0BV293fb+M8bBzHAHz3zeDkZhVwkFvBUomTNGwZ5GMqJS2AV21yIfBI0vgFbsXK8cB21/1eBpwsIuPcqpaT3bGyxo3B3wn8RVV/mPTRSDn/ia5ng4hEgU/g5LGeBs52F0s/f++6nA08pU7muAU4163kmgQcBDw3ICfRR1T1alVtUNUDcf6fn1LV8xkB5w4gIqNFZC/vNc539kUG4rs/2NUSQ+EHp0rjbzgx7v8e7OMp4nn9EngTiOHEX7+EE5teAbwM/BYY7y4rwI/da/ACMCVpOxfhJEw3Al8c7PPK89wbceLY64C17s9pI+j8jwRa3fN/EbjWHX8Pzk1zI/AAUOWOj3Lfb3Q/f0/Stv7bvS4bgFMH+9wKvA4fZU+V2og4d/c8n3d/1nv3tIH47pu0jWEYhjEgWEjNMAzDGBDM4BiGYRgDghkcwzAMY0Awg2MYhmEMCGZwDMMwjAHBDI5h5IGI7CsiS0XkFVcO5AkReX+B25gpIoeW6hgNo9wxg2MYOXAnif4K+J2qvldVjwWuZo/WVL7MxFEYHjBEJDyQ+zOMbJjBMYzcfAyIqepPvQFVfR4Ie71UAETkf0TkC+7rheL02lknIt8XkQ8BM4DFbg+S94rIZBF51l3mV0n9R34nIjeKyGoR+YuITBWRh92eI99J2t/nxOlps1ZEbvOMi4jsEJEfiMjzwAfTj2UgLphh+GHinYaRm8OBNfkuLCITgE8Bh6iqikiNqnaISAvOrPYH3eXWAV9T1d+LyAJgHnCFu5luVZ0iTmO4R4BjcVpJvCIiNwL7AOcA01Q1JiK3AucD9wCjcXqWfNM9ljuTj6Wf18Iw+ox5OIZRfLYDu4A7ReQsoDN9AREZC9So6u/dobtxGuJ5eHp9LwDr1enfsxunydX+wEk4RmiV22LgJBzJEoBeHFHSvI7FMAYKMziGkZv1ODf3dHpI/R8aBYmeKcfhNOv6JPDrPuxzt/s7nvTae1+Bo291t6pOdn8OVtX57jK71Gk9UKxjMYyiYAbHMHLzFFCV1KgKETkS56Z/qKsWXIPjZXg9dsaq6hPAN4Cj3NX+jdPOGlXdDmwTkQ+7n30e8LydfFgBnO32M/H60b87faEsx2IYA47lcAwjB27u41PATSJyFU6I6u84+ZYmHLXl13DUl8ExKo+IyCgco3SlO74UuF1ELseRub8Q+KmIVOOEyr5YwDG9JCLX4HRtDOEofl8GvJ62aNCxGMaAY2rRhmEYxoBgITXDMAxjQDCDYxiGYQwIZnAMwzCMAcEMjmEYhjEgmMExDMMwBgQzOIZhGMaAYAbHMAzDGBD+f5us2IcapIbHAAAAAElFTkSuQmCC"/>


여기는 프로모션별...

<img src="data:image/png;base64,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"/>

##### 위 그래프 결론에서..

- 각 그래프는 각 판매와 소비자간 휴일등 시기, 이벤트에서 어떤 상관관계나 추이등을 보여줌

- 크리스마스 연휴가 공휴일보다 영향력이..

- 방학은 별로.

- 프로모션은 매장에 따라 어떤 영향인가 보여주는(마지막)


#### 5. 매장별 통계 데이터 확인

- 5.1 spc = sales/customer 로 새로운 열 생성  >> train.csv df 

- 5.2 Groupby 이용 store로 sales, customers, spc 평균값 만들어 보기 : store별 평균값 data 



5.1에서 spc라는 새로운 열을 df_train에 생성해서 내용은 sales/customer로 넣으라는것.  

5.2에서 Groupby라는 기능으로 store에 sales, customers, spc 평균값 정리. 단, store별로

- 시작하기전에 train encoding 진행. holiday 진행 필수!



먼저 train 인코딩  

<pre>
0    855087
0    131072
a     20260
b      6690
c      4100
Name: StateHoliday, dtype: int64
0    855087
0    131072
1     20260
2      6690
3      4100
Name: StateHoliday, dtype: int64
</pre>

Spc행 null 처리  

<pre>
Spc null count : 172869
Spc null count : 0
</pre>

데이터 확인  

<pre>
Store  DayOfWeek  Date        Sales  Customers  Open  Promo  StateHoliday  SchoolHoliday  Spc      
1      1          2013-01-07  7176   785        1     1      0             1              9.141401     1
745    5          2015-06-05  7622   711        1     1      0             0              10.720113    1
                  2015-03-06  7667   738        1     1      0             0              10.388889    1
                  2015-03-13  6268   668        1     0      0             0              9.383234     1
                  2015-03-20  7857   725        1     1      0             0              10.837241    1
                                                                                                      ..
372    7          2013-03-03  0      0          0     0      0             0              0.000000     1
                  2013-03-10  0      0          0     0      0             0              0.000000     1
                  2013-03-17  0      0          0     0      0             0              0.000000     1
                  2013-03-24  0      0          0     0      0             0              0.000000     1
1115   7          2015-07-26  0      0          0     0      0             0              0.000000     1
Length: 1017209, dtype: int64
</pre>

Store 그룹에서 평균 판매, 소비자, Spc 확인  

<pre>
:2: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  tostore_df = en_df_train.groupby(['Store'])['Sales', 'Customers', 'Spc'].mean()
</pre>
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
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
    <tr>
      <th>Store</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4122.991507</td>
      <td>486.045648</td>
      <td>6.998110</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8021.769639</td>
      <td>1100.057325</td>
      <td>6.033827</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3867.110403</td>
      <td>444.360934</td>
      <td>7.121176</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>4342.968153</td>
      <td>373.548832</td>
      <td>9.549273</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>8465.280255</td>
      <td>693.498938</td>
      <td>9.918483</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>5516.180467</td>
      <td>596.763270</td>
      <td>7.666212</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>17200.196391</td>
      <td>2664.057325</td>
      <td>5.372308</td>
    </tr>
    <tr>
      <th>1115</th>
      <td>5225.296178</td>
      <td>358.687898</td>
      <td>11.952081</td>
    </tr>
  </tbody>
</table>
<p>1115 rows × 3 columns</p>
</div>


#### 6. Store 데이터에 Spc등 데이터 합치고 정리

- Store df에 5번에서 생성된 컬럼 merge 시키기 >> store data frame columns : 

- 5번에 합칠 데이터라 한다면 5.2에 평균낸 데이터로 생각.

- 다만 먼저 합치기 전에 df_store 자체의 인코딩을 진행 분석에 문자열은 포함 못함

- 인코딩 대상은 ["StoreType", "Assortment", "PromoInterval"] 임.

- 디코딩은 필요없을듯함.. 애초 결과에 포함안되니..

인코딩 진행  
StoreType, Assortment는 a, b, c 순 순서대로 숫자 0, 1, 2로 인코딩  
PromoInterval의 경우 특정 개월 그룹으로 묶인 String그룹이라 그 그룹째로 0, 1, 2 지정.  
  
<pre>
a    602
d    348
c    148
b     17
Name: StoreType, dtype: int64
a    593
c    513
b      9
Name: Assortment, dtype: int64
0                   544
Jan,Apr,Jul,Oct     335
Feb,May,Aug,Nov     130
Mar,Jun,Sept,Dec    106
Name: PromoInterval, dtype: int64
0    602
3    348
2    148
1     17
Name: StoreType, dtype: int64
0    593
2    513
1      9
Name: Assortment, dtype: int64
0    544
1    335
2    130
3    106
Name: PromoInterval, dtype: int64

  
</pre>

작업했던 데이터들을 합쳐서 확인.  

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
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>1</td>
      <td>4122.991507</td>
      <td>486.045648</td>
      <td>6.998110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>1</td>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>620.0</td>
      <td>9.0</td>
      <td>2009.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>8021.769639</td>
      <td>1100.057325</td>
      <td>6.033827</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3867.110403</td>
      <td>444.360934</td>
      <td>7.121176</td>
    </tr>
  </tbody>
</table>
</div>


#### 7. Train 데이터에 store 데이터 merge 시키기(기준은 Store 명 기준으로)

- 기준은 Store명 기준

- 즉 Train + Store의 완전한 Train데이터 구축하라.



위를 바탕으로 합쳐서 데이터 확인  

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Spc</th>
      <th>...</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.482883</td>
      <td>...</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.702400</td>
      <td>...</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>1</td>
      <td>4122.991507</td>
      <td>486.045648</td>
      <td>6.998110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10.126675</td>
      <td>...</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>1</td>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.342457</td>
      <td>...</td>
      <td>620.0</td>
      <td>9.0</td>
      <td>2009.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>8021.769639</td>
      <td>1100.057325</td>
      <td>6.033827</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.626118</td>
      <td>...</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3867.110403</td>
      <td>444.360934</td>
      <td>7.121176</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>


#### 8. Train 데이터에서 날짜데이터를 Year Month Day Week 분리하고 정리

- Year, Month, Day, Week 등 각각 4개 데이터로 분리

- python TimeDate 등 기능을 쓴다면..

백문이 불여일견.. 분석하기 쉽도록 나눠서 처리..  
Week의 경우 dt의 isocalendar의 week를 통해 몇번째 주인지 출력가능  


```python
......
dc_df_train.head()
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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Spc</th>
      <th>StoreType</th>
      <th>...</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.482883</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
      <td>2015</td>
      <td>7</td>
      <td>31</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.702400</td>
      <td>0</td>
      <td>...</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>1</td>
      <td>4122.991507</td>
      <td>486.045648</td>
      <td>6.998110</td>
      <td>2015</td>
      <td>7</td>
      <td>31</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10.126675</td>
      <td>0</td>
      <td>...</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>1</td>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
      <td>2015</td>
      <td>7</td>
      <td>31</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.342457</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>8021.769639</td>
      <td>1100.057325</td>
      <td>6.033827</td>
      <td>2015</td>
      <td>7</td>
      <td>31</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.626118</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3867.110403</td>
      <td>444.360934</td>
      <td>7.121176</td>
      <td>2015</td>
      <td>7</td>
      <td>31</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


##### 여기까지 하면서

- 이와 같이 test도 test + store 데이터를 통해서 train처럼 작업필요

- 다만 test의 경우 customers가 없기에(답안 없는...)

- Spc customers가 없는 Sales만 계산해서 합쳐지는..

- 여기도 StateHoliday Encoding 진행



train과 작업방식은 같음  

<pre>
0    40908
a      180
Name: StateHoliday, dtype: int64
0    40908
1      180
Name: StateHoliday, dtype: int64
</pre>

위 작업을 test데이터에 붙임

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
      <th>Id</th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>24000.0</td>
      <td>4.0</td>
      <td>2013.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7520.0</td>
      <td>10.0</td>
      <td>2014.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2030.0</td>
      <td>8.0</td>
      <td>2000.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

똑같이 날짜를 분석하기쉽게 나눔.

```python
# step3

......

dc_df_test.head()
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
      <th>Id</th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2015</td>
      <td>9</td>
      <td>17</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>1</td>
      <td>2015</td>
      <td>9</td>
      <td>17</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>24000.0</td>
      <td>4.0</td>
      <td>2013.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2015</td>
      <td>9</td>
      <td>17</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7520.0</td>
      <td>10.0</td>
      <td>2014.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2015</td>
      <td>9</td>
      <td>17</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9</td>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2030.0</td>
      <td>8.0</td>
      <td>2000.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2015</td>
      <td>9</td>
      <td>17</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>


아래 정보를 바탕으로 test로 인공지능 예측시 활용예정.  

<pre>
<class 'pandas.core.frame.DataFrame'>
Int64Index: 41088 entries, 0 to 41087
Data columns (total 20 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Id                         41088 non-null  int64  
 1   Store                      41088 non-null  int64  
 2   DayOfWeek                  41088 non-null  int64  
 3   Open                       41088 non-null  float64
 4   Promo                      41088 non-null  int64  
 5   StateHoliday               41088 non-null  object 
 6   SchoolHoliday              41088 non-null  int64  
 7   StoreType                  41088 non-null  int64  
 8   Assortment                 41088 non-null  int64  
 9   CompetitionDistance        41088 non-null  float64
 10  CompetitionOpenSinceMonth  41088 non-null  float64
 11  CompetitionOpenSinceYear   41088 non-null  float64
 12  Promo2                     41088 non-null  int64  
 13  Promo2SinceWeek            41088 non-null  float64
 14  Promo2SinceYear            41088 non-null  float64
 15  PromoInterval              41088 non-null  int64  
 16  Year                       41088 non-null  int64  
 17  Month                      41088 non-null  int64  
 18  Day                        41088 non-null  int64  
 19  Week                       41088 non-null  UInt32 
dtypes: UInt32(1), float64(6), int64(12), object(1)
memory usage: 6.5+ MB
</pre>


#### 9. Train DF에서 Label, Features 컬럼 나누기

- X는 학습할 특성

- y는 Label임은 당연지사

- 그러면 라벨은 Spc관련 데이터로 예상. 

- 고로 test에는 없는 데이터. Sales, Customer가 라벨



컬럼 나누기 전에 이때까지 한 컬럼들 현상황 확인

<pre>
Index(['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'Spc', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval', 'AVG_Sales', 'AVG_Customers',
       'AVG_Spc', 'Year', 'Month', 'Day', 'Week'],
      dtype='object')
</pre>

여기서 실제 확인할 데이터들 판매, 소비자, Spc, 그리고 평균 판매, 소비자, Spc등을  
예측 결과값으로 분류  
나머지는 판단자료데이터이므로 X로 분류  

```python
# 실제 이렇게 코드가 안짜여있으나 결과가 없기에 대략적으로 코드로 표현.
X = [ 'etc...' ]
y = ['Sales', 'Customers', 'Spc', 'AVG_Sales', 'AVG_Customers', 'AVG_Spc']
```


따로 분리된 예측결과 데이터 확인

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
      <th>Sales</th>
      <th>Customers</th>
      <th>Spc</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5263</td>
      <td>555</td>
      <td>9.482883</td>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6064</td>
      <td>625</td>
      <td>9.702400</td>
      <td>4122.991507</td>
      <td>486.045648</td>
      <td>6.998110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8314</td>
      <td>821</td>
      <td>10.126675</td>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13995</td>
      <td>1498</td>
      <td>9.342457</td>
      <td>8021.769639</td>
      <td>1100.057325</td>
      <td>6.033827</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4822</td>
      <td>559</td>
      <td>8.626118</td>
      <td>3867.110403</td>
      <td>444.360934</td>
      <td>7.121176</td>
    </tr>
  </tbody>
</table>
</div>


#### 10. X, y 데이터셋을 Train, test로 나누기

- X, y 로 8:2로 나누기

8:2 데이터 분리결과

<pre>
x_train values count: 813767
y_train values count: 813767
x_test values count: 203442
y_test values count: 203442
</pre>

그 데이터 일부  

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>720893</th>
      <td>274</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3640.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10.0</td>
      <td>2013.0</td>
      <td>1</td>
      <td>2013</td>
      <td>9</td>
      <td>23</td>
      <td>39</td>
    </tr>
    <tr>
      <th>611704</th>
      <td>355</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>9720.0</td>
      <td>8.0</td>
      <td>2013.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2013</td>
      <td>12</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>390659</th>
      <td>5</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2014</td>
      <td>7</td>
      <td>19</td>
      <td>29</td>
    </tr>
    <tr>
      <th>477862</th>
      <td>313</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>14160.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2014</td>
      <td>4</td>
      <td>29</td>
      <td>18</td>
    </tr>
    <tr>
      <th>374227</th>
      <td>478</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1940.0</td>
      <td>3.0</td>
      <td>2012.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2014</td>
      <td>8</td>
      <td>6</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>


#### 11. 회귀(예측) AI 모델 선택

- Linear regression, Ridge regression, Lasso regression, Polynomial regression ... 등으로 한번 해보기

- 그외 신경망으로 예측이 가능한지 확인


##### 각 모델들 입장..

- 한꺼번에 진행.



```python
# 기본적으로 4개 모델 기본옵션으로
# LinearRegression
# Lasso
# Ridge
# PolynomialFeatures ( degree : 2, 분산포함)
# 이와같이 작업할 모델 선정.
models = [
  ......
]

# Polynomial Regression 에서는 예측해줄 모델을 따로 설정해줘야함.
# PolynomialFeatures는 전처리만 담당하는 모델이기 때문.
......

... = [
  'Linear Regression',
  'Lasso',
  'Ridge',
  'Polynomial Regression',
]
```

#### 12. 학습(훈련), 예측, 성능평가

> - 학습(fit)

> - 예측(predict)

> - 평가(Score)

- 위 3과정을 진행하면서 확인

- 참고로 f1-score등은 분류모델의 성능 확인을 위한 데이터이기 때문에 패스

- 여기서 기준은 R2



```python
# 평가를 위한 라이브러리 받아오기
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 모델별로 Recall 점수 저장
# 모델 Recall 점수 순서대로 바차트를 그려 모델별로 성능 확인 가능

# error list 작성
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
err_list = [
    ......
]

......

# 막대그래프 설정 색상모음
colors = [
  ......
         ]

# R2 도출 및 그래프화 함수.
# 모델명, 예측값, 실제값을 주면 위의 plot_predictions 함수 호출하여 Scatter 그래프 그리며
# 모델별 MSE값을 Bar chart로 그려줌
def R2_eval(name_, pred, actual):
    ......
```

훈련, 예측, 평가 진행



```python
# 'Polynomial Regression' 일경우을 제외하고는 단순 fit진행.
# 'Polynomial Regression'에서 선정된 submodel을 통해서 본격 훈련, 예측 진행.
for idx, model in enumerate(models):
  # 훈련
  ......
  
  # 예측
  ......
  
  # 평가
  print( "Model : " + ...... )
  for ...... :
    print( "name : model name"  )
  
  R2_eval(......)
```

<pre>
Model : Linear Regression
MeanAE : 600.8584387741995
MedianAE : 459.8686808152799
MSE : 1776361.3675434568
RMSE : 863.3503418997265
R2 : 0.4035107433683398
               model         R2
0  Linear Regression  40.351074
</pre>
<pre>
<Figure size 864x648 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
Model : Lasso
MeanAE : 600.8850792186124
MedianAE : 459.6573239646421
MSE : 1776504.4040635328
RMSE : 863.4085057486828
R2 : 0.39972120000178996
               model         R2
0  Linear Regression  40.351074
1              Lasso  39.972120
</pre>
<pre>
<Figure size 864x648 with 0 Axes>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtMAAACgCAYAAADO3eB3AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAAsTAAALEwEAmpwYAAAeBklEQVR4nO3debxVdbn48c/DIIioXATBAUS7TjQ48VOvWmpZall5TdObmKJCppjapA0gqfc65BVLqczhaqWZmalXSkUTx+QizooDJWOMMigyc57fH2uz2xwOcNgc3AfP5/167Rd7re93rfWsvViH53x51ndHZiJJkiRp3bWqdQCSJEnSxspkWpIkSaqSybQkSZJUJZNpSZIkqUom05IkSVKVTKYlSZKkKplMS5IkSVUymZYk1UxEHBIRWe81PyKejYjzIqJNRd+IiL4RcXtEjIuIBRExMSLujYj9ankeklquNmvvIknSBvdb4E9AAN2BrwJXAbsDA0p92gG/Bp4HbgfeArYBzgD+GhFfzczfvL9hS2rpwm9AlCTVSkQcAjwCfCczr6xYvxnwGrAd0C0zZ5ZGqQ/MzEfr7aMb8AqwHNgmM+vep/AlyTIPSVLzk5nvAU9TjFR/qLRuWf1EurR+OvAosHXpJUnvG5NpSVJz9aHSn7Mb0Xd7YAkwd4NFI0kNsGZaktQcdIiILvyzZvoMYC/g/zLzjTVtGBGfBfYFfp2ZizZ4pJJUwZppSVLNVNRMN+Qu4KzMnLaG7XemKAdZCOyVmTObOkZJWhNHpiVJzcEvgd8DbYGPAudTlG6sdqQ5InYEHgYSONJEWlItmExLkpqDNzPzodL7P0fEE8ATwC+AE+p3joheFCPaHYFPZeZL71egklTJBxAlSc1OZj5FMaf08RFxQGVbKZEeCWwJfDozn3vfA5SkEpNpSVJzdTHF3NEXrVgRETtQjEh3Aj6TmWNqE5okFSzzkCQ1S5k5LiJuB06MiI9TfPPhI0Av4Bpg14jYtd5mI0rzTkvS+8JkWpLUnP0n8B8Uo9P9gB1L689eTf9DAZNpSe8bp8aTJEmSqmTNtCRJklQlk2lJkiSpSibTkiRJUpVMpiVJkqQqmUxLkiRJVXJqPDW5Ll26ZK9evWodhiRJ0lqNGTNmVmZ2rXZ7k2k1uV69evHMM8/UOgxJkqS1iogJ67O9ZR6SJElSlUymJUmSpCqZTEuSJElVsmZaTe6FF6YT8aNahyG1ON26bca0ad+udRiS1KI4Mq0mt2zZ8lqHILVI06e/V+sQJKnFMZmWJEmSqmQyLUmSJFXJZFqSJEmqksm0JEmSVCWTaUmSJKlKJtOSJElSlUymJUmSpCqZTEuSJElVMpmWpBbq4Ycf5ogjjmD77benffv2bLrppvTu3Zvvf//7zJ8/v9xv8eLFDBo0iA996ENssskmbLvttpx55pm8/fbbjTpOXV0dQ4cOpXfv3rRr146uXbty4oknMnHixJX6HXLIIUTEal9DhgwBYNGiRVxyySUceuihbLfddrRv355evXrxla98hb/97W9N9vlIUmP4deKS1EKNHj2aBx54YKV1Y8eOZezYsTz33HP8+c9/pq6ujs9//vOMGDGi3Gfq1Kn8/Oc/5/HHH2fUqFF06NBhjcfp378/N910U3l51qxZ3HbbbTz66KOMHj2abbbZplHxduzYEYC5c+cyaNCgldomTJjAhAkTGD58OKNHj2aXXXZp1D4laX05Mi1JLdSee+7J73//e6ZMmcKCBQu45557aNeuHQD3338/s2fP5t577y0n0meddRbvvvsut9xyCwAvv/wyQ4cOXeMxnn/++XIiffTRRzN37lweeughWrVqxZQpUxg8eHC578iRI8nMlV5f+tKXAGjdujXHH398ue/OO+/MzTffzNtvv80//vEPDj/8cADeeecdfvKTnzTRJyRJaxeZWesY9AETsW3C12odhtQiZV64Xtvvs88+PPvss0CRmA4aNKicnL7yyiv07t0bgM6dOzNnzhx23313Xn311dXu7+qrr+a8884DYPjw4Xz2s58FYO+99+a5556jY8eOzJ07l9atW6+y7bRp0+jRowfLli3j85//PPfeey8AS5cupa6urpz4A4wZM4Y+ffoA8JnPfGaVEXdJWp2IGJOZfard3pFpSRILFy7knnvu4ZVXXgHgxBNPZPPNN2fhwoVr3O61115j0aJFa9zvmsyfP59x48Y12HbjjTeybNkyAM4444zy+rZt266USAMrxbDddtut8ZiS1JRMpiWpBZs/fz4RQYcOHTj66KNZvHgxxxxzTLk042Mf+1i5789+9jPmz5/Pr371K+bMmQNAZjJ79uzV7r9y++uvv5558+bx8MMP88ILL5TXN/QgY11dHddffz0AvXr14ogjjljtMerq6rjooovKy6eeeuraTluSmozJdDMTEUMiYlat45DUct11113069cPgJNOOokddtgBgGHDhrH55ptz8sknr9S/bdu2q93XEUccUS6/uPvuu+nUqROHHXYYdXV1a9z+/vvvZ8KECQAMGDCAVq0a/ucqMznjjDN48MEHAfjhD3/IQQcd1NhTlaT1ZjItSS1Yx44dyUzee+89Ro4cSY8ePQC47bbbGDNmDFtssQWPPfYYxx13HFtuuSVbbrklRx11FAceeCAAHTp0oHPnzqvdf+vWrXnggQc47bTT6NKlC5ttthkHH3wwn/vc58p9Vhyz0i9+8QugSLRPO+20BvedmQwYMKA8gn3OOedw8cUXV/dBSFKVTKYlSXTo0IGDDz64PHsGwJtvvglAz549ueOOO5g7dy5z587lzjvvZNKkSQAcdNBBDT48WKlz587ccMMNzJw5k/nz5/PII4+USzt22WUXunfvvlL/SZMm8ac//QmAY445hq233nqVfWYm/fv354YbbgDg/PPP5+qrr67u5CVpPZhMb0QiYrOIuDYiXo+IBRHxVkQMi4gt6vU7LSJejYiFETErIh6NiA9XtH8vIsZFxKKImB4R90dE94r2HSPi7oh4JyLejYj/jYh/fT/PVdKGN3DgQEaMGMGMGTNYtGgRTz75JH/4wx/K7TvttBNQ1Dq/8cYbLFq0iDfffHOlL1z5xje+Ue5/8803l79gZeTIkeX1v/nNb3jppZdYuHAhkyZNYuDAgTz99NOrbL/CDTfcwPLly4GVHzxcITM5/fTTufHGGwG48MILueyyy9bz05Ck6jg1XjMTEUOAgZnZpYG2rsBFwMPATKAH8ANgYmYeXurziVL7YOCvwBbAvwH/m5lPRcRXgWHA+cArwFbAJ4Ghmfm3iGgHjAWWlvaxDPgR8C/ARzNz9U8aleN0ajypVtZlarxOnToxb968Btu+8IUvcM899wCw/fbbM2XKlFX6DBw4kGuuuaa8fPPNN5drrR955BEOOeQQoBi9fvLJJxs8xh//+MeV6qGXLVtGr169mDJlCrvtthtjx45dZbvx48ez4447rva8dthhB8aPH7/adkmqtL5T4/kNiBuRzJwJfH3FckS0Ad4CnoiInpk5EdgXeDEzL63Y9N6K9/sCD2bmzyrW3VXxvh/QE9glM/9eOs4o4O8UGXLlfiVtxM466yxGjBjB3//+d+bOnUvHjh3p3bs3J5xwAl//evlHDcceeyzDhw9nypQptGrVio9+9KOceeaZnHTSSY06zlFHHcWcOXOYOHEiy5cvZ9ddd6Vfv36ceeaZqzxYeN9995UT9699zV/KJTV/jkw3M2samS61nwR8E9gZ2Kyi6dOZ+VBEHAY8CPwE+CPwdGYuqdj+dOAa4ApgODAmM5dXtN8EfCQz96133EeABZn5ORoQEQOAAcXSlvvAeetw1pKayvp+aYsktTR+aUsLEhH/DvyKonzjOGB/4N9Lze0BMvMhitHlTwAjgVmluuoVifdNwPeBLwOjgOkRcUlErHiCaBtgegOHnw6s9pH9zPxlZvYp/jJ2qP4kJUmSNiIm0xuX44BRmXlmZv45M0cBc+p3ysxbMnMfoBvwHYrkelCprS4zh2bm7hTlHFcC3wP6lzafCqz66Hyxr7XWS0uSJLUkJtMbl02BxfXWnbi6zpk5MzOvAx4HejfQPikzLwPGVbSPAvaJiPLTPRGxHXAA8MT6hS9JkvTB4gOIzdMmEXFsA+ufB4ZExA8okt7PAp+q7BARP6IoxxgJzAL2Ag4GLii1X0cxwvw0MA84lKL++vzSLm4uvf9zRAwGlgMXlvZ1XROdnyRJ0geCyXTztDnw+wbWHwb8N3AORY30COArFInxCqMpnv47obSfCcAQigcSoai37k8xM0d7ilHp/pl5N0BmLi49xHgVcCMQFIn5lxozLZ4kSVJL4mweanLOMy3VjrN5SNK6cTYPSZIkqUZMpiVJkqQqmUxLkiRJVTKZliRJkqpkMi1JkiRVyWRakiRJqpLJtCRJklQlk2lJkiSpSibTkiRJUpVMpiVJkqQqmUxLkiRJVTKZliRJkqpkMi1JkiRVyWRakiRJqpLJtCRJklQlk2k1uTZtWtc6BKlF6tZts1qHIEktTptaB6APnj326MYzz1xY6zAkSZI2OEemJUmSpCqZTEuSJElVMpmWJEmSqmQyLUmSJFXJZFqSJEmqksm0JEmSVCWTaUmSJKlKJtOSJElSlUymJUmSpCr5DYhqeotfgNei1lFILVfrbrDztFpHIUktgiPTanq5rNYRSC3b8um1jkCSWgyTaUmSJKlKJtOSJElSlUymJUmSpCqZTEuSJElVMpmWJEmSqmQyLUmSJFXJZFqSJEmqksm0JEmSVCWTaUnSSh566CEiovx64oknVmq/8cYb2WOPPWjfvj1du3alb9++TJo0qVH7Hjx4MPvssw9bbbUVbdq0oXPnzhx66KH84Q9/WKnf+PHjV4qh8tWpU6eV+o4YMYLjjz+eHj16lPscdthh6/UZSFJjmUxLksqWLl3K2Wefvdr2Sy65hNNPP50XX3yRxYsXM2vWLG699VYOOOAApk1b+1eY33HHHTz77LPMnj2b5cuXM2fOHEaOHMmxxx7L7373u6piHj58OHfccQeTJ0+uantJWh8m05KksqFDh/Laa6/RoUOHVdomTJjARRddBMB+++3H1KlT+fWvfw3A5MmTGTJkyFr3P2DAAJ5++mnmzZvHzJkzGTBgQLnttttua3Cbt956i8wsv+bOnbtSe58+fbjssst47LHHGnmWktR0TKYlSQBMmTKFiy++mK233pr+/fuv0n7nnXeydOlSAL75zW/SvXt3+vbty+677w7A7bffTl1d3RqP8c1vfpP99tuPLbbYgi5dujBw4MByW9u2bauKu2/fvpx//vl8/OMfr2p7SVofJtOSJAC+/e1vM3/+fC6//PJV6pIBnn322fL7XXbZZZX38+bN46233mr08aZPn84111wDQOvWrRtM4AH23Xdf2rZty7bbbku/fv2YMmVKo48hSRuaybQkiZEjR3L77bdzwAEHcPLJJzfYZ9asWeX3W2yxRYPvZ8yYsdZjXXvttUQE3bt35/rrr2eTTTbhlltu4fDDD2+w/8yZM1m2bBlTp07l5ptvZr/99mPmzJmNPTVJ2qDWmkxHxJCImLWG9kMiIiPiI00b2oYRESNL8WZELIuI8RFxXUR0rXVsTS0iTimdZ8daxyKp+Vq2bBlnn302rVu3ZtiwYUTEOm2fmeX367otwJIlSzj11FMZPnx4ed1mm23GpZdeyssvv8yCBQt49dVXOeCAA4CiHGXYsGHrfBxJ2hCaYmT6WeDfgL81wb7eL49QxHwIcBXwH8BvaxnQBjKc4jwX1DoQSc3X3Xffzcsvv8yRRx4JwPPPP7/SzBzjxo1j3LhxdOnSpbzunXfeKb9/9913y++7dl37uMTAgQOpq6tjxowZXHHFFUCRUF9wwQUr7eeCCy7gwx/+MJtuuim77747V155Zbl99OjRVZypJDW99U6mM/OdzHw6Mxc2RUBNISI2XUuX2aWYn8jMnwL/BXwqIrZ9H8JrTHxNIjNnls5zzU8ESWrR5s+fD8B9993HXnvtxV577cV1111Xbu/Xrx+nn346e++9d3ndG2+8scr7Lbfckh133LFRx4wIunbtyne+851yffabb75Zbm/oQcbKUe9qRsAlaUNY72S6oTKP0vI5EfFfETEzImZExLCIaFdv254RcXtEzI6IBRHxQETsWq/PZRHxUkTMj4jJEXFrRHSv12d8RPx3RAyKiMnAO6ybF0p/9qjYZ/uIuCIiJkXE4oh4ISI+W++47SLi5xExNyLejogfR8S5EZEVfVZ8PodHxL0RMR+4dh3O/3sRMS4iFkXE9Ii4f8X5R0TbiLgyIiaWYvxHRPwxIjYpta9S5hERXSLillK8C0plL30a+DyvjIjzSp/5nFKcndbxc5X0AXLccceVZ9y46qqrmDZtGrfeeitjx44F4IQTTqBVq+KflSFDhpS/QGX8+PEAPPXUU1x88cW8+OKLvPfee8yePZurrrqqPNXdTjvtVD7WoEGD+O53v8tLL73EkiVLeO211/jWt75Vbj/wwAPL7xcsWMCsWbNWquleunRped3aZhiRpPWxIR9A/BawLdAX+DHwNeCcFY0R0Rl4AtgVOAP4MrAZ8FC9kdutKUaOPwecC+wE/CUi6sf+FeBg4Ezg+HWMtSdQB0yoWHcncErp2J8HRgP3RsSeFX2uKPX5EXBiaT/fomE3UiTtXwBubMz5R8RXge9TlKIcDnwdGFfqB/C90nEHAZ+m+HzmAa3XcK53l/b1bYrPqRXwSET8a71+XwY+BQwAzgeOKn0Wkj5gTjnllJXmcc5MLrzwwnL7448/zsiRI+nZsyeDBw8GYNSoUWyzzTb07dsXgO22226t80zPmDGDwYMHs8cee9CxY0e22mqrcoLcqlWr8hzWAO+99x4//vGP+djHPka7du3YfffdeeqppwDYbbfdOOuss8p9r7jiCrp27bpSicljjz1WXjdx4sT1+4AkaQ3abMB9j8/MU0rvH4iIA4FjKBJQgPMoksI9M3M2QEQ8CYwHTgWGAWTmqSt2GBGtgb8Ck4GDgPoz9B+VmYsaEVtERBuKpHMfiqT0l5k5rdT4KYrk/ZDMfLS0zYMRsQvwA+C4iNiKItEcnJlDS9s9ALy8mmP+PjMHVQRwcSPOf1/gwcz8WcV+7qp4vy9wW2beUrHujjWc9BHAgZXnFRF/KR3zOxS/8KywFDg6M5eV+vUGTqD4ZUVSC/XDH/6QbbbZhp/+9Ke8/vrrdOzYkcMPP5xLL72U7t27r3Hb3r17c+KJJzJq1CimTp3KkiVL2Hrrrdl///0599xzOeigg8p9TznlFJYtW8bIkSOZPHkyCxcupGfPnhx99NH84Ac/WGkGEUmqpah8CrvBDhFDgIGZ2WU17YdQPND30cx8ubQugUGZeUlFv/8CvpqZ25eW/0oxEty33i4fBCZkZr9SvyMpRl4/DFT+9OyfmTeU+owHnsjM+vtqKN6RFCPYlUYBH8/MpaU+l1KMOPeo1+8HwCmZuWPFee+Wma9X7P8y4PzMjHqfz6cz86GKfms9/4g4HbiG4heQ4cCYzFxesY9LKEarLwfuB17KigsaEacA/wNsnpnzI2IwcFZmdqv3mfwPsG9mfri0PB54KDNPr+gzAPgF0G7F51RvHwMofrmg57bsM+Hh+j0kva92W/PPdklSISLGZGaftfds2IYs85hbb3kJ0L5iuQtFmcHSeq9DKSWxEfH/gHspRqJPopiZYv/S9pX7Api+DrH9Bfh/FKPblwP7AZdUtHcBujcQ2xD+mWCvGIKpP9np6iY/rR/fWs8fuImizOPLFAn/9Ii4pDRCTynmYRSjxS8AkyKiXErTgG2AhiaBnQ50rrdubr3lJUAA7WhAZv4yM/tkZp+u/7KGCCRJkj5ANmSZx9rMpkiUL26gbcU8S/9OkZwev2LENSJ2WM3+1mUYZk5mPlN6/2QUc0yfGxHXZuakUmxTgKPXsI8V80Z1LfWnYrkx8a31/EuzcAwFhkZED4r66P+k+OXiF6WSlsHA4IjYmaL2+uqIeD0z729gv1MpatDr61bvHCRJktQItfwGxIcpSjdeycxn6r1WlE1sCiytLF2gSCib2oonbc6riK07ML+B2FYk4S8Bi4AvrthJRATFw4qN0ZjzL8vMSZl5GcUDiL0baH+T4qHCxQ21l4wCto6IT1TE3IGiPvyJRsYtSZKkksaOTG8SEcc2sP7RBtY11lUU9cJ/iYhrKEaCu1HUMz+Rmb8FRlCMGF8N/C9wAKvWGK+3zJwcEbcA/SPiotJxHwBGRMTlwCsU9dp7Au0z83uZ+XZEXA/8KCKWAmOBfqV+jRklX+v5R8R1FCPGT1PM0nEosDPF7BpExB+BMcBzwELgWIprWv/BzBXn+UBEPAX8LiIuAN6mSMA3pZhxRZIkSeugscn05sDvG1h/aLUHzsxZEbE/RdnCUKATRRnCE8CLpT5/iojzgbOB/hQzeRwFvNHQPtfTZRTJ8Ncz89KIOIaiXvlciinvZgPPUzwQuMJ3gbYUtdR1wK8ppsA7d20Ha8z5U5xvf4pZNtpTjEr3z8y7S+1PUdRdf4fifxleBb5UMXrekKOB/wauLu3z/4BPZua4tcUsSZKkla11Ng+tm4h4CGibmfVnDGkx+nwk8pk7ax2F1MI5m4ckNcr6zuZRywcQN3oRcSjFTCDPUoxQH0/xRSfH1TIuSZIkvT9MptfPfIqyie9RlEy8STEPteOykiRJLYDJ9HrIzNH8c95rSZIktTC1nBpPkiRJ2qiZTEuSJElVMpmWJEmSqmQyLUmSJFXJZFqSJEmqksm0JEmSVCWTaUmSJKlKJtOSJElSlUymJUmSpCqZTEuSJElVMpmWJEmSqmQyLUmSJFXJZFpNL9rUOgKpZWvdrdYRSFKLYdajptduD9jtmVpHIUmStME5Mi1JkiRVyWRakiRJqpLJtCRJklSlyMxax6APmIh4F3i91nGoKl2AWbUOQlXz+m28vHYbN6/fxm3XzNy82o19AFEbwuuZ2afWQWjdRcQzXruNl9dv4+W127h5/TZuEbFesyZY5iFJkiRVyWRakiRJqpLJtDaEX9Y6AFXNa7dx8/ptvLx2Gzev38Ztva6fDyBKkiRJVXJkWpIkSaqSybSaTEQcERGvR8S4iLig1vFozSKiR0Q8EhGvRsQrEXFOaX3niBgREW+W/vyXWseqhkVE64h4LiLuKy3vGBGjSvfg7yJik1rHqIZFRKeIuDMiXouIsRHxb957G4eIOK/0M/PliPhtRLT33mu+IuKmiJgRES9XrGvwXovCT0vX8cWI2LsxxzCZVpOIiNbAMOBIoDfwHxHRu7ZRaS2WAd/KzN7A/sBZpWt2AfBwZu4MPFxaVvN0DjC2YvlyYGhm/iswBzitJlGpMX4C3J+ZuwF7UFxH771mLiK2A74B9MnMjwCtgRPw3mvObgaOqLdudffakcDOpdcA4OeNOYDJtJrKvsC4zPx7Zi4Bbge+WOOYtAaZOTUzny29f5fiH/PtKK7bLaVutwBH1yRArVFEbA98DrihtBzAJ4E7S128ds1URGwJfAK4ESAzl2TmXLz3NhZtgE0jog3QAZiK916zlZmPAbPrrV7dvfZF4FdZeBroFBHbrO0YJtNqKtsBkyqWJ5fWaSMQEb2AvYBRQLfMnFpqmgZ0q1VcWqOrge8CdaXlrYC5mbmstOw92HztCMwE/qdUpnNDRGyG916zl5lTgCuBiRRJ9DxgDN57G5vV3WtV5TIm01ILFxEdgT8A52bmO5VtWUz345Q/zUxEHAXMyMwxtY5FVWkD7A38PDP3At6jXkmH917zVKqt/SLFL0TbApuxagmBNiJNca+ZTKupTAF6VCxvX1qnZiwi2lIk0rdm5l2l1dNX/LdW6c8ZtYpPq3Ug8IWIGE9RUvVJihrcTqX/egbvweZsMjA5M0eVlu+kSK6995q/w4C3MnNmZi4F7qK4H733Ni6ru9eqymVMptVURgM7l55o3oTigYx7axyT1qBUY3sjMDYzr6pouhc4ufT+ZOCe9zs2rVlmfi8zt8/MXhT32l8y80TgEeDYUjevXTOVmdOASRGxa2nVp4BX8d7bGEwE9o+IDqWfoSuunffexmV199q9wFdLs3rsD8yrKAdZLb+0RU0mIj5LUcfZGrgpM/+zthFpTSLiIOBx4CX+WXf7fYq66TuAnsAE4MuZWf/hDTUTEXEI8O3MPCoidqIYqe4MPAf0zczFNQxPqxERe1I8PLoJ8HegH8UAl/deMxcRPwKOp5gR6TngdIq6Wu+9ZigifgscAnQBpgMXAnfTwL1W+gXpWorSnQVAv8x8Zq3HMJmWJEmSqmOZhyRJklQlk2lJkiSpSibTkiRJUpVMpiVJkqQqmUxLkiRJVTKZliRJkqpkMi1JkiRVyWRakiRJqtL/B6QH0w25GEX3AAAAAElFTkSuQmCC"/>

<pre>
Model : Ridge
MeanAE : 600.8565338754206
MedianAE : 459.7666609792696
MSE : 1776354.0141114946
RMSE : 863.3505996037321
R2 : 0.4034714371240787
               model         R2
0  Linear Regression  40.351074
1              Ridge  40.347144
2              Lasso  39.972120
</pre>
<pre>
<Figure size 864x648 with 0 Axes>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtMAAADWCAYAAAATpg6rAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAAsTAAALEwEAmpwYAAAnIElEQVR4nO3de7xVZZnA8d/DRS4iEDdBFFFTE6fUNHXQRkwbtNQcw8skqZiQKU6mluYFSWy85ECllA5SmqJoampRKhrHSyaDIHnDCyUIDAjIRZE7550/1mbP5nA457A4uA+e3/fz2Z+z9vu+613P2osFz3l517sjpYQkSZKkzdek3AFIkiRJ2yqTaUmSJCknk2lJkiQpJ5NpSZIkKSeTaUmSJCknk2lJkiQpJ5NpSZIkKSeTaUlS2UREn4hIVV7LImJKRHwvIpqVtI2I6B8RYyNiekQsj4h3I+LRiDiknOchqfFqVnsTSZK2unuBPwIBdAXOAIYD+wCDCm1aAHcBU4GxwDtAN+Bc4K8RcUZK6e6PN2xJjV34DYiSpHKJiD7ABOD7KaWbSsq3B94AugM7ppQWFEapD0spPV2ljx2B14B1QLeUUuXHFL4kOc1DktTwpJQ+Al4gG6neo1C2tmoiXSh/D3ga6FJ4SdLHxmRaktRQ7VH4uagObXcGVgNLtlo0klQN50xLkhqC1hHRif+fM30ucADwPymlt2raMSK+AhwM3JVSWrnVI5WkEs6ZliSVTcmc6eo8BJyfUppXw/57kk0HWQEckFJaUN8xSlJNHJmWJDUE/w38FmgOfBa4lGzqxiZHmiNiN+ApIAHHmkhLKgeTaUlSQ/B2SunJwvafIuI54DngVuC0qo0joifZiHYb4KiU0isfV6CSVMoHECVJDU5K6XmyNaVPjYjepXWFRLoCaAd8OaX00sceoCQVmExLkhqqYWRrR1+zviAidiUbkW4P/GtKaXJ5QpOkjNM8JEkNUkppekSMBU6PiC+SffPhBKAncDOwd0TsXWW38YV1pyXpY2EyLUlqyH4M/DvZ6PQAYLdC+QWbaH8kYDIt6WPj0niSJElSTs6ZliRJknIymZYkSZJyMpmWJEmScjKZliRJknIymZYkSZJycmk81btOnTqlnj17ljsMSZKkWk2ePHlhSqlz3v1NplXvevbsyYsvvljuMCRJkmoVETO3ZH+neUiSJEk5mUxLkiRJOZlMS5IkSTmZTEuSJEk5mUxLkiRJObmah+rdB9Om8fs99ih3GFKj06JTJ/514sRyhyFJjYoj06p3ae3acocgNUqrFi4sdwiS1OiYTEuSJEk5mUxLkiRJOZlMS5IkSTmZTEuSJEk5mUxLkiRJOZlMS5IkSTmZTEuSJEk5mUxLkiRJOZlMS1Ij9dRTT3HMMcew884707JlS1q1akWvXr24/PLLWbZsWbHdqlWruOqqq9hjjz3Ybrvt2GmnnTjvvPN4//3363ScyspKRowYQa9evWjRogWdO3fm9NNP5913392gXZ8+fYiITb6GDh0KwMqVK7n22ms58sgj6d69Oy1btqRnz5584xvf4O9//3u9fT6SVBd+nbgkNVKTJk3i8ccf36Bs2rRpTJs2jZdeeok//elPVFZWcvzxxzN+/Phim7lz5/LLX/6SZ599lokTJ9K6desajzNw4EB+9atfFd8vXLiQe+65h6effppJkybRrVu3OsXbpk0bAJYsWcJVV121Qd3MmTOZOXMm48aNY9KkSey111516lOStpQj05LUSO2///789re/Zc6cOSxfvpxHHnmEFi1aAPDYY4+xaNEiHn300WIiff755/Phhx9y5513AvDqq68yYsSIGo8xderUYiJ94oknsmTJEp588kmaNGnCnDlzGDJkSLFtRUUFKaUNXl//+tcBaNq0Kaeeemqx7Z577skdd9zB+++/z//+7//St29fAD744AN+9rOf1dMnJEm1M5mWpEbqmGOOoV+/fuy00060atWKE044gX333bdY37x5cyoqKorvzzvvPNq0acMZZ5zBpz71KQDGjBlT4zFK9x84cCDt2rXjqKOOYr/99gNg7NixrFu3rtp9582bxyOPPALAV77yFXbZZRcAOnbsyCuvvMKZZ55Jhw4d6NatGz/+8Y+L+02fPr3uH4IkbSGTaUkSK1as4JFHHuG1114D4PTTT2eHHXZgxYoVNe73xhtvsHLlyhr7rcmyZcs2mfyOHj2atWvXAnDuuecWy5s3b14cQV+vNIbu3bvXeExJqk8m05LUiC1btoyIoHXr1px44omsWrWKk046qTg143Of+1yx7S9+8QuWLVvGb37zGxYvXgxASolFixZtsv/S/UeNGsXSpUt56qmn+Nvf/lYsr+5BxsrKSkaNGgVAz549OeaYYzZ5jMrKSq65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j5YtW9K5c2f69+/PrFmz6tT3kCFDOPDAA+nYsSPNmjWjQ4cOHHnkkTz44IOb3GfBggV06NChGM+1115brKuoqNgg1upekrS1mUxLkgBYs2YNF1xwwSbrr732Ws455xxefvllVq1axcKFCxkzZgy9e/dm3rx5tfZ///33M2XKFBYtWsS6detYvHgxFRUV9OvXj/vuu6/afS677DIWL16c63y23377XPtJ0uYwmZYkATBixAjeeOMNWrduvVHdzJkzueaaawA45JBDmDt3LnfddRcAs2fPZujQobX2P2jQIF544QWWLl3KggULGDRoULHunnvu2aj9xIkT+fWvf11tPAB9+vQhpbTB65VXXinWf+Mb36g1JknaUibTkiTmzJnDsGHD6NKlCwMHDtyo/oEHHmDNmjUAXHTRRXTt2pX+/fuzzz77ADB27FgqKytrPMZFF13EIYccQtu2benUqRODBw8u1jVv3nyDtpWVlZx//vkAXHHFFXU+j1tvvbW4fe6559Z5P0nKy2RaksQll1zCsmXLuOGGG2jfvv1G9VOmTClu77XXXhttL126lHfeeafOx3vvvfe4+eabAWjatOlGCfyoUaOYPHkyZ5xxBr17965Tn8uXL+fuu+8G4Atf+AKf//zn6xyPJOVlMi1JjVxFRQVjx46ld+/enHnmmdW2WbhwYXG7bdu21W7Pnz+/1mPdcsstRARdu3Zl1KhRbLfddtx555307du32Ob999/n8ssvp127dtx44411Po97772XpUuXAvCd73ynzvtJ0pYwmW5gImJoRKSS17yI+ENEfK6kTZ9C3T/V0tdNETFjqwctaZu1du1aLrjgApo2bcrIkSM3ewWMlFJxO8/qGatXr+bss89m3LhxxbIrrriCRYsWFaed1NVtt90GQPv27TnttNM2OxZJysNkumFaCvxz4XUhsBcwPiI6FOqnFOr+XpboJH1iPPzww7z66qsce+yxAEydOnWDlTmmT5/O9OnT6dSpU7Hsgw8+KG5/+OGHxe3OnTvXerzBgwdTWVnJ/Pnzi6POq1ev5rLLLgOy0e1Ro0bRvXt3evfuzdSpU5k+fXpx/3nz5jF16tSN+p0yZQqTJk0C4IwzzqBVq1Z1OX1J2mJROqqg8ouIocDglFKnkrJDgb8Cp6eUNn7kfdN93QT0Syn1rO84a/Lpg1qnG1/cq/aGkraa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"/>

<pre>
Model : Polynomial Regression
MeanAE : 567.0672357671167
MedianAE : 424.3426106043767
MSE : 1605768.94745449
RMSE : 814.6784490255732
R2 : 0.5152320248978711
                   model         R2
0  Polynomial Regression  51.523202
1      Linear Regression  40.351074
2                  Ridge  40.347144
3                  Lasso  39.972120
</pre>
<pre>
<Figure size 864x648 with 0 Axes>
</pre>
<img src="data:image/png;base64,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w4eTlZXFhx9+yNdffx3bn+wm3NzcXIYPHw5Aeno6HTt2LLCP3Nxc7rvvvtj2VVddVdRpi4jsNUrspVBmNsDMVqc6DhEpm9566y26d+8OwF/+8heaNWsGwLBhwzjooIO48sor89WvUKFCgW117NiRE044AYDx48dTq1YtzjrrLHJzcws9fuLEiSxatAiAnj17Uq5c8v863Z3rrruODz74AIC//e1vnHbaacU9VRGRPabEXkRESoXq1avj7mzatImMjIzYPPZXX32VGTNmUKNGDT755BMuvvhiatasSc2aNTnvvPM49dRTAahatSq1a9cusP20tDTef/99rr76aurWrUu1atU4/fTTOffcc2N18vqM98wzzwBB0n/11Vcnbdvd6dmzZ2xk/6abbmLgwIEluxAiIiWkxF5EREqVqlWrcvrpp8dWoQGYN28eAE2bNmXcuHGsX7+e9evX88Ybb7BkyRIATjvtNNLS0gptu3bt2jz//POsWrWK7OxspkyZEpt+07JlSxo2bJiv/pIlS/j3v/8NwEUXXUT9+vV3adPd6dGjB88//zwAffr04bHHHivZyYuI7AEl9lJiZlbNzIaa2Q9mlmNmC8xsmJnVSKh3tZl9Z2abzWy1mX1sZkfFlf/VzOab2RYzW2FmE82sYVx5czMbb2YbzGyjmf3LzA7bn+cqIvtW7969mTRpEitXrmTLli18/vnnvPnmm7HyFi1aAMHc+B9//JEtW7Ywb968fF8u9X//93+x+iNHjox9mVT8ajivvPIKs2fPZvPmzSxZsoTevXszderUXY7P8/zzz8dWyom/aTaPu3PNNdcwYsQIAPr378/999+/h1dDRKRkzN1THYOUYmY2AOjt7nWTlNUD7gM+BFYBTYC7gcXu3iGs8/uwvB/wJVADOAX4l7t/YWZXAMOAPsC3QB3gDGCIu/9kZpWAucD2sI0dwL3AwcDR7l7wnXLACU3Mv7p5T66AiJTYbcX//6VWrVpkZWUlLevUqVNsqclDDz2UZcuW7VKnd+/ePPnkk7HtkSNHxubmT5kyhXbt2gHBqP7nn3+etI9//vOf+ebP79ixg/T0dJYtW8aRRx7J3Llzdzlu4cKFNG/evMDzatasGQsXLiywXEQknpnNcPcTSnq8vnlWSszdVwHX522bWXlgAfCZmTV198XAScA37j447tB34p6fBHzg7k/F7Xsr7nl3oCnQ0t1/DvuZBvwMXAvEt5sXR0+gJ0DTg0t+fiKy//Tq1YtJkybx888/s379eqpXr07r1q3p0qUL118f+2eGzp07M2HCBJYtW0a5cuU4+uijueGGG/jLX/5SrH7OO+881q1bx+LFi9m5cydHHHEE3bt354Ybbtjlpth333039iHi2muv3XsnKyKyj2jEXgpV2Ih9WP4X4FbgcKBaXNEf3H2ymZ0FfAA8DvwTmOru2+KOvwZ4EngQmADMcPedceUvAL9295MS+p0C5Lj7uRRCI/YiKbQbI/YiIrLnI/aaYy8lZmYXAi8RTLG5GDgZuDAsrgzg7pMJRt1/D2QAq8N5+HkfAl4A7gIuAaYBK8xskJnl3QHXCFiRpPsVQMHLX4iIiIiUMUrsZU9cDExz9xvc/T13nwasS6zk7qPc/XigAXAHQaJ/T1iW6+5D3L0VwZSbh4G/Aj3Cw5cDuy5DEbRV6Px6ERERkbJEib3siSrA1oR9lxdU2d1XufuzwKdA6yTlS9z9fmB+XPk04Hgzi92dZmaNgbbAZ3sWvoiIiEh06OZZKY6KZtY5yf5ZwAAzu5sgAT8HODO+gpndSzBlJgNYDRwHnA70DcufJRh5nwpkAe0J5uv3CZsYGT5/z8z6ATuB/mFbz+6l8xMRERE54Cmxl+I4CHg9yf6zgEeAmwjm1E8CLiNI0vNMB24BuoTtLAIGENxMC8H8/B4EK9xUJhit7+Hu4wHcfWt4A+6jwAjACD4k/KmopS5FREREyhKtiiORplVxRFJIq+KIiOwWrYojIiIiIiJK7EVEREREokCJvYiIiIhIBCixFxERERGJACX2IiIiIiIRoMReRERERCQClNiLiIiIiESAEnsRERERkQhQYi8iIiIiEgFK7EVEREREIkCJvYiIiIhIBCixFxERERGJACX2IiIiIiIRoMReoq1c+VRHIFI2VW2Q6ghERMocZT0SbfWOgdu+SnUUIiIiIvucRuxFRERERCJAib2IiIiISAQosRcRERERiQAl9iIiIiIiEaDEXkREREQkApTYi4iIiIhEgBJ7EREREZEIUGIvIiIiIhIB+oIqibSvs8EyUh2FSNnVoAJknprqKEREygaN2Euk7fBURyBStq3YnuoIRETKDiX2IiIiIiIRoMReRERERCQClNiLiIiIiESAEnsRERERkQhQYi8iIiIiEgFK7EVEREREIkCJvYiIiIhIBCixFxERERGJACX2IiJSak2ePBkziz0+++yzfOUjRozgmGOOoXLlytSrV4+uXbuyZMmSYrXdr18/jj/+eOrUqUP58uWpXbs27du358033yzwmFWrVlG7du1YPIMGDYqVZWRk5Is12UNEZF9SYi8iIqXS9u3bufHGGwssHzRoENdccw3ffPMNW7duZfXq1YwePZq2bduSmZlZZPvjxo1j5syZrF27lp07d7Ju3ToyMjLo3Lkzr732WtJj+vbty7p160p0PtWqVSvRcSIixaXEXkRESqUhQ4bw/fffU7Vq1V3KFi1axH333QdAmzZtWL58OS+//DIAS5cuZcCAAUW237NnT6ZOnUpWVharVq2iZ8+esbJXX311l/rTpk3jxRdfTBoPQLt27XD3fI/Zs2fHyi+77LIiYxIR2RNK7EVEpNRZtmwZAwcOpH79+vTo0WOX8jfeeIPt27cDcOutt9KwYUO6du1Kq1atABg7diy5ubmF9nHrrbfSpk0batSoQd26dendu3esrEKFCvnq5ubm0qtXLwDuvvvuYp/HM888E3t+3XXXFfs4EZGSUGIvIiKlzu233052djYPPPAAtWrV2qV85syZsectW7bc5XlWVhYLFiwodn8rVqzgySefBCAtLW2XDxPDhw9nxowZXHHFFbRt27ZYbebk5PDKK68AcOKJJ/Lb3/622PGIiJSEEnsRESlVMjIyGDt2LG3btuXKK69MWmf16tWx5zVq1Ej6fOXKlUX2NXToUMyMhg0bMnz4cCpWrMioUaPo0KFDrM6aNWu46667qFmzJg8++GCxz2PMmDFkZWUBcP311xf7OBGRklJiLwUyswFm5nGPTDN718x+E1enXVj26yLaetjMFu7zoEXkgLZjxw5uvPFG0tLSGDZs2G6vJOPuseclWYVm27ZtXHXVVUyYMCG27+6772bt2rWxqUHF9eyzzwJQq1YtunTpstuxiIjsLiX2UpQs4JTwcTPQEphkZrXD8plh2U8piU5EImX8+PHMmTOHs88+G4BZs2blW+Fm/vz5zJ8/n7p168b2bdiwIfZ848aNsef16tUrsr/evXuTm5vLypUrY6Px27Zto2/fvkAw6j98+HAaN25M27ZtmTVrFvPnz48dn5mZyaxZs3Zpd+bMmUyfPh2AK664gipVqhTn9EVE9ojF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"/>

#### 13. 결과 보고

- 일단 성능좋은 Polynomial Regression 으로 결과보고 만들기

- test 데이터는 아까 저번에 만들어둔걸로..

- Polynomial 특징상 LR도 같이...



```python
# 그나마 성능 제일 좋은 애로 진행해서, id제거, 훈련 및 예측 등으로 진행
......

[predict].shape
```

<pre>
(41088, 6)
</pre>

결과로 나올 행 이름을 추가해서 데이터 보여주기

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
      <th>Sales</th>
      <th>Customers</th>
      <th>Spc</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7607.752650</td>
      <td>889.715885</td>
      <td>8.842535</td>
      <td>6029.960503</td>
      <td>760.633890</td>
      <td>6.731576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6120.761986</td>
      <td>565.340003</td>
      <td>10.511444</td>
      <td>4590.800948</td>
      <td>461.041049</td>
      <td>7.980048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7475.062033</td>
      <td>688.157977</td>
      <td>10.359298</td>
      <td>5474.933978</td>
      <td>540.822080</td>
      <td>8.026779</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6533.838109</td>
      <td>705.976699</td>
      <td>9.304709</td>
      <td>5040.406667</td>
      <td>568.408090</td>
      <td>7.396305</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8645.445125</td>
      <td>954.430660</td>
      <td>9.374381</td>
      <td>6633.371202</td>
      <td>770.817615</td>
      <td>7.359387</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41083</th>
      <td>5685.018501</td>
      <td>642.265425</td>
      <td>9.209060</td>
      <td>5226.790307</td>
      <td>607.817054</td>
      <td>7.347172</td>
    </tr>
    <tr>
      <th>41084</th>
      <td>8344.936185</td>
      <td>980.593709</td>
      <td>8.781420</td>
      <td>7013.484881</td>
      <td>847.581939</td>
      <td>6.920039</td>
    </tr>
    <tr>
      <th>41085</th>
      <td>7070.259068</td>
      <td>798.421054</td>
      <td>9.299123</td>
      <td>6308.945487</td>
      <td>722.586681</td>
      <td>7.508190</td>
    </tr>
    <tr>
      <th>41086</th>
      <td>7876.067103</td>
      <td>969.457115</td>
      <td>8.611529</td>
      <td>6923.436591</td>
      <td>867.374927</td>
      <td>6.852334</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>5478.998990</td>
      <td>431.183754</td>
      <td>11.537297</td>
      <td>3942.053162</td>
      <td>292.918551</td>
      <td>9.495523</td>
    </tr>
  </tbody>
</table>
<p>41088 rows × 6 columns</p>
</div>


##### 답안지 나옴!!

- 이제 이걸로 답안지 샘플대로 test Store id와 위 Sales, Customers와 조합하면 될듯함.

- Customers는 상수화 



소숫점으로 되어있는 데이터를 상수화 하면 아래와 같이 됨  

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
      <th>Sales</th>
      <th>Customers</th>
      <th>Spc</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7607.752650</td>
      <td>889</td>
      <td>8.842535</td>
      <td>6029.960503</td>
      <td>760.633890</td>
      <td>6.731576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6120.761986</td>
      <td>565</td>
      <td>10.511444</td>
      <td>4590.800948</td>
      <td>461.041049</td>
      <td>7.980048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7475.062033</td>
      <td>688</td>
      <td>10.359298</td>
      <td>5474.933978</td>
      <td>540.822080</td>
      <td>8.026779</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6533.838109</td>
      <td>705</td>
      <td>9.304709</td>
      <td>5040.406667</td>
      <td>568.408090</td>
      <td>7.396305</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8645.445125</td>
      <td>954</td>
      <td>9.374381</td>
      <td>6633.371202</td>
      <td>770.817615</td>
      <td>7.359387</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41083</th>
      <td>5685.018501</td>
      <td>642</td>
      <td>9.209060</td>
      <td>5226.790307</td>
      <td>607.817054</td>
      <td>7.347172</td>
    </tr>
    <tr>
      <th>41084</th>
      <td>8344.936185</td>
      <td>980</td>
      <td>8.781420</td>
      <td>7013.484881</td>
      <td>847.581939</td>
      <td>6.920039</td>
    </tr>
    <tr>
      <th>41085</th>
      <td>7070.259068</td>
      <td>798</td>
      <td>9.299123</td>
      <td>6308.945487</td>
      <td>722.586681</td>
      <td>7.508190</td>
    </tr>
    <tr>
      <th>41086</th>
      <td>7876.067103</td>
      <td>969</td>
      <td>8.611529</td>
      <td>6923.436591</td>
      <td>867.374927</td>
      <td>6.852334</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>5478.998990</td>
      <td>431</td>
      <td>11.537297</td>
      <td>3942.053162</td>
      <td>292.918551</td>
      <td>9.495523</td>
    </tr>
  </tbody>
</table>
<p>41088 rows × 6 columns</p>
</div>



test에서는 판매량과 소비자만 표시

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
      <th>Store</th>
      <th>Sales</th>
      <th>Customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7607.752650</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>6120.761986</td>
      <td>565</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>7475.062033</td>
      <td>688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>6533.838109</td>
      <td>705</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>8645.445125</td>
      <td>954</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41083</th>
      <td>1111</td>
      <td>5685.018501</td>
      <td>642</td>
    </tr>
    <tr>
      <th>41084</th>
      <td>1112</td>
      <td>8344.936185</td>
      <td>980</td>
    </tr>
    <tr>
      <th>41085</th>
      <td>1113</td>
      <td>7070.259068</td>
      <td>798</td>
    </tr>
    <tr>
      <th>41086</th>
      <td>1114</td>
      <td>7876.067103</td>
      <td>969</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>1115</td>
      <td>5478.998990</td>
      <td>431</td>
    </tr>
  </tbody>
</table>
<p>41088 rows × 3 columns</p>
</div>



이제 위 데이터로 csv파일을 뽑아서 제출하면 종료.  
  

#### RE 11~13 : 성능문제로 다시..

- 막상 결과물데이터를 보니, 음수값으로 예측되는 이상현상이 발생.

- 원인은 모델성능이 낮아서...

- 50%로는 안됨.. 80~90%의 성능을 요함.

- 결론은 랜덤포레스트등 성능이 괜찮을 수 있는 모델등으로 하기로 함.

- 그외 Arima, LSTM등으로도 해볼려고 한다.


- 모델 정의 및 훈련은 위와 다르게 별도로..

- 물론 그래프 구별은 같이..


성능 수치는 n수 5으로만 설정해서 진행.  

```python
model = RandomForestRegressor(.......)

......

# 평가
print( "Model : model name" )

# 해당 모델의 성능 지표 출력

R2_eval(......) # 그래프 출력
```

<pre>
Model : RandomForestRegressor
MeanAE : 101.26543946837957
MedianAE : 60.41119834006054
MSE : 141548.80777202942
RMSE : 191.73388691972642
R2 : 0.9776348312126862
                   model         R2
0  RandomForestRegressor  97.763483
1  Polynomial Regression  51.523202
2      Linear Regression  40.351074
3                  Ridge  40.347144
4                  Lasso  39.972120
</pre>
<pre>
<Figure size 864x648 with 0 Axes>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAzMAAAFDCAYAAAAOFlBeAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAAsTAAALEwEAmpwYAABFEElEQVR4nO3dd5xU1f3/8deH3kE6IlKiKKhRY0e+ESw/scQWVKJEQQUbxq5EI6KQYIvYG6KgotjRSERBWTuIGAQUEVSq9N7bfn5/nDvD7DBbqMPdfT8fj3nszD3n3vO5d3ZgPnvKNXdHREREREQkbkplOwAREREREZFtoWRGRERERERiScmMiIiIiIjEkpIZERERERGJJSUzIiIiIiISS0pmREREREQklpTMiIiIiIhILCmZERER2U5m1sbMPO2x0sy+NbPrzaxMSl0zs45mNtjMpprZajObYWbvmtlR2TwPEZG4KVN4FRERESmiV4D/AgbUBy4CHgRaAF2jOuWBF4FxwGDgV6ABcAXwlZld5O4v7dqwRUTiydw92zGIiIjEmpm1AUYCN7v7AynbKwM/Ag2Beu6+IOqlOdbdP0k7Rj3ge2AT0MDdc3dR+CIisaVhZiIiIjuJu68CRhF6an4XbduYnshE2+cBnwB1o4eIiBRCyYyIiMjO9bvo5+Ii1N0LWA8s3WnRiIgUI5ozIyIisuNUMrPabJ4zcwVwKPC1u/9U0I5mdipwJPCiu6/d6ZGKiBQDmjMjIiKynVLmzGTyFnC1u88tYP99CcPR1gCHuvuCHR2jiEhxpJ4ZERGRHecZ4HWgLHAQcCth6Fi+PS1m1hT4CHDgFCUyIiJFp2RGRERkx5ni7iOi5++b2efA58BTQIf0ymbWhNCjUwU4wd0n7KpARUSKAy0AICIispO4+5eEe8qcb2atUsuiRCYHqA6c5O7/2+UBiojEnJIZERGRnasX4d4xdyc2mFljQo9MDeD/ufvY7IQmIhJvGmYmIiKyE7n7VDMbDFxoZv8HjCMkMk2AR4H9zGy/tN2GR/edERGRAiiZERER2fn+CfyF0DvTGWgabb8mn/ptASUzIiKF0NLMIiIiIiISS5ozIyIiIiIisaRkRkREREREYknJjIiIiIiIxJKSGRERERERiSUlMyIiIiIiEktamlkkg9q1a3uTJk2yHYaIiIhIocaOHbvQ3etkO45sUDIjkkGTJk345ptvsh2GiIiISKHMbHq2Y8gWDTMTEREREZFYUjIjIiIiIiKxpGRGRERERERiScmMiIiIiIjEkpIZERERERGJJSUzIiIiIiISS0pmREREREQklpTMiIiIiIhILCmZERERERGRWFIyIyIiIiIisVQm2wGI7I5WfreSHMvJdhgiJU7ZemU5du6x2Q5DRERiQj0zIhn4Rs92CCIl0oZ5G7IdgoiIxIiSGRERERERiSUlMyIiIiIiEktKZkREREREJJaUzIiIiIiISCwpmRERERERkVhSMiMiIiIiIrGkZEZERERERGJJyYyIiMTWRx99RLt27dhrr72oUKECFStWpGXLltx2222sXLkyWW/dunXccccd/O53v6NcuXLsueeeXHXVVSxatKhI7eTm5tK3b19atmxJ+fLlqVOnDhdeeCEzZszIU69NmzaYWb6Pnj17ArB27Vp69+5N27ZtadiwIRUqVKBJkyZccMEF/Pzzzzvs+oiIFHdlsh2AiIjIthozZgwffPBBnm2TJk1i0qRJ/O9//+P9998nNzeXP/3pTwwfPjxZZ86cOTz55JN89tlnjB49mkqVKhXYTpcuXXjuueeSrxcuXMjLL7/MJ598wpgxY2jQoEGR4q1SpQoAS5cu5Y477shTNn36dKZPn87QoUMZM2YMzZs3L9IxRURKMvXMiIhIbB1yyCG8/vrrzJ49m9WrV/POO+9Qvnx5AIYNG8bixYt59913k4nM1VdfzYoVKxg4cCAAEydOpG/fvgW2MW7cuGQic9ZZZ7F06VJGjBhBqVKlmD17Nj169EjWzcnJwd3zPP785z8DULp0ac4///xk3X333ZcBAwawaNEifvvtN04++WQAli9fzsMPP7yDrpCISPGmZEZERGKrXbt2tG/fnj333JOKFStyxhlncMABByTLy5YtS05OTvL1VVddRZUqVbjooovYY489ABg0aFCBbaTu36VLF6pXr84JJ5zAwQcfDMDgwYPZtGlTxn3nzp3LO++8A8Cpp55Ko0aNAKhVqxYTJkzg4osvpmbNmjRo0IB//vOfyf2mTp1a9IsgIlKCKZkREZFiYc2aNbzzzjt8//33AFx44YVUrVqVNWvWFLjfjz/+yNq1aws8bkFWrlyZb/LRv39/Nm7cCMAVV1yR3F62bNlkD1JCagwNGzYssE0REQmUzIiISKytXLkSM6NSpUqcddZZrFu3jnPOOSc5NOz3v/99su4TTzzBypUreeGFF1iyZAkA7s7ixYvzPX7q/v369WPZsmV89NFHfPfdd8ntmRYSyM3NpV+/fgA0adKEdu3a5dtGbm4ud999d/L1JZdcUthpi4gISmZkJzGznma2MNtxiEjJ9NZbb9G5c2cA/vrXv9K4cWMAHn/8capWrcrFF1+cp37ZsmXzPVa7du04/PDDARgyZAg1atTgxBNPJDc3t8D9hw0bxvTp0wHo2rUrpUpl/i/X3bniiiv48MMPAfjHP/5B69ati3qqIiIlmpIZERGJtSpVquDurFq1ipycnOS8lJdffpmxY8dSrVo1Pv30U84991yqV69O9erVOf300zn22GMBqFSpEjVr1sz3+KVLl+aDDz7g0ksvpXbt2lSuXJnjjjuO0047LVkn0Waqp556CgiJzqWXXprx2O5O165dkz041157Lb169dq2CyEiUgIpmRERkWKhUqVKHHfcccnVwwCmTJkCwN57781rr73G0qVLWbp0KW+88QYzZ84EoHXr1pQuXbrAY9esWZNnn32WBQsWsHLlSkaOHJkcWta8eXPq16+fp/7MmTP573//C8A555xD3bp1tzimu9OlSxeeffZZAG699VYeeuihbTt5EZESSsmM7HJmVtnMHjOzyWa22sx+NbPHzaxaWr1LzewHM1tjZgvN7BMzOyCl/O9mNtXM1prZPDMbZmb1U8qbmtkQM1tuZivM7D9mts+uPFcR2bm6devG8OHDmT9/PmvXruWLL77gzTffTJY3a9YMCHNdfvrpJ9auXcuUKVPy3PDyb3/7W7L+gAEDkje4TF3F7KWXXmLChAmsWbOGmTNn0q1bN0aNGrXF/gnPPvtscoWz1In/Ce7OZZddRv/+/QG48847ueeee7bzaoiIlDzm7tmOQYohM+sJdHP32hnK6gB3Ax8BC4BGwO3ADHc/Oarzx6i8B/AVUA04BviPu39pZhcBjwO3At8DtYDjgb7u/rOZlQcmARuiY2wE7gL2AA5y9/xn+wL72X7+NE9v1zUQkW3TxtsUuW6NGjVYtmxZxrIzzjgjuSzyXnvtxezZs7eo061bNx599NHk6wEDBiTn2owcOZI2bUIsrVu35osvvsjYxttvv51nPszGjRtp0qQJs2fPZv/992fSpElb7Ddt2jSaNm2a73k1btyYadOm5VsuIpLKzMa6++HZjiMbymQ7ACl53H0BcGXitZmVAX4FPjezvd19BnAkMN7d+6Ts+m7K8yOBD939iZRtb6U87wzsDTR391+idkYDvwCXA6nHFZGYuvrqqxk+fDi//PILS5cupUqVKrRs2ZIOHTpw5ZXJf2Zo3749Q4cOZfbs2ZQqVYqDDjqIq666ir/+9a9Fauf0009nyZIlzJgxg02bNrHffvvRuXNnrrrqqi0m9r/33nvJxOnyyy/fcScrIiJbUM+M7BQF9cxE5X8FbgD2BSqnFJ3k7iPM7ETgQ+Bh4G1glLuvT9n/MuBR4D5gKDDW3TellD8HHOjuR6a1OxJY7e6nkcbMugJdAepR77DBDN7q8xaR7bc1PTMiIlKye2Y0Z0Z2OTM7G3iBMHzsXOBo4OyouAKAu48g9K78EcgBFkbzahKJz3PAbcB5wGhgnpn1NrPELN4GwLwMzc8DMi5b5O7PuPvh7n54dapv30mKiIiIyE6nZEay4VxgtLtf5e7vu/toYEl6JXcf6O6HAfWAmwnJzR1RWa6793X3FoThZA8Afwe6RLvPAbZcPigcq8D5MiIiIiISD0pmJBsqAuvStl2YX2V3X+DuTwOfAS0zlM9093uAqSnlo4HDzCw5w9bMGgKtgM+3L3wRERER2R1oAQDZmcqZWfsM28cBPc3sdkLScSpwQmoFM7uLMBwsB1gIHAocB3SPyp8m9LCMApYBbQnzb26NDjEgev6+mfUANgF3RsfSMmUiIiIixYCSGdmZqgKvZ9h+IvBv4FrCHJnhwAWExCRhDHA90CE6znSgJ2FBAAjzbboQViarQOiV6eLuQwDcfV20iMCDQH/ACInRnwtblllERERE4kGrmYlkoPvMiGSPVjMTEdk6Ws1MREREREQkZpTMiIiIiIhILCmZERERERGRWFIyIyIiIiIisaRkRkREREREYknJjIiIiIiIxJKSGRERERERiSUlMyIiIiIiEktKZkREREREJJaUzIiIiIiISCwpmRERERERkVhSMiMiIiIiIrGkZEZERERERGJJyYyIiIiIiMSSkhmRDKyMZTsEkRKpbL2y2Q5BRERipEy2AxDZHVU5uAptvmmT7TBEREREpADqmRERERERkVhSMiMiIiIiIrGkZEZERERERGJJyYyIiIiIiMSSkhkREREREYklJTMiIiIiIhJLSmZERERERCSWlMyIiIiIiEgs6aaZIhlMWjCJ3z3wu2yHIVJi1a5Um9FXjc52GCIisptTz4xIBhtzN2Y7BJESbeHqhdkOQUREYkDJjIiIiIiIxJKSGRERERERiSUlMyIiIiIiEktKZkREREREJJaUzIiIiIiISCwpmRERERERkVhSMiMiIiIiIrGkZEZERIqdESNGYGbJx+eff56nvH///hx88MFUqFCBOnXq0LFjR2bOnFmkY/fo0YPDDjuMWrVqUaZMGWrWrEnbtm158803891nwYIF1KxZMxlP7969k2U5OTl5Ys30EBGRzJTMiIhIsbJhwwauueaafMt79+7NZZddxvjx41m3bh0LFy5k0KBBtGrVirlz5xZ6/Ndee41vv/2WxYsXs2nTJpYsWUJOTg7t27fn1VdfzbhP9+7dWbJkyTadT+XKlbdpPxGRkkDJjIiIFCt9+/blxx9/pFKlSluUTZ8+nbvvvhuAo446ijlz5vDiiy8CMGvWLHr27Fno8bt27cqoUaNYtmwZCxYsoGvXrsmyl19+eYv6o0eP5vnnn88YD0CbNm1w9zyPCRMmJMsvuOCCQmMSESmplMyIiEixMXv2bHr16kXdunXp0qXLFuVvvPEGGzZsAOCGG26gfv36dOzYkRYtWgAwePBgcnNzC2zjhhtu4KijjqJatWrUrl2bbt26JcvKli2bp25ubi5XX301ALfffnuRz+Opp55KPr/iiiuKvJ+ISEmjZEZERIqNm266iZUrV3LvvfdSo0aNLcq//fbb5PPmzZtv8XzZsmX8+uuvRW5v3rx5PProowCULl16iwSqX79+jB07losuuohWrVoV6ZirV6/mpZdeAuCII47gD3/4Q5HjEREpaZTMiIhIsZCTk8PgwYNp1aoVF198ccY6CxcuTD6vVq1axufz588vtK3HHnsMM6N+/fr069ePcuXKMXDgQE4++eRknUWLFnHbbbdRvXp17rvvviKfxyuvvMKyZcsAuPLKK4u8n4hISaRkRnY4M+tpZp7ymGtm75nZ71PqtInKDizkWA+Y2bSdHrSIxNrGjRu55pprKF26NI8//vhWrwDm7snn27J62Pr167nkkksYOnRoctvtt9/O4sWLk8Peiurpp58GoEaNGnTo0GGrYxERKUmUzMjOsgw4JnpcBzQHhptZzaj826js56xEJyLFypAhQ5g4cSKnnHIKAOPGjcuzMtnUqVOZOnUqtWvXTm5bvnx58vmKFSuSz+vUqVNoe926dSM3N5f58+cne13Wr19P9+7dgdC7069fPxo2bEirVq0YN24cU6dOTe4/d+5cxo0bt8Vxv/32W8aMGQPARRddRMWKFYty+iIiJZal/jVKZEcws55AN3evnbLtaOAr4EJ333K5n/yP9QDQ3t2b7Og4C1K+UXnf69q9dmWTIpLm55uK/reOAQMG0Llz5wLrHHfccfzpT3/ipptuAuDVV1/lvPPOA6Bly5ZMmjSJ6tWrs3jxYkqV2rq/9e2xxx4sXbqU8uXLs3btWqZNm0bTpk0L3S/9/+CuXbvSr18/AH744YfkwgQiIgUxs7Hufni248gG9czIrvJd9LMRZB5mZmY1zOxlM1tpZnPMLOPSP9G+481srZmNMbMjzWxhlESl1jvTzL6J6s01s/vMrGymY4pIyXDuuecmVxx78MEHmTt3LoMGDWLSpEkAdOjQIZnI9OzZM3nTymnTpgHw5Zdf0qtXL8aPH8+qVatYvHgxDz74IEuXLgWgWbNm2xzbihUreOWVV4CQeCmREREpnJIZ2VX2jn4WtEzQ88ApwPVAV+D/AXkGjJtZQ+C/wHygPfA0MAiomFbvPOAt4GvgDOCu6Jh9tvM8RGQ31KlTpy3u1XLnnXcmyz/77DNycnLYe++96dGjBxDu/9KgQQM6duwIQMOGDQu9z8z8+fPp0aMHBx98MFWqVKFWrVrceOONAJQqVSp5D5smTZpsEc/IkSOTx+nVq9cWvTIvvfQSK1euBLQcs4hIUZXJdgBSfJlZ4verMfAYMA54J5+6BwBnAR3c/dVo20hgBrA8pep1wGrgT+6+Jqq3HHg15VgG3A+84O5XpWxfBzxuZn3cfVGGGLoSEh7K1NBHQ6S4+sc//kGDBg145JFHmDx5MlWqVOHkk0+mT58+1K9fv8B9W7ZsyYUXXsjo0aOZM2cO69evp27duhx99NFcd911tG7depvjSkz8r1u3Luecc842H0dEpCTRnBnZ4aLhXnembV4EHOHuv0Z12gAjgYPcfaKZdSL0zFR097Upx3oVOCoxZ8bMcoA57v6XlDoVgDXAXe7e08z2A34ETgWGp8SwF6FnqI27f1LQOWjOjEj2bc2cGRGRkkxzZkR2vGXAEcDRwOVAOeBlM8vvd64+sCI1kYmk3/ChPrAgdUO0z8qUTYmFB/4LbEh5JIa4NSr6aYiIiIjI7kpjaWRn2eju30TPR5vZGuAF4FxShoSlmAtUNbMKaQlN+s0Z5gJ51k2NemaqpGxaHP3sCvwvQ1tFv723iIiIiOy21DMju8pLwPfArfmUj4l+npnYYGZVgJMy1DvJzFIn/J+RVmcyMBto4u7fZHhsMV9GREREROJHPTOyS7i7m9m/gEFmdgKwKa38ezN7F3jSzKoBc4CbCZP9Uz0EXA38x8z6EoaddY/q5UbHyjWzG4EXo2O9D6wHmhEWGWjv7unHFREREZGYUc+M7EqvAlOAW/Ip7wR8SEhY+gMfAYNTK7j7bOA0wvCzt4BrgEuA0qSsehatiHYmcAjwelT3KuBbQmIjIiIiIjGn1cwk9sysNfAZcLy7jyysflFoNTOR7NNqZiIiRVOSVzPTMDOJHTO7lzCxfy6wH3AHMB4ocLllERERESlelMxIHJUn3BSzHrCCMDTtBnfPzWpUIiIiIrJLKZmR2HH364DrshyGiIiIiGSZFgAQEREREZFYUjIjIiIiIiKxpGRGRERERERiScmMiIiIiIjEkpIZERERERGJJSUzIiIiIiISS0pmREREREQklpTMiIiIiIhILCmZERERERGRWFIyIyIiIiIisaRkRiSDMqXKZDsEkRKtdqXa2Q5BRERiQN/YRDJoUacF39z0TbbDEBEREZECqGdGRERERERiScmMiIiIiIjEkpIZERERERGJJSUzIiIiIiISS0pmREREREQklpTMiIiIiIhILCmZERERERGRWFIyIyIiIiIisaSbZopksO63tUy5flK2wxApsUpXLU2zu5tnOwwREdnNqWdGJJNN2Q5ApGTbtEIfQhERKZySGRERERERiSUlMyIiIiIiEktKZkREREREJJaUzIiIiIiISCwpmRERERERkVhSMiMiIiIiIrGkZEZERERERGJJyYyIiBQ7I0aMwMySj88//zxPef/+/Tn44IOpUKECderUoWPHjsycObNIx+7RoweHHXYYtWrVokyZMtSsWZO2bdvy5ptv5qk3bdq0PDGkPmrUqJGn7vDhwzn//PNp1KhRss6JJ564XddARKQkUDIjIiLFyoYNG7jmmmvyLe/duzeXXXYZ48ePZ926dSxcuJBBgwbRqlUr5s6dW+jxX3vtNb799lsWL17Mpk2bWLJk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6pmRDMysLWEltG8JPTTnE240eW424xIRERERKYySGVlJGLb1d8KQrSmE+9C8kc2gREREREQKo2SmhHP3MWy+742IiIiISGxoaWYREREREYklJTMiIiIiIhJLSmZERERERCSWlMyIiIiIiEgsKZkREREREZFYUjIjIiIiIiKxpGRGRERERERiScmMiIiIiIjEkpIZERERERGJJSUzIiIiIiISS0pmRDIpne0AREq20lX1IRQRkcKVyXYAIruj8ntWYN++LbIdhoiIiIgUQD0zIiIiIiISS0pmREREREQklpTMiIiIiIhILCmZERERERGRWFIyIyIiIiIisaRkRkREREREYknJjIiIiIiIxJKSGRERERERiSUlMyIiIiIiEktlsh2AyO5o7axJTLiqcbbDEJESpkzV2rS4d2y2wxARiQ31zIhk4Lkbsx2CiJRAG1cszHYIIiKxomRGRERERERiScmMiIiIiIjEkpIZERERERGJJSUzIiIiIiISS0pmREREREQklpTMiIiIiIhILCmZERERERGRWFIyIyIiEmMDBgzAzDI+zjrrrGS9J554gnbt2lGjRo1kee/evYvcTpMmTfJtZ9y4ccl6L7zwAmeffTZNmzalUqVK1KtXjxNOOIGRI0fmOd7KlSvp2LEj+++/P9WqVaNs2bI0aNCA9u3b5zmeiEhBymQ7ABEREdn5nnnmGb777rud3s6//vUvJk+enHy9Zs0aPv74Yz7++GNeeeUVOnToAIRkZtCgQXn2nTt3Lm+++Sbvv/8+EydOpGnTpjs9XhGJN/XMiIiIFAONGzfG3fM8hgwZkiw/++yzeeKJJ3jyySe3q53nn39+i3YOOeSQZHmNGjXo3bs306ZNY/ny5dx2223Jsl69eiWfV6hQgT59+jBp0iRWr17NlClTOProowFYvXp1nthFRPKjnhkREZES4M477wQgJydnp7YzYsQIqlSpknzdu3dvHnvsMZYvX87UqVOT22vUqEH37t2Tr/fZZx8uuOACRo0aBUDZsmV3apwiUjyoZ0ZERKQY+O2336hVqxblypWjefPm9OjRg3Xr1u3wdm666SbKlSvHHnvswamnnspXX32Vpzw1kQFYv349mzZtAqBhw4YZj5mbm8tPP/2UHHZWu3Zt2rdvv8NjF5HiR8mMiIhIMbBhwwYWL17Mhg0bmDJlCr169eLMM8/c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"/>

##### RFR의 넘사벽...

- 위 모델로 다시 만들어보자.



```python
# model # RFR

# id 제거..
......

[predict].shape
```

<pre>
(41088, 6)
</pre>

이전과 같이 위에 출력된 예측데이터에 행 라벨 붙임.  

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
      <th>Sales</th>
      <th>Customers</th>
      <th>Spc</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4305.8</td>
      <td>490.4</td>
      <td>8.759757</td>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7649.0</td>
      <td>767.0</td>
      <td>9.972677</td>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9212.8</td>
      <td>968.2</td>
      <td>9.520230</td>
      <td>7364.866987</td>
      <td>785.740040</td>
      <td>7.743082</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7302.6</td>
      <td>811.2</td>
      <td>9.096203</td>
      <td>4685.878132</td>
      <td>539.836730</td>
      <td>7.119593</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8240.6</td>
      <td>647.6</td>
      <td>12.722666</td>
      <td>5426.816348</td>
      <td>479.487261</td>
      <td>9.267299</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41083</th>
      <td>3268.0</td>
      <td>290.0</td>
      <td>11.539755</td>
      <td>4334.747082</td>
      <td>392.967034</td>
      <td>9.128899</td>
    </tr>
    <tr>
      <th>41084</th>
      <td>9438.8</td>
      <td>792.4</td>
      <td>11.883049</td>
      <td>8465.280255</td>
      <td>693.498938</td>
      <td>9.918483</td>
    </tr>
    <tr>
      <th>41085</th>
      <td>6509.0</td>
      <td>649.8</td>
      <td>9.985102</td>
      <td>5516.180467</td>
      <td>596.763270</td>
      <td>7.666212</td>
    </tr>
    <tr>
      <th>41086</th>
      <td>22468.0</td>
      <td>3754.8</td>
      <td>5.926716</td>
      <td>17200.196391</td>
      <td>2664.057325</td>
      <td>5.372308</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>8412.2</td>
      <td>587.8</td>
      <td>14.223439</td>
      <td>5293.188323</td>
      <td>373.936730</td>
      <td>11.648778</td>
    </tr>
  </tbody>
</table>
<p>41088 rows × 6 columns</p>
</div>



```python
# 상수화 처리부터
df_predict = df_predict.astype({'Customers': 'int64'})
df_predict
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
      <th>Sales</th>
      <th>Customers</th>
      <th>Spc</th>
      <th>AVG_Sales</th>
      <th>AVG_Customers</th>
      <th>AVG_Spc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4305.8</td>
      <td>490</td>
      <td>8.759757</td>
      <td>3945.704883</td>
      <td>467.646497</td>
      <td>6.958559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7649.0</td>
      <td>767</td>
      <td>9.972677</td>
      <td>5741.253715</td>
      <td>620.286624</td>
      <td>7.539925</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9212.8</td>
      <td>968</td>
      <td>9.520230</td>
      <td>7364.866987</td>
      <td>785.740040</td>
      <td>7.743082</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7302.6</td>
      <td>811</td>
      <td>9.096203</td>
      <td>4685.878132</td>
      <td>539.836730</td>
      <td>7.119593</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8240.6</td>
      <td>647</td>
      <td>12.722666</td>
      <td>5426.816348</td>
      <td>479.487261</td>
      <td>9.267299</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41083</th>
      <td>3268.0</td>
      <td>290</td>
      <td>11.539755</td>
      <td>4334.747082</td>
      <td>392.967034</td>
      <td>9.128899</td>
    </tr>
    <tr>
      <th>41084</th>
      <td>9438.8</td>
      <td>792</td>
      <td>11.883049</td>
      <td>8465.280255</td>
      <td>693.498938</td>
      <td>9.918483</td>
    </tr>
    <tr>
      <th>41085</th>
      <td>6509.0</td>
      <td>649</td>
      <td>9.985102</td>
      <td>5516.180467</td>
      <td>596.763270</td>
      <td>7.666212</td>
    </tr>
    <tr>
      <th>41086</th>
      <td>22468.0</td>
      <td>3754</td>
      <td>5.926716</td>
      <td>17200.196391</td>
      <td>2664.057325</td>
      <td>5.372308</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>8412.2</td>
      <td>587</td>
      <td>14.223439</td>
      <td>5293.188323</td>
      <td>373.936730</td>
      <td>11.648778</td>
    </tr>
  </tbody>
</table>
<p>41088 rows × 6 columns</p>
</div>



소비자 상수화 처리

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
      <th>Store</th>
      <th>Sales</th>
      <th>Customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4305.8</td>
      <td>490</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>7649.0</td>
      <td>767</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>9212.8</td>
      <td>968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>7302.6</td>
      <td>811</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>8240.6</td>
      <td>647</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41083</th>
      <td>1111</td>
      <td>3268.0</td>
      <td>290</td>
    </tr>
    <tr>
      <th>41084</th>
      <td>1112</td>
      <td>9438.8</td>
      <td>792</td>
    </tr>
    <tr>
      <th>41085</th>
      <td>1113</td>
      <td>6509.0</td>
      <td>649</td>
    </tr>
    <tr>
      <th>41086</th>
      <td>1114</td>
      <td>22468.0</td>
      <td>3754</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>1115</td>
      <td>8412.2</td>
      <td>587</td>
    </tr>
  </tbody>
</table>
<p>41088 rows × 3 columns</p>
</div>



csv파일 출력  



##### 이렇게 해피엔딩이긴 합니다만..

- 확실히 결과를 보니 음수 등 이상사태 없고 잘나와 보인다.

- 하지만 RFR기본 옵션만 가지고 좋은 성능이긴 한데..

- 더 좋은 결과의 옵션을 탐색해서 최고의 답을 내야되지 않겠나?

- 해서 아래처럼 준비..



```python
# 각 모델에 대한 정확도와 이웃 수를 저장하기 위해 두 빈 목록을 만듭니다.
......

# ii를 사용하여 값 1에서 15까지 반복합니다. 이것은 RFR 관련 수가 됩니다.
for ii in range(1,16):
    # 이웃 수를 ii로 설정
    
    # 데이터로 모델 훈련 또는 피팅
    
    # .score는 테스트 데이터를 기반으로 모델의 정확도를 제공합니다. 정확도를 목록에 저장합니다.
   
    
    
    # 목록에 이웃 수 추가
    
    ......

#그래프 보여주기
......
```

<img src="data:image/png;base64,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"/>

##### 더 진행했으면 좋겠는데...

- 시간상 n= 14에서 멈추도록하자.

- 이 기반으로 다시 출력


이전 RFR처럼 그대로 진행. 다른점은 n최선의 수 14로 진행.

<pre>
Model : RandomForestRegressor
MeanAE : 95.29082905903987
MedianAE : 56.82775681298893
MSE : 125750.19816998013
RMSE : 180.10760475556484
R2 : 0.9806635172796763
                   model         R2
0  RandomForestRegressor  98.066352
1  Polynomial Regression  51.523202
2      Linear Regression  40.351074
3                  Ridge  40.347144
4                  Lasso  39.972120
</pre>
<pre>
<Figure size 864x648 with 0 Axes>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAzUAAAFDCAYAAAADCCAZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAAsTAAALEwEAmpwYAABFjElEQVR4nO3dd5xU1fnH8c9DkyYgHRFFolgTJWJDfxHURCyxBaNRoqCCDWNXYqEICXbUgA1RMIJYo0YUBWXtEMSgaCygdKnSe9nn98e5M9wdZgt1uLvf9+s1r51777nnPHdmB+bZU665OyIiIiIiIklVLtcBiIiIiIiIbA0lNSIiIiIikmhKakREREREJNGU1IiIiIiISKIpqRERERERkURTUiMiIiIiIommpEZERERERBJNSY2IiMhWMrPWZuYZj+Vm9rmZXWdmFWJlzczam9kwM5tsZivNbLqZvW5mR+byOkREkqpC8UVERESkhJ4D3gQMaAhcCDwAHAB0jsrsAvwTmAAMA6YAjYDLgU/N7EJ3f3bHhi0ikmzm7rmOQUREJNHMrDUwGrjJ3e+L7a8GfAs0Bhq4+/yo1+YYd38/o44GwNfABqCRu+fvoPBFRBJPw89ERES2E3dfAYwh9Nz8Itq3PjOhifbPBd4H6kcPEREpISU1IiIi29cvop8LS1B2D2AtsHi7RSMiUgppTo2IiMi2U9XM6rJxTs3lQAvgP+7+fVEnmtkpwBHAP9199XaPVESkFNGcGhERka0Um1OTzSvAVe4+p4jz9yUMU1sFtHD3+ds6RhGR0kw9NSIiItvOE8CLQEXgl8AthCFlhfa8mNnewLuAAycroRER2XxKakRERLadSe4+Knr+lpl9BHwEPAacl1nYzJoSeniqAye4+8QdFaiISGmihQJERES2E3f/hHBPmnPNrFX8WJTQ5AE1gd+6+393eIAiIqWEkhoREZHtqxfh3jN3pnaY2V6EHppawO/cfXxuQhMRKR00/ExERGQ7cvfJZjYMuMDM/g+YQEhomgL/APYzs/0yThsZ3bdGRERKQEmNiIjI9vc34E+E3pqOwN7R/qsLKd8GUFIjIlJCWtJZREREREQSTXNqREREREQk0ZTUiIiIiIhIoimpERERERGRRFNSIyIiIiIiiaakRkREREREEk1LOotkUbduXW/atGmuwxAREREp1vjx4xe4e71cx5FLSmpEsmjatCmfffZZrsMQERERKZaZTct1DLmm4WciIiIiIpJoSmpERERERCTRlNSIiIiIiEiiKakREREREZFEU1IjIiIiIiKJpqRGREREREQSTUmNiIiIiIgkmpIaERERERFJNCU1IiIiIiKSaEpqREREREQk0SrkOgCRndEXc+diPXvmOgyRMqdBtWrMufHGXIchIiIJo54akSzWb9iQ6xBEyqS5K1bkOgQREUkgJTUiIiIiIpJoSmpERERERCTRlNSIiIiIiEiiKakREREREZFEU1IjIiIiIiKJpqRGREREREQSTUmNiIiIiIgkmpIaERFJrHfffZe2bduyxx57ULlyZapUqcKBBx7IrbfeyvLly9Pl1qxZwx133MEvfvELKlWqxO67786VV17Jzz//XKJ28vPz6du3LwceeCC77LIL9erV44ILLmD69OkFyrVu3RozK/TRo0cPAFavXk3v3r1p06YNjRs3pnLlyjRt2pTzzz+fH374YZu9PiIiZUWFXAcgIiKypcaNG8fbb79dYN8333zDN998w3//+1/eeust8vPz+f3vf8/IkSPTZWbPns2jjz7Khx9+yNixY6latWqR7XTq1Imnnnoqvb1gwQKGDh3K+++/z7hx42jUqFGJ4q1evToAixcv5o477ihwbNq0aUybNo3hw4czbtw4mjdvXqI6RUREPTUiIpJghx56KC+++CKzZs1i5cqVvPbaa+yyyy4AjBgxgoULF/L666+nE5qrrrqKZcuWMXjwYAC++uor+vbtW2QbEyZMSCc0Z555JosXL2bUqFGUK1eOWbNm0a1bt3TZvLw83L3A4w9/+AMA5cuX59xzz02X3XfffRk0aBA///wzP/30EyeddBIAS5cu5aGHHtpGr5CISNmgpEZERBKrbdu2tGvXjt13350qVapw+umnc9BBB6WPV6xYkby8vPT2lVdeSfXq1bnwwgvZbbfdABgyZEiRbcTP79SpEzVr1uSEE07gkEMOAWDYsGFs2LAh67lz5szhtddeA+CUU06hSZMmANSpU4eJEydy0UUXUbt2bRo1asTf/va39HmTJ08u+YsgIiJKakREpHRYtWoVr732Gl9//TUAF1xwAbvuuiurVq0q8rxvv/2W1atXF1lvUZYvX15oEjJw4EDWr18PwOWXX57eX7FixXSPUko8hsaNGxfZpoiIFKSkRkREEm358uWYGVWrVuXMM89kzZo1nH322ekhY7/61a/SZR955BGWL1/OM888w6JFiwBwdxYuXFho/fHzBwwYwJIlS3j33Xf54osv0vuzLTiQn5/PgAEDAGjatClt27YttI38/HzuvPPO9PbFF19c3GWLiEiMkhrZLsysh5ktyHUcIlI2vfLKK3Ts2BGAP//5z+y1114A9O/fn1133ZWLLrqoQPmKFSsWWlfbtm1p2bIlAK+++iq1atXixBNPJD8/v8jzR4wYwbRp0wDo3Lkz5cpl/y/X3bn88st55513ALj99ts59thjS3qpIiKCkhoREUm46tWr4+6sWLGCvLy89LyVoUOHMn78eGrUqMEHH3zAOeecQ82aNalZsyannXYaxxxzDABVq1aldu3ahdZfvnx53n77bS655BLq1q1LtWrVOO644zj11FPTZVJtxj322GNASHguueSSrHW7O507d0736FxzzTX06tVry14IEZEyTEmNiIiUClWrVuW4445LrzYGMGnSJAD23HNPXnjhBRYvXszixYt56aWXmDFjBgDHHnss5cuXL7Lu2rVr8+STTzJ//nyWL1/O6NGj00POmjdvTsOGDQuUnzFjBm+++SYAZ599NvXr19+kTnenU6dOPPnkkwDccsstPPjgg1t28SIiZZySGtnhzKyamfUzs+/MbKWZTTGz/mZWI6PcJWb2PzNbZWYLzOx9MzsodvyvZjbZzFab2VwzG2FmDWPH9zazV81sqZktM7N/m9k+O/JaRWT76tKlCyNHjmTevHmsXr2ajz/+mJdffjl9vFmzZkCYC/P999+zevVqJk2aVODGmX/5y1/S5QcNGpS+UWZ81bNnn32WiRMnsmrVKmbMmEGXLl0YM2bMJuenPPnkk+kV0eILBKS4O5deeikDBw4EoHv37tx1111b+WqIiJRduvmm5EJVoDxwGzAfaBI9fxE4CcDMfgM8BnQDPgVqAEcDNaPjFwK3ArcAXwN1gOOBatHxXYB3gXVAJ2A90BN438x+6e6FzwoWkcR49tln6d+/f9Zjp59+OkcccQQAPXv2ZNasWZuU6dKlS4FhZIV57LHH+Pjjj7O2ccUVVxTYt379+nSysv/++9O6detNzps2bVqBm3n27NmTnj17prf32msvpk6dWmxcIiISKKmRHc7d5wPpbwFmVgGYAnxkZnu6+3TgCOBL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fPhwXnjhBWbOnLlF54uIlHVKakREpFTp27cv3377LVWrVt3k2LRp07jzzjsBOPLII5k9ezb//Oc/AZg5cyY9evQotv7OnTszZswYlixZwvz58+ncuXP62NChQ7OeM2XKFNw9/Vi8eHGB4y1btuSuu+7igw8+KOFViohInJIaEREpNWbNmkWvXr2oX78+nTp12uT4Sy+9xLp16wC4/vrradiwIe3bt+eAAw4AYNiwYeTn5xfZxvXXX8+RRx5JjRo1qFu3Ll26dEkfq1ix4hbF3b59e2655Rb+7//+b4vOFxEp65TUiIhIqXHjjTeyfPly7r777k3mrQB8/vnn6efNmzff5PmSJUuYMmVKidubO3cu//jHPwAoX7581kQK4IgjjqBixYrsvvvudOzYkVmzZpW4DRERKZ6SGhERKRXy8vIYNmwYrVq14qKLLspaZsGCBennNWrUyPp83rx5xbbVr18/zIyGDRsyYMAAKlWqxODBgznppJOylp8/fz7r169n9uzZDBo0iCOPPJL58+eX9NJERKQYZSKpMbMeZragiOOtzczN7OAdGdeWMrO8KF43s/VmNtXMHjezermObVszsw7RdVbPdSwisvNav349V199NeXLl6d///6Y2Wad7+7p55t7LsDatWu5+OKLGT58eHpftWrV6NOnD1999RUrV67kf//7H61atQLCMLn+/ftvdjsiIpJdmUhqSuBz4Gjgh1wHshlGE2JuDTwA/Al4LpcBbSfDCde5MteBiMjO69VXX+Wrr77i5JNPBmDChAkFVjKbPHkykydPpm7duul9S5cuTT9ftmxZ+nm9esX/fahLly7k5+czb9487rnnHiAkNl27di1QT9euXTnooIOoUqUKBxxwAPfdd1/6+Lhx47bgSkVEJBslNYC7L3X3Me6+KtexpJhZlWKKLIxi/sjdHwb+DpxgZrvvgPBKEt824e7zo+sseuauiJRpy5cvB+CNN96gRYsWtGjRgscffzx9vGPHjlx66aX8+te/Tu/7/vvvN3les2ZN9t577xK1aWbUq1ePm266KT1/Z9KkSenj2RYciPcCbUmPkIiIZKekhuzDz6Lta8zs72Y238zmmVl/M9sl49w9zWyYmS00s5Vm9raZ7ZdR5i4zm2hmy81sppkNMbOGGWWmmtn9ZnaHmc0ElrJ5voh+NonVWdnM7jGzGWa2xsy+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m3MJN6w8J5dxiYiIiIiUlJIaWU4YzvVXwlCuSYT72LyUy6BEREREREpKSU0Z5+7j2HjfHBERERGRxNGSziIiIiIikmhKakREREREJNGU1IiIiIiISKIpqRERERERkURTUiMiIiIiIommpEZERERERBJNSY2IiIiIiCSakhoREREREUk0JTUiIiIiIpJoSmpERERERCTRlNSIZGHl9NEQyaVKVavmOgQREUmQCrkOQGRnVL1ePU649tpchyEiIiIiJaA/R4uIiIiISKIpqRERERERkURTUiMiIiIiIommpEZERERERBJNSY2IiIiIiCSakhoREREREUk0JTUiIiIiIpJoSmpERERERCTRlNSIiIiIiEiiVch1ACI7o/xFM1gxpGOuwxCRsqZyDar94aFcRyEikjjqqRHJxjfkOgIRKYtWL811BCIiiaSkRkREREREEk1JjYiIiIiIJJqSGhERERERSTQlNSIiIiIikmhKakREREREJNGU1IiIiIiISKIpqRERERERkURTUiMiIpJggwYNwsyyPs4888x0uUceeYS2bdtSq1at9PHevXuXuJ2mTZsW2s6ECRPS5Z555hnOOuss9t57b6pWrUqDBg044YQTGD16dIH6li9fTvv27dl///2pUaMGFStWpFGjRrRr165AfSIiJVEh1wGIiIjI9vfEE0/wxRdfbPd2/v73v/Pdd9+lt1etWsV7773He++9x3PPPcd5550HhKRmyJAhBc6dM2cOL7/8Mm+99RZfffUVe++993aPV0RKB/XUiIiIlAJ77bUX7l7g8eqrr6aPn3XWWTzyyCM8+uijW9XO008/vUk7hx56aPp4rVq16N27N1OnTmXp0qXceuut6WO9evVKP69cuTJ9+vThm2++YeXKlUyaNImjjjoKgJUrVxaIXUSkOOqpERERKQO6d+8OQF5e3nZtZ9SoUVSvXj293bt3b/r168fSpUuZPHlyen+tWrXo2rVrenufffbh/PPPZ8yYMQBUrFhxu8YpIqWLempERERKgZ9++ok6depQqVIlmjdvTrdu3VizZs02b+fGG2+kUqVK7Lbbbpxyyil8+umnBY7HExqAtWvXsmHDBgAaN26ctc78/Hy+//779HC0unXr0q5du20eu4iUXkpqRERESoF169axcOFC1q1bx6RJk+jVqxdnnHHGNm/n559/Zt26dSxevJi33nqL4447jg8++KDQ8vfddx8rVqwA4JJLLtnkeLt27Shfvjz77bcfY8eOZffdd2fUqFE0bNhwm8cuIqXXdktqzKyHmXns8ZOZvWxmv9jMeqaa2X3bK87txcxaR9d98Gael2dmLxVTZlDsdc03s5lm9pyZNd2qoHdCW/o6ioiUFfvuuy8DBw5k6tSprFy5ktGjR9OgQQMA3n777W023Ozyyy/n448/ZsmSJcyZM4fLLrsMCMlUt27dsp7zzDPPpI+1adOGm2++udh2fvrpJ0455RSmTp26TeIWkbJhe/fULAGOjh43AocC75pZte3c7s7gc8J1/7Cd6v82qv9YoBvQGnjTzCptp/ZyZXu/jiIiiXbMMcdw8cUXs9dee1GlShVat27NNddckz4+bty4bdJO165dadWqFTVq1KBBgwb069ePqlWrFtrG4MGD6dixI/n5+Rx77LG89tprWefJvPTSS6xfv54ffviBP/3pT0BIbO6///5tEreIlA3bO6lZ7+5josdQ4CJgL+CU7dxuzrn70ui6V22nJlZE9X/i7k8B1wEHAC23U3sFmFmVHdHODngdRUQSLT8/f5N9Zpb1+bZuI1V3ZhuDBg3i4osvJj8/n+OPP54RI0aw6667Flp/+fLladasGbfcckt636RJk7Y6bhEpO3b0nJrx0c+mAGZW18wGm9nPZrYyGnpV6JdyMzslGm61d8b+vaP9Z0TbeWb2kpmdb2aTzWypmb1lZntknFds+6nhb2bW1cxmm9kSM7vfglPM7GszW2Zmr5rZbrHzNhk2ZWY3mNm4qI65ZvZvM9tnS1/MDKmbDzSJtVcuinuyma0xs+/N7KKM6zMz62Vm86LX6SkzOy+KvWlUpmm0fYGZPWNmi4F/R8dqm9kT0fWsNrNPzOzIjDYuMbP/mdkqM1tgZu+b2UGx43+NYlwd1TPCzBoW8TpWNbOHzWxOdM44M/tdRpsl+h0QEUm6008/nYcffpjp06ezevVq8vLyePDBB9PHjznmGACWLFnCggULWLJkSfrYypUrWbBgAYsWLUrv69ChQ4GEBeCNN97g3HPPJS8vj5UrVzJ37lyuuuqq9FyZVBsQlny+5JJLyM/Pp23btgwfPpxq1TYdoDFw4EAGDBjADz/8wJo1a5g+fTr33ntv+nizZs22/sURkTJjRy/p3DT6OSf6+SqwD2Fo2gLgJmC0mbVw98mbnA1vAz8Renx6xPZ3AOYBw2P7jgR2B24AqgAPAU9QsJeopO2fB/wH6AgcBvQmJIS/Ae6I6u8H9AEuL+L694jKTQNqRGU/MbN93X1JEeeVxJ7Rzymxff8gvFZ3EoZx/RZ4ysx+dvc3ojLXArcCfwM+As4A7imkjfuAV4BzgA1mtgswCqhFeO3mAVcAo6JrmmNmvwEeIwyR+zS67qOBmgBmdmHU/i3A10Ad4HigqCGKA4DTo/MmA52A4WbWxt0/ipUrye+AiEiizZw5k2uuuabAkLOU888/n6OPPhqAM844g/fff7/A8T59+tCnTx/22muvIuew5Ofn88ILL/DCCy9scqxatWrcdddd6e2ePXume3ZGjBhBlSoFO/anTJlC06ZNmThxIg899FDW9mrXrs31119faDwiIpm2e1JjZqk2mgGPAMsIX3rbAscArd39/ajse8BUwhfkyzLrcvcNZjYIuMjMerq7W/hT0kXAs+6+Pla8BnCquy+K6m4I9DWzKu6+ajPbXw2c4+4bgBFRj9DVwL7uPiU695AojkKTGne/Lva6lAdGEhKBM4BnCn8Vs4teWyMMO7sLGOHu/4mO7UNIMDq6++DolFFm1gjoDrwRxXAz8Ji7p2Z5vmOhJ6wJmxrj7lfF2r8EOBg4yN0nRftGAd8REombgCOAL929T6ye12PPjwDecfdHYvteKeKaDwD+FL8uM3sb+JKQYJ4UK17k70CWujsDnQGa1CkL075EpDS48847ee655xg3bhw//fQTZsYBBxxAx44dueKKK7ZJG0cffTTdu3dnxIgR/PjjjyxatIh69erRpk0bunXrxn777bfZdZ500klMnjyZL774gvnz52NmNGnShBNOOIFbbrmFpk2bbpPYRaRs2N5JTR1gXWx7OnCuu882s07AvFRCAeDuK8zsDcLk98I8RfgLfWtgNNCGME/n6Yxy41JfZiP/i342Jvx1/4jNaD8vSmhSJgO1UwlNbF89M6vk7muzBW5mRwG9gF8DtWOHmhdyrUU5jIKv7Y+E1yLlBCAf+FcssQR4F/hTlNA0ARpSMMkg2j45S5vDM7ZPJAwpnJLRxvtsnNszAbjHzPoC/yIkRvHXZwJwiZn1jOofn/FaZzqckMi9mNrh7vlm9iIhQYsr7negAHd/gtCTw6+b1fUiYhAR2WmcfvrpnH766cWWK+kqaIMGDWLQoEEF9jVo0IAePXrQo0ePYs8v6aplJ598MiefnO2/GhGRzbcjVj87nPAFdw+gqbu/FR1rROilyDSXgl/4C3D3H4E8wlAwop//cfevM4ouzthOfZGuvAXtZ6sr2z4Dsq4+ZmZ7Au9EZS4j9BIdHsVQOds5xfgmOr8V4cv8nsDjseN1gfKE92Bd7DGIkMw2IiQ0APMz6s7cTpmbsV0XOCqj/nWE96QJgLuPirZ/Q3jfFphZf9u4Al4qSf0jMBaYa2a9o6Qrm0bAcndfmSW2qtGQuJTFGWUyfwdEREREpBTY3j016939s0KOzQbqZ9nfAFhYTL1PAgPM7K/A2YShTptra9rfEm2BqsAZ7r4C0sPHCk3girEy9tp+amaVgTvN7AF3H0u4hvWE5GnTZWtCMpV6/+tlHMvcTsnsvVgIfEYY5pYpfRvraJjYYDOrR3i/+hKGIXZ19/xou6+ZNQEuIMzvmUmYi5NpNlDdzKpmJDYNCK/Jtr99toiIiIjs1Hb06mdxY4H60URyIKxqBZxKmLBelFcIf3UfRriGYTu4/S1RhZBcxOf9/JFtl1jeT1jsILUe5nuEnpqa7v5ZlsdaYAZh0YbMW04XP44heJew0ML0LPVPzCzs7vPd/XHgQ+DALMdnuPtdhKFhmxyPjCMkV+1SO6J5Ve3YPu+biIiIiOzkdvTqZ2nu/raZfQI8b2ZdgZ8Jq5BVAe4t5tzVZjYEuAp4zt0X78j2t1AqyXjazAYCB0XtLd4Wlbv7ymjeSq9o5bHvzOwxYJiZ3UPoUakctdvc3S+NFl64F7jXzOYDHxMSml9G1Wbr4Yl7hrAwQp6Z3UeY11OHMF9pjrv3jebK1CYaega0AI4DugKY2eOEHp8xhKFybYB92ZicZV7nN2b2HNDPzHYl3JSzE7A/2XuMRERERKSUy2VPDcCZhBXAHiRM/Dbg+EKWc870avTzqRy1v1minosOhGWG3wDOJyyNvLVLOcf1A5YSkiUISV8v4ELgTcJ8mlOBD2Ln9CUsRX0l8DKwG/D36NjSohpz99WEJGQk0JMwZ+ghQlLyn6jYOEKvy2OEJbmvICzHnVrH81PCfJunoxjPAjq5+6tFNN0JGExYJvo1wkIRp2Us5ywiIiIiZYS5J3ORp6j34Y9As2hehmwjZvYk8Ft33yvXseTKr5vV9Q97/T7XYYhIGVTtgszFPEVEimZm49290BvYlwU5G362pcxsP8Jf/q8Aeiqh2TpmdjBwLvAJYbjZyYTVyrIO/xIRERER2dkkLqkhLFt8JOFeKg/nOJbSYAXhvjxdgGrANEJCc38ugxIRERERKanEJTXu3jrXMZQm0Q1E2xRbUERERERkJ5XrhQJERERERES2ipIaERERERFJNCU1IiIiIiKSaEpqREREREQk0ZTUiIiIiIhIoimpERERERGRRFNSIyIiIiIiiaakRkREREREEk1JjYiIiIiIJJqSGhERERERSTQlNSLZWPlcRyAiZVHlGrmOQEQkkSrkOgCRnVG53ZpQ7YKncx2GiIiIiJSAempERERERCTRlNSIiIiIiEiiKakREREREZFEU1IjIiIiIiKJpqRGREREREQSTUmNiIiIiIgkmpIaERERERFJNCU1IiIiIiKSaLr5pkg2a76Aby3XUYiIiEhZVb4B7Dsn11EkhnpqRLLx9bmOQERERMqyDXNzHUGiKKkREREREZFEU1IjIiIiIiKJpqRGREREREQSTUmNiIiIiIgkmpIaERERERFJNCU1IiIiIiKSaEpqREREREQk0ZTUiIiIiIiUEmZ2qpl9aGYrzGyZmb1nZsdmlGlqZk+Y2Q9mtsrM5pjZKDP73Wa0c4mZfWFmq81svpk9a2ZNCin7ZzP71MyWR3F9a2b3ZCl3gJkNM7O5ZrbGzH4ys9fM7BfFxuPuJY1dpMxoebD5Zy/lOgoREREp0/Yv2fd0Mxvv7i3N7EJgEGAZRdYBJ7v7u2ZWA/gWaJSlKgdOc/c3i2nvdqBXlkMzgcPdfU6sbD/gqixlZ7n7HrFyvwHeAqpmKftbdx9VVEzqqRERERERSTgzqwDcT0hofgT2Bn4BTAMqAo+amQEnsDGheQ2oAZyfqgboUEw7ewHdos2xUV1/jrb3AHrEyp7GxoTmRaA5IWk5GPh7rFxlYEh0bBpwErArsDvQHphR3PUrqRERERERSb6DgbrR85fdfaq7/wi8Eu3bFzgcWB875w13XwbEx6dUKaaddoQkCeABd5/j7s8C30T7zjOzVI7xl+jnVKC9u09y91Xu/rW7PxKr82xCQgTQ2d3fcffl7j7b3Ye4+3fFxKSkRkRERESkFCguGQFoAbwLTIm2TzOzXYFzYmXeKaaOX8eef5/leU1gbzMrDxwT7ZsJ/MvMFpvZQjMbYmbx4W/HRT83AL8zs2nRXJ3/mNnJJbguKpSkkIiIiIiI7NS+IcydqQj8wcz6Ezowzo6VqePuK6OFA94GzgCWRsdWAX2B/sW0Uzf2fGkhz+sDy9g4P6bAQgWE4W4tzayFu68EUgsMlAduiJU7HHjDzE5x97eLCqpEPTVm1sPMPPaYY2ZvmNmvSnL+tmBmp0VtN91B7TXNuOb4Y4/ia9j+zOxmM2udZf/UWKxrzWySmd1tZtV2fJQiIiIisr25+2KgX7TZjDDk60dgr1ixdWZWHfgXYbha3C4UHMK2ueKLEzgbh6ilXESYv/N4tN0cuCB6Hi/7BrAbcCKQT8hXulGMzRl+tgQ4OnpcGwUy0sxqb0YdSXQjG6879ZiX04g2uhloXcixoYRYTwSeAa4DHtoxYYmIiIhIDtwI3EpIaNYAXwFPxI7PAC4Fjoi2/w5UA44i9NScDgwspo0Fsec1Ys93jT2fDywiJDcAi9z9mWj+TnwuzSHRz59j+wa4+2J3fxf4IqNcoTYnqVnv7mOixzDgQkLXUtvNqCOJvotdd+qxdksrM7OSjHfcFmZHsX7g7r2Ap4ALYhO3tqtoFYud1g58H0RERER2CHfPd/c+7r63u1d2918SkgsIvR4fAfvHThns7ivdfSzwZbTv+GKa+Tz2vHmW50uAKdGwsuIm+K+Kfv63hOUKtTVfcFOZUxMAMzvazF43s9nRTXUmmNkF8RPMrEM0JOqXZjYydvOdszPKWTTkbV5006BnKJgJpsrVNbPBZvazma00szwza5lRZqqZ3WdmXaPYlpjZ/VEbp5jZ11Ebr5rZbpvzAmxG+/eb2R1mNpNovKGZlYtimhzdXOh7M7so49xjLdw8aWn0mGBm56TqBeoA3WNDzVoXEe4XQGWgXqz+2hZuvDQ3moz1iZkdmRHDbhZugrTCwg2Qbolez6mxMqn39YjoNVgF3BQdO9jMhkev8TIze9HMGsbOrRjVN9023mTpX2ZWKTpey8yejPavjsoNyIjxeDMbGx2fa2aPRF2rqeOto/hOin5Hl7Oxe1ZERESkVDCzE83sODOrYWZ1zOwywmgdgFfcfSYwO3bKRWZWNfr+l5pWsjhW36DU98zYOS8S5u4AXG9mDaPv/AdE+4a5e37qefRzNzO7MPp+dmWsrvejny8RFgkA6GRmNc3seDb20LxPMbYmqdkz+plaPWEv4GPgEuD3wMvA02b2pyznDgVeB84CJgHDrOA8lb8Qxs49QVg2bhWwyV1HgVcJ61jfCJxLuJ7RZrZPRrnzCN1sHaN6rgceINw06A7gcsKqC32ytFHOzCrEHvHXrKTtnx/Vf2VUDuAfwO3RNZ5KGNv4lIX1vLFwY6Q3CGMh/xC9Dv8EakXnn0XIhAeycVhcPHPOtCdhwtaCqP5dgFGE4Wk3AWcSugpHxZMOwg2cfgtcA3QGfhe7hkzPAf8GTiFM6tqH8DtRmbDGeAfgIODfZpYad/lXwnjKO6J2ro2uq3x0/AHC5LLrCK/1rWzsysTMDgJGRNf1B6A74fXOduvMgYTkriRdqyIiIiJJ0xrII3yXWgA8BlQCJgNXR2Wejo5D+F61AhhDGIYGBYeHbcLdpwN3RptHEpKkZ6PtWcTuU0O4b05qqefBhO+il0XbI4HhUZ0/sPG+NacREqt3Cd+tlxG+JxZps1Y/s3BTHwgJTD9gAuGmPURD0lLlDPiAsN50J8KX3bi+7v5UVHY8MDe6gMcsLP92C/C4u98elX/bzEYCjWNttCUsE9fa3d+P9r1HGEN4ExtfMIDVwDnuvgEYYWZnEN7Yfd19SnTuIYQJTJdnxPpaxvYQoP1mtg/h7qyro3L7AFcAHd19cHR8lIWl7boTkpnmhCXxukTjDyG2xJ67/9fM1gMz3X0Mm7Lo/aoE/Ca6rr9FrwGEJONg4CB3nxSdMIrQTXgDcJOZHUxIAP7o7i9GZd4ljMdcnqXNh909PW/HzP4JzCHcwXZttO9Lwl1sTyH8Ih8BDI29DgAvxJ4fAfR39+dj+56NPb+DcJOm01PXZmYLgefN7Gh3/zRW9kV3L/ZDISIiIpJQYwkJyn6EJGUW4Q/nf3P3hRCSEjM7mvCd8zjCKJ7VwP+Age7+eLaK49y9t5nNJnRE7Ef4Xvg28Fd3nxMrt9zMjgN6E1Zaq034HjkU6O3uHivbzcxmAV0I34NXAKOB2909lRgVanOSmjps7GqCMKHncHdfA2GYEtAzCrgxG//SPitLXfEv5z+b2Tw23nCnCeHOpJnJxCuEXoWUI4B5qYQiqmuFmb3BpsvG5cW+zEPIVmunEprYvnpmViljzsx1hPGH8eve3PbfTSU0kRMI4xr/FUsUIWSkf4oSux8IvyBDzexJ4P1oVYuSuj56pLzi7nfHtk8ExgNTMmJ4H0gNoUv9/HfsGldFyc9RWdocnrF9IiErz4+1MYWQ+LWMyk8ArjCzuYQel4nxX/Do+E1mtgEY5e7x9dAhvA8vZby/LxNuLHUsEE9qMuMrwMw6E3qj2HP3okqKiIiI7Hzc/d/EvrcVUe4bwkim4sp1IIy0yXZsICUY+eLu8wl/7M/8g3+2so+zcXW0zbK5q58dTvgyexmhB2BobDjWIMKwpHsJQ5QOJ0xOzzZhfHHG9tpYudTQp8wVxjK3G2XZB6HXJ3NFtmztZdtnhOuKm+zun8UeqURoc9qfm7Fdl5D0LSEkiqnHIEKi2cjdFxGGY1Uk9FzMj+amNMvSZjbPEt6D1oRuxrPN7IqMGI7KaH8dYYheaq3whsCyjIQMwjC1bLJd5y1Z2mgWa6M3YT30KwlDw2aY2TWxOroQhvl1A76zsDx1/EPYKLPdKMH5meLfhwLc/Ql3b+nuLett1uwqEREREcmlzempWe/un0XPx0aTwZ8BzjGz1wjDx65y98dSJ9iWrbSV6rKqn7E/c3t2ln0ADYCFW9Du5tqc9j1jeyGhJ+EYQo9NpnkA0bCythZW6jqRML9kKNl7STLNjb1f75vZXsCdZvaMu6+IYviMMAwu05ro5xxgVzOrnJHY1MtyDmS/zn8BT2YpuwAgqrcb0M3M9iUMk3vQzL5z9xFR79RfgL9YuC/SzcAQM/vS3f9Hlvch6umqQ/Hvg4iIiIiUAluzUMCzwNeEv8TvEtWV+jKMme1KmI+xuWYQvkyfkbH/7IztsUB9M/tNrM2qhEn3H7H9bU377xF6ampm9AKlHgWWjHb3VVF34lPAgbFD8R6u4vyV0HNySbT9LrAPMD1L+xOjMqmkKP0+RgnWb0vY5ruEhQHGZ2ljambhaG7PjYTfowOzHP+SMF+pHBuXIxwLnBUlMilnExL2HfF7ICIiIiI5tlkLBcS5u5vZ3wkT51sC4wh/bV9K6H3oShhetclSzMXUu8HM7gHuM7MFwIeEVa0OyCj3tpl9QpgQ3pUw3OhGoAphCNx2tTXtu/t3ZvYYYdW3ewjJQ2VCAtDc3S81s1OBiwlDr6YT5ildRkiIUr4FTjWzEYT5N9/FFhXIbPM/0WIL15lZf0Iv2+VAnpndR1hlrQ5hjsocd+/r7l+Z2b+BR6MkdQ5hns5KsvcwZeoB/AcYbmZPEXpnGhOSokHunmdm/yLM7fkvYZW7doTfyw8AzOwjQm/PV4Selk6EiWP/idroHZ37qpk9SpibdTfwdsYiASIiIiJSSm3tjRifJyzJfDNhGd0fCV+WHyJM1n5mC+t9kLCs2+VRPdWjNjKdSVgO7kHCmtkGHO/uk7ew3c21Ne1fRVhS+kLgTcJ8mlOJvswTFi5wwuvwDmEp6hGERCflJsIX/OGEpPKwYtrsDTQlrGa2GmgTxd8zauMhYF82JgwQJoeNAh4m9BS9H8WxtLgLjCb1H0VIgp4A3oraWhNdH8AnhNdxKGFxiMOAP8SGzn0axfASYW5RXcJqajOjNr4GTiYMQXslusbnCMmRiIiIiJQBVnChKZGiRauYfQWMdfeLiiufVC0PNv8s251uRERERHaU/Uv2Pd3Mxrt7y+JLll5bPPxMygYzOwfYHZhIGErYidCbc2Eu4xIRERERSVFSI8VZQVjmeR/C4gYTgd+7+3+KPEtEREREZAdRUiNFcvc3CXN+RERERER2Slu7UICIiIiIiEhOKakREREREZFEU1IjIiIiIiKJpqRGREREREQSTUmNiIiIiIgkmpIaERERERFJNCU1IiIiIiKSaEpqREREREQk0ZTUiIiIiIhIoimpERERERGRRFNSI5KNVch1BCIiIlKWlW+Q6wgSRd/cRLLZ5RDY/7NcRyEiIiIiJaCeGhERERERSTQlNSIiIiIikmhKakREREREJNGU1IiIiIiISKIpqRERERERkURTUiMiIiIiIommpEZERERERBJNSY2IiIiIiCSakhoREREREUk0JTUiIiIiIpJoSmpERERERCTRzN1zHYPITsfMlgHf5ToO2SJ1gQW5DkK2mN6/ZNP7l1x675JtP3ffNddB5FKFXAcgspP6zt1b5joI2Xxm9pneu+TS+5dsev+SS+9dspnZZ7mOIdc0/ExERERERBJNSY2IiIiIiCSakhqR7J7IdQCyxfTeJZvev2TT+5dceu+Srcy/f1ooQEREREREEk09NSIiIiIikmhKakRizKytmX1nZpPNrGuu45GimVkTMxttZv8zs6/N7Jpof20zG2lmk6Kfu+U6VsnOzMqb2X/N7I1oe28zGxt9Bp83s0q5jlGyM7NaZvaSmX1rZt+Y2dH67CWHmV0X/bv5lZk9Z2aV9fnbeZnZU2Y2z8y+iu3L+nmz4OHoffzSzH6du8h3HCU1IhEzKw/0B04GDgT+ZGYH5jYqKcZ64AZ3PxA4Crgqes+6Au+6+77Au9G27JyuAb6Jbd8N9HX3fYBFwCU5iUpK4iFghLvvDxxCeB/12UsAM2sM/AVo6e4HA+WB89Dnb2c2CGibsa+wz9vJwL7RozPw6A6KMaeU1IhsdAQw2d1/dPe1wDDgjBzHJEVw99nu/nn0fBnhS1Vjwvs2OCo2GDgzJwFKkcxsD+BU4Mlo24DjgZeiInrvdlJmVhP4DTAQwN3Xuvti9NlLkgpAFTOrAFQFZqPP307L3T8AFmbsLuzzdgbwjAdjgFpm1miHBJpDSmpENmoMzIhtz4z2SQKYWVOgBTAWaODus6NDc4AGuYpLivQgcDOQH23XARa7+/poW5/BndfewHzg6Wj44JNmVg199hLB3WcB9wHTCcnMEmA8+vwlTWGftzL5fUZJjYgknplVB14GrnX3pfFjHpZ41DKPOxkzOw2Y5+7jcx2LbJEKwK+BR929BbCCjKFm+uztvKK5F2cQktPdgWpsOrRJEkSfNyU1InGzgCax7T2ifbITM7OKhIRmiLu/Eu2em+pqj37Oy1V8UqhjgNPNbCphqOfxhDkataLhMKDP4M5sJjDT3cdG2y8Rkhx99pLhRGCKu89393XAK4TPpD5/yVLY561Mfp9RUiOy0Thg32j1l0qESZOv5zgmKUI0B2Mg8I27PxA79DpwUfT8IuC1HR2bFM3d/+rue7h7U8Jn7T13vwAYDbSLium920m5+xxghpntF+06Afgf+uwlxXTgKDOrGv07mnr/9PlLlsI+b68DF0aroB0FLIkNUyu1dPNNkRgzO4Uwzr888JS7/y23EUlRzOxY4ENgIhvnZdxKmFfzArAnMA34o7tnTrCUnYSZtQZudPfTzKwZoeemNvBfoL27r8lheFIIMzuUsMhDJeBHoCPhj6X67CWAmfUEziWsIvlf4FLCvAt9/nZCZvYc0BqoC8wFugOvkuXzFiWq/QhDClcCHd39sxyEvUMpqRERERERkUTT8DMREREREUk0JTUiIiIiIpJoSmpERERERCTRlNSIiIiIiEiiKakREREREZFEU1IjIiIiIiKJpqRGREREREQSTUmNiIiIiIgk2v8DXaKdYLwFfPMAAAAASUVORK5CYII="/>


이를 바탕으로 실전데이터처리로 상수화등으로 진행후, csv파일을 출력 보고하면 종료.



#### 그 외 참조한 사이트



- 날짜 조건에 따른 추출방법  

https://kibua20.tistory.com/195  

  

- pandas 데이터 계산  

https://nalara12200.tistory.com/162  

  

- pandas groupby 참조 사이트  

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html  

  

- pandas merge 에 관해서  

https://yganalyst.github.io/data_handling/Pd_12/#2-1-merge%ED%95%A8%EC%88%98%EC%9D%98-%EC%98%B5%EC%85%98  

  

- pandas 날짜 분리 및 처리에 관해서  

https://steadiness-193.tistory.com/60  

  

- pandas 몇주차 출력에 관해서  

https://moondol-ai.tistory.com/180  

여기서 경고에서 나온 노하우 : pandas dt.isocalendar().week 이 변수를 통해 몇 주차인지 출력이 가능  

  

- Polynomial regression 등에 관해서  

https://data36.com/polynomial-regression-python-scikit-learn/  


#### 마무리

- MSE 수치는 원래 이렇게 높게 나오나 보다 하면서 깨달음

- 그렇다 해도 예측의 위대함을 경험한 귀중한 기회

- RFR이외에 다른 더 좋은 예측 모델도 존재하는데..  

거기에 예측 신경망;;; 일단 이렇게 업로드 했으나 추후 업데이트 예정.

