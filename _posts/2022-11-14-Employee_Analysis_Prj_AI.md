---
layout: single
title:  "이직률 분석에 사용한 회귀분석"
categories: coding
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


# Employee Attrition Rate using Regression



# 회귀 분석을 사용한 직원 감소율



## Introduction



인공지능은 프로세스를 자동화하고, 비즈니스에 대한 통찰력을 모으고, 프로세스 속도를 높이기 위해 다양한 산업에서 사용되고 있습니다. 인공지능이 실제로 산업에 어떤 영향을 미치는지 실제 시나리오에서 인공지능의 사용을 연구하기 위해 Python을 사용할 것입니다.



직원은 조직에서 가장 중요한 존재입니다. 성공적인 직원들은 조직에 많은 것을 제공합니다. 이 노트북에서는 AI를 사용하여 직원의 이직률이나 회사가 직원을 유지할 수 있는 빈도를 예측해 볼 것입니다.





## Context



Hackerearth가 수집하여 [Kaggle]에 업로드한 직원 감소율을 포함한 데이터 세트를 사용합니다. 회귀 분석을 사용하여 감소율을 예측하고 우리 모델이 얼마나 성공적인지 확인할 것입니다.





## Use Python to open csv files



[scikit-learn](https://scikit-learn.org/stable/)과 [pandas](https://pandas.pydata.org/)를 사용하여 데이터 세트를 작업합니다. Scikit-learn은 예측 데이터 분석을 위한 효율적인 도구를 제공하는 매우 유용한 기계 학습 라이브러리입니다. Pandas는 데이터 과학을 위한 인기 있는 Python 라이브러리입니다. 강력하고 유연한 데이터 구조를 제공하여 데이터 조작 및 분석을 더 쉽게 만듭니다.





## Import Libraries




```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
```

### Dataset 가져오기



데이터 세트에는 직원 이직률이 포함되어 있습니다. 데이터 세트를 시각화해 보겠습니다.



```python
# train 변수(데이터프레임)로 [Dataset]_Module11_Train_(Employee).csv 가져오기
# your code here 
df_train = pd.read_csv("./[Dataset]_Module11_Train_(Employee).csv")
```

## Task 1: training set의 column 출력



```python
# training set의 column 출력
# your code here 
df_train.columns
```

<pre>
Index(['Employee_ID', 'Gender', 'Age', 'Education_Level',
       'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
       'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate',
       'Post_Level', 'Pay_Scale', 'Compensation_and_Benefits',
       'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6',
       'VAR7', 'Attrition_rate'],
      dtype='object')
</pre>

```python
# train 데이터 세트 크기 및 첫 5행 확인하기
# your code here 
df_train.head(5)
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
      <th>Employee_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Education_Level</th>
      <th>Relationship_Status</th>
      <th>Hometown</th>
      <th>Unit</th>
      <th>Decision_skill_possess</th>
      <th>Time_of_service</th>
      <th>Time_since_promotion</th>
      <th>...</th>
      <th>Compensation_and_Benefits</th>
      <th>Work_Life_balance</th>
      <th>VAR1</th>
      <th>VAR2</th>
      <th>VAR3</th>
      <th>VAR4</th>
      <th>VAR5</th>
      <th>VAR6</th>
      <th>VAR7</th>
      <th>Attrition_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EID_23371</td>
      <td>F</td>
      <td>42.0</td>
      <td>4</td>
      <td>Married</td>
      <td>Franklin</td>
      <td>IT</td>
      <td>Conceptual</td>
      <td>4.0</td>
      <td>4</td>
      <td>...</td>
      <td>type2</td>
      <td>3.0</td>
      <td>4</td>
      <td>0.7516</td>
      <td>1.8688</td>
      <td>2.0</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>0.1841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EID_18000</td>
      <td>M</td>
      <td>24.0</td>
      <td>3</td>
      <td>Single</td>
      <td>Springfield</td>
      <td>Logistics</td>
      <td>Analytical</td>
      <td>5.0</td>
      <td>4</td>
      <td>...</td>
      <td>type2</td>
      <td>4.0</td>
      <td>3</td>
      <td>-0.9612</td>
      <td>-0.4537</td>
      <td>2.0</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0.0670</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EID_3891</td>
      <td>F</td>
      <td>58.0</td>
      <td>3</td>
      <td>Married</td>
      <td>Clinton</td>
      <td>Quality</td>
      <td>Conceptual</td>
      <td>27.0</td>
      <td>3</td>
      <td>...</td>
      <td>type2</td>
      <td>1.0</td>
      <td>4</td>
      <td>-0.9612</td>
      <td>-0.4537</td>
      <td>3.0</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>0.0851</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EID_17492</td>
      <td>F</td>
      <td>26.0</td>
      <td>3</td>
      <td>Single</td>
      <td>Lebanon</td>
      <td>Human Resource Management</td>
      <td>Behavioral</td>
      <td>4.0</td>
      <td>3</td>
      <td>...</td>
      <td>type2</td>
      <td>1.0</td>
      <td>3</td>
      <td>-1.8176</td>
      <td>-0.4537</td>
      <td>NaN</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>0.0668</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EID_22534</td>
      <td>F</td>
      <td>31.0</td>
      <td>1</td>
      <td>Married</td>
      <td>Springfield</td>
      <td>Logistics</td>
      <td>Conceptual</td>
      <td>5.0</td>
      <td>4</td>
      <td>...</td>
      <td>type3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.7516</td>
      <td>-0.4537</td>
      <td>2.0</td>
      <td>2</td>
      <td>8</td>
      <td>2</td>
      <td>0.1827</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



```python
# train 데이터 세트 정보 확인하기
# your code here 
df_train.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7000 entries, 0 to 6999
Data columns (total 24 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Employee_ID                7000 non-null   object 
 1   Gender                     7000 non-null   object 
 2   Age                        6588 non-null   float64
 3   Education_Level            7000 non-null   int64  
 4   Relationship_Status        7000 non-null   object 
 5   Hometown                   7000 non-null   object 
 6   Unit                       7000 non-null   object 
 7   Decision_skill_possess     7000 non-null   object 
 8   Time_of_service            6856 non-null   float64
 9   Time_since_promotion       7000 non-null   int64  
 10  growth_rate                7000 non-null   int64  
 11  Travel_Rate                7000 non-null   int64  
 12  Post_Level                 7000 non-null   int64  
 13  Pay_Scale                  6991 non-null   float64
 14  Compensation_and_Benefits  7000 non-null   object 
 15  Work_Life_balance          6989 non-null   float64
 16  VAR1                       7000 non-null   int64  
 17  VAR2                       6423 non-null   float64
 18  VAR3                       7000 non-null   float64
 19  VAR4                       6344 non-null   float64
 20  VAR5                       7000 non-null   int64  
 21  VAR6                       7000 non-null   int64  
 22  VAR7                       7000 non-null   int64  
 23  Attrition_rate             7000 non-null   float64
dtypes: float64(8), int64(9), object(7)
memory usage: 1.3+ MB
</pre>

```python
# train 데이터 세트 데이터 타입 확인하기
# your code here 
df_train.dtypes
```

<pre>
Employee_ID                   object
Gender                        object
Age                          float64
Education_Level                int64
Relationship_Status           object
Hometown                      object
Unit                          object
Decision_skill_possess        object
Time_of_service              float64
Time_since_promotion           int64
growth_rate                    int64
Travel_Rate                    int64
Post_Level                     int64
Pay_Scale                    float64
Compensation_and_Benefits     object
Work_Life_balance            float64
VAR1                           int64
VAR2                         float64
VAR3                         float64
VAR4                         float64
VAR5                           int64
VAR6                           int64
VAR7                           int64
Attrition_rate               float64
dtype: object
</pre>

```python
# train 데이터 세트 유니크 아이템의 개수를 확인합니다.
# your code here 
df_train.nunique(dropna=False)
```

<pre>
Employee_ID                  7000
Gender                          2
Age                            48
Education_Level                 5
Relationship_Status             2
Hometown                        5
Unit                           12
Decision_skill_possess          4
Time_of_service                45
Time_since_promotion            5
growth_rate                    55
Travel_Rate                     3
Post_Level                      5
Pay_Scale                      11
Compensation_and_Benefits       5
Work_Life_balance               6
VAR1                            5
VAR2                            6
VAR3                            5
VAR4                            4
VAR5                            5
VAR6                            5
VAR7                            5
Attrition_rate               3317
dtype: int64
</pre>

```python
# Attrition_rate 컬럼에 대하 히스토그램 그리기 
# your code here 
df_train['Attrition_rate'].plot.hist()
```

<pre>
<AxesSubplot: ylabel='Frequency'>
</pre>
<img src="data:image/png;base64,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fPmykpKS1KhRI9ntdrVu3Zp/P25Tecf6jTfeUJs2bVS7dm1FR0dr4sSJunbtmo+6rZp27typAQMGKCoqSgEBAfroo49+8TXbt29Xly5dZLfb1bJlS6WmplZcgxYq1MqVKy2bzWYtXbrUOnTokDVmzBgrLCzMysvLu2n9rl27rMDAQGvevHnW4cOHrZkzZ1o1a9a0Dhw44OPOq5byjvPQoUOtRYsWWfv27bOOHDlijR492goNDbXOnTvn486rnvKOdZlTp05Zv/rVr6xevXpZjz76qG+arcLKO86FhYVWt27drP79+1ufffaZderUKWv79u1WVlaWjzuveso71itWrLDsdru1YsUK69SpU9amTZusRo0aWRMnTvRx51XLJ598Ys2YMcNau3atJclat27dLetPnjxp1alTx0pOTrYOHz5svfXWW1ZgYKC1cePGCumPQFTB7r//fispKcm1XlJSYkVFRVkpKSk3rf/9739vJSQkuG3r3r279cc//rFC+6zqyjvOP1VcXGwFBwdby5cvr6gWqw1Pxrq4uNh68MEHrb/+9a/WqFGjCES3obzjvHjxYqt58+bW9evXfdVitVHesU5KSrIeeught23JyclWjx49KrTP6uR2AtGUKVOsDh06uG17/PHHrfj4+Arpia/MKtD169eVmZmp2NhY17YaNWooNjZWGRkZN31NRkaGW70kxcfH/2w9PBvnn/r+++9VVFSk+vXrV1Sb1YKnY/3SSy8pPDxciYmJvmizyvNknD/++GPFxMQoKSlJERER6tixo1555RWVlJT4qu0qyZOxfvDBB5WZmen6Wu3kyZP65JNP1L9/f5/0bApffx4a86Rqf/j2229VUlJyw9OwIyIidPTo0Zu+xuFw3LTe4XBUWJ9VnSfj/FNTp05VVFTUDX/54M6Tsf7ss8/03nvvKSsrywcdVg+ejPPJkye1detWDRs2TJ988omOHz+uZ555RkVFRZo9e7Yv2q6SPBnroUOH6ttvv1XPnj1lWZaKi4s1btw4/b//9/980bIxfu7z0Ol06ocfflDt2rW9+n5cIYLx5s6dq5UrV2rdunWqVauWv9upVq5cuaIRI0bo3XffVYMGDfzdTrVWWlqq8PBwvfPOO+ratasef/xxzZgxQ0uWLPF3a9XO9u3b9corr+jtt9/WV199pbVr12rDhg2aM2eOv1vDHeAKUQVq0KCBAgMDlZeX57Y9Ly9PkZGRN31NZGRkuerh2TiXee211zR37lxt2bJF9957b0W2WS2Ud6xPnDih06dPa8CAAa5tpaWlkqSgoCAdO3ZMLVq0qNimqyBP/kw3atRINWvWVGBgoGtbu3bt5HA4dP36ddlstgrtuaryZKz/9Kc/acSIEXrqqackSZ06dVJBQYHGjh2rGTNmqEYNrjV4w899HoaEhHj96pDEFaIKZbPZ1LVrV6Wnp7u2lZaWKj09XTExMTd9TUxMjFu9JKWlpf1sPTwbZ0maN2+e5syZo40bN6pbt26+aLXKK+9Yt23bVgcOHFBWVpZr+d3vfqc+ffooKytL0dHRvmy/yvDkz3SPHj10/PhxV+CUpG+++UaNGjUiDN2CJ2P9/fff3xB6yoKoxc+Deo3PPw8rZKo2XFauXGnZ7XYrNTXVOnz4sDV27FgrLCzMcjgclmVZ1ogRI6xp06a56nft2mUFBQVZr732mnXkyBFr9uzZ3HZ/G8o7znPnzrVsNpv1n//5n1Zubq5ruXLlir9Oocoo71j/FHeZ3Z7yjnN2drYVHBxsjR8/3jp27Ji1fv16Kzw83Hr55Zf9dQpVRnnHevbs2VZwcLD14YcfWidPnrQ2b95stWjRwvr973/vr1OoEq5cuWLt27fP2rdvnyXJWrBggbVv3z7rzJkzlmVZ1rRp06wRI0a46stuu588ebJ15MgRa9GiRdx2X9W99dZb1j333GPZbDbr/vvvt/bs2ePa95vf/MYaNWqUW/3q1aut1q1bWzabzerQoYO1YcMGH3dcNZVnnJs0aWJJumGZPXu27xuvgsr7Z/qfEYhuX3nHeffu3Vb37t0tu91uNW/e3Pr3f/93q7i42MddV03lGeuioiLrhRdesFq0aGHVqlXLio6Otp555hnr0qVLvm+8Ctm2bdtN/90tG9tRo0ZZv/nNb254TefOnS2bzWY1b97cWrZsWYX1F2BZXN8DAABmYw4RAAAwHoEIAAAYj0AEAACMRyACAADGIxABAADjEYgAAIDxCEQAAMB4BCIAAGA8AhEAADAegQgAABiPQAQAAIxHIAIAAMb7/wZooEDlaS+JAAAAAElFTkSuQmCC"/>


```python
# 성별이 직원의 성과에 미치는 영향을 확인해 봅니다.

# your code here 
# dft_g_gr = df_train.groupby('Gender')['growth_rate'].mean()
# dft_g_gr

dft_g_gr = df_train[['Gender','growth_rate']].groupby(['Gender']).agg('median')
dft_g_gr
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
      <th>growth_rate</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>48.0</td>
    </tr>
    <tr>
      <th>M</th>
      <td>47.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# your code here 
# sns.barplot(
#     data= dft_g_gr,
#     x= "growth_rate",
#     y= 'Rate'
# )

dft_g_gr.T.plot(kind='bar', figsize=(10, 5) )
plt.ylabel('Rate')
plt.legend(loc="upper left") # 범례표..
plt.xticks(rotation=0); # 세로가 아닌 가로로 표시
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA0kAAAGvCAYAAACO+572AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAjIElEQVR4nO3de5CV9X3H8c9yWwTcRVBZiYtY75fgBW/rLWhI8VKrES+J2qhhmtiiiaBJZOI9neLYKJp6SzoGdKaUyFRJTEatoqIiWsVLvGusCqkCUYddQFkInP7hePpsRAVceJbl9Zo5Mzy/85znfA8zjL7nOc9zaiqVSiUAAAAkSbqUPQAAAEBHIpIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKupX55pdeemkuu+yyNms77bRTXn755STJ0qVLc95552XKlClpbW3NiBEjcsMNN2TAgAGr/R4rV67M22+/nU033TQ1NTXtOj8AALDhqFQqWbRoUQYOHJguXT79fFGpkZQku+22W+67777qdrdu/z/SmDFj8rvf/S5Tp05NfX19zj777Bx//PGZOXPmah//7bffTmNjY7vODAAAbLjmzp2brbfe+lOfLz2SunXrloaGhk+sNzc35+abb87kyZNz+OGHJ0kmTpyYXXbZJY899lgOOOCA1Tr+pptumuSjv4i6urr2GxwAANigtLS0pLGxsdoIn6b0SHrttdcycODA9OzZM01NTRk/fnwGDRqU2bNnZ/ny5Rk+fHh135133jmDBg3KrFmzPjWSWltb09raWt1etGhRkqSurk4kAQAAn3sZTqk3bth///0zadKk3H333bnxxhvzxhtv5JBDDsmiRYsyb9689OjRI3379m3zmgEDBmTevHmfeszx48envr6++vBVOwAAYE2UeibpyCOPrP55yJAh2X///bPNNtvktttuyyabbLJWxxw3blzGjh1b3f74lBoAAMDq6FC3AO/bt2923HHH/OEPf0hDQ0OWLVuWhQsXttln/vz5q7yG6WO1tbXVr9b5ih0AALCmSr8mqWjx4sV5/fXX83d/93cZOnRounfvnunTp2fkyJFJkldeeSVz5sxJU1NTu7/3ihUrsnz58nY/bkfTvXv3dO3atewxAACgwyo1ks4///wcc8wx2WabbfL222/nkksu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k/r6+gwZMiT33HNPvva1ryVJJkyYkC5dumTkyJFpbW3NiBEjcsMNN5Q5MgAA0Ml1uN9Jam9+Jwk+4neS2Nj5nSSI30lio7fB/U4SAABARyCSAAAACkQSAABAgUgCAAAoEEkAAAAFIgkAAKBAJAEAABSIJAAAgAKRBAAAUCCSAAAACkQSAABAgUgCAAAoEEkAAAAFIgkAAKBAJAEAABSIJAAAgAKRBAAAUPCFIukPf/hD7rnnnnz44YdJkkql0i5DAQAAlGWtIum9997L8OHDs+OOO+aoo47KO++8kyQZNWpUzjvvvHYdEAAAYH1aq0gaM2ZMunXrljlz5qRXr17V9ZNPPjl33313uw0HAACwvnVbmxf913/9V+65555svfXWbdZ32GGHvPXWW+0yGAAAQBnW6kzSkiVL2pxB+tj777+f2traLzwUAABAWdYqkg455JDceuut1e2ampqsXLkyV155ZQ477LB2Gw4AAGB9W6uv21155ZX56le/mieffDLLli3LD3/4w7zwwgt5//33M3PmzPaeEQAAYL1ZqzNJu+++e1599dUcfPDBOfbYY7NkyZIcf/zxefrpp7Pddtu194wAAADrzVqdSZozZ04aGxvz4x//eJXPDRo06AsPBgAAUIa1OpO07bbb5k9/+tMn1t97771su+22X3goAACAsqxVJFUqldTU1HxiffHixenZs+cXHgoAAKAsa/R1u7Fjxyb56G52F110UZvbgK9YsSKPP/549txzz3YdEAAAYH1ao0h6+umnk3x0Jum5555Ljx49qs/16NEje+yxR84///z2nRAAAGA9WqNIeuCBB5IkZ555Zq699trU1dWtk6EAAADKslZ3t5s4cWJ7zwEAANAhrFUkJcmTTz6Z2267LXPmzMmyZcvaPHf77bd/4cEAAADKsFZ3t5syZUoOPPDAvPTSS7njjjuyfPnyvPDCC7n//vtTX1/f3jMCAACsN2sVSf/8z/+cCRMm5M4770yPHj1y7bXX5uWXX85JJ53kh2QBAIAN2lpF0uuvv56jjz46yUd3tVuyZElqamoyZsyY/OIXv2jXAQEAANantYqkzTbbLIsWLUqSfOlLX8rzzz+fJFm4cGE++OCD9psOAABgPVurGzcceuihuffee/PlL385J554Yr7//e/n/vvvz7333pvDDz+8vWcEAABYb9Yqkq677rosXbo0SfLjH/843bt3z6OPPpqRI0f6MVkAAGCDtlZft+vXr18GDhz40QG6dMkFF1yQ2267LQMHDsxee+3VrgMCAACsT2sUSa2trRk3blz22WefHHjggZk2bVqSj35cdrvttsu1116bMWPGrIs5AQAA1os1+rrdxRdfnJ///OcZPnx4Hn300Zx44ok588wz89hjj+Wqq67KiSeemK5du66rWQEAANa5NYqkqVOn5tZbb83f/u3f5vnnn8+QIUPy5z//Oc8++2xqamrW1YwAAADrzRp93e6Pf/xjhg4dmiTZfffdU1tbmzFjxggkAACg01ijSFqxYkV69OhR3e7WrVv69OnT7kMBAACUZY2+blepVHLGGWektrY2SbJ06dKcddZZ6d27d5v9br/99vabEAAAYD1ao0g6/fTT22yfdtpp7ToMAABA2dYokiZOnLiu5gAAAOgQ1urHZAEAADorkQQAAFAgkgAAAApEEgAAQIFIAgAAKBBJAAAABSIJAACgQCQBAAAUiCQAAIACkQQAAFAgkgAAAApEEgAAQIFIAgAAKBBJAAAABSIJAACgQCQBAAAUiCQAAICCUiNp/Pjx2XfffbPppptmyy23zHHHHZdXXnmlzT5Lly7N6NGj079///Tp0ycjR47M/PnzS5oYAADo7EqNpBkzZmT06NF57LHHcu+992b58uX567/+6yxZsqS6z5gxY3LnnXdm6tSpmTFjRt5+++0cf/zxJU4NAAB0Zt3KfPO77767zfakSZOy5ZZbZvbs2Tn00EPT3Nycm2++OZMnT87hhx+eJJk4cWJ22WWXPPbYYznggAPKGBsAAOjEOtQ1Sc3NzUmSfv36JUlmz56d5cuXZ/jw4dV9dt555wwaNCizZs1a5TFaW1vT0tLS5gEAALC6OkwkrVy5Mueee24OOuig7L777kmSefPmpUePHunbt2+bfQcMGJB58+at8jjjx49PfX199dHY2LiuRwcAADqRDhNJo0ePzvPPP58pU6Z8oeOMGzcuzc3N1cfcuXPbaUIAAGBjUOo1SR87++yz89vf/jYPPfRQtt566+p6Q0NDli1bloULF7Y5mzR//vw0NDSs8li1tbWpra1d1yMDAACdVKlnkiqVSs4+++zccccduf/++7Ptttu2eX7o0KHp3r17pk+fXl175ZVXMmfOnDQ1Na3vcQEAgI1AqWeSRo8encmTJ+fXv/51Nt100+p1RvX19dlkk01SX1+fUaNGZezYsenXr1/q6upyzjnnpKmpyZ3tAACAdaLUSLrxxhuTJMOGDWuzPnHixJxxxhlJkgkTJqRLly4ZOXJkWltbM2LEiNxwww3reVIAAGBjUWokVSqVz92nZ8+euf7663P99devh4kAAICNXYe5ux0AAEBHIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKSo2khx56KMccc0wGDhyYmpqaTJs2rc3zlUolF198cbbaaqtssskmGT58eF577bVyhgUAADYKpUbSkiVLsscee+T6669f5fNXXnllfvazn+Wmm27K448/nt69e2fEiBFZunTpep4UAADYWHQr882PPPLIHHnkkat8rlKp5JprrsmFF16YY489Nkly6623ZsCAAZk2bVq+8Y1vrM9RAQCAjUSHvSbpjTfeyLx58zJ8+PDqWn19ffbff//MmjXrU1/X2tqalpaWNg8AAIDV1WEjad68eUmSAQMGtFkfMGBA9blVGT9+fOrr66uPxsbGdTonAADQuXTYSFpb48aNS3Nzc/Uxd+7cskcCAAA2IB02khoaGpIk8+fPb7M+f/786nOrUltbm7q6ujYPAACA1dVhI2nbbbdNQ0NDpk+fXl1raWnJ448/nqamphInAwAAOrNS7263ePHi/OEPf6huv/HGG3nmmWfSr1+/DBo0KOeee27+6Z/+KTvssEO23XbbXHTRRRk4cGCOO+648oYGAAA6tVIj6cknn8xhhx1W3R47dmyS5PTTT8+kSZPywx/+MEuWLMl3vvOdLFy4MAcffHDuvvvu9OzZs6yRAQCATq6mUqlUyh5iXWppaUl9fX2am5tdn8RGbfAFvyt7BCjVmz1PKXsEKN+lzWVPAKVa3TbosNckAQAAlEEkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoEAkAQAAFIgkAACAApEEAABQIJIAAAAKRBIAAECBSAIAACgQSQAAAAUiCQAAoGCDiKTrr78+gwcPTs+ePbP//vvnv//7v8seCQAA6KQ6fCT96le/ytixY3PJJZfkqaeeyh577JERI0ZkwYIFZY8GAAB0Qh0+kq6++ur8/d//fc4888zsuuuuuemmm9KrV6/88pe/LHs0AACgE+pW9gCfZdmyZZk9e3bGjRtXXevSpUuGDx+eWbNmrfI1ra2taW1trW43NzcnSVpaWtbtsNDBrWz9oOwRoFQtNZWyR4Dy+f8hNnIfN0Gl8tn/TejQkfTuu+9mxYoVGTBgQJv1AQMG5OWXX17la8aPH5/LLrvsE+uNjY3rZEYANgz1ZQ8AHcEV/iVAkixatCj19Z/+76FDR9LaGDduXMaOHVvdXrlyZd5///30798/NTU1JU4GQFlaWlrS2NiYuXPnpq6uruxxAChJpVLJokWLMnDgwM/cr0NH0uabb56uXbtm/vz5bdbnz5+fhoaGVb6mtrY2tbW1bdb69u27rkYEYANSV1cnkgA2cp91BuljHfrGDT169MjQoUMzffr06trKlSszffr0NDU1lTgZAADQWXXoM0lJMnbs2Jx++unZZ599st9+++Waa67JkiVLcuaZZ5Y9GgAA0Al1+Eg6+eST86c//SkXX3xx5s2blz333DN33333J27mAACfpra2Npdccsknvo4NAKtSU/m8+98BAABsRDr0NUkAAADrm0gCAAAoEEkAAAAFIgmATq+mpibTpk0rewwANhAiCYBO49JLL82ee+5Z9hhVgwcPzjXXXFP2GACsIZEEQCmWLVtW9ghrpVKp5M9//nPZYwCwDokkANrFokWLcuqpp6Z3797ZaqutMmHChAwbNiznnntuko/OqvzkJz/Jt771rdTV1eU73/lOkuQ///M/s9tuu6W2tjaDBw/OVVddVT3mddddl9133726PW3atNTU1OSmm26qrg0fPjwXXnhhJk2alMsuuyzPPvtsampqUlNTk0mTJlX3e/fdd/P1r389vXr1yg477JDf/OY3q/W5HnzwwdTU1OSuu+7K0KFDU1tbm0ceeSSvv/56jj322AwYMCB9+vTJvvvum/vuu6/6umHDhuWtt97KmDFjqvN87JFHHskhhxySTTbZJI2Njfne976XJUuWrNHfNwDrjkgCoF2MHTs2M2fOzG9+85vce++9efjhh/PUU0+12eenP/1p9thjjzz99NO56KKLMnv27Jx00kn5xje+keeeey6XXnppLrroomrcfOUrX8mLL76YP/3pT0mSGTNmZPPNN8+DDz6YJFm+fHlmzZqVYcOG5eSTT855552X3XbbLe+8807eeeednHzyydX3vuyyy3LSSSfl97//fY466qiceuqpef/991f7811wwQW54oor8tJLL2XIkCFZvHhxjjrqqEyfPj1PP/10jjjiiBxzzDGZM2dOkuT222/P1ltvncsvv7w6T5K8/vrrOeKIIzJy5Mj8/ve/z69+9as88sgjOfvss9f2rx6A9lYBgC+opaWl0r1798rUqVOrawsXLqz06tWr8v3vf79SqVQq22yzTeW4445r87pTTjml8rWvfa3N2g9+8IPKrrvuWqlUKpWVK1dW+vfvXz3unnvuWRk/fnyloaGhUqlUKo888kile/fulSVLllQqlUrlkksuqeyxxx6fmC9J5cILL6xuL168uJKkctddd33uZ3vggQcqSSrTpk373H132223yr/+679Wt7fZZpvKhAkT2uwzatSoyne+8502aw8//HClS5culQ8//PBz3wOAdc+ZJAC+sP/5n//J8uXLs99++1XX6uvrs9NOO7XZb5999mmz/dJLL+Wggw5qs3bQQQfltddey4oVK1JTU5NDDz00Dz74YBYuXJgXX3wx//iP/5jW1ta8/PLLmTFjRvbdd9/06tXrc2ccMmRI9c+9e/dOXV1dFixYsNqf8S9nX7x4cc4///zssssu6du3b/r06ZOXXnqpeibp0zz77LOZNGlS+vTpU32MGDEiK1euzBtvvLHa8wCw7nQrewAANh69e/de49cMGzYsv/jFL/Lwww9nr732Sl1dXTWcZsyYka985SurdZzu3bu32a6pqcnKlStXe46/nP3888/Pvffem5/+9KfZfvvts8kmm+SEE0743BtSLF68ON/97nfzve997xPPDRo0aLXnAWDdcSYJgC/sr/7qr9K9e/c88cQT1bXm5ua8+uqrn/m6XXbZJTNnzmyzNnPmzOy4447p2rVrkv+/Lmnq1KkZNmxYko/C6b777svMmTOra0nSo0ePrFixon0+1OeYOXNmzjjjjHz961/Pl7/85TQ0NOTNN99ss8+q5tl7773z4osvZvvtt//Eo0ePHutldgA+m0gC4AvbdNNNc/rpp+cHP/hBHnjggbzwwgsZNWpUunTp0uaubn/pvPPOy/Tp0/OTn/wkr776am655ZZcd911Of/886v7DBkyJJtttlkmT57cJpKmTZuW1tbWNl/XGzx4cN54440888wzeffdd9Pa2rrOPvMOO+yQ22+/Pc8880yeffbZnHLKKZ84MzV48OA89NBD+d///d+8++67SZIf/ehHefTRR3P22WfnmWeeyWuvvZZf//rXbtwA0IGIJADaxdVXX52mpqb8zd/8TYYPH56DDjoou+yyS3r27Pmpr9l7771z2223ZcqUKdl9991z8cUX5/LLL88ZZ5xR3aempiaHHHJIampqcvDBByf5KJzq6uqyzz77tPka3MiRI3PEEUfksMMOyxZbbJH/+I//WKefd7PNNsuBBx6YY445JiNGjMjee+/dZp/LL788b775ZrbbbrtsscUW1dlnzJiRV199NYccckj22muvXHzxxRk4cOA6mxWANVNTqVQqZQ8BQOezZMmSfOlLX8pVV12VUaNGlT0OAKw2N24AoF08/fTTefnll7Pffvulubk5l19+eZLk2GOPLXkyAFgzvm4HQLv5+Mdihw8fniVLluThhx/O5ptvXvZYn+mss85qczvu4uOss84qezwASuDrdgBs1BYsWJCWlpZVPldXV5ctt9xyPU8EQNlEEgAAQIGv2wEAABSIJAAAgAKRBAAAUCCSAAAACkQSAABAgUgCAAAoEEkAAAAFIgkAAKDg/wDhHb3fCuyaGQAAAABJRU5ErkJggg=="/>


```python
# 데이터 세트에서 남성과 여성의 수 시각화
# your code here 
plt.figure(figsize=(10, 5))
sns.countplot(x=df_train['Gender'], palette = 'bone')
plt.title('Comparison of Males and Females', fontweight = 30)
plt.xlabel('Gender')
plt.ylabel('Count')
```

<pre>
Text(0, 0.5, 'Count')
</pre>
<img src="data:image/png;base64,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1Sr9z//8jxISErRv3z4tXLhQsbGxOnny5CXf++STT+r9999Xhw4dNGLECMXGxurYsWP64osvtH79eh07dkyS1KNHD4WGhqpjx44KCQnRN998o7lz5yohIeGSD3P5Lb1dTGpqqhISEnTLLbdo6NChOnbsmPUZYGVd5/kGDRqkN954QyNHjtT777+vjh07qrCwULt379Ybb7yhNWvWqF27dr95OzExMWrUqJHGjRunQ4cOyel0auXKlb96z6D0y/8jHnvsMaWkpGj//v3q27evqlevrn379umtt97Sgw8+qHHjxmnjxo1KTk7Wvffeq5tuuknnzp3TK6+8YgVtAJXUVX/+IQBUQP/+97/NiBEjTIMGDYyfn5+pXr266dixo3nuuefMmTNnrLqzZ8+aadOmmaioKOPr62siIyNNSkqKW40xvzz6PSEhocR2dMGjxI3576O4n376aWssMTHRBAYGmr1791qfQxQSEmKmTJni9gh0Y4x56aWXzI033mgcDoeJiYkxixcvLvWzgkrb9vlzxY8pz8/PN+PHjzctW7Y01atXN4GBgaZly5alfibW66+/blq3bm0cDoepVauWGThwoPnhhx/caor35UKl9Xgxc+fONTExMcbX19eEhISYUaNGuX2W1/nru9JHv1/KhcesqKjIPPHEE6Z+/frG4XCY1q1bm7S0NJOYmFjiUejnH9NiOTk5JikpyURGRhpfX18TGhpqunXrZhYtWmTVPP/886Zz586mdu3axuFwmEaNGpnx48ebvLy8S/Z6ub2V9v12qZ5XrlxpmjRpYhwOh4mNjTWrVq0qdX9Lc+utt/7qY+wLCgrMU089ZZo2bWocDoepWbOmadu2rZk2bZrbPl/uz44xxrz//vslHsP+9ddfm+7du5tq1aqZOnXqmBEjRpivvvrKSDKLFy+26i72fbly5Upzyy23mMDAQBMYGGhiYmJMUlKSycrKMsYY891335mhQ4eaRo0aGX9/f1OrVi3TtWtXs379+l89TgCuX17GXMU7YQEAl2XIkCF68803y+XsAQAA8Azu2QIAAAAAGxC2AAAAAMAGhC0AAAAAsAH3bAEAAACADTizBQAAAAA2IGwBAAAAgA34UOPLUFRUpMOHD6t69ery8vLydDsAAAAAPMQYoxMnTig8PFze3pc+d0XYugyHDx9WZGSkp9sAAAAAUEEcPHhQ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ZQpbs2fPLnV86tSpOnny5G9qCAAAAACuB+V6z9b999+vl19+uTxXCQAAAADXpHINWxkZGfL39y/PVQIAAADANalMlxHefffdbq+NMTpy5Ig+//xz/e1vfyuXxgAAAADgWlamsBUUFOT22tvbW40bN9b06dPVo0ePcmkMAAAAAK5lZQpbixcvLu8+AAAAAOC6UqawVWzbtm365ptvJElNmzZV69aty6UpAAAAALjWlSls5ebmqn///tq0aZNq1KghSTp+/Li6du2q1157TXXr1i3PHgEAAADgmlOmpxE+/PDDOnHihHbt2qVjx47p2LFj2rlzp1wul/70pz+Vd48AAAAAcM0p05mt1atXa/369WrSpIk1Fhsbq3nz5vGADAAAAABQGc9sFRUVydfXt8S4r6+vioqKfnNTAAAAAHCtK1PYuu222/TnP/9Zhw8ftsYOHTqkMWPGqFu3buXWHAAAAABcq8oUtubOnSuXy6UGDRqoUaNGatSokaKiouRyufTcc89d9npSU1PVvn17Va9eXcHBwerbt6+ysrLcas6cOaOkpCTVrl1b1apVU79+/ZSTk+NWc+DAASUkJKhq1aoKDg7W+PHjde7cObeaTZs2qU2bNnI4HIqOjtaSJUvKsusAAAAAcFnKdM9WZGSkvvjiC61fv167d++WJDVp0kTdu3e/ovVs3rxZSUlJat++vc6dO6e//OUv6tGjh77++msFBgZKksaMGaP09HStWLFCQUFBSk5O1t13362PP/5YklRYWKiEhASFhoZqy5YtOnLkiAYPHixfX1898cQTkqR9+/YpISFBI0eO1LJly7RhwwYNHz5cYWFhio+PL8shAAAAAIBL8jLGmMst3rhxo5KTk/XJJ5/I6XS6zeXl5en3v/+9Fi5cqE6dOpWpmaNHjyo4OFibN29W586dlZeXp7p162r58uW65557JEm7d+9WkyZNlJGRoZtvvlnvvfee7rzzTh0+fFghISGSpIULF2rixIk6evSo/Pz8NHHiRKWnp2vnzp3Wtvr376/jx49r9erVJfrIz89Xfn6+9drlcikyMlJ5eXkl9rsi6NNnhKdbAIBy9c47L3i6hWvSopXveroFAChXD/a7y9MtlOByuRQUFHRZ2eCKLiN89tlnNWLEiFJXGhQUpP/93//VrFmzrqzb8+Tl5UmSatWqJemXD00+e/as2xmz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gLAFAAAAADYgbAEAAACADQhbAACUoy5dumj06NGebgMAUAEQtgAA153s7Gz9+c9/VnR0tPz9/RUSEqKOHTtqwYIF+vnnnz3dHgCgkqji6QYAAChP3333nTp27KgaNWroiSeeUPPmzeVwOLRjxw4tWrRIN9xwg+666y5Pt3lRhYWF8vLykrc3/x4KANc6/k8OALiuPPTQQ6pSpYo+//xz/eEPf1CTJk3UsGFD9enTR+np6erdu7ck6fjx4xo+fLjq1q0rp9Op2267TV999ZW1nqlTp6pVq1Z65ZVX1KBBAwUFBal///46ceKEVXPq1CkNHjxY1apVU1hYmGbOnFmin/z8fI0bN0433HCDAgMD1aFDB23atMmaX7JkiWrUqKF3331XsbGxcjgcOnDggH0HCABw1RC2AADXjR9//FFr165VUlKSAgMDS63x8vKSJN17773Kzc3Ve++9p23btqlNmzbq1q2bjh07ZtXu3btXb7/9ttLS0pSWlqbNmzfrySeftObHjx+vzZs365133tHatWu1adMmffHFF27bS05OVkZGhl577TVt375d9957r3r27Klvv/3Wqvn555/11FNP6cUXX9SuXbsUHBxcnocFAOAhXEYIALhu7NmzR8YYNW7c2G28Tp06OnPmjCQpKSlJvXv31meffabc3Fw5HA5J0jPPPKO3335bb775ph588EFJUlFRkZYsWaLq1atLkgYNGqQNGzbo8ccf18mTJ/XSSy/p1VdfVbdu3SRJS5cuVUREhLXdAwcOaPHixTpw4IDCw8MlSePGjdPq1au1ePFiPfHEE5Kks2fPav78+WrZsqWNRwcAcLURtgAA173PPvtMRUVFGjhwoPLz8/XVV1/p5MmTql27tlvd6dOntXfvXut1gwYNrKAlSWFhYcrNzZX0y1mvgoICdejQwZqvVauWW9DbsWOHCgsLddNNN7ltJz8/323bfn5+atGiRfnsLACgwiBsAQCuG9HR0fLy8lJWVpbbeMOGDSVJAQEBkqSTJ08qLCzM7d6pYjVq1LC+9vX1dZvz8vJSUVHRZfdz8uRJ+fj4aNu2bfLx8XGbq1atmvV1QECAdXkjAOD6QdgCAFw3ateurdtvv11z587Vww8/fNH7ttq0aaPs7GxVqVJFDRo0KNO2GjVqJF9fX3366aeqV6+eJOmnn37Sv//9b916662SpNatW6uwsFC5ubnq1KlTmbYDALh28YAMAMB1Zf78+Tp37pzatWun119/Xd98842ysrL06quvavfu3fLx8VH37t0VFxenvn37au3atdq/f7+2bNmiv/71r/r8888vazvVqlXTsGHDNH78eG3cuFE7d+7UkCFD3B7ZftNNN2ngwIEaPHiwVq1apX379umzzz5Tamqq0tPT7ToEAIAKgjNbAIDrSqNGjfTll1/qiSeeUEpKin744Qc5HA7FxsZq3Lhxeuihh+Tl5aV//etf+utf/6oHHnhAR48eVWhoqDp37qyQkJDL3tbTTz+tkydPqnfv3qpevboeeeQR5eXludUsXrxYjz32mB555BEdOnRIderU0c0336w777yzvHcdAFDBeBljjKebAAAAAIDrDZcRAgAAAIANCFsAAAAAYAPCFgAAAADYgLAFAAAAADYgbAEAAACADQhbAAAAAGADwhYAAAAA2ICwBQAAAAA2IGwBAAAAgA0IWwAAAABgA8IWAAAAANjg/wOQ9Ux8b1vv8QAAAABJRU5ErkJggg=="/>


```python
# 데이터 세트에서 Hometown 그룹별시각화

# your code here 
plt.figure(figsize=(10, 5))
sns.countplot(x=df_train['Hometown'], palette = 'pastel')
plt.title('Comparison of various Groups', fontweight = 30)
plt.xlabel('Groups')
plt.ylabel('Count')
```

<pre>
Text(0, 0.5, 'Count')
</pre>
<img src="data:image/png;base64,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reHi4mL8/vvvZvnly5cdxpOammrUqFHDePzxxx3KJRlOTk7GwYMHs8xNkjF69Ghz2dvb24iKirrhc5Gammr4+voaNWrUMP7880+zfPny5YYkY9SoUVnmMnbsWIdt1K1b16hfv/4N+zAMwzh79qzh4uJihIeHG+np6Wb5jBkzDEnGnDlzzLLRo0cbkhyemxsJDQ3N0veOHTsMScZ//vMfs+z659YwDGPcuHGGzWZzeA0z5/h///d/Wdr36tXLKFu2rLm8bNkyQ5Lxz3/+06HdU089ZdhsNuPIkSOGYRhGQkKCIcmYO3dulm3m9PXKTub+37Fjxyx158+fN37//Xfzce3zcKO5WjGvzNe0ffv2Du1efPFFQ5Kxd+9ewzAMY+rUqbf92gNATnBkCwByyW63S5I8PT1vq/3KlSslKcu39MOHD5ekLEeSQkJCFBoaai5nHj14/PHHVaZMmSzlx44dy9LnwIEDzb8zTwNMTU3VmjVrzPIiRYqYf58/f17Jycl67LHHsj2FrGnTpgoJCbnFTP+67mn79u06depUtvW7du3S2bNn9eKLLzpc79W2bVtVrVo126NqL7zwgsPyY489lu2cr7VmzRqlpqZqyJAhDtcN9evXT15eXrm+nq5z587avXu3jh49apYtXrxYrq6u6tChg1l27XN76dIl/fe//9UjjzwiwzCynPopSQMGDLhl3ytXrlShQoX00ksvOZQPHz5chmHom2++yfF8bvV6ZSdz/y9atGiWumbNmsnHx8d8zJw5M0ub6+dqxbwyRUVFOSwPGjTI7FP633V6X375ZZYjxABwJwhbAJBLXl5ekv663uV2nDhxQk5OTlnudOfv769ixYpluQbo2kAlSd7e3pLkcP3LteWZ1wllcnJyUoUKFRzKqlSpIumv614yLV++XI0aNZKbm5tKlCghHx8fzZ49W8nJyVnmUL58+VtNU9Jf1/IcOHBAQUFBevjhhxUTE+MQjDLnGhwcnGXdqlWrZnku3Nzc5OPj41BWvHjxLHO+3o36cXFxUYUKFbK97up2PP3003JyctLixYsl/XVq35IlS9S6dWtzv5CkkydPqnfv3ipRooR5rVnTpk0lKcvzW7hwYZUuXfqWfZ84cUKBgYFZQn61atXM+py61euVncz+L168mKXu/fffV2xsrD7++ONs181urlbMK1PlypUdlitWrCgn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AAByIkdh67XXXtO5c+dUpUoVTZgwQV9++aW+/PJLvf322woODta5c+f06quv5snAypcvL39/f61du9Yss9vt2r59u0JDQyVJoaGhSkpKcrhObN26dcrIyFDDhg3NNhs3blRaWprZJjY2VsHBwSpevHiejBUAAAAArpejuxH6+flpy5YtGjBggKKjo2UYhiTJZrMpIiJCM2fOzHJ91M1cvHhRR44cMZcTEhIUFxenEiVKqEyZMhoyZIj++c9/qnLlyipfvrxef/11BQYGqmPHjpKkatWqqVWrVurXr5/ee+89paWlaeDAgerSpYsCAwMlSd26ddOYMWPUp08fjRw5UgcOHNC0adM0derUnEwdAAAAAHIkR2FLksqWLauVK1fq/PnzOnLkiAzDUOXKlXN1lGjXrl0KCwszl4cNGyZJ6tWrl+bNm6dXXnlFly5dUv/+/ZWUlKRHH31Uq1atkpubm7nOggULNHDgQDVv3lxOTk6KjIzUu+++a9Z7e3tr9erVioqKUv369VWqVCmNGjWK274DAAAAsJTNyDw8hRuy2+3y9vZWcnJyrq/f+mxH3tylEQXDUw/75FvfSaum51vfuPuKtRqU30MALHfxs8X5PQTcRUWf6pxvfR/YnPW+BLh/1WjsdutG2chJNsjRNVsAAAAAgNtD2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALHDPh61y5crJZrNleURFRUmSmjVrlqXuhRdecNjGyZMn1bZtW7m7u8vX11cjRozQ1atX82M6AAAAAB4QhfN7ALeyc+dOpaenm8sHDhxQy5Yt9fTTT5tl/fr109ixY81ld3d38+/09HS1bdtW/v7+2rJli06fPq2ePXvK2dlZb7311t2ZBAAAAIAHzj0ftnx8fByWx48fr4oVK6pp06Zmmbu7u/z9/bNdf/Xq1frxxx+1Zs0a+fn5qU6dOnrjjTc0cuRIxcTEyMXFxdLxAwAAAHgw3fOnEV4rNTVVH3/8sZ577jnZbDazfMGCBSpVqpRq1Kih6OhoXb582azbunWratasKT8/P7MsIiJCdrtdBw8ezLaflJQU2e12hwcAAAAA5MQ9f2TrWsuWLVNSUpJ69+5tlnXr1k1ly5ZVYGCg9u3bp5EjRyo+Pl5Lly6VJCUmJjoELUnmcmJiYrb9jBs3TmPGjLFmEgAAAAAeCAUqbH344Ydq3bq1AgMDzbL+/fubf9esWVMBAQFq3ry5jh49qooVK+aqn+joaA0bNsxcttvtCgoKyv3AAQAAADxwCkzYOnHihNasWWMesbqRhg0bSpKOHDmiihUryt/fXzt27HBoc+bMGUm64XVerq6ucnV1zYNRAwAAAHhQFZhrtubOnStfX1+1bdv2pu3i4uIkSQEBAZKk0NBQ7d+/X2fPnjXbxMbGysvLSyEhIZaNFwAAAMCDrUAc2crIyNDcuXPVq1cvFS78vyEfPXpUCxcuVJs2bVSyZEnt27dPQ4cOVZMmTVSrVi1JUnh4uEJCQtSjRw9NmDBBiYmJeu211xQVFcXRKwAAAACWKRBha82aNTp58qSee+45h3IXFxetWbNG77zzji5duqSgoCBFRkbqtddeM9sUKlRIy5cv14ABAxQaGioPDw/16tXL4Xe5AAAAACCvFYiwFR4eLsMwspQHBQXpu+++u+X6ZcuW1cqVK60YGgAAAABkq8BcswUAAAAABQlhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsUDi/BwAAKJje/3lRfg8Bd9Hzlbvk9xAAoMDhyBYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFrinw1ZMTIxsNpvDo2rVqmb9lStXFBUVpZIlS6po0aKKjIzUmTNnHLZx8uRJtW3bVu7u7vL19dWIESN09erVuz0VAAAAAA+Ye/53tqpXr641a9aYy4UL/2/IQ4cO1YoVK7RkyRJ5e3tr4MCBevLJJ7V582ZJUnp6utq2bSt/f39t2bJFp0+fVs+ePeXs7Ky33nrrrs8FAAAAwIPjng9bhQsXlr+/f5by5ORkffjhh1q4cKEef/xxSdLcuXNVrVo1bdu2TY0aNdLq1av1448/as2aNfLz81OdOnX0xhtvaOTIkYqJiZGLi8vdng4AAACAB8Q9fRqhJP38888KDAxUhQoV1L17d508eVKStHv3bqWlpalFixZm26pVq6pMmTLaunWrJGnr1q2qWbOm/Pz8zDYRERGy2+06ePDgDftMSUmR3W53eAAAAABATtzTYathw4aaN2+eVq1apdmzZyshIUGPPfaYLly4oMTERLm4uKhYsWIO6/j5+SkxMVGSlJiY6BC0Musz625k3Lhx8vb2Nh9BQUF5OzEAAAAA9717+jTC1q1bm3/XqlVLDRs2VNmyZfXpp5+qSJEilvUbHR2tYcOGmct2u53ABQAAACBH7ukjW9crVqyYqlSpoiNHjsjf31+pqalKSkpyaHPmzBnzGi9/f/8sdyfMXM7uOrBMrq6u8vLycngAAAAAQE4UqLB18eJFHT16VAEBAapfv76cnZ21du1asz4+Pl4nT55UaGioJCk0NFT79+/X2bNnzTaxsbHy8vJSSEjIXR8/AAAAgAfHPX0a4csvv6x27dqpbNmyOnXqlEaPHq1ChQqpa9eu8vb2Vp8+fTRs2DCVKFFCXl5eGjRokEJDQ9WoUSNJUnh4uEJCQtSjRw9NmDBBiYmJeu211xQVFSVXV9d8nh0AAACA+9k9HbZ+/fVXde3aVX/88Yd8fHz06KOPatu2bfLx8ZEkTZ06VU5OToqMjFRKSooiIiI0a9Ysc/1ChQpp+fLlGjBggEJDQ+Xh4aFevXpp7Nix+TUlAAAAAA+IezpsLVq06Kb1bm5umjlzpmbOnHnDNmXLltXKlSvzemgAAAAAcFMF6potAAAAACgoCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABggXs6bI0bN04PPfSQPD095evrq44dOyo+Pt6hTbNmzWSz2RweL7zwgkObkydPqm3btnJ3d5evr69GjBihq1ev3s2pAAAAAHjAFM7vAdzMd999p6ioKD300EO6evWq/vGPfyg8PFw//vijPDw8zHb9+vXT2LFjzWV3d3fz7/T0dLVt21b+/v7asmWLTp8+rZ49e8rZ2VlvvfXWXZ0PAAAAgAfHPR22Vq1a5bA8b948+fr6avfu3WrSpIlZ7u7uLn9//2y3sXr1av34449as2aN/Pz8VKdOHb3xxhsaOXKkYmJi5OLikmWdlJQUpaSkmMt2uz2PZgQAAADgQXFPn0Z4veTkZElSiRIlHMoXLFigUqVKqUaNGoqOjtbly5fNuq1bt6pmzZry8/MzyyIiImS323Xw4MFs+xk3bpy8vb3NR1BQkAWzAQAAAHA/u6ePbF0rIyNDQ4YMUePGjVWjRg2zvFu3bipbtqwCAwO1b98+jRw5UvHx8Vq6dKkkKTEx0SFoSTKXExMTs+0rOjpaw4YNM5ftdjuBCwAAAECOFJiwFRUVpQMHDuj77793KO/fv7/5d82aNRUQEKDmzZvr6NGjqlixYq76cnV1laur6x2NFwAAAMCDrUCcRjhw4EAtX75c69evV+nSpW/atmHDhpKkI0eOSJL8/f115swZhzaZyze6zgsAAAAA7tQ9HbYMw9DAgQP1xRdfaN26dSpfvvwt14mLi5MkBQQESJJCQ0O1f/9+nT171mwTGxsrLy8vhYSEWDJuAAAAALinTyOMiorSwoUL9eWXX8rT09O8xsrb21tFihTR0aNHtXDhQrVp00YlS5bUvn37NHToUDVp0kS1atWSJIWHhyskJEQ9evTQhAkTlJiYqNdee01RUVGcKggAAADAMvf0ka3Zs2crOTlZzZo1U0BAgPlYvHixJMnFxUVr1qxReHi4qlatquHDhysyMlJff/21uY1ChQpp+fLlKlSokEJDQ/XMM8+oZ8+eDr/LBQAAAAB57Z4+smUYxk3rg4KC9N13391yO2XLltXKlSvzalgAAAAAcEv39JEtAAAAACioCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUeqLA1c+ZMlStXTm5ubmrYsKF27NiR30MCAAAAcJ96YMLW4sWLNWzYMI0ePVp79uxR7dq1FRERobNnz+b30AAAAADchx6YsDVlyhT169dPzz77rEJCQvTee+/J3d1dc+bMye+hAQAAALgPFc7vAdwNqamp2r17t6Kjo80yJycntWjRQlu3bs3SPiUlRSkpKeZycnKyJMlut+d6DJcvXsj1uih47HbX/Ov70p/51jfuPqc7eF+6U39evJxvfePuu5PPwDt18TL72oMkIz/3tUtX8q1v3H12e2ou1/trHzUM45ZtH4iw9d///lfp6eny8/NzKPfz89Phw4eztB83bpzGjBmTpTwoKMiyMQJA7ozM7wHgATFUffJ7CHhgPJffAwBuy4ULF+Tt7X3TNg9E2Mqp6OhoDRs2zFzOyMjQuXPnVLJkSdlstnwcWcFit9sVFBSkX375RV5eXvk9HNzH2Ndwt7Cv4W5hX8Pdwr6Wc4Zh6MKFCwoMDLxl2wcibJUqVUqFChXSmTNnHMrPnDkjf3//LO1dXV3l6up4GlixYsWsHOJ9zcvLi3+8uCvY13C3sK/hbmFfw93CvpYztzqilemBuEGGi4uL6tevr7Vr15plGRkZWrt2rUJDQ/NxZAAAAADuVw/EkS1JGjZsmHr16qUGDRro4Ycf1jvvvKNLly7p2Wefze+hAQAAALgPPTBhq3Pnzvr99981atQoJSYmqk6dOlq1alWWm2Yg77i6umr06NFZTskE8hr7Gu4W9jXcLexruFvY16xlM27nnoUAAAAAgBx5IK7ZAgAAAIC7jbAFAAAAABYgbAEAAACABQhbuGt69+6tjh073rA+JiZGderUue32uD/NmzfPkt+1u3z5siIjI+Xl5SWbzaakpCSVK1dO77zzzm1v4/p9NDvstwWHzWbTsmXLJEnHjx+XzWZTXFxcvo4J969mzZppyJAh+T0MPCBu5z3Nqs9bOCJsPaB69+4tm82W5XHkyJH8Hppp2rRpmjdvXn4PAzfw+++/a8CAASpTpoxcXV3l7++viIgIbd68+Y6227lzZ/300095NMr/mT9/vjZt2qQtW7bo9OnT8vb21s6dO9W/f/887wv3hsTERA0aNEgVKlSQq6urgoKC1K5dO4ffXMwUFBSk06dPq0aNGre9/dsJ37i/8GUKcuu9996Tp6enrl69apZdvHhRzs7OatasmUPbDRs2yGaz6ejRo5aOyarP25x+kXm/e2Bu/Y6sWrVqpblz5zqU+fj4OCynpqbKxcXlbg7LdLu/zI38ERkZqdTUVM2fP18VKlTQmTNntHbtWv3xxx+53mZaWpqKFCmiIkWK5OFI/3L06FFVq1bN4T/T1+/vuH8cP35cjRs3VrFixTRx4kTVrFlTaWlp+vbbbxUVFaXDhw87tC9UqJD8/f3zabQA7ndhYWG6ePGidu3apUaNGkmSNm3aJH9/f23fvl1XrlyRm5ubJGn9+vUqU6aMKlasaOmYrPq8hSOObD3AMo9GXPto3ry5Bg4cqCFDhqhUqVKKiIiQJE2ZMkU1a9aUh4eHgoKC9OKLL+rixYvmtjIPRX/77beqVq2aihYtqlatWun06dM37H/nzp3y8fHR22+/nW399d8gNmvWTC+99JJeeeUVlShRQv7+/oqJicmT5wI5k5SUpE2bNuntt99WWFiYypYtq4cffljR0dFq3769pL9O0Zo9e7Zat26tIkWKqEKFCvrss8/MbWSe4rB48WI1bdpUbm5uWrBgQZbTGjKPHnz00UcqV66cvL291aVLF124cMFsc+HCBXXv3l0eHh4KCAjQ1KlTHU7ZadasmSZPnqyNGzfKZrOZ3yJe/+1bUlKS+vbtKx8fH3l5eenxxx/X3r17b/g8pKena9iwYSpWrJhKliypV155Rfyaxr3hxRdflM1m044dOxQZGakqVaqoevXqGjZsmLZt25al/fWn3GR+s7x27Vo1aNBA7u7ueuSRRxQfHy/pr/e8MWPGaO/eveaZAZlH4k+ePKkOHTqoaNGi8vLyUqdOnXTmzBmzr9vZp1HwHDhwQK1bt1bRokXl5+enHj166L///a9Dm6tXr2rgwIHy9vZWqVKl9Prrrzu8Z3z00Udq0KCBPD095e/vr27duuns2bNm/a32y0yzZ89WxYoV5eLiouDgYH300UcO9TabTf/+97/197//Xe7u7qpcubK++uorC54VZAoODlZAQIA2bNhglm3YsEEdOnRQ+fLlHd6XNmzYoLCwsFvuD+fPn1f37t3l4+OjIkWKqHLlylm+RD927JjCwsLk7u6u2rVra+vWrWadVZ+3J06c0NChQ833xkyff/65qlevLldXV5UrV06TJ092GGu5cuX01ltv6bnnnpOnp6fKlCmjDz74IFfP972EsIUs5s+fLxcXF23evFnvvfeeJMnJyUnvvvuuDh48qPnz52vdunV65ZVXHNa7fPmyJk2apI8++kgbN27UyZMn9fLLL2fbx7p169SyZUu9+eabGjlyZI7G5uHhoe3bt2vChAkaO3asYmNjcz9Z5ErRokVVtGhRLVu2TCkpKTds9/rrrysyMlJ79+5V9+7d1aVLFx06dMihzf/93/9p8ODBOnTokBnur3f06FEtW7ZMy5cv1/Lly/Xdd99p/PjxZv2wYcO0efNmffXVV4qNjdWmTZu0Z88es37p0qXq16+fQkNDdfr0aS1dujTbfp5++mmdPXtW33zzjXbv3q169eqpefPmOnfuXLbtJ0+erHnz5mnOnDn6/vvvde7cOX3xxRc3fD5wd5w7d06rVq1SVFSUPDw8stTn5BqFV199VZMnT9auXbtUuHBhPffcc5L+Ov1m+PDhql69uk6fPq3Tp0+rc+fOysjIUIcOHXTu3Dl99913io2N1bFjx9S5c2eH7d5qn0bBkpSUpMcff1x169bVrl27tGrVKp05c0adOnVyaDd//nwVLlxYO3bs0LRp0zRlyhT9+9//NuvT0tL0xhtvaO/evVq2bJmOHz+u3r17Z+nvRvulJH3xxRcaPHiwhg8frgMHDuj555/Xs88+q/Xr1ztsY8yYMerUqZP27dunNm3aqHv37jd8r0PeCAsLc3gd1q9fr2bNmqlp06Zm+Z9//qnt27crLCzslvvD66+/rh9//FHffPONDh06pNmzZ6tUqVIOfb766qt6+eWXFRcXpypVqqhr164OpzJe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bTP/Pn/+vH766SdVq1bNLAsKCtILL7ygpUuXavjw4frXv/6V4/ECAO5dXLMFALjvzJo1S40bN1aDBg0UExOjWrVqycnJSTt37tThw4dVv379G647YsQIderUSXXr1lWLFi309ddfa+nSpVqzZo0kqUiRImrUqJHGjx+v8uXL6+zZs3rttdey3dbYsWNVsmRJ+fn56dVXX1WpUqXUsWNHSdKQIUPUunVrValSRefPn9f69esdghgAoOAjbAEA7jsVK1bUDz/8oLfeekvR0dH69ddf5erqqpCQEL388svmzS+y07FjR02bNk2TJk3S4MGDVb58ec2dO1fNmjUz28yZM0d9+vRR/fr1FRwcrAkTJig8PDzLtsaPH6/Bgwfr559/Vp06dfT111/LxcVFkpSenq6oqCj9+uuv8vLyUqtWrTR16tQ8fy4AAPnHZlx70jkAALhjGzZsUFhYmM6fP69ixYrl93AAAPmEa7YAAAAAwAKELQAAAACwAKcRAgAAAIAFOLIFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFjg/wH2XN6iiTwZjQAAAABJRU5ErkJggg=="/>


```python
# 데이터 세트에서 결혼 유무에 대한 시각화

# your code here 
plt.figure(figsize=(10, 5))
sns.countplot(x=df_train['Relationship_Status'], palette = 'pastel')
plt.title('Comparison of various Groups', fontweight = 30)
plt.xlabel('Groups')
plt.ylabel('Count')
```

<pre>
Text(0, 0.5, 'Count')
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1sAAAHWCAYAAACBjZMqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA8lklEQVR4nO3deVyVZf7/8fdB4IDgARdEGXFPFLcSS8nKLBQNt9LUstxtNDSX0oap3Fosc8lcy3KZGf2lNuaUpOaS2STl0uCWWhoukwFOCqgJKNy/P3pwvh7BFOTqCL6ej8d5PM65ruu+7891DgVv7/u+js2yLEsAAAAAgGLl4e4CAAAAAKA0ImwBAAAAgAGELQAAAAAwgLAFAAAAAAYQtgAAAADAAMIWAAAAABhA2AIAAAAAAwhbAAAAAGAAYQsAAAAADCBsAQBumM1m04QJE9xdxg37+9//rvr168vLy0uBgYHuLkeS1K9fP9WsWdPdZQAAioCwBQDF4MiRI/rzn/+s2rVry8fHRw6HQ61atdLMmTN14cIFd5eH63Dw4EH169dPderU0YIFC/Tuu++6u6QSIykpScOGDVO9evVUtmxZlS1bVuHh4YqNjdWePXvcXR4AuI3NsizL3UUAQEkWHx+vRx99VHa7XX369FGjRo2UnZ2tf//73/rnP/+pfv36lfo/3DMzM+Xp6SlPT093l1Jk8+fP19ChQ/XDDz+obt267i7H6eLFi8rNzZXdbnd3KQVas2aNevbsKU9PT/Xu3VtNmzaVh4eHDh48qFWrVunYsWNKSkpSjRo13F0qAPzhSu5vRQC4CSQlJalXr16qUaOGNm/erKpVqzr7YmNjdfjwYcXHx7uxQnNyc3OVnZ0tHx8f+fj4uLucG5aamipJN83lg+fPn5efn5+8vLzcXcpVHTlyxPnzv2nTJpeff0l64403NHfuXHl4/P6FNHlzBYDShssIAeAGTJkyRefOndP777+f7w9NSapbt65GjBjhfH3p0iW9/PLLqlOnjux2u2rWrKm//vWvysrKctmuZs2a6tixo7Zs2aLmzZvL19dXjRs31pYtWyRJq1atUuPGjeXj46OIiAj95z//cdm+X79+8vf3148//qjo6Gj5+fkpJCREkyZN0pUXNEydOlV33323KlasKF9fX0VEROjDDz/MNxebzaZhw4Zp6dKlatiwoex2u9atW+fsu/yerbNnz2rkyJGqWbOm7Ha7KleurLZt2+rbb7912efKlSsVEREhX19fVapUSU888YR++umnAufy008/qWvXrvL391dQUJCee+455eTkXOWTcTV37lxnzSEhIYqNjVVaWprL+z1+/HhJUlBQ0O/egzZ16lTZbDYdO3YsX19cXJy8vb115swZSdKXX36pRx99VNWrV5fdbldoaKhGjRqV79LSvDkeOXJEDz30kMqVK6fevXs7+668Z+v8+fN69tlnFRoaKrvdrrCwME2dOtXlsz169KhsNpsWL16cr86ifl5XmjJlis6fP69FixYV+PPv6empZ555RqGhodc11+Ke14QJE2Sz2XTw4EH16NFDDodDFStW1IgRI5SZmemy7YYNG3TPPfcoMDBQ/v7+CgsL01//+tffnT8AXAthCwBuwCeffKLatWvr7rvvvq7xgwYN0rhx49SsWTPNmDFDrVu31uTJk9WrV698Yw8fPqzHH39cnTp10uTJk3XmzBl16tRJS5cu1ahRo/TEE09o4sSJOnLkiHr06KHc3FyX7XNyctS+fXsFBwdrypQpioiI0Pjx452hIs/MmTN1xx13aNKkSXrttdfk6empRx99tMAzcps3b9aoUaPUs2dPzZw586oLNwwZMkTz5s1Tt27dNHfuXD333HPy9fXVgQMHnGMWL16sHj16qEyZMpo8ebIGDx6sVatW6Z577nEJQnlziY6OVsWKFTV16lS1bt1a06ZNu67LMydMmKDY2FiFhIRo2rRp6tatm9555x21a9dOFy9elCS99dZbevjhhyVJ8+bN09///nc98sgjBe6vR48estlsWrFiRb6+FStWqF27dipfvryk38Lkr7/+qqFDh2rWrFmKjo7WrFmz1KdPn3zbXrp0SdHR0apcubKmTp2qbt26FXh8y7LUuXNnzZgxQ+3bt9f06dMVFhamMWPGaPTo0dd8PwpyPZ9XQdasWaO6deuqRYsWhTpeQXM1Ma88PXr0UGZmpiZPnqyHHnpIb7/9tp566iln//79+9WxY0dlZWVp0qRJmjZtmjp37qyvvvrqho4LALIAAEWSnp5uSbK6dOlyXeMTExMtSdagQYNc2p977jlLkrV582ZnW40aNSxJ1rZt25xt69evtyRZvr6+1rFjx5zt77zzjiXJ+vzzz51tffv2tSRZw4cPd7bl5uZaMTEx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7B07duipp54ysm8AwI07deqUhg4dqurVq8tut6tKlSqKjo7WV199JUn6+eef1aFDBzdXCZQ+hC2gFAkNDdUHH3ygCxcuONsyMzO1bNkyVa9e/Yb2ffHixXxt2dnZkqSgoCCVLVv2hvYPADCnW7du+s9//qMlS5bo+++/18cff6z7779fv/zyiySpSpUqLOUOGEDYAkqRZs2aKTQ0VKtWrXK2rVq1StWrV9cdd9zhbFu3bp3uueceBQYGqmLFiurYsaOOHDni7D969KhsNpuWL1+u1q1by8fHR0uXLlW/fv3UtWtXvfrqqwoJCVFYWJik/JcRpqWladCgQQoKCpLD4dADDzyg3bt3u9T6+uuvKzg4WOXKldPAgQOVmZlp6F0BgFtbWlqavvzyS73xxhtq06aNatSoobvuuktxcXHq3LmzJNfLCPN+B6xatUpt2rRR2bJl1bRpUyUkJLjsd8GCBQoNDVXZsmX18MMPa/r06QoMDPzdWt577z01aNBAPj4+ql+/vubOnWtiysBNg7AFlDIDBgzQokWLnK8XLlyo/v37u4w5f/68Ro8erZ07d2rTpk3y8PDQww8/rNzcXJdxf/nLXzRixAgdOHBA0dHRkqRNmzbp0KFD2rBhg9asWVNgDY8++qhSU1O1du1a7dq1S82aNdODDz6o06dPS5JWrFihCRMm6LXXXtPOnTtVtWpVfuECgCH+/v7y9/fX6tWrlZWVdd3bvfDCC3ruueeUmJioevXq6bHHHtOlS5ckSV999ZWGDBmiESNGKDExUW3bttWrr776u/tbunSpxo0bp1dffVUHDhzQa6+9ppdeeklLliy5ofkBNzULQKnQt29fq0uXLlZqaqplt9uto0ePWkePHrV8fHysU6dOWV26dLH69u1b4LanTp2yJFl79+61LMuykpKSLEnWW2+9le8YwcHBVlZWlkt7jRo1rBkzZliWZVlffvml5XA4rMzMTJcxderUsd555x3LsiwrMjLSevrpp136W7RoYTVt2rSIswcA/J4PP/zQKl++vOXj42PdfffdVlxcnLV7925nvyTro48+sizr/34HvPfee87+/fv3W5KsAwcOWJZlWT179rRiYmJcjtG7d28rICDA+Xr8+PEu/1+vU6eOtWzZMpdtXn75ZSsyMrKYZgncfDizBZQyQUFBiomJ0eLFi7Vo0SLFxMSoUqVK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"/>


```python
# 나이가 직원의 성과에 미치는 영향을 확인

# your code here 
dft_rs_ar = df_train[['Relationship_Status','Attrition_rate']].groupby(['Relationship_Status']).agg('median')
dft_rs_ar
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
      <th>Attrition_rate</th>
    </tr>
    <tr>
      <th>Relationship_Status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Married</th>
      <td>0.14155</td>
    </tr>
    <tr>
      <th>Single</th>
      <td>0.14470</td>
    </tr>
  </tbody>
</table>
</div>



```python
# your code here 
dft_rs_ar.T.plot(kind='bar', figsize=(10, 5) )
plt.ylabel('Rates')
plt.title('Relationship Status', fontweight = 30)
plt.legend(loc="upper right") # 범례표..
plt.xticks(rotation=0); # 세로가 아닌 가로로 표시
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1cAAAHDCAYAAADIquCMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABC4UlEQVR4nO3de3zP9f//8ft7m21sNqfZHMaQY8PYWM76tJpaZUhL9cVIKYRFjTJ0Wgc0RQ7lVPEhQkJqLVRMYg4fQj4+bMI2h2xsGbbX7w8/73pnG5uXvbHb9XJ5Xz57P9/P1/P1eL73Ke49X6/ny2IYhiEAAAAAwHVxsHcBAAAAAHA7IFwBAAAAgAkIVwAAAABgAsIVAAAAAJiAcAUAAAAAJiBcAQAAAIAJCFcAAAAAYALCFQAAAACYgHAFAAAAACYgXAEAimzdunWyWCxat26dqeNaLBaNGzfO1DGL6tChQ7JYLJowYcJV+44bN04Wi6UEqgIA3AoIVwBwm5s7d64sFov15eTkpBo1aqhv3746cuRIidezevVquweom9nx48c1dOhQNWrUSGXLllXVqlXVunVrvfTSSzp79qy134IFCxQXF1fs82RnZ2vcuHGmB2QAKM2c7F0AAKBkvPrqq6pTp47OnTunTZs2ae7cufrpp5+0a9cuubq6llgdq1ev1tSpU/MNWH/++aecnG6dP5peeeUVRUdHmzbeqVOnFBQUpMzMTPXr10+NGjXSyZMntXPnTk2bNk3PPvus3N3dJV0KV7t27dKwYcOKda7s7GyNHz9ektS5c2eTZgAApdut8ycYAOC63H///QoKCpIkPfXUU6pSpYrefvttrVixQo8++qidq7ukJEOeGZycnEwNg7NmzVJKSoo2bNigtm3b2nyWmZkpZ2dn084FADAflwUCQCnVoUMHSdKBAwds2vfu3atHHnlElSpVkqurq4KCgrRixYqrjvfjjz+qZ8+eqlWrllxcXOTr66vhw4frzz//tPbp27evpk6dKkk2lypelt89V9u2bdP9998vDw8Pubu765577tGmTZts+ly+9HHDhg2KioqSl5eX3Nzc1K1bNx0/ftym75YtWxQaGqoqVaqobNmyqlOnjvr165fvnGbOnKl69erJxcVFrVq10i+//GLzeX73XFksFg0ePFjz589Xw4YN5erqqsDAQP3www9X/Q4PHDggR0dH3XXXXVd85uHhYQ2fnTt31qpVq5ScnGz9Dv38/CRJ58+fV0xMjAIDA+Xp6Sk3Nzd16NBBa9eutY516NAheXl5SZLGjx9vHePyd9+5c+d8V7P69u1rPc9lCxcuVGBgoMqXLy8PDw81bdpUkydPvupcAeB2xMoVAJRShw4dkiRVrFjR2rZ79261a9dONWrUUHR0tNzc3PT5558rPDxcX3zxhbp161bgeIsXL1Z2draeffZZVa5cWZs3b9YHH3yg33//XYsXL5YkPfPMMzp69Kji4+P16aefXrXG3bt3q0OHDvLw8NCLL76oMmXKaMaMGercubPWr1+v4OBgm/5DhgxRxYoVNXbsWB06dEhxcXEaPHiwFi1aJElKT0/XfffdJy8vL0VHR6tChQo6dOiQli5desW5FyxYoDNnzuiZZ56RxWLRO++8o+7du+t///ufypQpU2jd69ev16JFi/T888/LxcVFH374obp06aLNmzfL39+/wONq166t3Nxcffrpp+rTp0+B/V5++WVlZGTo999/13vvvSdJ1ssFMzMz9fHHH6tXr14aMGCAzpw5o1mzZik0NFSbN29WQECAvLy8rJcZduvWTd27d5ckNWvWrNB5/VN8fLx69eqle+65R2+//bYkac+ePdqwYYOGDh1apLEA4LZgAABua3PmzDEkGd99951x/Phx4/Dhw8aSJUsMLy8vw8XFxTh8+LC17z333GM0bdrUOHfunLUtLy/PaNu2rVG/fn1r29q1aw1Jxtq1a61t2dnZV5w7NjbWsFgsRnJysrVt0KBBRkF//Egyxo4da30fHh5uODs7GwcOHLC2HT161ChfvrzRsWPHK+YYEhJi5OXlWduHDx9uODo6GqdPnzYMwzCWLVtmSDJ++eWXAr+vgwcPGpKMypUrG6dOnbK2f/nll4Yk46uvvrK2jR079oq5SDIkGVu2bLG2JScnG66urka3bt0KPK9hGEZqaqrh5eVlSDIaNWpkDBw40FiwYIG1/r8LCwszateufUX7xYsXjZycHJu2P/74w/D29jb69etnbTt+/PgV3/dlnTp1Mjp16nRFe58+fWzOOXToUMPDw8O4ePFiofMCgNKCywIBoJQICQmRl5eXfH199cgjj8jNzU0rVqxQzZo1JV3aTOH777/Xo48+qjNnzujEiRM6ceKETp48qdDQUO3fv7/Q3QXL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"/>

## Task2: describe 함수를 사용하여  training data set 에 대한 정보 가져오기


### Data 설명



데이터가 어떻게 분배되어 있는지 확인해 봅시다. 각 열의 평균값, 최대값, 최소값을 다른 특성들과 함께 시각화할 수 있습니다.



```python
# your code here 
df_train.describe()
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
      <th>Age</th>
      <th>Education_Level</th>
      <th>Time_of_service</th>
      <th>Time_since_promotion</th>
      <th>growth_rate</th>
      <th>Travel_Rate</th>
      <th>Post_Level</th>
      <th>Pay_Scale</th>
      <th>Work_Life_balance</th>
      <th>VAR1</th>
      <th>VAR2</th>
      <th>VAR3</th>
      <th>VAR4</th>
      <th>VAR5</th>
      <th>VAR6</th>
      <th>VAR7</th>
      <th>Attrition_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6588.000000</td>
      <td>7000.000000</td>
      <td>6856.000000</td>
      <td>7000.000000</td>
      <td>7000.000000</td>
      <td>7000.000000</td>
      <td>7000.000000</td>
      <td>6991.000000</td>
      <td>6989.000000</td>
      <td>7000.000000</td>
      <td>6423.000000</td>
      <td>7000.000000</td>
      <td>6344.000000</td>
      <td>7000.000000</td>
      <td>7000.000000</td>
      <td>7000.000000</td>
      <td>7000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.622799</td>
      <td>3.187857</td>
      <td>13.385064</td>
      <td>2.367143</td>
      <td>47.064286</td>
      <td>0.817857</td>
      <td>2.798000</td>
      <td>6.006294</td>
      <td>2.387895</td>
      <td>3.098571</td>
      <td>-0.008126</td>
      <td>-0.013606</td>
      <td>1.891078</td>
      <td>2.834143</td>
      <td>7.101286</td>
      <td>3.257000</td>
      <td>0.189376</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.606920</td>
      <td>1.065102</td>
      <td>10.364188</td>
      <td>1.149395</td>
      <td>15.761406</td>
      <td>0.648205</td>
      <td>1.163721</td>
      <td>2.058435</td>
      <td>1.122786</td>
      <td>0.836377</td>
      <td>0.989850</td>
      <td>0.986933</td>
      <td>0.529403</td>
      <td>0.938945</td>
      <td>1.164262</td>
      <td>0.925319</td>
      <td>0.185753</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-1.817600</td>
      <td>-2.776200</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>33.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>-0.961200</td>
      <td>-0.453700</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>0.070400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>3.000000</td>
      <td>10.000000</td>
      <td>2.000000</td>
      <td>47.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>-0.104800</td>
      <td>-0.453700</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>0.142650</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>52.000000</td>
      <td>4.000000</td>
      <td>21.000000</td>
      <td>3.000000</td>
      <td>61.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.751600</td>
      <td>0.707500</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>0.235000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>65.000000</td>
      <td>5.000000</td>
      <td>43.000000</td>
      <td>4.000000</td>
      <td>74.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1.608100</td>
      <td>1.868800</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>0.995900</td>
    </tr>
  </tbody>
</table>
</div>



```python
# training set에 누락된 값이 있는지 확인합니다.
# your code here 
df_train.isna().any()
```

<pre>
Employee_ID                  False
Gender                       False
Age                           True
Education_Level              False
Relationship_Status          False
Hometown                     False
Unit                         False
Decision_skill_possess       False
Time_of_service               True
Time_since_promotion         False
growth_rate                  False
Travel_Rate                  False
Post_Level                   False
Pay_Scale                     True
Compensation_and_Benefits    False
Work_Life_balance             True
VAR1                         False
VAR2                          True
VAR3                         False
VAR4                          True
VAR5                         False
VAR6                         False
VAR7                         False
Attrition_rate               False
dtype: bool
</pre>
### Data 시각화



이제, 상관 행렬을 이용하여 각 데이터 feature가 얼마나 관련되어 있는지 알아보겠습니다.



```python
plt.figure(figsize=(18,10))
cor = df_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Accent)
plt.show()
plt.savefig("main_correlation.png")
```

<pre>
C:\Users\User\AppData\Local\Temp\ipykernel_8388\2391686863.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
  cor = df_train.corr()
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABaEAAAPACAYAAAAsRrWWAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzdd1xT5/4H8E9CSBhhhxHZW2SjqDhxY1u9trZa6+2tdlzrva3dtbt299plp+1ttXp/tbW9tdpha+u24kBFUIaCIEMEBCHsBDJ+f+QSjYACJiS2n/frlZfm5IzneXjOc875nuc8R6DT6XQgIiIiIiIiIiIiIjIDoaUTQERERERERERERER/XAxCExEREREREREREZHZMAhNRERERERERERERGbDIDQRERERERERERERmQ2D0ERERERERERERERkNgxCExEREREREREREZHZMAhNRERERERERERERGbDIDQRERERERERERERmQ2D0ERERERERERERERkNgxCExEREREREREREZHZMAhNRERERERERERE9CewZ88ezJgxA4MGDYJAIMCmTZuuuMyuXbuQlJQEiUSCsLAwrFmzps/bZRCaiIiIiIiIiIiI6E+gpaUF8fHx+PDDD3s1/+nTp3H99ddjwoQJyMrKwoMPPoi7774bv/76a5+2K9DpdLr+JJiIiIiIiIiIiIiIrk0CgQAbN27ErFmzepxn6dKl2Lx5M3JycgzTbr31VigUCmzZsqXX22JPaCIiIiIiIiIiIqJrkEqlQmNjo9FHpVKZbP379+/H5MmTjaZNmzYN+/fv79N6RCZLEdEAW7RokaWTYBXkd8ktnQSLq1xVaekkWIVb05ZYOglWYbfvfy2dBCIiIiIiomvWsuHLLJ0Eq2StcSi5XI4XXnjBaNrzzz+PZcuWmWT9VVVV8Pb2Nprm7e2NxsZGtLW1wd7evlfrYRCaiIiIiIiIiIiI6Br05JNP4uGHHzaaJpFILJSanjEITURERERERERERHQNkkgkZg06+/j4oLq62mhadXU1nJ2de90LGuCY0ERERERERERERETUjZSUFGzfvt1o2tatW5GSktKn9TAITURERERERERERPQn0NzcjKysLGRlZQEATp8+jaysLJSVlQHQD+/xt7/9zTD/vffei+LiYjz++OM4ceIEPvroI3zzzTd46KGH+rRdBqGJ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iC5HoPuD9tNfsGBBt4GwyMhInDhxAgUFBbjrrruQkZGBoKAgvPfee0hLS8PGjRsxa9YsAEB6ejoWL16MwsJCJCcnY8mSJbjllltw+vRpQ6/g3bt346mnnsKRI0dgb2+PESNGYP369XB1dcW2bduwZMkSFBcXIzIyEu+99x5SU1ONtvHZZ5/hxRdfREVFBcaOHYtdu3ZhwYIFUCgU2LRpEwBAqVTi8ccfx1dffYWmpiYMGzYM77zzDpKT9XcId+3ahQkTJqC+vh6urq4A9L22ExMTjdLakzVr1mDhwoVdpoeGhuLUqVNoamrC008/jQ0bNqCmpgY+Pj4YN24cXnvtNfj7X3i8ae7cufjmm2+wevXqLusrLCzE0qVLsXPnTqhUKgQGBiItLQ1vv/02BAIBUlNTkZCQgBUrVlz+D3uRZRnLej3vH1nlqkpLJ8Hi5Hf98V+2QURERERERDQQlg1fZukkWKVFixZZOgnd+uSTTyydhF75wwah6Y+PQWg9BqEZhCYiIiIiIiIyFQahu8cg9NX5Q7+YkIiIiIiIiIiIiIgsi0HoP4Ho6GhIpdJuP+vWrbN08oiIiIiIiIiIiOgP7A/9YkLS+/nnn9HR0dHtb97eA/NWeSIiIiIiIiIiIvpzYhD6TyAwMNDSSSAiIiIiIiIiIqI/KQ7HQURERERERERERERmwyA0EREREREREREREZkNg9BEREREREREREREZDYMQhMRERERERERERGR2TAITURERERERERERERmwyA0EREREREREREREZkNg9BEREREREREREREZDYMQhMRERERERERERGR2TAITURERERERERERERmwyA0EREREREREREREZkNg9BEREREREREREREZDYMQhMRERERERERERGR2YgsnQCi/qpcVWnpJFgF+V1ySyfB4lgX9G5NW2LpJFiF3b7/tXQSiIiIiIiIiOgi7AlNRERERERERERERGbDIDQRERERERERERERmQ2D0ERERERERERERERkNgxCExEREREREREREZHZMAhNRERERERERERERGYjsnQCiIiIiIiIiIiIiKzZvJbvLZ2EHnxi6QT0CntCExEREREREREREZHZMAhNRERERERERERERGbDIDQRERERERERERERmQ2D0ERERERERERERERkNgxCExEREREREREREZHZMAhNRERERERE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SehW6hdVV57JCnBMaCIiIiIiIiIiIiIyGwahzaisrAxSqbTHT1lZWZ/Xee+99/a4vs7hLoiIiIiIiIiIiIisBceENqNBgwYhKyvrsr/31YsvvthljOFOlhpPmYiIiIiIiIiIiKgnDEKbkUgkQliYaV8Y5OXlBS8vL5Ouk4iIiIiIiIiIiMhcOBwHEREREREREREREZkNg9BEREREREREREREZDYMQhMRERERERERERGR2TAITURERERERERERERmwyA0EREREREREREREZkNg9BEREREREREREREZDYMQhMRERERERERERGR2TAITURERERERERERERmwyA0EREREREREREREZkNg9BEREREREREREREZDYMQhMRERERERERERGR2TAITURERERERERERERmwyA0EREREREREREREZmNyNIJIOqvW9OWWDoJVmE3/mvpJFgc64Le+i3vWToJVkF+l9zSSSAiIiIiIiKii7AnNBERERERERERERGZDYPQRERERERERERERGQ2DEITERERERERERERkdkwCE1EREREREREREREZsMgNBERERERERERERGZDYPQRERERERERERERGQ2DEITERERERERERERkdkwCE1EREREREREREREZsMgNBERERERERERERGZDYPQRERERERERERERH8SH374IYKCgmBnZ4cRI0YgIyPjsvOvWLECkZGRsLe3h7+/Px566CEolco+bZNBaCIiIiIiIiIiIqI/ga+//hoPP/wwnn/+eWRmZiI+Ph7Tpk3DuXPnup3/yy+/xBNPPIHnn38e+fn5WLVqFb7++ms89dRTfdoug9BEREREREREREREfwJvv/027rnnHixcuBBDhgzBxx9/DAcHB6xevbrb+fft24fRo0fjtttuQ1BQEKZOnYp58+Zdsff0pRiEJiIiIiIiIiIiIroGqVQqNDY2Gn1UKlW387a3t+PIkSOYPHmyYZpQKMTkyZOxf//+bpcZNWoUjhw5Ygg6FxcX4+eff8Z1113Xp3QyCE1ERERERERERER0DXrttdfg4uJi9Hnttde6nbe2thYajQbe3t5G0729vVFVVdXtMrfddhtefPFFjBkzBra2tggNDUVqaiqH4yAiIiIiIiIiIiL6M3jyySfR0NBg9HnyySdNtv5du3bh1VdfxUcffYTMzEx899132Lx5M156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7Qj+r9hJIDvBsmCielAldhLsQuk/S8VOguhYFoiIiIiIiMiecCQ0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBip2AkgIiIiIiIiIiIismf9vpsjdhLa9v/EToB1OBKaiIiIiIiIiIiIiATDTmgiIiIiIiIiIiIiEgw7oYmIiIiIiIiIiIhIMOyEJiIiIiIiIiIiIiLBsBP6JiGRSLB27Vqxk0FEREREREREREQ3GanYCSDbWrBgAdauXYusrCyxkwIACA8Px9NPP42nn35a7KSgYFMBjv5yFI01jfDu4Y3ke5PhG+Xb7vIl+0qQszoH9RX18AjwQN8ZfRHUN8j8udFoxOE1h1G4rRDNDc1QxiqRMjsFHoEe5mXy1uXhXNY5VJVUwUHqgOlfTrfYhrZOiz1/24Oa0zXQqrVwVjgjuH8w+tzVB04uTrYPQhvEiMvOD3aiqqQKmloNZK4yBCYEos+MPnD1dhU0r51JY1s6ipW+SY9D3x5C8b5iGJoNCEwMxIDZA+Di6QLAuv1dnl+OrUu3ttr2lE+mwMXLxYZR6Hz+rtRRWTh94DRObD2ByqJKNKmbkL4kHd5h3m2uy2g0Ysd7O1CaU4oRT41AyIAQm+dPaDExMRg/fjxCQ0Ph5eWFzz//HNnZ2WInyypi1AO1pbXI+j4LFwouwKAzwCvUC0nTkhDQO8C8zHf3fNdq20MfH4qwIWE2yrklseqGFid3nsTRjUdRV1YHJ2cnhA4MxYDZA8yfl+wrQd5Peagrq4PcQ47YcbGIuzXOtkGA+HEATPXlhlc2oLGqEdO+mAaZmwwA0FjdiEPfHkLlqUrUldchdnwskv+cbNsAtEGomJzYegLFvxejsqgSOo3OIq8trDlWhGCP502Xa6+MCE2MttKactLdGAcTW8bBoDMgZ1UOzmWfg/q8GjJXGQLiA1qdI3fmOOkuYtaRPz3zE+or6i3e63NXH/Se3Nt2GWyDmG2lNedHBZsKcHzzcdRfqIerryvib49HxPAIG+T86m7G9rKzebkSr6noZsKR0CJqamoSOwld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ObH1hGle7//ZsmQLAGDQw4MQOTLSqmNFCPZ63iQ2sdrKjspJd2McTGwdh4aqBpzNPAvAsp4DLI8BezxOxKojHZwcULK3BLk/5sLQbICbnxt6pve0mCf6estzR22lNedHRoMRRzccRV1pHRwcHeAf549xr4/rlkFON1t7eSVeUxFdncTYXfcm3ITq6+sRHByM999/Hw8++KDYybnhLNi/QOwkEJEdKv1nqdhJEJ3qQZXYSSAiIiIiouvUgoELxE6CXapxXCB2EtrkqV8gdhKswpHQNnTo0CEcPXoUAwcORE1NDRYtWgQAuP3220VOGREREREREREREZE4+GBCG3vvvffQp08fjB07FvX19di1axeUSvt+6MacOXPg7u7e5mvOnDliJ4+IiIiIiIiIiIiuYxwJbUP9+vXDH3/8IXYyOm3RokV47rnn2vxMoVB0c2qIiIiIiIiIiIjoRsJOaIK/vz/8/f3FTgYRERERERERERHdgDgdBxEREREREREREREJhp3QRERERERERERERCQYdkITERERERERERERkWDYCU1EREREREREREREgmEnNBEREREREREREREJhp3QRERERERERERERCQYdkITERERERERERERkWDYCU1EREREREREREREgmEnNBEREREREREREREJhp3QRERERERERERERCQYdkITERERERERERERkWDYCU1EREREREREREREgmEnNBEREREREREREREJRip2AoiIiGxJ9aBK7CSIrvSfpWInwS6wLBAREREREdkHjoQmIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IQmIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IQmIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IQmIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IS+iRQVFUEikSArK0vspBAREREREREREdFNQip2Am5EEonkqp+/8cYbWLBgQfckpouKiooQERFh/tvb2xuJiYlYsmQJRowYYfV6tm/fjrS0NFRV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H0LGo4VYx4SY7LWdsAdC1Qv9Z/WHRCLB7o93Q9+shypJhQH3WT5L51z2OeT9lAdDswFeoV4Y8cwIQQcwXE6sfFtzbmA0GHF0w1HUldbBwdEB/nH+GPf6OJtfY4gVA0eZIwq3FSJzZSYMzQa4+roiZECIxdy+3RWDttxsbWULMeuCIXOGIPP/ZWLH+zsgcZDAv5c/Up9PNXcwSxwlOPLzEdSV1QFGwFXpipixMeiV3kvgqIhXHqRyKU4fOI3Daw5Dp9XBxdMFqiQV4ufG845KsgsSo1D3vRMJbMH+BWIngYjILpX+s1TsJNgF1YOqjhciIiIiIiILCwYuEDsJdqnGcYHYSWiTp36B2EmwCueEJiIiIiIiIiIiIiLBsBP6JjVnzhy4u7u3+ZozZ47YySMiIiIiIiIiIqIbBOeEvkktWrQIzz33XJufKRSKbk4NERERERERERER3ajYCX2T8vf3h7+/v9jJICIiIiIiIiIiohscp+MgIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IQmIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IQmIiIiIiIiIiIiIsGwE5qIiIiIiIiIiIiIBMNOaCIiIiIiIiIiIiISDDuhiYiIiIiIiIiIiEgw7IQmIiIiIiIiIiIiIsFIxU4AERER2ZbqQZXYSbALpf8sFTsJdoHlgYiIiIiIxMaR0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzTZBYlEgrVr14qdDCIiIiIiIiIiIrIxqdgJuFnNnj0bK1asAAA4OTkhNDQU9957L15++WVIpV3fLdu3b0daWhqqqqrg5eVldVqqq6tv+E7ggk0FOPrLUTTWNMK7hzeS702G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Q1lZGZqamiCTWfecCnZC3+CkUimio23/QDMiIiIiIiIiIiK6fshkMiQnJ2PLli3mOaENBgO2bNmCuXPntvmdYcOG4dtvv4XBYICDg2lm54KCAqhUKqs7oAHOCU1ERERERERERER0U3j22Wfx1VdfYcWKFcjPz8djjz2G+vp63H///QCAe++9Fy+99JJ5+cceewyVlZV46qmnUFBQgPXr12Pp0qV44oknOrVdjoQmIiIiIiIiIiIiugnMmDEDFy5cwOuvv46ysjL07dsXGzduND+ssKSkxDziGQB69OiBX3/9Fc888wySkpIQHByMp556Ci+++GKntstOaCIiIiIiIiIiIqKbxNy5c9udfmP79u2t3hsyZAj27t17TdvkdBxEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYKRiJ4CIiIhICKoHVWInwS6U/rNU7CSIjmWBiIiIiEhcHAlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCExEREREREREREZFg2AlNRERERERERERERIJhJzQRERERERERERERCYad0EREREREREREREQkGHZCk+CKioogkUiQlZUldlKIiIiIiIiIiIiom0nFTgCZzJ49GytWrAAAODk5ITQ0FPfeey9efvllSKXC7qYdO3Zg4cKFyMrKgkajQXBwMIYOHYqvvvoKMplM0G2LqWBTAY7+chSNNY3w7uGN5HuT4Rvl2+7yJftKkLM6B/UV9fAI8EDfGX0R1DfI/PnpA6dxYusJVBZVokndhPQl6fAO87ZYx/6v96M8rxyNVY2QOkuhjFGi74y+UAQpBMvn5WydZ6PRiMNrDqNwWyGaG5qhjFUiZXYKPAI9LNZzNuss8n7MQ/Xp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vJKLy/X3kCMiIiIiIiIiIiK6vrAT+gbn4+MDHx8fsZNBRERERERERERENylOx0FEREREREREREREgmEnNBEREREREREREREJhp3QRERERERERERERCQYdkITERERERERERERkWDYCU1EREREREREREREgmEnNBEREREREREREREJhp3QRERERERERERERCQYdkITERERERERERERkWDYCU1EREREREREREREgmEnNBEREREREREREREJhp3QRERERERERERERCQYdkITERERERERERERkWDYCU1EREREREREREREgpGKnQAiIiIiEo7qQZXYSRBd6T9LxU6CXWBZICIiIiKxcCQ0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCuak7oRcsWIC+ffsKsu7Zs2djypQp5r+NRiMeeeQR+Pj4QCKRICsr65rWv337dkgkElRXV1/TelJTU/H0009f0zqIiIiIiIiIiIiI2iMVOwFX+uKLL/D888+jqqoKUqkpeWq1Gt7e3hg2bBi2b99uXnb79u1IS0vDiRMnEBUV1a3pbNl2VVUVvLy8Wn3+0UcfwWg0mv/euHEjvvnmG2zfvh2RkZFQKpXdmFoCgIJNBTj6y1E01jTCu4c3ku9Nhm+Ub7vLl+wrQc7qHNRX1MMjwAN9Z/RFUN8g8+dGoxGH1xxG4bZCNDc0QxmrRMrsFHgEepiX2fnBTlSVVEFTq4HMVYbAhED0mdEHrt6uAAB9kx4Hlh9AZVElas/VIqhvEEY+M1K4ILTBmny0paN4nth6AsW/F6OyqBI6jQ7TvpgGmZvMYh0/PfMT6ivqLd7rc1cf9J7c23YZbINQedY36XHo20Mo3lcMQ7MBgYmBGDB7AFw8XQAAVcVVOPLzWCNUzAABAABJREFUEVQUVEBbp4WbnxuiR0ej5y09Lbaj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b4JngCeGZg+1mafLGV0vX4BlISldW8+F4fXKFwDO5py1zG/8vZ0v7QRgmfszJjNmEHJmX1qFqAsyhQzVR6tRsKUABp0B7t7uCEkLQfKjyXBx7SnXc/vPIe/DPJjNZqjj1Zj23DT4x/a82nWrl4M9zu07B2OXEaf+cwqn/nPK+nlgYqBgAfkrCVEuNbk1OPzPw9Z/H3zzIAAgZVEKUv/H8kqcmOeHGHVB4aVA1pNZOPnxSeSsy4HJYIJ3uDcmr5g8oNdvhSZUe5hxVwYkEgn2/3k/jHojQtJCMOreUb2OX76nHOGjwgftokGsdmHEj0dAKpXiu7e+sywoFOuP6c9Ot+bbRe6Csl1lyH0/Fya9CR7+HggfFS7o/L9i/u2PvH0El05fsv57+6rtAIAFry+wBt2LPitCR30HpC5SqEJUmPDoBESMiXBoGYhRH2oLa9Fe1472unZsfXyrTXru/L87AQASFwlOfXEKbbVtgBnwUHsgfkY8ErOde3wFAJGRkXjiiSes/16yZAkA4ODBg3j33XfFStaAidFXNlY0oqGsAQDwxa+/sEnPlefGYBKjHKSuUlw+cxlnvj4DfYcebt5uCBgagJm/nXnT0/qIlWd72tuCTwpwbv8567+728Vpz01D0LAgqEJVyFyRicLPCrHjxR2QSCTwjfw+bQI8Beys42hgcPqH/jjzddZgE2Ic4YztINFAScyOfj+FaJCsPrJa7CQQERER0S3m4vqLYidBdCHLQ8ROAhERObHVY1aLnQQn9Q+xE9CPB8VOgF34JDQRERERERERERHRNZj2NIidhD5Jp4idAvs498RpRERERERERERERHRLYxCaiIiIiIiIiIiIiATDIDQRERERERERERERCYZBaCIiIiIiIiIiIiISDIPQRERERERERERERCQYBqGJiIiIiIiIiIiISDAMQhMRERERERERERGRYBiEJiIiIiIiIiIiIiLBMAhNRERERERERERERIJhEJqIiIiIiIiIiIiIBMMgNBEREREREREREREJhkFoIiIiIiIiIiIiIhIMg9BEREREREREREREJBgGoYmIiIiIiIiIiIhIMAxCExEREREREREREZFgGIQmIiIiIiIiIiIiIsHIxE4AERERERHRYAlZHiJ2EkR3cf1FsZPgFFgXiIiIBg+fhCYiIiIiIiIiIiIiwTAITURERERERERE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b0xeMdnhQXZ7iTVmKPikAOf2n7P+u/vvPO25aQgaFmTX9RaRGCRmod5rJhLY6iOrxU4CEREREdEt5+L6i2InwSmELOciXUREfVk9ZrXYSXBKpj1rxU5Cn6RTnhU7CXa5dd7dIiIiIiIiIiIiIqJbDoPQRERERERERERERCQYBqGJiIiIiIiIiIiISDAMQhMRERERERERERGRYBiEJiIiIiIiIiIiIiLBMAhNRERERERERERERIJhEJqIiIiIiIiIiIiIBMMgNBEREREREREREREJhkFoIiIiIiIiIiIiIhIMg9BEREREREREREREJBgGoYmIiIiIiIiIiIhIMAxCExEREREREREREZFgGIQmIiIiIiIiIiIiIsEwCE1EREREREREREREgmEQmoiIiIiIiIiIiIgEwyA0EREREREREREREQmGQWgiIiIiIiIiIiIiEoxM7AQQERERERHR4AlZHiJ2EpzCxfUXxU6CU2B9ICKiwcAnoYmIiIiIiIiIiIhIMAxCExEREREREREREZFgGIQmIiIiIiIiIiIiIsEwCE1EREREREREREREgmEQmoiIiIiIiIiIiIgEwyA0EREREREREREREQmGQWgiIiIiIiIiIiIiEgyD0EREREREREREREQkGAahiYiIiIiIiIiIiEgwDEITERERERERERERkWAYhCYiIiIiIiIiIiIiwTAITURERERERERERESCYRCaiIiIiIiIiIiIiATDIDQRERERERERERERCYZBaCIiIiIiIiIiIiISDIPQRERERERERERERCQYBqGJiIiIiIiIiIiISDAysRNA4luwYAH0ej22b9/e67t9+/YhMzMT+fn5SEtLw0MPPYS3334bmzZtwh133GGz7erVq/HCCy8AAKRSKUJDQzFnzhysW7cOfn5+vfZtNpsxd+5cbN++HZ9++ikWLlwoSP66lewowekvT0PTooHvEF+MvGck/GP9+92+6nAVTn5yEh31HfAK8kL60nSEpofapL9gSwHKdpVB36mHOkGN0ctGwyvYy7pN0dYiXMi7gKaqJkhlUtz+99t7HaehvAH5/85HY0UjAMA/1h/pS9PhG+nrwNxbiFEG3Yx6I75Z/Q2aq5qR/VK2NX/tl9vxn5X/6bX9zOdnQh2ndkCu7TOQvFzpemVq7DLixAcnUHm4Eia9CcGpwRi1bBTcvd1t9lO+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JdOetWInoU/SKc+KnQS7OOf7SURERERERERERET0X4FBaCIiIiIiIiIiIiISDIPQRERERERERERERCQYBqGJiIiIiIiIiIiISDAMQhMRERERERERERGRYGRiJ4CIiIiIiIiIiIjIme1pvE3sJPRpqtgJsBOfhCYiIiIiIiIiIiIiwTAITURERERERERERESCYRCaiIiIiIiIiIiIiATDIDQRERERERERERERCYZBaCIiIiIiIiIiIiISDIPQRERERERERERERCQYBqGJiIiIiIiIiIiISDAMQhMRERERERERERGRYBiEJiIiIiIiIiIiIiLBMAhNRERERERERERERIJhEJqIiIiIiIiIiIiIBCMTOwFEREREREREgy1keYjYSXAKF9dfFDsJomNdICISHp+EJiIiIiIiIiIiIiLBMAhNRERERERERERERIJhEJqIiIiIiIiIiIiIBMMgNBEREREREREREREJhkFoIiIiIiIiIiIiIhIMg9BEREREREREREREJBgGoYmIiIiIiIiIiIh+IN58801ERUXBzc0NY8eOxZEjR/rd9p///CcmT54MX19f+Pr6YsaMGdfcvj8MQhMRERERERERERH9APz73//GypUr8fzzzyM3NxfDhw/H7NmzcenSpT633717N+68807s2rUL3333HYYMGYJZs2ahpqZmQMdlEJqIiIiIiIiIiIjoB+D111/HAw88gPvuuw9JSUl466234OHhgQ0bNvS5/fvvv4+f//znSE9PR2JiIt5++22YTCbs3LlzQMdlEJqIiIiIiIiIiIjoFqTT6dDa2mrzo9Pp+ty2q6sLx48fx4wZM6yfSaVSzJgxA999951dx+vs7IRer4efn9+A0skgNBEREREREREREdEtaO3atfD29rb5Wbt2bZ/b1tfXw2g0IigoyObzoKAg1NbW2nW8p59+GqGhoTaBbHvIBrQ1ERERERERERERETmFZ599FitXrrT5TKFQCHKsdevWYdOmTdi9ezfc3NwG9LsMQhMRERERERERERHdgv4/e/cdH0WZ/wH8s5vNbnrdkArpgfTQO4QeQP3BoXCcDeE8zjuUA/uJCsoBh+XUU0/vBERPRBEUFEFp0nsahEBCQgohCel1d5Mtvz/2srAkgSTs7ET4vF+vfWlmZ2ee58szzzzzzDPPKBSKDnc6K5VK2NjYoLS01Gx5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RfSCaKvO5ydWWeiT1Af6Zj1SvkiBpl4D917uGPP8mFYX39Z0t54vO5qXzpaFxqpGFCUXAbh2jmsx9q9j4R1pbOCf2XwGlw5dMn3Xsu716wiNbYa2CRGXyrxKVORUAAB+eOYHs/3d+/a9Zp1L1iZEfdiRGN7qGCg8WQhNnQZ5h/OQdzjPtJ6j0hH3/eM+gaJh7m48V4pVL3REyKgQaNVaZO3OQsqXKZA7yNEjqocg71QQKw42chvk7MtB8hfJ0Dfr4eDpgIABAYi6x/wp6OvPL5WXKpF/NN8qx8bd3ma4HYGBgXj66adNf8+cOROAcS7e9evXi5Us+pVoawpbhUL8J+1uJDFY+5k2IgtZemKp2EkgIiIiIiIiojtE8ZpisZPQLXz88cdiJ6Fb6q79UEsHLe3wuk1NTXBwcMA333yDadOmmZY/+uijqK6uxtatW1v9plevXli8eDH+8pe/mJa9+uqr+O6775CWltbhfXNOaCIiIiIiIiIiIqI7nFwuR//+/bFnzx7TMr1ejz179mDo0KFt/mbo0KFm6wPArl272l2/PeyEJvzxj3+Ek5NTm58//vGPYiePiIiIiIiIiIiILKBlyuD169cjMzMTTzzxBBoaGvDYY48BAB555BG8+OKLpvUXLlyInTt34q233sL58+exdOlSnDp1CgsWLOjUfjknNOG1117DM8880+Z3Li4uVk4NERERERERERERCWHWrFkoKyvDK6+8gpKSEiQkJGDnzp2mlw8WFBRAKr02bnnYsGHYsGEDlixZgr/+9a8IDw/Hd999h5iYmE7tl3NC069Wd52Lh4iIiIiIiIh+fTgntBHnhG5bd+2H6syc0GLidBxEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBh2QhMRERERERERERGRYNgJTURERERERERERESCYSc0EREREREREREREQmGndBEREREREREREREJBwDEXWJWq02vPrqqwa1Wi12UkTFODAGLRgHI8aBMWjBODAGLRgHI8aBMWjBOBgxDoxBC8bBiHFgDOjOJjEYDAaxO8KJfo1qa2vh6uqK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0qULtm7diqpVq6qvGRkZoV69enq3C/jfvvrqK0yZMgWFhYWoXr06xowZg48//hgmJiaio2nFm2++iRo1auC7775Tz3/+66+/8N577+H27dtPjLQiIiIi/ePg4IApU6Zg9OjRGusrVqzAokWLkJKSIigZ0avHJjQRlVtMTAymT5+O0NDQpx5Ip1KpBCWrWP45G1cXcX5ZqS+++AKenp548803NUYw6Kv8/Hzk5ubq1eiJf8rIyEDdunX1bkzRf7l58ya+++47REVFISMjA/3798eIESOQlZWF+fPnw9bWFvv27RMdUyuMjY2RkJAAJycnjfWzZ8+idevWejUrnYiIiJ7uf//7Hz766CMEBgaiffv2AErvqaKiorB06dInmtNErzM2oYmo3MoOpPt3k0VfDiZ8UbrehG7SpAnef/99jfllABAeHo5vv/1WvTta13Xv3h2//fYbioqK0Lp1a3h4eKBz587o0KEDjI2NRceTxezZs+Hv74/69euLjiLcnj17YGZmpn5SZPny5fj222/RtGlTLF++XL0jVtdFR0djzZo12Lt3L5o2bYqRI0di6NChsLS0VL8nLS0NTZo0QWFhobigWtSsWTMsXrwYXbp00ViPiYnBhAkTcObMGUHJiIiIqCLZtm0bFi1apL5/atKkCaZMmYK+ffsKTkb0arEJTUTldujQoWde9/DwkClJxabrTWjOL/tbUVERfv/9d8TFxeHQoUM4evQoHj16hNatW+PIkSOi42lds2bNcPbsWbRt2xZDhw7FwIED9XYWuouLC+bPnw9vb2+cOXMGrVq1QnBwMGJjY9G4cWOsWbNGdERZWFhYYNCgQRg5ciRat2791Pc8fPgQYWFh+PLLL2VOpz33799Xf3/kyBFMnToVM2bMQLt27QAAx44dQ0hICObNmwdvb29RMYmIiIiIZMcmNBGRluh6E5rzy5506dIlxMbGYv/+/di+fTssLCxw584d0bFkce7cOaxfvx4//PADsrKy0L17d/j7+6Nfv346O/P3aczMzHD27FnY2dlhxowZOHv2LLZs2YJTp07B29sbN27cEB1RFvn5+Xr1/3uZfx9QWfZrdtnaP1/zqSEiIiIi0ie6fSw5EWlNTk4OIiMj1Y8MOTk5ITAwEBYWFoKTkVyCg4MRFBSExMTEp84v0xerVq3CwYMHcejQITx69Aju7u7o3Lkz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iOil3L17F35+foiJiYFCoUBKSgrs7e0RGBgIKysrLFq0SHREIiLZVapUCTY2NujXrx86d+4MDw8PVKtWTXQsIiIiIiKhlKIDEBHR62nixIkwNDREZmYmTExM1Ot+fn7Ys2ePwGREROLk5ORg1apVMDExwfz582FrawsXFxd88MEH2LJlC27fvi06IhERERGR7LgTmoiIXkqtWrWwd+9eNGvWDObm5khKSoK9vT3S09Ph6uqK3Nxc0RGJiIR78OABjhw5op4PnZSUBEdHR5w9e1Z0NCIiIiIi2XAnNBERvZS8vDyNHdBlsrOzUblyZQGJiIgqHlNTU1StWhVVq1aFlZUVDA0NceHCBdGxiIiIiIhkxSY0ERG9FHd3d6xdu1b9WqFQoKSkBGFhYfD09BSYjIhInJKSEhw/fhxhYWHo1asXLC0t0b59e3zzzTeoVasWli9fjvT0dNExiYiIiIhkxXEcRET0Us6ePYuuXbvCzc0NMTEx8PHxwblz55CdnY34+Hg0aNBAdEQiItmpVCrk5eWhVq1a8PT0hKenJzp37syfiURERESk19iEJiKil3bv3j18/fXXSEpKQm5uLtzc3DB+/HjY2NiIjkZEJMTKlSvh6emJhg0bio5CRERERFRhsAlNRERERERERERERFpjKDoAERG9vgoKCpCcnIxbt26hpKRE45qPj4+gVERERERERERUkbAJTUREL2XPnj0ICAjAnTt3nrimUChQXFwsIBURERERERERVTRK0QGIiOj19OGHH2LAgAG4fv06SkpKNL7YgCYiIiIiIiKiMpwJTUREL0WlUuH06dNo0KCB6ChEREREREREVIFxJzQREb2Ud955BwcPHhQdg4iIiIiIiIgqOO6EJiKil5Kfn48BAwagRo0acHFxQaVKlTSuBwUFCUpGRERERERERBUJm9BERPRSIiMjMWbMGFSpUgXVqlWDQqFQX1MoFEhPTxeYjoiIiIiIiIgqCjahiYjopdSqVQtBQUH4+OOPoVRyuhMRERERERERPR27BkRE9FIKCwvh5+fHBjQRERERERERPRM7B0RE9FKGDRuGTZs2iY5BRERERERERBWcoegARET0eiouLkZYWBj27t0LV1fXJw4mDA8PF5SMiIiIiIiIiCoSzoQmIqKX4unp+Z/XFAoFYmJiZExDRERERERERBUVm9BEREREREREREREpDWcCU1EREREREREREREWsOZ0ERE9MJ8fX0RFRUFlUoFX1/fZ743OjpaplREREREREREVJGxCU1ERC/MwsICCoVC/T0RERERERER0fNwJjQREZVLSEgIJk+eDBMTE9FRiIiIiIiIiOg1wCY0ERGVi4GBAa5fvw5ra2vRUYiIiIiIiIjoNcCDCYmIqFz42SURERERERERlQeb0EREVG5lc6GJiIiIiIiIiJ6H4ziIiKhclEqlxgGF/yU7O1umRERERERERERUkRmKDkBERK+fmTNnwsLCQnQMIiIiIiIiInoNcCc0ERGVi1KpxI0bN3gwIRERERERERG9EM6EJiKicuE8aCIiIiIiIiIqDzahiYioXPgADRERERERERGVB8dxEBEREREREREREZHWcCc0EREREREREREREWkNm9BEREREREREREREpDVsQhMRERERERERERGR1rAJTURERERERERERERawyY0EREREREREREREWkNm9BEREREREREREREpDVsQhMRERERERERERGR1vw/UQRF40lfekEAAAAASUVORK5CYII="/>

<pre>
<Figure size 640x480 with 0 Axes>
</pre>
### 모델 준비



이제 훈련을 위한 데이터를 마무리하고 모델을 준비합니다.



```python
# Create evaluation function
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test, y_preds))
# Create function to evaluate our model
def show_scores(y_test, val_preds):
    scores = {"Valid MAE": mean_absolute_error(y_test, val_preds),
              "Valid RMSLE": rmsle(y_test, val_preds)}
    return scores
```


```python
# Attrition_rate는 예측할 레이블 또는 출력입니다.
# features는 Attrition_rate를 예측하는 데 사용됩니다.
label = ["Attrition_rate"]
features = ['VAR7','VAR6','VAR5','VAR1','VAR3','growth_rate','Time_of_service','Time_since_promotion','Travel_Rate','Post_Level','Education_Level']
```


```python
featured_data = df_train.loc[:,features+label]
# your code here 
featured_data.shape
```

<pre>
(7000, 12)
</pre>

```python
# dropna 함수를 사용하여 누락된 값이 있는 열을 제거합니다.
# your code here 
featured_data = featured_data.dropna()
featured_data.shape
```

<pre>
(6856, 12)
</pre>

```python
X = featured_data.loc[:,features]
y = featured_data.loc[:,label]
```


```python
# test size가 0.55이므로 training과 test data를 55%:45%로 분할합니다.
# your code here 
# 위치 잘맞출것;;
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=110)
print( "x_train values count: " + str(x_train.shape[0]) )
print( "y_train values count: " + str(y_train.shape[0]) )
print( "x_test values count: " + str(x_test.shape[0]) )
print( "y_test values count: " + str(y_test.shape[0]) )
# print( "x_train values count: " + str(len(x_train)))
# print( "y_train values count: " + str(len(y_train)))
# print( "x_test values count: " + str(len(x_test)))
# print( "y_test values count: " + str(len(y_test)))
```

<pre>
x_train values count: 3770
y_train values count: 3770
x_test values count: 3086
y_test values count: 3086
</pre>

```python
# LinearRegression 모델을 사용하여 학습(fit)하고 예측(predict) 합니다
model = LinearRegression()
# your code here 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```


```python
# score 를 출력해 봅니다. : error(MAE, RMSLE)
# your code here 
show_scores(y_test, y_pred)
```

<pre>
{'Valid MAE': 0.1280385863084669, 'Valid RMSLE': 0.14088870335452292}
</pre>
### 예측 해보기



```python
# 예측 (아래 선언 이전에 한게 있어서 필요없을지도..)
import pandas as pd
```


```python
# sample 데이터 [Dataset]_Module11_sample_(Employee).csv 가져오기
sample = pd.read_csv("./[Dataset]_Module11_sample_(Employee).csv")

c=[]
for i in range(len(y_pred)):
    c.append((y_pred[i][0].round(5)))
pf=c[:3000]

sample.head(5)
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
      <th>Employee_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Education_Level</th>
      <th>Relationship_Status</th>
      <th>Hometown</th>
      <th>Unit</th>
      <th>Decision_skill_possess</th>
      <th>Time_of_service</th>
      <th>Time_since_promotion</th>
      <th>...</th>
      <th>Pay_Scale</th>
      <th>Compensation_and_Benefits</th>
      <th>Work_Life_balance</th>
      <th>VAR1</th>
      <th>VAR2</th>
      <th>VAR3</th>
      <th>VAR4</th>
      <th>VAR5</th>
      <th>VAR6</th>
      <th>VAR7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EID_22713</td>
      <td>F</td>
      <td>32.0</td>
      <td>5</td>
      <td>Single</td>
      <td>Springfield</td>
      <td>R&amp;D</td>
      <td>Conceptual</td>
      <td>7.0</td>
      <td>4</td>
      <td>...</td>
      <td>4.0</td>
      <td>type2</td>
      <td>1.0</td>
      <td>3</td>
      <td>-0.9612</td>
      <td>-0.4537</td>
      <td>2.0</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EID_9658</td>
      <td>M</td>
      <td>65.0</td>
      <td>2</td>
      <td>Single</td>
      <td>Lebanon</td>
      <td>IT</td>
      <td>Directive</td>
      <td>41.0</td>
      <td>2</td>
      <td>...</td>
      <td>1.0</td>
      <td>type2</td>
      <td>1.0</td>
      <td>4</td>
      <td>-0.9612</td>
      <td>0.7075</td>
      <td>1.0</td>
      <td>2</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EID_22203</td>
      <td>M</td>
      <td>52.0</td>
      <td>3</td>
      <td>Married</td>
      <td>Springfield</td>
      <td>Sales</td>
      <td>Directive</td>
      <td>21.0</td>
      <td>3</td>
      <td>...</td>
      <td>8.0</td>
      <td>type3</td>
      <td>1.0</td>
      <td>4</td>
      <td>-0.1048</td>
      <td>0.7075</td>
      <td>2.0</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EID_7652</td>
      <td>M</td>
      <td>50.0</td>
      <td>5</td>
      <td>Single</td>
      <td>Washington</td>
      <td>Marketing</td>
      <td>Analytical</td>
      <td>11.0</td>
      <td>4</td>
      <td>...</td>
      <td>2.0</td>
      <td>type0</td>
      <td>4.0</td>
      <td>3</td>
      <td>-0.1048</td>
      <td>0.7075</td>
      <td>2.0</td>
      <td>2</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EID_6516</td>
      <td>F</td>
      <td>44.0</td>
      <td>3</td>
      <td>Married</td>
      <td>Franklin</td>
      <td>R&amp;D</td>
      <td>Conceptual</td>
      <td>12.0</td>
      <td>4</td>
      <td>...</td>
      <td>2.0</td>
      <td>type2</td>
      <td>4.0</td>
      <td>4</td>
      <td>1.6081</td>
      <td>0.7075</td>
      <td>2.0</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



```python
# your code here 
dff = pd.DataFrame({'Employee_ID':sample['Employee_ID'],'Attrition_rate':pf})
dff.head()
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
      <th>Employee_ID</th>
      <th>Attrition_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EID_22713</td>
      <td>0.18430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EID_9658</td>
      <td>0.18544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EID_22203</td>
      <td>0.18532</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EID_7652</td>
      <td>0.20305</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EID_6516</td>
      <td>0.20507</td>
    </tr>
  </tbody>
</table>
</div>


## Task 3: 예측된 결과의 Attrition_rate이 높은 20개 열 값 출력




```python
# your code here 
dff.sort_values('Attrition_rate', ascending=False).head(20)
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
      <th>Employee_ID</th>
      <th>Attrition_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1809</th>
      <td>EID_5873</td>
      <td>0.21646</td>
    </tr>
    <tr>
      <th>2036</th>
      <td>EID_10338</td>
      <td>0.21526</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>EID_19140</td>
      <td>0.21519</td>
    </tr>
    <tr>
      <th>373</th>
      <td>EID_4261</td>
      <td>0.21500</td>
    </tr>
    <tr>
      <th>2986</th>
      <td>EID_17284</td>
      <td>0.21488</td>
    </tr>
    <tr>
      <th>2513</th>
      <td>EID_24214</td>
      <td>0.21315</td>
    </tr>
    <tr>
      <th>281</th>
      <td>EID_14096</td>
      <td>0.21303</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>EID_15641</td>
      <td>0.21291</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>EID_10684</td>
      <td>0.21227</td>
    </tr>
    <tr>
      <th>2431</th>
      <td>EID_22724</td>
      <td>0.21154</td>
    </tr>
    <tr>
      <th>34</th>
      <td>EID_19046</td>
      <td>0.21149</td>
    </tr>
    <tr>
      <th>51</th>
      <td>EID_17967</td>
      <td>0.21116</td>
    </tr>
    <tr>
      <th>2173</th>
      <td>EID_18844</td>
      <td>0.21107</td>
    </tr>
    <tr>
      <th>691</th>
      <td>EID_14595</td>
      <td>0.21069</td>
    </tr>
    <tr>
      <th>2516</th>
      <td>EID_2152</td>
      <td>0.21055</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>EID_23260</td>
      <td>0.21048</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>EID_25701</td>
      <td>0.21040</td>
    </tr>
    <tr>
      <th>2388</th>
      <td>EID_6168</td>
      <td>0.21012</td>
    </tr>
    <tr>
      <th>1747</th>
      <td>EID_20012</td>
      <td>0.21011</td>
    </tr>
    <tr>
      <th>2944</th>
      <td>EID_1930</td>
      <td>0.21009</td>
    </tr>
  </tbody>
</table>
</div>


#### 추가로...



강사님께서 아래처럼 주어진 방식으로 예측이 주어졌을 경우.. 차이와 이직률을 분석해본다면..



```python
ID          = ["Employee_ID"]
pred_data   = sample.loc[:,features+ID]
pred_data   = pred_data.dropna(axis=0)
y = pred_data.loc[:,ID]
sample_data = sample.loc[:,features]
sample_data = sample_data.dropna(axis=0)
y_hat = model.predict(sample_data)
size = len(y_hat)
c=[]
for i in range(len(y_hat)):
    c.append((y_hat[i][0].round(5)))
pf=c[:size]
sample_data.head(5)
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
      <th>VAR7</th>
      <th>VAR6</th>
      <th>VAR5</th>
      <th>VAR1</th>
      <th>VAR3</th>
      <th>growth_rate</th>
      <th>Time_of_service</th>
      <th>Time_since_promotion</th>
      <th>Travel_Rate</th>
      <th>Post_Level</th>
      <th>Education_Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>-0.4537</td>
      <td>30</td>
      <td>7.0</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>0.7075</td>
      <td>72</td>
      <td>41.0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>0.7075</td>
      <td>25</td>
      <td>21.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8</td>
      <td>2</td>
      <td>3</td>
      <td>0.7075</td>
      <td>28</td>
      <td>11.0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>0.7075</td>
      <td>47</td>
      <td>12.0</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



```python
y.head(5)
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
      <th>Employee_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EID_22713</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EID_9658</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EID_22203</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EID_7652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EID_6516</td>
    </tr>
  </tbody>
</table>
</div>



```python
pf[:10]
```

<pre>
[0.19277,
 0.17537,
 0.17751,
 0.17771,
 0.19025,
 0.19642,
 0.19457,
 0.1931,
 0.19157,
 0.17776]
</pre>

```python
dff1 = pd.DataFrame({'Employee_ID':y['Employee_ID'], 'Attrition_rate':pf})
dff1.head()
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
      <th>Employee_ID</th>
      <th>Attrition_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EID_22713</td>
      <td>0.19277</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EID_9658</td>
      <td>0.17537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EID_22203</td>
      <td>0.17751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EID_7652</td>
      <td>0.17771</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EID_6516</td>
      <td>0.19025</td>
    </tr>
  </tbody>
</table>
</div>



```python
dff1.sort_values('Attrition_rate', ascending=False).head(20)
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
      <th>Employee_ID</th>
      <th>Attrition_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1695</th>
      <td>EID_21702</td>
      <td>0.21477</td>
    </tr>
    <tr>
      <th>2791</th>
      <td>EID_17304</td>
      <td>0.21387</td>
    </tr>
    <tr>
      <th>52</th>
      <td>EID_13270</td>
      <td>0.21373</td>
    </tr>
    <tr>
      <th>2546</th>
      <td>EID_13443</td>
      <td>0.21370</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>EID_21042</td>
      <td>0.21336</td>
    </tr>
    <tr>
      <th>988</th>
      <td>EID_23350</td>
      <td>0.21214</td>
    </tr>
    <tr>
      <th>631</th>
      <td>EID_13681</td>
      <td>0.21191</td>
    </tr>
    <tr>
      <th>1819</th>
      <td>EID_19366</td>
      <td>0.21178</td>
    </tr>
    <tr>
      <th>638</th>
      <td>EID_7609</td>
      <td>0.21153</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>EID_15348</td>
      <td>0.21148</td>
    </tr>
    <tr>
      <th>1041</th>
      <td>EID_5420</td>
      <td>0.21137</td>
    </tr>
    <tr>
      <th>846</th>
      <td>EID_24603</td>
      <td>0.21125</td>
    </tr>
    <tr>
      <th>2327</th>
      <td>EID_18242</td>
      <td>0.21122</td>
    </tr>
    <tr>
      <th>512</th>
      <td>EID_16224</td>
      <td>0.21088</td>
    </tr>
    <tr>
      <th>578</th>
      <td>EID_22377</td>
      <td>0.21073</td>
    </tr>
    <tr>
      <th>305</th>
      <td>EID_18954</td>
      <td>0.21072</td>
    </tr>
    <tr>
      <th>1521</th>
      <td>EID_23613</td>
      <td>0.21064</td>
    </tr>
    <tr>
      <th>1541</th>
      <td>EID_11601</td>
      <td>0.21055</td>
    </tr>
    <tr>
      <th>1859</th>
      <td>EID_15161</td>
      <td>0.21050</td>
    </tr>
    <tr>
      <th>561</th>
      <td>EID_13536</td>
      <td>0.21035</td>
    </tr>
  </tbody>
</table>
</div>


#### 위 차이를 보기 편하게 아래 표로..



두 대조표를 합쳐본다면...



```python
dff = dff.rename(columns={ "Attrition_rate": "Attrition_rate1"})
dff1 = dff1.rename(columns={"Attrition_rate": "Attrition_rate2"})
dff_t = pd.concat([dff, dff1["Attrition_rate2"]], axis=1)
dff_t.sort_values('Attrition_rate1', ascending=False).head(20)
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
      <th>Employee_ID</th>
      <th>Attrition_rate1</th>
      <th>Attrition_rate2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1809</th>
      <td>EID_5873</td>
      <td>0.21646</td>
      <td>0.19128</td>
    </tr>
    <tr>
      <th>2036</th>
      <td>EID_10338</td>
      <td>0.21526</td>
      <td>0.19797</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>EID_19140</td>
      <td>0.21519</td>
      <td>0.18473</td>
    </tr>
    <tr>
      <th>373</th>
      <td>EID_4261</td>
      <td>0.21500</td>
      <td>0.16753</td>
    </tr>
    <tr>
      <th>2986</th>
      <td>EID_17284</td>
      <td>0.21488</td>
      <td>0.19579</td>
    </tr>
    <tr>
      <th>2513</th>
      <td>EID_24214</td>
      <td>0.21315</td>
      <td>0.18597</td>
    </tr>
    <tr>
      <th>281</th>
      <td>EID_14096</td>
      <td>0.21303</td>
      <td>0.19472</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>EID_15641</td>
      <td>0.21291</td>
      <td>0.18291</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>EID_10684</td>
      <td>0.21227</td>
      <td>0.18685</td>
    </tr>
    <tr>
      <th>2431</th>
      <td>EID_22724</td>
      <td>0.21154</td>
      <td>0.19074</td>
    </tr>
    <tr>
      <th>34</th>
      <td>EID_19046</td>
      <td>0.21149</td>
      <td>0.19697</td>
    </tr>
    <tr>
      <th>51</th>
      <td>EID_17967</td>
      <td>0.21116</td>
      <td>0.19278</td>
    </tr>
    <tr>
      <th>2173</th>
      <td>EID_18844</td>
      <td>0.21107</td>
      <td>0.19377</td>
    </tr>
    <tr>
      <th>691</th>
      <td>EID_14595</td>
      <td>0.21069</td>
      <td>0.19196</td>
    </tr>
    <tr>
      <th>2516</th>
      <td>EID_2152</td>
      <td>0.21055</td>
      <td>0.19799</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>EID_23260</td>
      <td>0.21048</td>
      <td>0.17965</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>EID_25701</td>
      <td>0.21040</td>
      <td>0.18777</td>
    </tr>
    <tr>
      <th>2388</th>
      <td>EID_6168</td>
      <td>0.21012</td>
      <td>0.18116</td>
    </tr>
    <tr>
      <th>1747</th>
      <td>EID_20012</td>
      <td>0.21011</td>
      <td>0.18441</td>
    </tr>
    <tr>
      <th>2944</th>
      <td>EID_1930</td>
      <td>0.21009</td>
      <td>0.17772</td>
    </tr>
  </tbody>
</table>
</div>



```python
dff_t.sort_values('Attrition_rate2', ascending=False).head(20)
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
      <th>Employee_ID</th>
      <th>Attrition_rate1</th>
      <th>Attrition_rate2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1695</th>
      <td>EID_21702</td>
      <td>0.18664</td>
      <td>0.21477</td>
    </tr>
    <tr>
      <th>2791</th>
      <td>EID_17304</td>
      <td>0.18761</td>
      <td>0.21387</td>
    </tr>
    <tr>
      <th>52</th>
      <td>EID_13270</td>
      <td>0.19970</td>
      <td>0.21373</td>
    </tr>
    <tr>
      <th>2546</th>
      <td>EID_13443</td>
      <td>0.20360</td>
      <td>0.21370</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>EID_21042</td>
      <td>0.18628</td>
      <td>0.21336</td>
    </tr>
    <tr>
      <th>988</th>
      <td>EID_23350</td>
      <td>0.19529</td>
      <td>0.21214</td>
    </tr>
    <tr>
      <th>631</th>
      <td>EID_13681</td>
      <td>0.17672</td>
      <td>0.21191</td>
    </tr>
    <tr>
      <th>1819</th>
      <td>EID_19366</td>
      <td>0.18475</td>
      <td>0.21178</td>
    </tr>
    <tr>
      <th>638</th>
      <td>EID_7609</td>
      <td>0.19132</td>
      <td>0.21153</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>EID_15348</td>
      <td>0.19424</td>
      <td>0.21148</td>
    </tr>
    <tr>
      <th>1041</th>
      <td>EID_5420</td>
      <td>0.17756</td>
      <td>0.21137</td>
    </tr>
    <tr>
      <th>846</th>
      <td>EID_24603</td>
      <td>0.18627</td>
      <td>0.21125</td>
    </tr>
    <tr>
      <th>2327</th>
      <td>EID_18242</td>
      <td>0.19149</td>
      <td>0.21122</td>
    </tr>
    <tr>
      <th>512</th>
      <td>EID_16224</td>
      <td>0.18563</td>
      <td>0.21088</td>
    </tr>
    <tr>
      <th>578</th>
      <td>EID_22377</td>
      <td>0.17252</td>
      <td>0.21073</td>
    </tr>
    <tr>
      <th>305</th>
      <td>EID_18954</td>
      <td>0.18692</td>
      <td>0.21072</td>
    </tr>
    <tr>
      <th>1521</th>
      <td>EID_23613</td>
      <td>0.19274</td>
      <td>0.21064</td>
    </tr>
    <tr>
      <th>1541</th>
      <td>EID_11601</td>
      <td>0.17665</td>
      <td>0.21055</td>
    </tr>
    <tr>
      <th>1859</th>
      <td>EID_15161</td>
      <td>0.19136</td>
      <td>0.21050</td>
    </tr>
    <tr>
      <th>561</th>
      <td>EID_13536</td>
      <td>0.19840</td>
      <td>0.21035</td>
    </tr>
  </tbody>
</table>
</div>


#### 혹시 LinearRegression에 옵션을 더 줘서 성능 향상이 가능할까?



순서대로 진행해보자.



```python
# LinearRegression 모델을 사용하여 학습(fit)하고 예측(predict) 합니다
# x_train, y_train 등 데이터는 위 그대로..
model = LinearRegression(copy_X=False)
# your code here 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# score 를 출력해 봅니다. : error(MAE, RMSLE)
# your code here 
show_scores(y_test, y_pred)
```

<pre>
{'Valid MAE': 0.1280385863084669, 'Valid RMSLE': 0.14088870335452292}
</pre>
#### LR 옵션변경 결과



별로 바뀐게 없어서 이상으로 마무리를... 다음!<br>

아무래도 옵션변경보다.. 데이터를 손봐야..


#### 그 외 모델을 적용한다면...



예로.. Ridge, Lesso, RandomForest..

음.... 시간없을거 같아서 RandomForest만...

비교해서 최대한 오류가 적은 방식을 최우선...;;


##### 먼저 모델 적용부터..



```python
# Classifier로 하기에 데이터자체 특성상 오류가 나서 아래 모델로..
# 물론 불연속적으로 정제해서 하는 방법도 있으나 일단...
from sklearn.ensemble import RandomForestRegressor
```


```python
### RandomForestClassifier 모델을 사용하여 학습(fit)하고 예측(predict) 합니다
# x_train, y_train 등 데이터는 위 그대로..
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=950, max_samples = 1 )
# bootstrap=True, n_jobs=10, min_samples_split=10, warm_start=False
# max_leaf_nodes=2, min_weight_fraction_leaf=0.23, max_features="log2", min_impurity_decrease=0.1,
# ccp_alpha=0.05, 
# your code here 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# score 를 출력해 봅니다. : error(MAE, RMSLE)
# your code here 
show_scores(y_test, y_pred)
```

<pre>
C:\Users\User\AppData\Local\Temp\ipykernel_8388\957247849.py:8: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  model.fit(x_train, y_train)
</pre>
<pre>
{'Valid MAE': 0.12574740764744005, 'Valid RMSLE': 0.1403597189649096}
</pre>
#### Random Forest 모델을 끄적이면서...

음... 아무리 옵션을 조정을 해봐도 0.14... 미만으로 내려가지 않는다.<br>

좀 더 확인이 필요하지만 이 데이터에서는 Random Forest로는 한계가 있을듯 하다;;;



#### 그러면... 다른 모델로...

다른 사람이 한걸 봤는데.. Lasso Reg 모델이 0.13 초반정도의 에러율을 낮춘 결과를 보였다.<br>

좀 더 단순한 Ridge, Lasso 등 모델로 낮게  에러율을 낮출 수 있는 듯하다.



#### 데이터의 경우?

데이터 자체 수집 등은 손 대기 힘든 부분이라 한계가 있음.<br>

대신에 Random_state를 110으로 조정되어있는데 더 올려보거나 낮춰보거나..<br>

test, train 비율 조정등의 경우도 있으나<br>

랜덤포레스트를 이리저리 굴려본다면 차라리 Ridge, Lasso등을 해봐서 확인하는게 더 확실하다는 결론을 생각해냈다.


### Conclusion



이 노트북에서 우리는 기업에서 AI를 사용하여 충성할 직원을 예측하는 방법을 살펴보았습니다. 우리는 직원 감소율을 예측하기 위해 선형 회귀 모델을 만들었습니다.


#### 그 외 참조한 링크



https://minorman.tistory.com/84<br>

판다스 모든 열 확인용...<br>

<br>

https://eunjin3786.tistory.com/204<br>

판다스 정보 등 확인용..<br>

<br>

https://zephyrus1111.tistory.com/163<br>

유니크 갯수 확인용<br>

<br>

https://rfriend.tistory.com/383<br>

그룹by 확인<br>

<br>

https://zephyrus1111.tistory.com/70<br>

그룹by 특정열 출력 확인<br>

<br>

https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette<br>

막대 그래프 설정 가능한 색상 확인



https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html<br>

train_test_split X, y 나눠서 적용방법 확인<br>

<br>

https://seaborn.pydata.org/generated/seaborn.countplot.html<br>

countplot 확인...<br>

<br>

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html<br>

pandas dropna 확인.. <br>

<br>

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html<br>

dataframe의 column 이름을 바꾸고 싶다면..<br>

<br>

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html<br>

Linear_Regression 옵션 확인<br>

<br>



#### 마무리

- 이 과제로 각 회사내 사람들의 이직률을 이렇게 분석해서 필요한 정보를 전해주는구나 하고 깨닫게 되었다.

- 바뀌어진 버전에 따라 도큐먼트를 잘 봐둬야 되어야 문제가 벌어져도 쓸 수 있다는 교훈.

- 성별 growth_rate 비교에 데이터가 예시

- 이번엔 이전 과제에 비해 참고하려는 링크들이 많았다.

- 오류율 등 품질 향상을 위해 LR.. 및 다른 모델 등 중에서 랜덤포레스트를 해보았으나 0.14이하로 못내렸다.

- 이 후 Lasso, Ridge 등 모델로 결과를 해볼 필요가 있을 거 같다.


#### 주의사항



본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.

