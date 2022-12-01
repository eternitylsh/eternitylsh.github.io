---
layout: single
title:  "랜덤포레스트를 통한 보험사기 추측해보기"
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


# Insurance Fraud Detection using Random Forest



# 랜덤 포레스트를 이용한 보험 사기 탐지



## Introduction



인공지능은 프로세스를 자동화하고, 비즈니스에 대한 통찰력을 모으고, 프로세스 속도를 높이기 위해 다양한 산업에서 사용되고 있습니다. 인공지능이 실제로 산업에 어떤 영향을 미치는지 실제 시나리오에서 인공지능의 사용을 연구하기 위해 Python을 사용할 것입니다.



보험 사기는 매우 크고 중요한 문제입니다. 다양한 사기가 계속 발생하고 있으며, 일부 수법은 일반화 되어있습니다. 따라서 미리 예측하면 많은 피해를 막을 수 있으며, 비용을 절약할 수 있습니다. 이러한 문제에 대하여 AI는 우리를 도울 수 있습니다.



이 노트북에서는 랜덤 포레스트를 사용한 보험 사기 탐지에 중점을 둘 것입니다.



## Context



우리는 [Kaggle]에서 얻은 자동차 보험 청구 데이터로 실습할 것입니다. Kaggle은 데이터 전문가들이 모여 지식을 공유하고 서로 경쟁하여 보상을 받을 수 있는 데이터 공유 플랫폼입니다. 정리된 데이터가 Insurance.csv에 포함되어 있습니다.





### Side note: Random Forest란 무엇인가?



랜덤 포레스트는 많은 의사 결정 트리의 결정을 결합하여 데이터 포인트의 클래스를 결정하는 분류 알고리즘입니다. 우리는 다양한 트리를 기반으로 결정을 내리고 과반수 투표를 수행하고 최종 클래스를 결정합니다. 다음 다이어그램을 보면 좀 더 명확하게 이해할 수 있을 것입니다.



![Random Forest](https://miro.medium.com/max/888/1*i0o8mjFfCn-uD79-F1Cqkw.png)



## Use Python to open csv files





[scikit-learn](https://scikit-learn.org/stable/)과 [pandas](https://pandas.pydata.org/)를 사용하여 데이터 세트를 작업합니다. Scikit-learn은 예측 데이터 분석을 위한 효율적인 도구를 제공하는 매우 유용한 기계 학습 라이브러리입니다. Pandas는 데이터 과학을 위한 인기 있는 Python 라이브러리입니다. 강력하고 유연한 데이터 구조를 제공하여 데이터 조작 및 분석을 더 쉽게 만듭니다.





## Import Libraries




```python
import numpy as np 
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

### TASK1.  [Dataset]_Module11_(Insurance).csv 파일을 df 변수로 읽어 오세요.



```python
df =  pd.read_csv("./[Dataset]_Module11_(Insurance).csv")# your code here
df.head(5)# your code here
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
      <th>Unnamed: 0</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>witnesses</th>
      <th>total_claim_amount</th>
      <th>fraud_reported</th>
      <th>insured_sex_FEMALE</th>
      <th>insured_sex_MALE</th>
      <th>...</th>
      <th>months_as_customer_groups_301-350</th>
      <th>months_as_customer_groups_351-400</th>
      <th>months_as_customer_groups_401-450</th>
      <th>months_as_customer_groups_451-500</th>
      <th>months_as_customer_groups_51-100</th>
      <th>policy_annual_premium_groups_high</th>
      <th>policy_annual_premium_groups_low</th>
      <th>policy_annual_premium_groups_medium</th>
      <th>policy_annual_premium_groups_very high</th>
      <th>policy_annual_premium_groups_very low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>53300</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>71610</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>5070</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>35100</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>34650</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>48900</td>
      <td>-62400</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>63400</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>66000</td>
      <td>-46000</td>
      <td>20</td>
      <td>1</td>
      <td>1</td>
      <td>6500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69 columns</p>
</div>



```python
# Shape of dataframe

# your code here
df.shape
```

<pre>
(1000, 69)
</pre>

```python
# df의 컬럼 값을 확인하세요.
# your code here
df.columns
```

<pre>
Index(['Unnamed: 0', 'capital-gains', 'capital-loss',
       'incident_hour_of_the_day', 'number_of_vehicles_involved', 'witnesses',
       'total_claim_amount', 'fraud_reported', 'insured_sex_FEMALE',
       'insured_sex_MALE', 'insured_occupation_adm-clerical',
       'insured_occupation_armed-forces', 'insured_occupation_craft-repair',
       'insured_occupation_exec-managerial',
       'insured_occupation_farming-fishing',
       'insured_occupation_handlers-cleaners',
       'insured_occupation_machine-op-inspct',
       'insured_occupation_other-service',
       'insured_occupation_priv-house-serv',
       'insured_occupation_prof-specialty',
       'insured_occupation_protective-serv', 'insured_occupation_sales',
       'insured_occupation_tech-support',
       'insured_occupation_transport-moving', 'insured_hobbies_chess',
       'insured_hobbies_cross-fit', 'insured_hobbies_other',
       'incident_type_Multi-vehicle Collision', 'incident_type_Parked Car',
       'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft',
       'collision_type_?', 'collision_type_Front Collision',
       'collision_type_Rear Collision', 'collision_type_Side Collision',
       'incident_severity_Major Damage', 'incident_severity_Minor Damage',
       'incident_severity_Total Loss', 'incident_severity_Trivial Damage',
       'authorities_contacted_Ambulance', 'authorities_contacted_Fire',
       'authorities_contacted_None', 'authorities_contacted_Other',
       'authorities_contacted_Police', 'age_group_15-20', 'age_group_21-25',
       'age_group_26-30', 'age_group_31-35', 'age_group_36-40',
       'age_group_41-45', 'age_group_46-50', 'age_group_51-55',
       'age_group_56-60', 'age_group_61-65', 'months_as_customer_groups_0-50',
       'months_as_customer_groups_101-150',
       'months_as_customer_groups_151-200',
       'months_as_customer_groups_201-250',
       'months_as_customer_groups_251-300',
       'months_as_customer_groups_301-350',
       'months_as_customer_groups_351-400',
       'months_as_customer_groups_401-450',
       'months_as_customer_groups_451-500', 'months_as_customer_groups_51-100',
       'policy_annual_premium_groups_high', 'policy_annual_premium_groups_low',
       'policy_annual_premium_groups_medium',
       'policy_annual_premium_groups_very high',
       'policy_annual_premium_groups_very low'],
      dtype='object')
</pre>

```python
# 다양한 Feature에 대하여 null 값의 수를 확인합니다.
# your code here
df.isna().sum()
```

<pre>
Unnamed: 0                                0
capital-gains                             0
capital-loss                              0
incident_hour_of_the_day                  0
number_of_vehicles_involved               0
                                         ..
policy_annual_premium_groups_high         0
policy_annual_premium_groups_low          0
policy_annual_premium_groups_medium       0
policy_annual_premium_groups_very high    0
policy_annual_premium_groups_very low     0
Length: 69, dtype: int64
</pre>

```python
# Dataset에 대한 추가 정보를 확인합니다.
# your code here
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 69 columns):
 #   Column                                  Non-Null Count  Dtype
---  ------                                  --------------  -----
 0   Unnamed: 0                              1000 non-null   int64
 1   capital-gains                           1000 non-null   int64
 2   capital-loss                            1000 non-null   int64
 3   incident_hour_of_the_day                1000 non-null   int64
 4   number_of_vehicles_involved             1000 non-null   int64
 5   witnesses                               1000 non-null   int64
 6   total_claim_amount                      1000 non-null   int64
 7   fraud_reported                          1000 non-null   int64
 8   insured_sex_FEMALE                      1000 non-null   int64
 9   insured_sex_MALE                        1000 non-null   int64
 10  insured_occupation_adm-clerical         1000 non-null   int64
 11  insured_occupation_armed-forces         1000 non-null   int64
 12  insured_occupation_craft-repair         1000 non-null   int64
 13  insured_occupation_exec-managerial      1000 non-null   int64
 14  insured_occupation_farming-fishing      1000 non-null   int64
 15  insured_occupation_handlers-cleaners    1000 non-null   int64
 16  insured_occupation_machine-op-inspct    1000 non-null   int64
 17  insured_occupation_other-service        1000 non-null   int64
 18  insured_occupation_priv-house-serv      1000 non-null   int64
 19  insured_occupation_prof-specialty       1000 non-null   int64
 20  insured_occupation_protective-serv      1000 non-null   int64
 21  insured_occupation_sales                1000 non-null   int64
 22  insured_occupation_tech-support         1000 non-null   int64
 23  insured_occupation_transport-moving     1000 non-null   int64
 24  insured_hobbies_chess                   1000 non-null   int64
 25  insured_hobbies_cross-fit               1000 non-null   int64
 26  insured_hobbies_other                   1000 non-null   int64
 27  incident_type_Multi-vehicle Collision   1000 non-null   int64
 28  incident_type_Parked Car                1000 non-null   int64
 29  incident_type_Single Vehicle Collision  1000 non-null   int64
 30  incident_type_Vehicle Theft             1000 non-null   int64
 31  collision_type_?                        1000 non-null   int64
 32  collision_type_Front Collision          1000 non-null   int64
 33  collision_type_Rear Collision           1000 non-null   int64
 34  collision_type_Side Collision           1000 non-null   int64
 35  incident_severity_Major Damage          1000 non-null   int64
 36  incident_severity_Minor Damage          1000 non-null   int64
 37  incident_severity_Total Loss            1000 non-null   int64
 38  incident_severity_Trivial Damage        1000 non-null   int64
 39  authorities_contacted_Ambulance         1000 non-null   int64
 40  authorities_contacted_Fire              1000 non-null   int64
 41  authorities_contacted_None              1000 non-null   int64
 42  authorities_contacted_Other             1000 non-null   int64
 43  authorities_contacted_Police            1000 non-null   int64
 44  age_group_15-20                         1000 non-null   int64
 45  age_group_21-25                         1000 non-null   int64
 46  age_group_26-30                         1000 non-null   int64
 47  age_group_31-35                         1000 non-null   int64
 48  age_group_36-40                         1000 non-null   int64
 49  age_group_41-45                         1000 non-null   int64
 50  age_group_46-50                         1000 non-null   int64
 51  age_group_51-55                         1000 non-null   int64
 52  age_group_56-60                         1000 non-null   int64
 53  age_group_61-65                         1000 non-null   int64
 54  months_as_customer_groups_0-50          1000 non-null   int64
 55  months_as_customer_groups_101-150       1000 non-null   int64
 56  months_as_customer_groups_151-200       1000 non-null   int64
 57  months_as_customer_groups_201-250       1000 non-null   int64
 58  months_as_customer_groups_251-300       1000 non-null   int64
 59  months_as_customer_groups_301-350       1000 non-null   int64
 60  months_as_customer_groups_351-400       1000 non-null   int64
 61  months_as_customer_groups_401-450       1000 non-null   int64
 62  months_as_customer_groups_451-500       1000 non-null   int64
 63  months_as_customer_groups_51-100        1000 non-null   int64
 64  policy_annual_premium_groups_high       1000 non-null   int64
 65  policy_annual_premium_groups_low        1000 non-null   int64
 66  policy_annual_premium_groups_medium     1000 non-null   int64
 67  policy_annual_premium_groups_very high  1000 non-null   int64
 68  policy_annual_premium_groups_very low   1000 non-null   int64
dtypes: int64(69)
memory usage: 539.2 KB
</pre>

```python
# df의 각 컬럼의 unique값을 확인합니다.
# your code here
df.nunique(dropna=False)
```

<pre>
Unnamed: 0                                1000
capital-gains                              338
capital-loss                               354
incident_hour_of_the_day                    24
number_of_vehicles_involved                  4
                                          ... 
policy_annual_premium_groups_high            2
policy_annual_premium_groups_low             2
policy_annual_premium_groups_medium          2
policy_annual_premium_groups_very high       2
policy_annual_premium_groups_very low        2
Length: 69, dtype: int64
</pre>
### Task 2: describe 함수를 사용하여 데이터 세트에 대한 정보 표시



```python
# your code here
df.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>1000.0</td>
      <td>499.500</td>
      <td>288.819436</td>
      <td>0.0</td>
      <td>249.75</td>
      <td>499.5</td>
      <td>749.25</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>capital-gains</th>
      <td>1000.0</td>
      <td>25126.100</td>
      <td>27872.187708</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>51025.00</td>
      <td>100500.0</td>
    </tr>
    <tr>
      <th>capital-loss</th>
      <td>1000.0</td>
      <td>-26793.700</td>
      <td>28104.096686</td>
      <td>-111100.0</td>
      <td>-51500.00</td>
      <td>-23250.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>incident_hour_of_the_day</th>
      <td>1000.0</td>
      <td>11.644</td>
      <td>6.951373</td>
      <td>0.0</td>
      <td>6.00</td>
      <td>12.0</td>
      <td>17.00</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>number_of_vehicles_involved</th>
      <td>1000.0</td>
      <td>1.839</td>
      <td>1.018880</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>3.00</td>
      <td>4.0</td>
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
    </tr>
    <tr>
      <th>policy_annual_premium_groups_high</th>
      <td>1000.0</td>
      <td>0.153</td>
      <td>0.360168</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>policy_annual_premium_groups_low</th>
      <td>1000.0</td>
      <td>0.151</td>
      <td>0.358228</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>policy_annual_premium_groups_medium</th>
      <td>1000.0</td>
      <td>0.693</td>
      <td>0.461480</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>policy_annual_premium_groups_very high</th>
      <td>1000.0</td>
      <td>0.001</td>
      <td>0.031623</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>policy_annual_premium_groups_very low</th>
      <td>1000.0</td>
      <td>0.002</td>
      <td>0.044699</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 8 columns</p>
</div>



```python
# Fraud_reported가 목표 열입니다. Fraud_reported의 고유값을 확인해 봅시다.
# your code here
df['fraud_reported'].unique()
```

<pre>
array([1, 0], dtype=int64)
</pre>

```python
# 다음으로 fraud_reported 열의 0과 1의 분포를 확인할 수 있습니다.
sns.countplot(x=df['fraud_reported'])
```

<pre>
<AxesSubplot: xlabel='fraud_reported', ylabel='count'>
</pre>
<img src="data:image/png;base64,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Q5SZmamqqiq7RwNwgU6fPq0hQ4Zo6dKldo+CEMSnsXDJSEtL0zXXXKMXXnhB0t+/aTspKUnTpk3Tr371K5unA9BWHA6H1q5da32rPsCdHVwS6uvrVVJSooyMDGtdWFiYMjIyVFxcbONkAID2RuzgknDixAk1NTW1+FZtt9stn89n01QAgIuB2AEAAEYjdnBJ6NGjhzp16qTKysqg9ZWVlfJ4PDZNBQC4GIgdXBIiIiKUmpqqoqIia11zc7OKiork9XptnAwA0N6M+UOgwA/Jz89XTk6Ohg0bpmuvvVaLFy/W6dOndd9999k9GoALdOrUKX3++efW46NHj2rv3r2KjY1V7969bZwMoYCPnuOS8sILL+jpp5+Wz+fT0KFDtWTJEqWlpdk9FoALtHXrVt18880t1ufk5GjlypUXfyCEFGIHAAAYjffsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AA4q0AgoClTpig2NlYOh0N79+69qOefMGGC7rzzzot6zlDFzwK4MPxtLABntXHjRq1cuVJbt27VZZddph49etg9UocyYcIEVVdXa926dXaPAlzyiB0AZ3XkyBH17NlT11133Vm319fXKyIi4iJPdX7snK2pqUkOh8OWcwM4O17GAtDChAkTNG3aNJWVlcnhcKhv37666aablJeXp+nTp6tHjx7KzMyUJD377LMaNGiQunXrpqSkJN1///06deqUdaxHHnlEQ4cODTr+4sWL1bdvX+txU1OT8vPzFRMTo7i4OM2aNUs/5s/2nWu2zz77TCNHjlR0dLTcbrfGjRunEydOtHheXl6eXC6XevTooblz5wad+6uvvtL48eP1k5/8RF27dtXIkSN1+PBha/vKlSsVExOjt956SykpKYqMjNTEiRO1atUqvfnmm3I4HHI4HNq6daskqby8XGPGjFFMTIxiY2M1atQo/fWvf22znwWAlogdAC0899xzWrBggXr16qWKigrt3r1bkrRq1SpFRERox44dWr58uSQpLCxMS5Ys0f79+7Vq1Sq99957mjVr1o863zPPPKOVK1fq1Vdf1YcffqiTJ09q7dq1P+oY352turpaw4cP11VXXaU9e/Zo48aNqqys1JgxY1o8Lzw8XLt27dJzzz2nZ599Vr/97W+t7RMmTNCePXv01ltvqbi4WIFAQLfddpsaGhqsfb755hs99dRT+u1vf6v9+/dryZIlGjNmjG699VZVVFSooqJC1113nRoaGpSZmanu3bvrgw8+0I4dOxQdHa1bb71V9fX1bfazAPAdAQA4i0WLFgX69OljPf6Xf/mXwFVXXfWDz1uzZk0gLi7Oejx//vzAkCFDvvfYPXv2DCxcuNB63NDQEOjVq1dg1KhR5zXr2WZ77LHHAiNGjAhaV15eHpAUKC0ttZ6XnJwcaG5utvaZPXt2IDk5ORAIBAJ//vOfA5ICO3bssLafOHEiEBUVFfjDH/4QCAQCgRUrVgQkBfbu3Rt0rpycnBbz/+53vwsMGDAg6Hx1dXWBqKiowKZNm9rkZwGgJd6zA+C8paamtli3ZcsWFRYW6tChQ/L7/WpsbFRtba2++eYbde3a9QePWVNTo4qKCqWlpVnrwsPDNWzYsB/18s13Z/v000/1/vvvKzo6usW+R44c0RVXXCFJSk9PD3qPjdfr1TPPPKOmpiYdPHhQ4eHhQbPFxcVpwIABOnjwoLUuIiJCgwcP/sEZP/30U33++efq3r170Pra2lodOXKkzX4WAIIROwDOW7du3YIe//Wvf9Xtt9+uqVOn6te//rViY2P14YcfatKkSaqvr1fXrl0VFhbW4hf1t18Caq/ZTp06pTvuuENPPfVUi3179uzZpueOioo6rzclnzp1SqmpqXr99ddbbIuPj2/TmQD8A+/ZAdBqJSUlam5u1jPPPKP09HRdccUVOnbsWNA+8fHx8vl8QcHz7e/scblc6tmzp3bu3Gmta2xsVElJyQXNdvXVV2v//v3q27evLr/88qDl22H07fNK0scff6z+/furU6dOSk5OVmNjY9A+X375pUpLS5WSkvK954+IiFBTU1OLmQ4fPqyEhIQWM7lcrnb7WQCXOmIHQKtdfvnlamho0PPPP6+//OUv+t3vfme9cfmMm266ScePH9fChQt15MgRLV26VBs2bAja58EHH9STTz6pdevW6dChQ7r//vtVXV19QbPl5ubq5MmTuvfee7V7924dOXJEmzZt0n333RcUIWVlZcrPz1dpaan+8z//U88//7wefPBBSVL//v01atQoTZ48WR9++KE+/fRT/eIXv9BPf/pTjRo16nvP37dvX/3pT39SaWmpTpw4oYaGBo0dO1Y9evTQqFGj9MEHH+jo0aPaunWrHnjgAX3xxRft9rMALnXEDoBWGzJkiJ599lk99dRTuvLKK/X666+rsLAwaJ/k5GQtW7ZMS5cu1ZAhQ7Rr1y49/PDDQfs89NBDGjdunHJycuT1etW9e3f927/92wXNlpiYqB07dqipqUkjRozQoEGDNH36dMXExCgs7B//6xs/frz+9re/6dprr1Vubq4efPBBTZkyxdq+YsUKpaam6vbbb5fX61UgENAf//hHde7c+XvPP3nyZA0YMEDDhg1TfHy8duzYoa5du2r79u3q3bu3Ro8ereTkZE2aNEm1tbVyOp3t9rMALnWOAO96A3CJuummmzR06FAtXrzY7lEAtCPu7AAAAKMROwBCWllZmaKjo8+5lJWV2T0igBDHy1gAQlpjY2PQn1P4rr59+yo8nG/RAHBuxA4AADAaL2MBAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjPb/X33Mm+FBHvsAAAAASUVORK5CYII="/>


```python
# plotly 설치.
%pip install plotly==5.11.0
```

<pre>
Collecting plotly==5.11.0Note: you may need to restart the kernel to use updated packages.

  Downloading plotly-5.11.0-py2.py3-none-any.whl (15.3 MB)
     --------------------------------------- 15.3/15.3 MB 40.9 MB/s eta 0:00:00
Collecting tenacity>=6.2.0
  Downloading tenacity-8.1.0-py3-none-any.whl (23 kB)
Installing collected packages: tenacity, plotly
Successfully installed plotly-5.11.0 tenacity-8.1.0
</pre>

```python
# fraud_reported 열의 분포를 원형 그래프로 그려 보세요.(pie 함수 이용)
import plotly.express as px

# your code here
fig = px.pie(df['fraud_reported'], names='fraud_reported')
fig.show()
```

![](../assets/images/newplot_01_circle01.png){: .align-center}

##### jupyter 파일로는 상관없는데 md로 할 때..

파일 용량이 괜히 커지고 해서 삭제해서 이미지로 대체 양해바람


## feature간 상관관계 확인



여기서 우리는 상관 다이어그램을 그리기 위하여 plotly 라이브러리를 사용합니다.



상관 행렬은 변수 간의 관계, 즉 다른 변수가 변경될 때 한 변수가 어떻게 변경되는지를 보여주는 테이블입니다. 5개의 변수가 있는 경우 상관 행렬에는 5 곱하기 5 또는 25개의 항목이 있으며 각 항목은 두 변수 간의 상관 관계를 보여줍니다.



기계 학습 알고리즘의 정확도는 알고리즘이 얼마나 잘 수행되고 있는지, 즉 알고리즘이 데이터 포인트를 올바르게 분류하는 빈도를 측정하는 것입니다. 정확도는 다음과 같이 주어집니다:



![정확도](https://miro.medium.com/max/1050/1*O5eXoV-SePhZ30AbCikXHw.png)



정밀도는 관련성 있는 결과의 %를 의미하고, 재현율은 알고리즘에 의해 올바르게 분류된 전체 관련 결과의 %를 의미합니다.





![Precision and Recall](https://miro.medium.com/max/1050/1*pOtBHai4jFd-ujaNXPilRg.png)





True positive: 모델이 긍정 클래스를 올바르게 예측합니다.



True negative: 모델이 부정 클래스를 올바르게 예측합니다.



False positive: 모델이 긍정 클래스를 잘못 예측합니다.



False negative: 모델이 부정 클래스를 잘못 예측합니다.


우리는  plotly 라이브러리를 사용합니다. <br>

라이브러리가 설치되어 있지 않은 경우 터미널에서 다음 단계를 수행하여 설치해주세요: <br>



pip install plotly **<원형 그릴때 설치완료>**



```python
import plotly.express as px
import plotly.graph_objects as go

# pandas의 corr() 함수를 사용하여 상관 행렬 가져오기
corr_matrix = df.corr()

fig = go.Figure(data = go.Heatmap(
                                z = corr_matrix.values,
                                x = list(corr_matrix.columns),
                                y = list(corr_matrix.index)))

fig.update_layout(title = 'Correlation_Insurance_Fraud')

fig.show()
```

![](../assets/images/Correlation_Insurance_Fraud.png){: .align-center}

##### jupyter 파일로는 상관없는데 md로 할 때..

파일 용량이 괜히 커지고 해서 삭제해서 이미지로 대체 양해바람



```python
# fraud_reported를 Target으로 Dataset 나누기

X = df.loc[:, (df.columns != 'Unnamed: 0') & (df.columns != 'fraud_reported')] # your code here
y = df.loc[:, ['fraud_reported']] # your code here
```


```python
from sklearn.preprocessing import StandardScaler

# Data 정규화 하기
sc = StandardScaler()
X = sc.fit_transform(X)
```


```python
# 데이터를 test와 train 데이터로 나눕니다.
# your code here
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 랜덤 포레스트 분류기 적용하기



랜덤 포레스트는 많은 의사 결정 트리의 결정을 결합하여 데이터 포인트의 클래스를 결정하는 분류 알고리즘입니다. 우리는 다양한 트리를 기반으로 결정을 내리고 과반수 투표를 수행하고 최종 클래스를 결정합니다.



```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 1)
# your code here
rfc.fit(x_train, y_train)
rfc
```

<pre>
C:\Users\User\AppData\Local\Temp\ipykernel_13696\2780899869.py:4: DataConversionWarning:

A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().

</pre>
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=1)</pre></div></div></div></div></div>


### Task 3: 랜덤 포레스트 분류기를 사용하여 훈련 데이터를 예측하고 결과를 변수 preds에 저장




```python
# X_test로 test 결과값 만들기

# your code here
y_pred = rfc.predict(x_test)
```


```python
y_pred
```

<pre>
array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1,
       0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int64)
</pre>

```python
# score 함수로 Mean accuracy 확인하기 
# your code here
score = accuracy_score(y_test, y_pred)
print(score*100)

# classification_report 함수로 모델 평가하기
# your code here
print(classification_report(y_test, y_pred, target_names=['0', '1']))
```

<pre>
82.33333333333334
              precision    recall  f1-score   support

           0       0.86      0.90      0.88       220
           1       0.70      0.60      0.64        80

    accuracy                           0.82       300
   macro avg       0.78      0.75      0.76       300
weighted avg       0.82      0.82      0.82       300

</pre>

```python
from sklearn.metrics import roc_auc_score
# auc  score 확인하기 

# your code here
roc_auc_score(y_test, y_pred)
```

<pre>
0.7522727272727273
</pre>

```python
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
f, ax = plt.subplots(figsize=(10, 10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1kAAANXCAYAAADHC5VDAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAC0nElEQVR4nOzdd3iU1dbG4ScJJPTQi4B0KYKAKIgC0gRRQBQpgoIgKNLx2BW7cjwepSMIAoLioaogiEpVECuCoBTpRXpJaElI5v3+WF8SIsUEZvJO+d3XlSvz7kwmi0DCPLP3XjvMcRxHAAAAAACvCHe7AAAAAAAIJoQsAAAAAPAiQhYAAAAAeBEhCwAAAAC8iJAFAAAAAF5EyAIAAAAALyJkAQAAAIAXEbIAAAAAwIsIWQAAAADgRYQsAMhkpUuX1oMPPuh2GSGnYcOGatiwodtl/KOXXnpJYWFhOnz4sNul+J2wsDC99NJLXnmsHTt2KCwsTJMnT/bK4wHAuQhZAILK5MmTFRYWlvKWJUsWFS9eXA8++KD27t3rdnl+7dSpU3r11Vd13XXXKUeOHIqOjlb9+vU1ZcoUOY7jdnnp8scff+ill17Sjh073C7lPElJSZo0aZIaNmyo/PnzKyoqSqVLl1a3bt30888/u12eV0ybNk3Dhg1zu4w0/LEmAMEvi9sFAIAvvPLKKypTpozi4uL0/fffa/LkyVqxYoXWr1+vbNmyuVrbpk2bFB7uX69xHThwQE2aNNGGDRvUsWNH9e3bV3FxcZo9e7a6du2qBQsW6KOPPlJERITbpV7SH3/8oZdfflkNGzZU6dKl03zsq6++cqcoSWfOnNE999yjhQsXqkGDBnr22WeVP39+7dixQzNmzNAHH3ygXbt2qUSJEq7V6A3Tpk3T+vXrNXDgQJ88/pkzZ5QlS8aeulysplKlSunMmTPKmjWrFysEAEPIAhCUWrRooRtuuEGS1KNHDxUsWFBvvvmm5s6dq/bt27taW1RUVKZ/zbi4OEVGRl403HXt2lUbNmzQJ598otatW6eM9+/fX0888YT++9//qmbNmnrqqacyq2RJNruWM2dOrzxWZGSkVx7ncjzxxBNauHChhg4det6T/RdffFFDhw7N1Hocx1FcXJyyZ8+eqV/3cng8HiUkJChbtmxefYEkLCzM9RdcAAQv/3opFQB8pH79+pKkrVu3phnfuHGj7r33XuXPn1/ZsmXTDTfcoLlz5573+cePH9egQYNUunRpRUVFqUSJEurSpUuafTPx8fF68cUXVb58eUVFRalkyZJ68sknFR8fn+axzt2T9fPPPyssLEwffPDBeV/zyy+/VFhYmD7//POUsb1796p79+4qUqSIoqKidO2112rixIlpPm/ZsmUKCwvT//73Pz3//PMqXry4cuTIodjY2At+b77//nt9+eWXevDBB9MErGRDhgxRhQoV9Oabb+rMmTOSUvez/Pe//9XQoUNVqlQpZc+eXbfeeqvWr19/3mOk5/ucvNRz+fLl6t27twoXLpwys7Nz50717t1bFStWVPbs2VWgQAG1a9cuzbLAyZMnq127dpKkRo0apSwZXbZsmaTz92Qlf59mzJih119/XSVKlFC2bNnUpEkTbdmy5bw/w+jRo1W2bFllz55dtWvX1rfffpuufV579uzRuHHjdNttt11whiciIkKPP/74ebNYx48f14MPPqi8efMqOjpa3bp10+nTp9PcZ9KkSWrcuLEKFy6sqKgoValSRe++++55X6N06dJq2bKlvvzyS91www3Knj27xo0bl6HHkKQvvvhCt956q3Lnzq08efLoxhtv1LRp0yTZ93f+/PnauXNnyvf+3NnE9P58hIWFqW/fvvroo4907bXXKioqSgsXLkz52Ll7sk6cOKGBAwem/FwWLlxYt912m1avXv2PNV1sT9bGjRvVvn17FSpUSNmzZ1fFihX13HPPXfD7AQAXw0wWgJCQ/GQ8X758KWO///67brnlFhUvXlxPP/20cubMqRkzZqhNmzaaPXu27r77bknSyZMnVb9+fW3YsEHdu3fX9ddfr8OHD2vu3Lnas2ePChYsKI/Ho9atW2vFihV6+OGHVblyZa1bt05Dhw7V5s2b9emnn16wrhtuuEFly5bVjBkz1LVr1zQfmz59uvLly6fmzZtLsiV9N910U8qT0EKFCumLL77QQw89pNjY2POewL/66quKjIzU448/rvj4+IvO5MybN0+S1KVLlwt+PEuWLOrUqZNefvllrVy5Uk2bNk352JQpU3TixAn16dNHcXFxGj58uBo3bqx169apSJEiGfo+J+vdu7cKFSqkF154QadOnZIk/fTTT/ruu+/UsWNHlShRQjt27NC7776rhg0b6o8//lCOHDnUoEED9e/fXyNG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E3311Vd666239Oabb2rOnDlq0aLFFdcNAJeLkAUg6EVERGjIkCFq1KiRRo0apaeffjrlle6sWbOmaeRwIeXKlbtgx7y/32ft2rVq0qTJZS0t6tChg15++WXNnj1bRYoUUWxsrDp27Jjy8UKFCil37txKSkr6x3ozqmXLlhoyZIimTJlywZCVlJSkadOmKV++fLrlllvSfOzPP/887/6bN29OmeHJyPf5UmbNmqWuXbvq7bffThmLi4vT8ePH09zPF8u6SpUqJclm5Ro1apQynpiYqB07dpwXbv+uRYsWioiI0Icffpjh5heXMm/ePMXHx2vu3LlpZr0utTT1ch+jXLlykqT169df8sWHi33/r/Tn41KKFSum3r17q3fv3jp48KCuv/56vf766ykhK71fL/nf6j/9rANAerAnC0BIaNiwoWrXrq1hw4YpLi5OhQsXVsOGDTVu3Djt27fvvPsfOnQo5Xbbtm21du1affLJJ+fdL3lWoX379tq7d6/Gjx9/3n3OnDmT0iXvYipXrqxq1app+vTpmj59uooVK5Ym8ERERKht27aaPXv2BZ8EnltvRt18881q2rSpJk2alKZdfLLnnntOmzdv1pNPPnneDMOnn36aZk/Vjz/+qB9++CHlCW5Gvs+XEhERcd7M0siRI5WUlJRmLPlMrb+Hrytxww03qECBAho/frwSExNTxj/66KOUJYWXUrJkSfXs2VNfffWVRo4ced7HPR6P3n77be3ZsydDdSXPdP196eSkSZO8/hjNmjVT7ty5NWTIEMXFxaX52LmfmzNnzgsu37zSn48LSUpKOu9rFS5cWFdddVWatvAXq+nvChUqpAYNGmjixInatWtXmo95a1YTQOhgJgtAyHjiiSfUrl07TZ48Wb169dLo0aNVr149VatWTT179lTZsmV14MABrVq1Snv27NHatWtTPm/WrFlq166dunfvrlq1auno0aOaO3euxo4dq+rVq+uBBx7QjBkz1KtXLy1dulS33HKLkpKStHHjRs2YMSPlfKJL6dChg1544QVly5ZNDz300HkHB//73//W0qVLVadOHfXs2VNVqlTR0aNHtXr1ai1atEhHjx697O/NlClT1KRJE911113q1KmT6tevr/j4eM2ZM0fLli1Thw4d9MQTT5z3eeXLl1e9evX06KOPKj4+XsOGDVOBAgX05JNPptwnvd/nS2nZsqWmTp2q6OhoValSRatWrdKiRYtUoECBNPerUaOGIiIi9OabbyomJkZRUVEpZ0BdrsjISL300kvq16+fGjdurPbt22vHjh2aPHmyypUrl66Zkrfffltbt25V//79NWfOHLVs2VL58uXTrl27NHPmTG3cuDHNzGV6NGvWTJGRkWrVqpUeeeQRnTx5UuPHj1fhwoUvGGiv5DHy5MmjoUOHqkePHrrxxhvVqVMn5cuXT2vXrtXp06dTznmrVauWpk+frscee0w33nijcuXKpVatWnnl5+PvTpw4oRIlSujee+9V9erVlStXLi1atEg//fRTmhnPi9V0ISNGjFC9evV0/fXX6+GHH1aZMmW0Y8cOzZ8/X2vWrMlQfQBCnCs9DQHAR5LbgP/000/nfSwpKckpV66cU65cuZQW4Vu3bnW6dOniFC1a1MmaNatTvHhxp2XLls6sWbPSfO6RI0ecvn37OsWLF3ciIyOdEiVKOF27dk3TTj0hIcF58803nWuvvdaJiopy8uXL59SqVct5+eWXnZiYmJT7/b2Fe7I///zTkeRIclasWHHBP9+BAwecPn36OCVLlnSyZs3qFC1a1GnSpInz3nvvpdwnuTX5zJkzM/S9O3HihPPSSy851157rZM9e3Ynd+7czi233OJMnjz5vBbWye2v33rrLeftt992SpYs6URFRTn169d31q5de95jp+f7fKm/u2PHjjndunVzChYs6OTKlctp3ry5s3Hjxgt+L8ePH++ULVvWiYiISNPO/WIt3P/+fbpQa2/HcZwRI0Y4pUqVcqKiopzatWs7K1eudGrVquXcfvvt6fjuOk5iYqIzYcIEp379+k50dLSTNWtWp1SpUk63bt3StHdPbuF+6NChNJ+f/P3Zvn17ytjcuXOd6667zsmWLZtTunRp580333QmTpx43v1KlSrl3HnnnResK72PkXzfm2++2cmePbuTJ08ep3bt2s7HH3+c8vGTJ086nTp1cvLmzetIStM6Pb0/H5Iu2pZd57Rwj4+Pd5544gmnevXqTu7cuZ2cOXM61atXd8aMGZPmcy5W08X+ntevX+/cfffdTt68eZ1s2bI5FStWdAYPHnzBegDgYsIchzlwAEDG7NixQ2XKlNFbb72lxx9/3O1yXOHxeFSoUCHdc889F1wGBwAIXezJAgDgH8TFxZ23L2fKlCk6evSoGjZs6E5RAAC/xZ4sAAD+wffff69BgwapXbt2KlCggFavXq33339fVatWVbt27dwuDwDgZwhZAAD8g9KlS6tkyZIaMWKEjh49qvz586tLly7697//rcjISLfLAwD4GVf3ZH3zzTd666239Msvv2jfvn365JNP1KZNm0t+zrJly/TYY4/p999/V8mSJfX888/rwQcfzJR6AQAAAOCfuLon69SpU6pevbpGjx6drvtv375dd955pxo1aqQ1a9Zo4MCB6tGjh7788ksfVwoAAAAA6eM33QXDwsL+cSbrqaee0vz589McxNmxY0cdP35cCxcuzIQqAQAAAODSAmpP1qpVq9S0adM0Y82bN9fAgQMv+jnx8fFpTn73eDw6evSoChQokK4DJAEAAAAEJ8dxdOLECV111VUKD/feIr+ACln79+9XkSJF0owVKVJEsbGxOnPmjLJnz37e5wwZMkQvv/xyZpUIAAAAIMDs3r1bJUqU8NrjBVTIuhzPPPOMHnvssZTrmJgYXX311dq9e7fy5MnjYmUAAAAAfC0mRvrwQ2nsWGnXLqmkdmq22uqLBv/WDU/V1p13llTu3Lm9+jUDKmQVLVpUBw4cSDN24MAB5cmT54KzWJIUFRWlqKio88bz5MlDyAIAAACC1Nat0siR0sSJ0okTNlYj7w4tcVopX8xO3XjsJcXWXS5JXt9G5Gp3wYyqW7euFi9enGbs66+/Vt26dV2qCAAAAIC/cBxp+XLp7rulChWk4cMtYFWpIn38xnb9kvtW5YvZaR/84gspIsIndbg6k3Xy5Elt2bIl5Xr79u1as2aN8ufPr6uvvlrPPPOM9u7dqylTpkiSevXqpVGjRunJJ59U9+7dtWTJEs2YMUPz5893648AAAAAwGXx8dL06dKwYdKvv6aOt2ghDRwo3VZum8IaNZR275auuUZaulS66iopNtYn9bgasn7++Wc1atQo5Tp571TXrl01efJk7du3T7t27Ur5eJkyZTR//nwNGjRIw4cPV4kSJTRhwgQ1b94802sHAAAA4K5Dh2yv1Zgx0v79NpY9u9SlizRggFS5smzdYKNGFrAqVrSAVayYT+vym3OyMktsbKyio6MVExPDniwAAAAgAK1fb7NWH35os1iSVLy41Lev1LOnVKDAOXd+7DFp6FCpUiULWEWLpnzIV9kgoBpfAAAAAAhNHo+0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4aAQAICf8Hik+fOloUNt31WyOnWsBfs991jXQCAk7NplAWvbNjs1e+lSm9IFvMxX2cC15YIJCQn65Zdf1LRp09RiwsPVtGlTrVq16oKfc/PNN+uXX35JWVK4bds2LViwQHfcccdFv058fLxiY2PTvAEA4C9OnpRGjZIqVpRat7bnkhERUocO0qpV0vff220CFkLGrl22RHDbNjtFe9kyAhYCjmvLBQ8fPqykpCQVKVIkzXiRIkW0cePGC35Op06ddPjwYdWrV0+O4ygxMVG9evXSs88+e9GvM2TIEL388sterR0AgCu1c6eFq/HjpZgYG8ubV3r4YalPH+nqq10tD3DPTz/ZD0i5cvaqQ8mSblcEZJjrjS8yYtmyZXrjjTc0ZswYrV69WnPmzNH8+fP16quvXvRznnnmGcXExKS87d69OxMrBgAgleNI330ntWtnL9D/978WsCpUkEaPti0ob75JwEKIa9tWmjHDZrAIWAhQrs1kFSxYUBERETpw4ECa8QMHDqjoRU5QHDx4sB544AH16NFDklStWjWdOnVKDz/8sJ577jmFh5+fGaOiohQVFeX9PwAAAOl09qw0a5Y1szi3iW6TJrbfqkUL6QL/hQGhY8cOKTJSuuoqu27b1tVygCvl2q/0yMhI1apVK00TC4/Ho8WLF6tu3boX/JzTp0+fF6QiIiIkSS727wAA4IKOHpX+/W/bt9+pkwWsqCg70+q336RFi+ysKwIWQtr27dKtt1qji3373K4G8ApXW7g/9thj6tq1q2644QbVrl1bw4YN06lTp9StWzdJUpcuXVS8eHENGTJEktSqVSu98847qlmzpurUqaMtW7Zo8ODBatWqVUrYAgDAbRs3SsOHSx98IJ05Y2NFitj5Vr16SRc4pQQITdu2WZOL3bula66xNptAEHA1ZHXo0EGHDh3SCy+8oP3796tGjRpauHBhSjOMXbt2pZm5ev755xUWFqbnn39ee/fuVaFChdSqVSu9/vrrbv0RAACQZPutvv7algR+8UXqeI0a0sCBUseONosF4P9t3WoBa88ea6+5ZEnqckEgwLl6TpYbOCcLAOBNZ85IH35o4eqPP2wsLMzasQ8aJDVoYNcAzrFliy0P3LNHqlTJAlaxYm5XhRDkq2zg6kwWAACBat8+6wg4dqx05IiN5colde8u9e9v3acBXMCWLTaDtXev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srKzzYC1Zs2FgNW+fQlfFAAAwLMZhqHs7Gzl5ORYXQosUL58efm4ZLraBYQsuTBklStnjlp9/70ZsFhFEAAAoERlZmYqOTlZZ8+etboUWMTLy0vXXXedKleu7LprGoZhuOxqpUB6erqCgoKUlpamwMBAGYb5eKr0dOnHH6Ubb3RBEYcPS6GhLrgQAACA57LZbNq3b598fHx07bXXytfXV15eXlaXBRcyDEPHjx/X2bNn1bBhw3wjWpdmA2fx+JGs5GQzYHl7Sw0blsAFzpyRxo+XJk6UctMzAQsAAKDEZWZmymazKTQ0VBUrVrS6HFjk2muvVVJSkrKyslw2bdDjQ1buVMEGDSQ/Pyef/PRpqXt36ZtvzAdxff65ky8AAACAK/H2Zq03T2bF6CUhq6Tuxzp9WrrzTmnDBikwUHruOSdfAAAAAEBp5PGxvkRC1p9/St26mQErKMhcrr1tWydeAAAAAEBpxUiWs0NWbsDauPFCwGrd2kknBwAAAFDaMZLl7JA1cKAZsKpUkVavJmABAACgWBISEuTj46Pu3bvn+2zdunXy8vLSqVOn8n0WFhamV199Nc++tWvX6s4779Q111yjihUr6v/+7//05JNP6siRIyVUvZSRkaERI0bommuuUeXKldW7d2+lpqZe9hgvL68CXy+//LK9TVhYWL7Pp06dWmLfozg8OmSdOiWlpJjbjRo56aSTJpnLFK5eLbVq5aSTAgAAwNPExcVp1KhR+uabb3T06NFin+edd95RVFSUgoOD9d///le7du3S22+/rbS0NE2fPt2JFec1ZswYff7551q8eLHWr1+vo0eP6t57773sMcnJyXlec+fOlZeXl3r37p2n3QsvvJCn3ahRo0rsexSHR08XzB3Fql3bXJui2AxDyl21pGlTadcu88HDAAAAKFUMQ7LiucQVK17452JRnD59WvHx8dqyZYtSUlI0b948Pfvssw5f97ffftPo0aM1evRozZw5074/LCxMt912W4EjYc6QlpamuLg4ffjhh+rUqZMk6b333tMNN9yg//3vf7r55psLPC44ODjP+08//VQdO3ZU/fr18+wPCAjI17Y08eiRLKdMFTx1SoqKkr799sI+AhYAAECpdPas+ehSV78cDXaLFi1So0aNFBERoYEDB2ru3LkyDMPh77t48WJlZmbq6aefLvDzKlWqFHpst27d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wN7eMAzZbDYdPHhQN9xwQ4G1paWlqXLlyrLZbMrIyNCtt96qd999V+np6Tp69KhuueWWPO1vueUW/fDDD5LMqX533HGHIiIi1LVrV911113q3LnzVf2uBgwYoL///e9688035efnpwULFuj++++Xt7e3/Xtu3Lgxz8hVTk7OZX9vkhQREaHPPvtMGRkZ+s9//qMdO3Zo1KhRedrMmjVLc+fO1aFDh3Tu3DllZmaqefPml633hx9+0P79+xUQEJBnf0ZGhn755Zdi/AYAAEVByAIAD1epUiVdf/31efYlJSXprrvu0mOPPabJkyerWrVq+vbbbxUdHa3MzMwCw8KECRP0wAMPaPny5friiy8UGxurhQsXqlevXjp9+rQeeeQRjR49Ot9xderUKbS2gIAAbdu2Td7e3qpVq5YqVKggSUpPT7/i92rRooUOHjyoL774QqtXr1bfvn0VFRWljz/++IrHFubuu++WYRhavny5WrdurQ0bNmjmzJn2z0+fPq2JEyfq3nvvzXesv79/oef19fW1/xlMnTpV3bt318SJEzVp0iRJ0sKFC/XUU09p+vTpioyMVEBAgF5++WV9//33l6339OnTatmyZZ5wm6u0LG4CAGURIQsAkM/WrVtls9k0ffp0+yhN7v0/lxMeHq7w8HCNGTNG/fv313vvvadevXqpRYsW2rVrV74wdyXe3t4FHhMYGKiQkBBt3LhRHTp0sO/fuHGj2rRpk6ddv3791K9fP/Xp00ddu3bV77//rmrVquU5X+79Tzk5OZetx9/fX/fee68WLFig/fv3KyIiQi1atLB/3qJFCyUmJjr8PS/13HPPqVOnTnrsscfs37Ndu3YaPny4vc2lI1G+vr756m/RooXi4+NVo0YNBQYGXlVNAICiY+ELAEA+119/vbKysvTvf/9bBw4c0AcffKC333670Pbnzp3TyJEjtW7dOv3666/auHGjNm/ebJ8GOHbsWH333XcaOXKkduzYoX379unTTz91eOGLi/3jH//QtGnTFB8fr8TERI0bN047duzQ448/LkmaMWOGPvroI+3Zs0d79+7V4sWLFRwcXOADlGvUqKEKFSpo5cqVSk1NVVpaWqHXHTBggJYvX665c+faF7zINX78eL3//vuaOHGifv75Z+3evVsLFy7Uc88959B3i4yMVNOmTfXiiy9Kkho2bKgtW7Zo1apV2rt3r55//nlt3rw5zzFhYWHauXOnEhMTdeLECWVlZWnAgAGqXr26evTooQ0bNujgwYNat26dRo8erd9++82hmgAARUfIAgDk06xZM82YMUPTpk3TjTfeqAULFuRZ/vxSPj4+OnnypAYPHqzw8HD17dtX3bp108SJEyVJTZs21fr167V37161b99eN910k8aPH6+QkJBi1zh69GjFxMToySefVJMmTbRy5Up99tlnatiwoSRzquFLL72kVq1aqXXr1kpKStKKFSvsI3MXK1eunF5//XW98847CgkJUY8ePQq9bqdOnVStWjUlJibqgQceyPNZly5dtGzZMn355Zdq3bq1br75Zs2cOVN169Z1+PuNGTNG7777rg4fPqxHHnlE9957r/r166e2bdvq5MmTeUa1JOnvf/+7IiIi1KpVK1177bXauHGjKlasqG+++UZ16tTRvffeqxtuuEHR0dHKyMhgZAsASpCXYRiG1UUAAAAAQFnBSBYAAAAAOBEhCwAAAACciJAFAAAAAE5EyAIAAAAAJyJkAQAAAIATEbIAAAAAwIkIWQAAAADgRIQsAAAAAHAiQhYAAAAAOBEhCwAAAACciJAFAAAAAE70/+ob/CmgmAe6AAAAAElFTkSuQmCC"/>

### Task 4: 랜덤 포레스트 분류기의 결과를 보았습니다. 이제 의사 결정 트리 분류기를 사용하여 데이터를 분류해 보세요.



```python
dtc = DecisionTreeClassifier()

# your code here
dtc.fit(x_train, y_train)
# your code here
y_pred = dtc.predict(x_test)
# your code here
score = accuracy_score(y_test, y_pred)
print(score*100)

# your code here
print(classification_report(y_test, y_pred, target_names=['0', '1']))
```

<pre>
73.0
              precision    recall  f1-score   support

           0       0.80      0.84      0.82       220
           1       0.49      0.44      0.46        80

    accuracy                           0.73       300
   macro avg       0.65      0.64      0.64       300
weighted avg       0.72      0.73      0.72       300

</pre>
### Task 5: 의사 결정 트리 분류기의 결과를 보았습니다. 이제 로지스틱 회귀 분류기를 사용하여 데이터를 분류해 보세요.



```python
lr = LogisticRegression()

# your code here
lr.fit(x_train, y_train)
# your code here
y_pred = lr.predict(x_test)
# your code here
score = accuracy_score(y_test, y_pred)
print(score*100)

# your code here
print(classification_report(y_test, y_pred, target_names=['0', '1']))
```

<pre>
83.66666666666667
              precision    recall  f1-score   support

           0       0.87      0.92      0.89       220
           1       0.73      0.61      0.67        80

    accuracy                           0.84       300
   macro avg       0.80      0.77      0.78       300
weighted avg       0.83      0.84      0.83       300

</pre>
<pre>
C:\Users\User\.conda\envs\myai\lib\site-packages\sklearn\utils\validation.py:1111: DataConversionWarning:

A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().

</pre>
### Task 6: Gradient Boosting Classifier로 데이터를 분류해 보세요.



```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
# your code here
gb.fit(x_train, y_train)
# your code here
y_pred = gb.predict(x_test)
# your code here
score = accuracy_score(y_test, y_pred)
print(score*100)

# your code here
print(classification_report(y_test, y_pred, target_names=['0', '1']))
```

<pre>
82.0
              precision    recall  f1-score   support

           0       0.87      0.89      0.88       220
           1       0.67      0.64      0.65        80

    accuracy                           0.82       300
   macro avg       0.77      0.76      0.77       300
weighted avg       0.82      0.82      0.82       300

</pre>
<pre>
C:\Users\User\.conda\envs\myai\lib\site-packages\sklearn\ensemble\_gb.py:570: DataConversionWarning:

A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().

</pre>

```python
%pip install xgboost
```

<pre>
Collecting xgboost
  Downloading xgboost-1.7.1-py3-none-win_amd64.whl (89.1 MB)
     --------------------------------------- 89.1/89.1 MB 34.4 MB/s eta 0:00:00
Requirement already satisfied: numpy in c:\users\user\.conda\envs\myai\lib\site-packages (from xgboost) (1.23.4)
Requirement already satisfied: scipy in c:\users\user\.conda\envs\myai\lib\site-packages (from xgboost) (1.9.3)
Installing collected packages: xgboost
Successfully installed xgboost-1.7.1
Note: you may need to restart the kernel to use updated packages.
</pre>
### Task 7: XGB Classifier로 데이터를 분류해 보세요.



```python
from xgboost import XGBClassifier

xgb = XGBClassifier()
# your code here
xgb.fit(x_train, y_train)
# your code here
y_pred = xgb.predict(x_test)
# your code here
score = accuracy_score(y_test, y_pred)
print(score*100)

# your code here
print(classification_report(y_test, y_pred, target_names=['0', '1']))
```

<pre>
81.66666666666667
              precision    recall  f1-score   support

           0       0.85      0.91      0.88       220
           1       0.70      0.55      0.62        80

    accuracy                           0.82       300
   macro avg       0.77      0.73      0.75       300
weighted avg       0.81      0.82      0.81       300

</pre>
### Conclusion





인공 지능은 다양한 현대 사회의 문제를 해결하는데 널리 사용되고 있습니다. 이 노트북에서는 랜덤 포레스트 알고리즘을 사용하여 사기 탐지에 인공 지능을 사용하는 방법의 예를 보았습니다. 같은 목적으로 다른 모델을 사용할 수도 있습니다. 추가적으로 다른 모델을 적용하였을 때 정확도를 비교해 보는 연습을 해보세요.


#### 그 외 참고한 리스트들..



https://stackoverflow.com/questions/26266362/how-do-i-count-the-nan-values-in-a-column-in-pandas-dataframe

isna().sum() 등 null값 조회 및 분석,정리 참조..<br>

<br>

https://seaborn.pydata.org/generated/seaborn.countplot.html<br>

countplot 확인<br>

<br>

https://plotly.com/python/pie-charts/<br>

plotly, pie함수의 원형 그래프 그리는 방법 확인<br>

<br>

https://hmiiing.tistory.com/entry/jupyter-notebook%EC%97%90-plotly-%EA%B7%B8%EB%9E%98%ED%94%84%EA%B0%80-%EB%82%98%EC%98%A4%EC%A7%80-%EC%95%8A%EC%9D%84-%EB%95%8C-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95<br>

만일 plotly 그래프가 안나올때;;;;<br>

<br>

https://jhryu1208.github.io/data/2020/12/25/pandas_practice_loc/<br>

특정 열, 행 자르기 및 조회<br>

<br>

https://datascienceschool.net/03%20machine%20learning/09.04%20%EB%B6%84%EB%A5%98%20%EC%84%B1%EB%8A%A5%ED%8F%89%EA%B0%80.html<br>

classification_report를 통한 평가방법 확인 및 정확도 평가 확인<br>

<br>

https://sumniya.tistory.com/26<br>

f1등 인공지능 평가에 관한 수학기초<br>

<br>


#### 마무리;;

- (임시) 바삐 작성한 내용이라 좀 더 조절없이 그대로 하게된 점 아쉽..

- 랜덤 포레스트에 이어 돌리고 평가하고 하는 방식이 뭔가 복붙 술술이 신기.

- 평가 기준 리스트에서 정밀도, 재현율, f1, 감지량(support) 이 데이터 중에서 f1 제일 높은쪽이 좋은 모델로..

- 특히 f1 관련 성능 정보에 대해서는 추후 찾아서 정리.



- f1이란..<br>

Precision과 Recall의 조화평균, 데이터 label이 불균형 구조일 때, 모델의 성능을 정확하게 평가할 수 있으며, 성능을 하나의 숫자로 표현할 수 있다 등..


#### 주의사항



본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.

