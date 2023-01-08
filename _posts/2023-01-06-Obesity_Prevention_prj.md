---
layout: single
title:  "Intel AI 캡스톤1 비만 예방을 위한 행동분류예측"
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


비만 예방을 위한 분석 AI모델 구축
==================================



1. 비만 예방을 위해 비만에 걱정하시는 분들을 위한 AI솔루션
2. 어떻게 예방할지 AI를 통해 적기에 사람들에게 솔루션 제공이 중요.
3. 백문이 불여일견 진행시작


## 진행 순서



1. .csv 파일 읽어오기 (test.csv, train.csv): df_test / df_train

2. Data 크기 확인하기(test, train)

3. Data SET 첫 5행 확인하기(test, train)

4. Data SET 통계값, Null 값 확인하기 (null 값이 있는 경우 행 삭제)

5. “activity” 컬럼의 Unique 값, Unique count 값 확인 하기

6. “activity” 컬럼의 Unique count 값 그래프 그리기

7. rn 열 drop (df_test / df_train)

8. “activity” 컬럼에 대하여 인코딩 하기

9. df_train 으로 X = “특성 컬럼”, y = “activity” 만들기

10. X_train, X_test, y_train, y_test 만들기

11. X_train, X_test 정규화 하기(옵션)

12. AI Model : 분류

13. .fit // .predict // score (정확도, 정밀도, 재현율, F1 Score)

14. df_test 에 대한 분류 하여 “activity” 컬럼 작성하여 test_result.csv 파일 만들기 (6가지 클래스로만...)  

<'df_test'의 경우 정답 없는 문제지라 AI에게 테스트로 predict로 던져보는... 디코딩해서 만든게 test_result.csv>

15. 결과 보고


#### 0. 필요한 라이브러리..


- 두말할 거 없이 이번 캡스톤1에 필요한 라이브러리를 가져오는건 기본.

- 여기서 KNN, GNB, MNB, LR, RFC 그 외 XGBoost 등등 들어본 머신러닝들을 활용.  

이웃, 네이브베어, 선형모델, 앙상블 랜덤포레스트분류 등  



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

from sklearn import tree # Decision Tree
from sklearn import svm # support vector machine
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.naive_bayes import GaussianNB # GNB
from sklearn.naive_bayes import MultinomialNB # MNB
from sklearn.linear_model import LogisticRegression # LR..
from sklearn.ensemble import RandomForestClassifier # RFC

from xgboost import XGBClassifier # Xgboost
```

#### 1. .csv 파일 읽어오기

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
      <th>rn</th>
      <th>activity</th>
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>STANDING</td>
      <td>0.279</td>
      <td>-0.0196</td>
      <td>-0.1100</td>
      <td>-0.997</td>
      <td>-0.967</td>
      <td>-0.983</td>
      <td>-0.997</td>
      <td>-0.966</td>
      <td>...</td>
      <td>0.146</td>
      <td>-0.217</td>
      <td>-0.5640</td>
      <td>-0.2130</td>
      <td>-0.2310</td>
      <td>0.0146</td>
      <td>-0.190</td>
      <td>-0.852</td>
      <td>0.182</td>
      <td>-0.0430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>STANDING</td>
      <td>0.277</td>
      <td>-0.0127</td>
      <td>-0.1030</td>
      <td>-0.995</td>
      <td>-0.973</td>
      <td>-0.985</td>
      <td>-0.996</td>
      <td>-0.974</td>
      <td>...</td>
      <td>0.121</td>
      <td>0.349</td>
      <td>0.0577</td>
      <td>0.0807</td>
      <td>0.5960</td>
      <td>-0.4760</td>
      <td>0.116</td>
      <td>-0.852</td>
      <td>0.188</td>
      <td>-0.0347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>STANDING</td>
      <td>0.277</td>
      <td>-0.0147</td>
      <td>-0.1070</td>
      <td>-0.999</td>
      <td>-0.991</td>
      <td>-0.993</td>
      <td>-0.999</td>
      <td>-0.991</td>
      <td>...</td>
      <td>0.740</td>
      <td>-0.564</td>
      <td>-0.7660</td>
      <td>0.1060</td>
      <td>-0.0903</td>
      <td>-0.1320</td>
      <td>0.499</td>
      <td>-0.850</td>
      <td>0.189</td>
      <td>-0.0351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>STANDING</td>
      <td>0.298</td>
      <td>0.0271</td>
      <td>-0.0617</td>
      <td>-0.989</td>
      <td>-0.817</td>
      <td>-0.902</td>
      <td>-0.989</td>
      <td>-0.794</td>
      <td>...</td>
      <td>0.131</td>
      <td>0.208</td>
      <td>-0.0681</td>
      <td>0.0623</td>
      <td>-0.0587</td>
      <td>0.0312</td>
      <td>-0.269</td>
      <td>-0.731</td>
      <td>0.283</td>
      <td>0.0364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>STANDING</td>
      <td>0.276</td>
      <td>-0.0170</td>
      <td>-0.1110</td>
      <td>-0.998</td>
      <td>-0.991</td>
      <td>-0.998</td>
      <td>-0.998</td>
      <td>-0.989</td>
      <td>...</td>
      <td>0.667</td>
      <td>-0.942</td>
      <td>-0.9660</td>
      <td>0.2450</td>
      <td>0.1030</td>
      <td>0.0661</td>
      <td>-0.412</td>
      <td>-0.761</td>
      <td>0.263</td>
      <td>0.0296</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 563 columns</p>
</div>


<pre>
Index(['rn', 'activity', 'tBodyAcc.mean.X', 'tBodyAcc.mean.Y',
       'tBodyAcc.mean.Z', 'tBodyAcc.std.X', 'tBodyAcc.std.Y', 'tBodyAcc.std.Z',
       'tBodyAcc.mad.X', 'tBodyAcc.mad.Y',
       ...
       'fBodyBodyGyroJerkMag.meanFreq', 'fBodyBodyGyroJerkMag.skewness',
       'fBodyBodyGyroJerkMag.kurtosis', 'angle.tBodyAccMean.gravity',
       'angle.tBodyAccJerkMean.gravityMean', 'angle.tBodyGyroMean.gravityMean',
       'angle.tBodyGyroJerkMean.gravityMean', 'angle.X.gravityMean',
       'angle.Y.gravityMean', 'angle.Z.gravityMean'],
      dtype='object', length=563)
</pre>
방금 진행한 작업은 위 head데이터에서 어떠한 목록들이 있는가 한번 확인해 보기 위해 넣어본 것.


<pre>
<bound method DataFrame.info of          rn            activity  tBodyAcc.mean.X  tBodyAcc.mean.Y  \
0         7            STANDING            0.279         -0.01960   
1        11            STANDING            0.277         -0.01270   
2        14            STANDING            0.277         -0.01470   
3        15            STANDING            0.298          0.02710   
4        20            STANDING            0.276         -0.01700   
...     ...                 ...              ...              ...   
3604  10277    WALKING_UPSTAIRS            0.357         -0.04460   
3605  10278    WALKING_UPSTAIRS            0.344          0.00479   
3606  10279    WALKING_UPSTAIRS            0.284         -0.00796   
3607  10280    WALKING_UPSTAIRS            0.207          0.02460   
3608  10281  WALKING_DOWNSTAIRS            0.393         -0.01780   

......

[3609 rows x 563 columns]>
</pre>
test도 동일하게 진행.


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
      <th>rn</th>
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>tBodyAcc.mad.Z</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.280</td>
      <td>-0.0195</td>
      <td>-0.113</td>
      <td>-0.995</td>
      <td>-0.967</td>
      <td>-0.979</td>
      <td>-0.997</td>
      <td>-0.964</td>
      <td>-0.977</td>
      <td>...</td>
      <td>0.4150</td>
      <td>-0.391</td>
      <td>-0.760</td>
      <td>-0.11900</td>
      <td>0.1780</td>
      <td>0.101</td>
      <td>0.809</td>
      <td>-0.849</td>
      <td>0.181</td>
      <td>-0.0491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.277</td>
      <td>-0.0166</td>
      <td>-0.115</td>
      <td>-0.998</td>
      <td>-0.981</td>
      <td>-0.990</td>
      <td>-0.998</td>
      <td>-0.980</td>
      <td>-0.990</td>
      <td>...</td>
      <td>0.0878</td>
      <td>-0.351</td>
      <td>-0.699</td>
      <td>0.12300</td>
      <td>0.1230</td>
      <td>0.694</td>
      <td>-0.616</td>
      <td>-0.848</td>
      <td>0.185</td>
      <td>-0.0439</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>0.277</td>
      <td>-0.0218</td>
      <td>-0.121</td>
      <td>-0.997</td>
      <td>-0.961</td>
      <td>-0.984</td>
      <td>-0.998</td>
      <td>-0.957</td>
      <td>-0.984</td>
      <td>...</td>
      <td>0.3140</td>
      <td>-0.269</td>
      <td>-0.573</td>
      <td>0.01300</td>
      <td>0.0809</td>
      <td>-0.234</td>
      <td>0.118</td>
      <td>-0.848</td>
      <td>0.189</td>
      <td>-0.0374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>0.279</td>
      <td>-0.0148</td>
      <td>-0.117</td>
      <td>-0.997</td>
      <td>-0.982</td>
      <td>-0.983</td>
      <td>-0.997</td>
      <td>-0.982</td>
      <td>-0.981</td>
      <td>...</td>
      <td>0.5610</td>
      <td>-0.779</td>
      <td>-0.940</td>
      <td>-0.00145</td>
      <td>-0.0481</td>
      <td>-0.340</td>
      <td>-0.229</td>
      <td>-0.759</td>
      <td>0.264</td>
      <td>0.0270</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>0.279</td>
      <td>-0.0145</td>
      <td>-0.107</td>
      <td>-0.998</td>
      <td>-0.986</td>
      <td>-0.993</td>
      <td>-0.998</td>
      <td>-0.985</td>
      <td>-0.995</td>
      <td>...</td>
      <td>0.6770</td>
      <td>-0.715</td>
      <td>-0.937</td>
      <td>0.02570</td>
      <td>0.0665</td>
      <td>-0.226</td>
      <td>-0.225</td>
      <td>-0.762</td>
      <td>0.262</td>
      <td>0.0294</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 562 columns</p>
</div>



<pre>
Index(['rn', 'tBodyAcc.mean.X', 'tBodyAcc.mean.Y', 'tBodyAcc.mean.Z',
       'tBodyAcc.std.X', 'tBodyAcc.std.Y', 'tBodyAcc.std.Z', 'tBodyAcc.mad.X',
       'tBodyAcc.mad.Y', 'tBodyAcc.mad.Z',
       ...
       'fBodyBodyGyroJerkMag.meanFreq', 'fBodyBodyGyroJerkMag.skewness',
       'fBodyBodyGyroJerkMag.kurtosis', 'angle.tBodyAccMean.gravity',
       'angle.tBodyAccJerkMean.gravityMean', 'angle.tBodyGyroMean.gravityMean',
       'angle.tBodyGyroJerkMean.gravityMean', 'angle.X.gravityMean',
       'angle.Y.gravityMean', 'angle.Z.gravityMean'],
      dtype='object', length=562)
</pre>

<pre>
<bound method DataFrame.info of          rn  tBodyAcc.mean.X  tBodyAcc.mean.Y  tBodyAcc.mean.Z  \
0         3            0.280         -0.01950          -0.1130   
1         5            0.277         -0.01660          -0.1150   
2         9            0.277         -0.02180          -0.1210   
3        17            0.279         -0.01480          -0.1170   
4        26            0.279         -0.01450          -0.1070   
...     ...              ...              ...              ...   
1536  10255            0.289         -0.02810          -0.0943   
1537  10270            0.377         -0.01810          -0.1100   
1538  10272            0.253         -0.02490          -0.1700   
1539  10289            0.277          0.00108          -0.0740   
1540  10294            0.192         -0.03360          -0.1060   

......

[1541 rows x 562 columns]>
</pre>
#### 2. Data 크기 확인하기

- 대략 적인 크기 및 정보는 위에 info등을 통해 나와버렸으나 자세한 부분을 여기 담음

- 세부적인 그래프 비교는 추후에..


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
      <th>rn</th>
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>tBodyAcc.mad.Z</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>...</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
      <td>3609.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5152.430590</td>
      <td>0.274544</td>
      <td>-0.017415</td>
      <td>-0.109195</td>
      <td>-0.608457</td>
      <td>-0.506265</td>
      <td>-0.614482</td>
      <td>-0.634634</td>
      <td>-0.521660</td>
      <td>-0.616047</td>
      <td>...</td>
      <td>0.128804</td>
      <td>-0.300815</td>
      <td>-0.619400</td>
      <td>0.007561</td>
      <td>0.009484</td>
      <td>0.029185</td>
      <td>-0.010632</td>
      <td>-0.496977</td>
      <td>0.060040</td>
      <td>-0.050202</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2975.767839</td>
      <td>0.063589</td>
      <td>0.042589</td>
      <td>0.056218</td>
      <td>0.439157</td>
      <td>0.501627</td>
      <td>0.399514</td>
      <td>0.413194</td>
      <td>0.485282</td>
      <td>0.394932</td>
      <td>...</td>
      <td>0.240278</td>
      <td>0.317963</td>
      <td>0.308303</td>
      <td>0.332249</td>
      <td>0.448971</td>
      <td>0.613615</td>
      <td>0.490830</td>
      <td>0.509336</td>
      <td>0.311308</td>
      <td>0.263935</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>-0.521000</td>
      <td>-1.000000</td>
      <td>-0.926000</td>
      <td>-1.000000</td>
      <td>-0.999000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-0.999000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>-0.786000</td>
      <td>-0.968000</td>
      <td>-0.995000</td>
      <td>-0.969000</td>
      <td>-0.997000</td>
      <td>-1.000000</td>
      <td>-0.993000</td>
      <td>-0.999000</td>
      <td>-1.000000</td>
      <td>-0.971000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2570.000000</td>
      <td>0.262000</td>
      <td>-0.025200</td>
      <td>-0.122000</td>
      <td>-0.992000</td>
      <td>-0.976000</td>
      <td>-0.979000</td>
      <td>-0.993000</td>
      <td>-0.976000</td>
      <td>-0.978000</td>
      <td>...</td>
      <td>-0.015800</td>
      <td>-0.533000</td>
      <td>-0.836000</td>
      <td>-0.118000</td>
      <td>-0.281000</td>
      <td>-0.478000</td>
      <td>-0.398000</td>
      <td>-0.816000</td>
      <td>-0.015600</td>
      <td>-0.122000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5158.000000</td>
      <td>0.277000</td>
      <td>-0.017200</td>
      <td>-0.109000</td>
      <td>-0.939000</td>
      <td>-0.812000</td>
      <td>-0.844000</td>
      <td>-0.946000</td>
      <td>-0.816000</td>
      <td>-0.837000</td>
      <td>...</td>
      <td>0.132000</td>
      <td>-0.341000</td>
      <td>-0.706000</td>
      <td>0.007740</td>
      <td>0.009830</td>
      <td>0.029600</td>
      <td>-0.013400</td>
      <td>-0.716000</td>
      <td>0.183000</td>
      <td>-0.005260</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7727.000000</td>
      <td>0.287000</td>
      <td>-0.011000</td>
      <td>-0.098000</td>
      <td>-0.254000</td>
      <td>-0.051700</td>
      <td>-0.283000</td>
      <td>-0.306000</td>
      <td>-0.084500</td>
      <td>-0.288000</td>
      <td>...</td>
      <td>0.290000</td>
      <td>-0.118000</td>
      <td>-0.501000</td>
      <td>0.142000</td>
      <td>0.309000</td>
      <td>0.554000</td>
      <td>0.374000</td>
      <td>-0.522000</td>
      <td>0.252000</td>
      <td>0.104000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10281.000000</td>
      <td>0.693000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.980000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.988000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.871000</td>
      <td>0.990000</td>
      <td>0.957000</td>
      <td>0.981000</td>
      <td>0.997000</td>
      <td>0.999000</td>
      <td>0.996000</td>
      <td>0.977000</td>
      <td>1.000000</td>
      <td>0.998000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 562 columns</p>
</div>

- 열 확인

<pre>
rn                                       int64
activity                                object
tBodyAcc.mean.X                        float64
tBodyAcc.mean.Y                        float64
tBodyAcc.mean.Z                        float64
                                        ...   
angle.tBodyGyroMean.gravityMean        float64
angle.tBodyGyroJerkMean.gravityMean    float64
angle.X.gravityMean                    float64
angle.Y.gravityMean                    float64
angle.Z.gravityMean                    float64
Length: 563, dtype: object
</pre>

- 유니크 개수 확인

<pre>
rn                                     3609
activity                                  6
tBodyAcc.mean.X                         375
tBodyAcc.mean.Y                        1243
tBodyAcc.mean.Z                         689
                                       ... 
angle.tBodyGyroMean.gravityMean        1843
angle.tBodyGyroJerkMean.gravityMean    1847
angle.X.gravityMean                     940
angle.Y.gravityMean                    1172
angle.Z.gravityMean                    1892
Length: 563, dtype: int64
</pre>

- 요약.

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
      <th>rn</th>
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>tBodyAcc.mad.Z</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1541.000000</td>
      <td>1541.00000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>...</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
      <td>1541.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5161.052563</td>
      <td>0.27582</td>
      <td>-0.018046</td>
      <td>-0.109217</td>
      <td>-0.607266</td>
      <td>-0.516822</td>
      <td>-0.617703</td>
      <td>-0.632406</td>
      <td>-0.532077</td>
      <td>-0.619526</td>
      <td>...</td>
      <td>0.132199</td>
      <td>-0.316954</td>
      <td>-0.634735</td>
      <td>0.008390</td>
      <td>0.014311</td>
      <td>-0.000550</td>
      <td>0.006301</td>
      <td>-0.493145</td>
      <td>0.061068</td>
      <td>-0.054510</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3018.400705</td>
      <td>0.06066</td>
      <td>0.024662</td>
      <td>0.046271</td>
      <td>0.440922</td>
      <td>0.499209</td>
      <td>0.400243</td>
      <td>0.416450</td>
      <td>0.483753</td>
      <td>0.396243</td>
      <td>...</td>
      <td>0.246591</td>
      <td>0.313443</td>
      <td>0.297231</td>
      <td>0.338114</td>
      <td>0.450936</td>
      <td>0.621036</td>
      <td>0.482773</td>
      <td>0.511159</td>
      <td>0.303507</td>
      <td>0.270579</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>-0.41000</td>
      <td>-0.268000</td>
      <td>-0.347000</td>
      <td>-0.999000</td>
      <td>-1.000000</td>
      <td>-0.999000</td>
      <td>-0.999000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>-0.786000</td>
      <td>-1.000000</td>
      <td>-0.993000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-0.999000</td>
      <td>-0.991000</td>
      <td>-0.997000</td>
      <td>-0.987000</td>
      <td>-0.971000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2504.000000</td>
      <td>0.26300</td>
      <td>-0.024800</td>
      <td>-0.121000</td>
      <td>-0.992000</td>
      <td>-0.978000</td>
      <td>-0.980000</td>
      <td>-0.993000</td>
      <td>-0.979000</td>
      <td>-0.979000</td>
      <td>...</td>
      <td>-0.014900</td>
      <td>-0.551000</td>
      <td>-0.848000</td>
      <td>-0.129000</td>
      <td>-0.279000</td>
      <td>-0.526000</td>
      <td>-0.370000</td>
      <td>-0.814000</td>
      <td>-0.018400</td>
      <td>-0.136000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5064.000000</td>
      <td>0.27700</td>
      <td>-0.017200</td>
      <td>-0.109000</td>
      <td>-0.945000</td>
      <td>-0.865000</td>
      <td>-0.862000</td>
      <td>-0.951000</td>
      <td>-0.869000</td>
      <td>-0.857000</td>
      <td>...</td>
      <td>0.141000</td>
      <td>-0.353000</td>
      <td>-0.718000</td>
      <td>0.011900</td>
      <td>0.027500</td>
      <td>-0.004190</td>
      <td>0.001220</td>
      <td>-0.723000</td>
      <td>0.183000</td>
      <td>-0.002240</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7888.000000</td>
      <td>0.28900</td>
      <td>-0.010400</td>
      <td>-0.098300</td>
      <td>-0.239000</td>
      <td>-0.062700</td>
      <td>-0.273000</td>
      <td>-0.290000</td>
      <td>-0.082100</td>
      <td>-0.287000</td>
      <td>...</td>
      <td>0.293000</td>
      <td>-0.140000</td>
      <td>-0.523000</td>
      <td>0.148000</td>
      <td>0.301000</td>
      <td>0.527000</td>
      <td>0.377000</td>
      <td>-0.512000</td>
      <td>0.249000</td>
      <td>0.103000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10294.000000</td>
      <td>0.56400</td>
      <td>0.153000</td>
      <td>0.719000</td>
      <td>0.718000</td>
      <td>1.000000</td>
      <td>0.753000</td>
      <td>0.722000</td>
      <td>1.000000</td>
      <td>0.728000</td>
      <td>...</td>
      <td>0.947000</td>
      <td>0.941000</td>
      <td>0.927000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.993000</td>
      <td>0.991000</td>
      <td>0.912000</td>
      <td>0.901000</td>
      <td>0.991000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 562 columns</p>
</div>


- 타입 체크.

<pre>
rn                                       int64
tBodyAcc.mean.X                        float64
tBodyAcc.mean.Y                        float64
tBodyAcc.mean.Z                        float64
tBodyAcc.std.X                         float64
                                        ...   
angle.tBodyGyroMean.gravityMean        float64
angle.tBodyGyroJerkMean.gravityMean    float64
angle.X.gravityMean                    float64
angle.Y.gravityMean                    float64
angle.Z.gravityMean                    float64
Length: 562, dtype: object
</pre>

- 유니크 개수 확인

<pre>
rn                                     1541
tBodyAcc.mean.X                         288
tBodyAcc.mean.Y                         729
tBodyAcc.mean.Z                         417
tBodyAcc.std.X                          593
                                       ... 
angle.tBodyGyroMean.gravityMean        1091
angle.tBodyGyroJerkMean.gravityMean    1097
angle.X.gravityMean                     683
angle.Y.gravityMean                     725
angle.Z.gravityMean                    1084
Length: 562, dtype: int64
</pre>
#### 3. Data SET 첫 5행 확인하기(test, train)

- 이미 위에 확인했기에 패스


#### 4. Data SET 통계값, Null 값 확인하기 (null 값이 있는 경우 행 삭제)

- 통계값 확인은 위 데이터를 근거로 괜찮을 것이라 생각.

- null값 확인은 해보아야..



- null값 확인 결과

<pre>
rn                                     0
activity                               0
tBodyAcc.mean.X                        0
tBodyAcc.mean.Y                        0
tBodyAcc.mean.Z                        0
                                      ..
angle.tBodyGyroMean.gravityMean        0
angle.tBodyGyroJerkMean.gravityMean    0
angle.X.gravityMean                    0
angle.Y.gravityMean                    0
angle.Z.gravityMean                    0
Length: 563, dtype: int64
</pre>

- test값도 확인

<pre>
rn                                     0
tBodyAcc.mean.X                        0
tBodyAcc.mean.Y                        0
tBodyAcc.mean.Z                        0
tBodyAcc.std.X                         0
                                      ..
angle.tBodyGyroMean.gravityMean        0
angle.tBodyGyroJerkMean.gravityMean    0
angle.X.gravityMean                    0
angle.Y.gravityMean                    0
angle.Z.gravityMean                    0
Length: 562, dtype: int64
</pre>
#### 5. “activity” 컬럼의 Unique 값, Unique count 값 확인 하기

- 전체적인 대상의 활동 데이터를 보고 앞으로 어떻게 행동할지 예측 및 조언할 데이터가 activity

- 여기서 정보 확인은 중요..


<pre>
array(['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS',
       'WALKING_UPSTAIRS'], dtype=object)
</pre>

- 유니크 개수 확인

<pre>
6
</pre>

- 왜 강사님께서 6클래스 진행하면 된다고 하는지 위 데이터 답지들을 보고 이해.


#### 6. “activity” 컬럼의 Unique count 값 그래프 그리기

- 어떤 차트가 나은지 고찰은 진행해봐야..

- 일단 진행.
- solution에 관한 차트 비교도 진행.


<pre>
LAYING                681
STANDING              668
SITTING               623
WALKING               603
WALKING_UPSTAIRS      541
WALKING_DOWNSTAIRS    493
Name: activity, dtype: int64
</pre>

- 꺾은선 그래프 등은 추후에 여유되면.

<img src="data:image/png;base64,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"/>

#### 7. rn 열 drop (df_test / df_train)

- rn열은 대상의 유니크번호이겠으나 분석등 판단에는 필요없어보이기에 삭제.

- 아래 2표는 기존 데이터와 삭제된 열 데이터로 구분


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
      <th>rn</th>
      <th>activity</th>
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>STANDING</td>
      <td>0.279</td>
      <td>-0.01960</td>
      <td>-0.1100</td>
      <td>-0.9970</td>
      <td>-0.9670</td>
      <td>-0.983</td>
      <td>-0.997</td>
      <td>-0.9660</td>
      <td>...</td>
      <td>0.1460</td>
      <td>-0.2170</td>
      <td>-0.5640</td>
      <td>-0.2130</td>
      <td>-0.2310</td>
      <td>0.0146</td>
      <td>-0.1900</td>
      <td>-0.852</td>
      <td>0.182</td>
      <td>-0.0430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>STANDING</td>
      <td>0.277</td>
      <td>-0.01270</td>
      <td>-0.1030</td>
      <td>-0.9950</td>
      <td>-0.9730</td>
      <td>-0.985</td>
      <td>-0.996</td>
      <td>-0.9740</td>
      <td>...</td>
      <td>0.1210</td>
      <td>0.3490</td>
      <td>0.0577</td>
      <td>0.0807</td>
      <td>0.5960</td>
      <td>-0.4760</td>
      <td>0.1160</td>
      <td>-0.852</td>
      <td>0.188</td>
      <td>-0.0347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>STANDING</td>
      <td>0.277</td>
      <td>-0.01470</td>
      <td>-0.1070</td>
      <td>-0.9990</td>
      <td>-0.9910</td>
      <td>-0.993</td>
      <td>-0.999</td>
      <td>-0.9910</td>
      <td>...</td>
      <td>0.7400</td>
      <td>-0.5640</td>
      <td>-0.7660</td>
      <td>0.1060</td>
      <td>-0.0903</td>
      <td>-0.1320</td>
      <td>0.4990</td>
      <td>-0.850</td>
      <td>0.189</td>
      <td>-0.0351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>STANDING</td>
      <td>0.298</td>
      <td>0.02710</td>
      <td>-0.0617</td>
      <td>-0.9890</td>
      <td>-0.8170</td>
      <td>-0.902</td>
      <td>-0.989</td>
      <td>-0.7940</td>
      <td>...</td>
      <td>0.1310</td>
      <td>0.2080</td>
      <td>-0.0681</td>
      <td>0.0623</td>
      <td>-0.0587</td>
      <td>0.0312</td>
      <td>-0.2690</td>
      <td>-0.731</td>
      <td>0.283</td>
      <td>0.0364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>STANDING</td>
      <td>0.276</td>
      <td>-0.01700</td>
      <td>-0.1110</td>
      <td>-0.9980</td>
      <td>-0.9910</td>
      <td>-0.998</td>
      <td>-0.998</td>
      <td>-0.9890</td>
      <td>...</td>
      <td>0.6670</td>
      <td>-0.9420</td>
      <td>-0.9660</td>
      <td>0.2450</td>
      <td>0.1030</td>
      <td>0.0661</td>
      <td>-0.4120</td>
      <td>-0.761</td>
      <td>0.263</td>
      <td>0.0296</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>3604</th>
      <td>10277</td>
      <td>WALKING_UPSTAIRS</td>
      <td>0.357</td>
      <td>-0.04460</td>
      <td>-0.1300</td>
      <td>-0.3140</td>
      <td>-0.0556</td>
      <td>-0.173</td>
      <td>-0.386</td>
      <td>-0.0575</td>
      <td>...</td>
      <td>0.0168</td>
      <td>-0.1630</td>
      <td>-0.5930</td>
      <td>-0.7110</td>
      <td>-0.0612</td>
      <td>-0.7060</td>
      <td>0.0646</td>
      <td>-0.660</td>
      <td>0.274</td>
      <td>0.1760</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>10278</td>
      <td>WALKING_UPSTAIRS</td>
      <td>0.344</td>
      <td>0.00479</td>
      <td>-0.1220</td>
      <td>-0.3200</td>
      <td>-0.0667</td>
      <td>-0.182</td>
      <td>-0.380</td>
      <td>-0.0710</td>
      <td>...</td>
      <td>-0.0292</td>
      <td>0.1810</td>
      <td>-0.2500</td>
      <td>-0.4030</td>
      <td>-0.7060</td>
      <td>0.7390</td>
      <td>0.8710</td>
      <td>-0.653</td>
      <td>0.278</td>
      <td>0.1800</td>
    </tr>
    <tr>
      <th>3606</th>
      <td>10279</td>
      <td>WALKING_UPSTAIRS</td>
      <td>0.284</td>
      <td>-0.00796</td>
      <td>-0.1190</td>
      <td>-0.3090</td>
      <td>-0.0804</td>
      <td>-0.211</td>
      <td>-0.369</td>
      <td>-0.0971</td>
      <td>...</td>
      <td>-0.1100</td>
      <td>0.0245</td>
      <td>-0.3930</td>
      <td>-0.0761</td>
      <td>-0.2390</td>
      <td>0.9600</td>
      <td>0.0866</td>
      <td>-0.657</td>
      <td>0.272</td>
      <td>0.1830</td>
    </tr>
    <tr>
      <th>3607</th>
      <td>10280</td>
      <td>WALKING_UPSTAIRS</td>
      <td>0.207</td>
      <td>0.02460</td>
      <td>-0.1040</td>
      <td>-0.3650</td>
      <td>-0.1690</td>
      <td>-0.216</td>
      <td>-0.449</td>
      <td>-0.1860</td>
      <td>...</td>
      <td>-0.2140</td>
      <td>-0.3520</td>
      <td>-0.7340</td>
      <td>0.5350</td>
      <td>-0.2570</td>
      <td>0.9270</td>
      <td>-0.0843</td>
      <td>-0.657</td>
      <td>0.267</td>
      <td>0.1880</td>
    </tr>
    <tr>
      <th>3608</th>
      <td>10281</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>0.393</td>
      <td>-0.01780</td>
      <td>-0.0902</td>
      <td>-0.0963</td>
      <td>-0.1740</td>
      <td>-0.257</td>
      <td>-0.153</td>
      <td>-0.2080</td>
      <td>...</td>
      <td>0.0894</td>
      <td>0.2740</td>
      <td>-0.0368</td>
      <td>-0.7430</td>
      <td>-0.0802</td>
      <td>0.9270</td>
      <td>-0.6520</td>
      <td>-0.807</td>
      <td>0.190</td>
      <td>0.1180</td>
    </tr>
  </tbody>
</table>
<p>3609 rows × 563 columns</p>
</div>



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
      <th>activity</th>
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>tBodyAcc.mad.Z</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>STANDING</td>
      <td>0.279</td>
      <td>-0.01960</td>
      <td>-0.1100</td>
      <td>-0.9970</td>
      <td>-0.9670</td>
      <td>-0.983</td>
      <td>-0.997</td>
      <td>-0.9660</td>
      <td>-0.983</td>
      <td>...</td>
      <td>0.1460</td>
      <td>-0.2170</td>
      <td>-0.5640</td>
      <td>-0.2130</td>
      <td>-0.2310</td>
      <td>0.0146</td>
      <td>-0.1900</td>
      <td>-0.852</td>
      <td>0.182</td>
      <td>-0.0430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>STANDING</td>
      <td>0.277</td>
      <td>-0.01270</td>
      <td>-0.1030</td>
      <td>-0.9950</td>
      <td>-0.9730</td>
      <td>-0.985</td>
      <td>-0.996</td>
      <td>-0.9740</td>
      <td>-0.985</td>
      <td>...</td>
      <td>0.1210</td>
      <td>0.3490</td>
      <td>0.0577</td>
      <td>0.0807</td>
      <td>0.5960</td>
      <td>-0.4760</td>
      <td>0.1160</td>
      <td>-0.852</td>
      <td>0.188</td>
      <td>-0.0347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>STANDING</td>
      <td>0.277</td>
      <td>-0.01470</td>
      <td>-0.1070</td>
      <td>-0.9990</td>
      <td>-0.9910</td>
      <td>-0.993</td>
      <td>-0.999</td>
      <td>-0.9910</td>
      <td>-0.992</td>
      <td>...</td>
      <td>0.7400</td>
      <td>-0.5640</td>
      <td>-0.7660</td>
      <td>0.1060</td>
      <td>-0.0903</td>
      <td>-0.1320</td>
      <td>0.4990</td>
      <td>-0.850</td>
      <td>0.189</td>
      <td>-0.0351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>STANDING</td>
      <td>0.298</td>
      <td>0.02710</td>
      <td>-0.0617</td>
      <td>-0.9890</td>
      <td>-0.8170</td>
      <td>-0.902</td>
      <td>-0.989</td>
      <td>-0.7940</td>
      <td>-0.888</td>
      <td>...</td>
      <td>0.1310</td>
      <td>0.2080</td>
      <td>-0.0681</td>
      <td>0.0623</td>
      <td>-0.0587</td>
      <td>0.0312</td>
      <td>-0.2690</td>
      <td>-0.731</td>
      <td>0.283</td>
      <td>0.0364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>STANDING</td>
      <td>0.276</td>
      <td>-0.01700</td>
      <td>-0.1110</td>
      <td>-0.9980</td>
      <td>-0.9910</td>
      <td>-0.998</td>
      <td>-0.998</td>
      <td>-0.9890</td>
      <td>-0.997</td>
      <td>...</td>
      <td>0.6670</td>
      <td>-0.9420</td>
      <td>-0.9660</td>
      <td>0.2450</td>
      <td>0.1030</td>
      <td>0.0661</td>
      <td>-0.4120</td>
      <td>-0.761</td>
      <td>0.263</td>
      <td>0.0296</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>3604</th>
      <td>WALKING_UPSTAIRS</td>
      <td>0.357</td>
      <td>-0.04460</td>
      <td>-0.1300</td>
      <td>-0.3140</td>
      <td>-0.0556</td>
      <td>-0.173</td>
      <td>-0.386</td>
      <td>-0.0575</td>
      <td>-0.217</td>
      <td>...</td>
      <td>0.0168</td>
      <td>-0.1630</td>
      <td>-0.5930</td>
      <td>-0.7110</td>
      <td>-0.0612</td>
      <td>-0.7060</td>
      <td>0.0646</td>
      <td>-0.660</td>
      <td>0.274</td>
      <td>0.1760</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>WALKING_UPSTAIRS</td>
      <td>0.344</td>
      <td>0.00479</td>
      <td>-0.1220</td>
      <td>-0.3200</td>
      <td>-0.0667</td>
      <td>-0.182</td>
      <td>-0.380</td>
      <td>-0.0710</td>
      <td>-0.245</td>
      <td>...</td>
      <td>-0.0292</td>
      <td>0.1810</td>
      <td>-0.2500</td>
      <td>-0.4030</td>
      <td>-0.7060</td>
      <td>0.7390</td>
      <td>0.8710</td>
      <td>-0.653</td>
      <td>0.278</td>
      <td>0.1800</td>
    </tr>
    <tr>
      <th>3606</th>
      <td>WALKING_UPSTAIRS</td>
      <td>0.284</td>
      <td>-0.00796</td>
      <td>-0.1190</td>
      <td>-0.3090</td>
      <td>-0.0804</td>
      <td>-0.211</td>
      <td>-0.369</td>
      <td>-0.0971</td>
      <td>-0.301</td>
      <td>...</td>
      <td>-0.1100</td>
      <td>0.0245</td>
      <td>-0.3930</td>
      <td>-0.0761</td>
      <td>-0.2390</td>
      <td>0.9600</td>
      <td>0.0866</td>
      <td>-0.657</td>
      <td>0.272</td>
      <td>0.1830</td>
    </tr>
    <tr>
      <th>3607</th>
      <td>WALKING_UPSTAIRS</td>
      <td>0.207</td>
      <td>0.02460</td>
      <td>-0.1040</td>
      <td>-0.3650</td>
      <td>-0.1690</td>
      <td>-0.216</td>
      <td>-0.449</td>
      <td>-0.1860</td>
      <td>-0.326</td>
      <td>...</td>
      <td>-0.2140</td>
      <td>-0.3520</td>
      <td>-0.7340</td>
      <td>0.5350</td>
      <td>-0.2570</td>
      <td>0.9270</td>
      <td>-0.0843</td>
      <td>-0.657</td>
      <td>0.267</td>
      <td>0.1880</td>
    </tr>
    <tr>
      <th>3608</th>
      <td>WALKING_DOWNSTAIRS</td>
      <td>0.393</td>
      <td>-0.01780</td>
      <td>-0.0902</td>
      <td>-0.0963</td>
      <td>-0.1740</td>
      <td>-0.257</td>
      <td>-0.153</td>
      <td>-0.2080</td>
      <td>-0.265</td>
      <td>...</td>
      <td>0.0894</td>
      <td>0.2740</td>
      <td>-0.0368</td>
      <td>-0.7430</td>
      <td>-0.0802</td>
      <td>0.9270</td>
      <td>-0.6520</td>
      <td>-0.807</td>
      <td>0.190</td>
      <td>0.1180</td>
    </tr>
  </tbody>
</table>
<p>3609 rows × 562 columns</p>
</div>


rn열 삭제, test도 진행


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
      <th>tBodyAcc.mean.X</th>
      <th>tBodyAcc.mean.Y</th>
      <th>tBodyAcc.mean.Z</th>
      <th>tBodyAcc.std.X</th>
      <th>tBodyAcc.std.Y</th>
      <th>tBodyAcc.std.Z</th>
      <th>tBodyAcc.mad.X</th>
      <th>tBodyAcc.mad.Y</th>
      <th>tBodyAcc.mad.Z</th>
      <th>tBodyAcc.max.X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag.meanFreq</th>
      <th>fBodyBodyGyroJerkMag.skewness</th>
      <th>fBodyBodyGyroJerkMag.kurtosis</th>
      <th>angle.tBodyAccMean.gravity</th>
      <th>angle.tBodyAccJerkMean.gravityMean</th>
      <th>angle.tBodyGyroMean.gravityMean</th>
      <th>angle.tBodyGyroJerkMean.gravityMean</th>
      <th>angle.X.gravityMean</th>
      <th>angle.Y.gravityMean</th>
      <th>angle.Z.gravityMean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.280</td>
      <td>-0.01950</td>
      <td>-0.1130</td>
      <td>-0.9950</td>
      <td>-0.9670</td>
      <td>-0.979</td>
      <td>-0.997</td>
      <td>-0.9640</td>
      <td>-0.977</td>
      <td>-0.9390</td>
      <td>...</td>
      <td>0.4150</td>
      <td>-0.391</td>
      <td>-0.760</td>
      <td>-0.11900</td>
      <td>0.1780</td>
      <td>0.101</td>
      <td>0.809</td>
      <td>-0.849</td>
      <td>0.181</td>
      <td>-0.0491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.277</td>
      <td>-0.01660</td>
      <td>-0.1150</td>
      <td>-0.9980</td>
      <td>-0.9810</td>
      <td>-0.990</td>
      <td>-0.998</td>
      <td>-0.9800</td>
      <td>-0.990</td>
      <td>-0.9420</td>
      <td>...</td>
      <td>0.0878</td>
      <td>-0.351</td>
      <td>-0.699</td>
      <td>0.12300</td>
      <td>0.1230</td>
      <td>0.694</td>
      <td>-0.616</td>
      <td>-0.848</td>
      <td>0.185</td>
      <td>-0.0439</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.277</td>
      <td>-0.02180</td>
      <td>-0.1210</td>
      <td>-0.9970</td>
      <td>-0.9610</td>
      <td>-0.984</td>
      <td>-0.998</td>
      <td>-0.9570</td>
      <td>-0.984</td>
      <td>-0.9410</td>
      <td>...</td>
      <td>0.3140</td>
      <td>-0.269</td>
      <td>-0.573</td>
      <td>0.01300</td>
      <td>0.0809</td>
      <td>-0.234</td>
      <td>0.118</td>
      <td>-0.848</td>
      <td>0.189</td>
      <td>-0.0374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.279</td>
      <td>-0.01480</td>
      <td>-0.1170</td>
      <td>-0.9970</td>
      <td>-0.9820</td>
      <td>-0.983</td>
      <td>-0.997</td>
      <td>-0.9820</td>
      <td>-0.981</td>
      <td>-0.9420</td>
      <td>...</td>
      <td>0.5610</td>
      <td>-0.779</td>
      <td>-0.940</td>
      <td>-0.00145</td>
      <td>-0.0481</td>
      <td>-0.340</td>
      <td>-0.229</td>
      <td>-0.759</td>
      <td>0.264</td>
      <td>0.0270</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.279</td>
      <td>-0.01450</td>
      <td>-0.1070</td>
      <td>-0.9980</td>
      <td>-0.9860</td>
      <td>-0.993</td>
      <td>-0.998</td>
      <td>-0.9850</td>
      <td>-0.995</td>
      <td>-0.9430</td>
      <td>...</td>
      <td>0.6770</td>
      <td>-0.715</td>
      <td>-0.937</td>
      <td>0.02570</td>
      <td>0.0665</td>
      <td>-0.226</td>
      <td>-0.225</td>
      <td>-0.762</td>
      <td>0.262</td>
      <td>0.0294</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>1536</th>
      <td>0.289</td>
      <td>-0.02810</td>
      <td>-0.0943</td>
      <td>-0.0623</td>
      <td>0.1140</td>
      <td>-0.190</td>
      <td>-0.114</td>
      <td>0.0393</td>
      <td>-0.207</td>
      <td>0.3300</td>
      <td>...</td>
      <td>0.2860</td>
      <td>-0.237</td>
      <td>-0.607</td>
      <td>-0.19600</td>
      <td>0.6980</td>
      <td>0.990</td>
      <td>-0.108</td>
      <td>-0.806</td>
      <td>0.190</td>
      <td>0.1200</td>
    </tr>
    <tr>
      <th>1537</th>
      <td>0.377</td>
      <td>-0.01810</td>
      <td>-0.1100</td>
      <td>-0.3140</td>
      <td>-0.1520</td>
      <td>-0.214</td>
      <td>-0.394</td>
      <td>-0.1810</td>
      <td>-0.266</td>
      <td>-0.0726</td>
      <td>...</td>
      <td>-0.1910</td>
      <td>-0.323</td>
      <td>-0.753</td>
      <td>-0.82900</td>
      <td>0.0483</td>
      <td>0.913</td>
      <td>-0.904</td>
      <td>-0.695</td>
      <td>0.246</td>
      <td>0.1730</td>
    </tr>
    <tr>
      <th>1538</th>
      <td>0.253</td>
      <td>-0.02490</td>
      <td>-0.1700</td>
      <td>-0.3080</td>
      <td>-0.1890</td>
      <td>-0.141</td>
      <td>-0.377</td>
      <td>-0.2260</td>
      <td>-0.221</td>
      <td>0.0920</td>
      <td>...</td>
      <td>-0.1780</td>
      <td>-0.142</td>
      <td>-0.564</td>
      <td>0.00451</td>
      <td>0.3570</td>
      <td>-0.946</td>
      <td>0.614</td>
      <td>-0.695</td>
      <td>0.259</td>
      <td>0.1580</td>
    </tr>
    <tr>
      <th>1539</th>
      <td>0.277</td>
      <td>0.00108</td>
      <td>-0.0740</td>
      <td>-0.0685</td>
      <td>-0.2450</td>
      <td>-0.145</td>
      <td>-0.149</td>
      <td>-0.3030</td>
      <td>-0.199</td>
      <td>0.4030</td>
      <td>...</td>
      <td>-0.1040</td>
      <td>0.161</td>
      <td>-0.126</td>
      <td>0.13400</td>
      <td>0.8830</td>
      <td>-0.994</td>
      <td>0.475</td>
      <td>-0.804</td>
      <td>0.197</td>
      <td>0.1140</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>0.192</td>
      <td>-0.03360</td>
      <td>-0.1060</td>
      <td>-0.3550</td>
      <td>-0.0925</td>
      <td>-0.313</td>
      <td>-0.434</td>
      <td>-0.0887</td>
      <td>-0.336</td>
      <td>-0.0416</td>
      <td>...</td>
      <td>0.1590</td>
      <td>-0.630</td>
      <td>-0.916</td>
      <td>0.53600</td>
      <td>0.6890</td>
      <td>-0.937</td>
      <td>0.562</td>
      <td>-0.647</td>
      <td>0.282</td>
      <td>0.1810</td>
    </tr>
  </tbody>
</table>
<p>1541 rows × 561 columns</p>
</div>


#### 8. “activity” 컬럼에 대하여 인코딩 하기

- 문자열로 구분하는데 여러모로 비용 및 한계가 있기에 이와 같이 숫자 구분으로 인코딩 필수

- 0: ..., 1:... 등으로 인코딩됨


<pre>
array(['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS',
       'WALKING_UPSTAIRS'], dtype=object)
</pre>
위 unique 라벨을 바탕으로 라벨링 진행


<pre>
LAYING                681
STANDING              668
SITTING               623
WALKING               603
WALKING_UPSTAIRS      541
WALKING_DOWNSTAIRS    493
Name: activity, dtype: int64
1    681
2    668
3    623
4    603
6    541
5    493
Name: activity, dtype: int64
</pre>


#### 9. df_train 으로 X = “특성 컬럼”, y = “activity” 만들기

- 이제 훈련데이터의 X와 y로 구분

- 전처리는 거의 완료 단계로..

- 참고로 train데이터만 하는 이유는 test에 'activity'라벨이 없다.

- 여기서 test는 고객에게 보여줄 훈련모델의 답지



```python
# ......
print(X.columns)
print(y.name)
```

<pre>
Index(['tBodyAcc.mean.X', 'tBodyAcc.mean.Y', 'tBodyAcc.mean.Z',
       'tBodyAcc.std.X', 'tBodyAcc.std.Y', 'tBodyAcc.std.Z', 'tBodyAcc.mad.X',
       'tBodyAcc.mad.Y', 'tBodyAcc.mad.Z', 'tBodyAcc.max.X',
       ...
       'fBodyBodyGyroJerkMag.meanFreq', 'fBodyBodyGyroJerkMag.skewness',
       'fBodyBodyGyroJerkMag.kurtosis', 'angle.tBodyAccMean.gravity',
       'angle.tBodyAccJerkMean.gravityMean', 'angle.tBodyGyroMean.gravityMean',
       'angle.tBodyGyroJerkMean.gravityMean', 'angle.X.gravityMean',
       'angle.Y.gravityMean', 'angle.Z.gravityMean'],
      dtype='object', length=561)
activity
</pre>

#### 10. X_train, X_test, y_train, y_test 만들기

- 더 말할건 없을듯.


<pre>
x_train values count: 2887
y_train values count: 2887
x_test values count: 722
y_test values count: 722
</pre>


#### 11. X_train, X_test 정규화 하기(옵션)

- 여기서 정규화와 안 된 데이터하고 비교가 중요

- 두 그룹을 만들필요가 있음


```python
# train을 test 정규화 데이터 확인.

X_train_stand.head(5)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>551</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.263369</td>
      <td>0.041429</td>
      <td>-0.473708</td>
      <td>0.558023</td>
      <td>0.688464</td>
      <td>0.823657</td>
      <td>0.505502</td>
      <td>0.752587</td>
      <td>0.822891</td>
      <td>1.082366</td>
      <td>...</td>
      <td>-0.288262</td>
      <td>0.264987</td>
      <td>0.073328</td>
      <td>-2.304810</td>
      <td>1.330785</td>
      <td>-1.136089</td>
      <td>1.079116</td>
      <td>-0.403241</td>
      <td>0.550853</td>
      <td>0.852714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.793488</td>
      <td>-1.771799</td>
      <td>-5.097944</td>
      <td>1.087796</td>
      <td>0.344323</td>
      <td>1.580836</td>
      <td>1.094525</td>
      <td>0.378718</td>
      <td>1.626735</td>
      <td>1.152838</td>
      <td>...</td>
      <td>0.133866</td>
      <td>-0.377673</td>
      <td>-0.526774</td>
      <td>-0.150440</td>
      <td>0.106660</td>
      <td>1.485142</td>
      <td>-1.543728</td>
      <td>-0.498093</td>
      <td>0.608812</td>
      <td>0.544066</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.179134</td>
      <td>0.148650</td>
      <td>0.266953</td>
      <td>0.364758</td>
      <td>0.579055</td>
      <td>0.753710</td>
      <td>0.322035</td>
      <td>0.522514</td>
      <td>0.979664</td>
      <td>0.535279</td>
      <td>...</td>
      <td>-0.764047</td>
      <td>0.522051</td>
      <td>0.486301</td>
      <td>-0.659841</td>
      <td>-1.414478</td>
      <td>-1.321346</td>
      <td>-0.859596</td>
      <td>-0.452643</td>
      <td>0.518654</td>
      <td>-0.298958</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.179134</td>
      <td>0.003306</td>
      <td>0.016147</td>
      <td>-0.883506</td>
      <td>-0.976539</td>
      <td>-0.945008</td>
      <td>-0.877738</td>
      <td>-0.979125</td>
      <td>-0.954719</td>
      <td>-0.874164</td>
      <td>...</td>
      <td>1.584276</td>
      <td>-0.587714</td>
      <td>-0.449341</td>
      <td>-0.224884</td>
      <td>1.646946</td>
      <td>-0.489313</td>
      <td>-1.411773</td>
      <td>1.902853</td>
      <td>-1.706329</td>
      <td>-1.846038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.064547</td>
      <td>0.352131</td>
      <td>0.678432</td>
      <td>0.908173</td>
      <td>0.764055</td>
      <td>1.003522</td>
      <td>0.879676</td>
      <td>0.880360</td>
      <td>1.101037</td>
      <td>0.798066</td>
      <td>...</td>
      <td>-0.661344</td>
      <td>0.076891</td>
      <td>0.028159</td>
      <td>1.519444</td>
      <td>-2.164805</td>
      <td>1.444516</td>
      <td>-0.934708</td>
      <td>-0.221441</td>
      <td>0.876068</td>
      <td>-0.229858</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>



```python
X_test_stand.head(5)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>551</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.096096</td>
      <td>-0.393721</td>
      <td>-0.331865</td>
      <td>0.810934</td>
      <td>1.437674</td>
      <td>0.840067</td>
      <td>0.833843</td>
      <td>1.473250</td>
      <td>0.918939</td>
      <td>0.674501</td>
      <td>...</td>
      <td>0.483094</td>
      <td>0.204334</td>
      <td>-0.009696</td>
      <td>-2.194124</td>
      <td>-1.329660</td>
      <td>0.973570</td>
      <td>0.488045</td>
      <td>-0.406303</td>
      <td>0.782000</td>
      <td>0.577781</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.784875</td>
      <td>0.529633</td>
      <td>0.675178</td>
      <td>0.728368</td>
      <td>1.359141</td>
      <td>2.711874</td>
      <td>0.660003</td>
      <td>1.607020</td>
      <td>2.751704</td>
      <td>0.586195</td>
      <td>...</td>
      <td>0.454734</td>
      <td>-1.060738</td>
      <td>-0.790499</td>
      <td>-0.726376</td>
      <td>0.817020</td>
      <td>-1.071162</td>
      <td>-1.914002</td>
      <td>0.351198</td>
      <td>0.861634</td>
      <td>1.851792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.946647</td>
      <td>-0.037385</td>
      <td>2.043828</td>
      <td>2.269595</td>
      <td>0.905257</td>
      <td>2.037417</td>
      <td>2.280880</td>
      <td>0.900549</td>
      <td>1.974245</td>
      <td>2.078204</td>
      <td>...</td>
      <td>0.527660</td>
      <td>-0.338752</td>
      <td>-0.428340</td>
      <td>0.179548</td>
      <td>0.879566</td>
      <td>-1.663406</td>
      <td>-0.887804</td>
      <td>-0.358360</td>
      <td>0.246859</td>
      <td>1.158059</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.039720</td>
      <td>0.065061</td>
      <td>-0.086245</td>
      <td>-0.858729</td>
      <td>-0.954582</td>
      <td>-0.930698</td>
      <td>-0.850695</td>
      <td>-0.957595</td>
      <td>-0.936767</td>
      <td>-0.848784</td>
      <td>...</td>
      <td>0.965214</td>
      <td>-0.265276</td>
      <td>-0.381824</td>
      <td>-0.023318</td>
      <td>0.256337</td>
      <td>1.084409</td>
      <td>-1.181595</td>
      <td>2.103997</td>
      <td>-2.865241</td>
      <td>-0.301670</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.144335</td>
      <td>0.033882</td>
      <td>0.036565</td>
      <td>-0.826620</td>
      <td>-0.932431</td>
      <td>-0.900385</td>
      <td>-0.818865</td>
      <td>-0.934603</td>
      <td>-0.911276</td>
      <td>-0.821188</td>
      <td>...</td>
      <td>0.082003</td>
      <td>-0.635853</td>
      <td>-0.621049</td>
      <td>-0.437089</td>
      <td>0.832657</td>
      <td>-0.113546</td>
      <td>-0.036226</td>
      <td>1.762642</td>
      <td>-1.126033</td>
      <td>-2.292538</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>


셀 따로 나누어서 실행하는 이유는 다를바 없이 데이터 양, 평균, 분산 등 요약데이터를 특정 vscode등에서 쉽게 보기 위해서



```python
X_train_stand.describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>551</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>...</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
      <td>2.887000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.472904e-16</td>
      <td>2.092003e-17</td>
      <td>1.052155e-16</td>
      <td>-1.230590e-17</td>
      <td>6.214480e-17</td>
      <td>-3.568711e-17</td>
      <td>1.421332e-16</td>
      <td>-4.307065e-18</td>
      <td>-6.399069e-17</td>
      <td>-4.060947e-17</td>
      <td>...</td>
      <td>-7.629659e-17</td>
      <td>1.107531e-16</td>
      <td>-1.267508e-16</td>
      <td>1.230590e-17</td>
      <td>1.476708e-17</td>
      <td>4.430124e-17</td>
      <td>2.707298e-17</td>
      <td>-6.276010e-17</td>
      <td>2.707298e-17</td>
      <td>-2.461180e-18</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>...</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
      <td>1.000173e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.268818e+01</td>
      <td>-2.342331e+01</td>
      <td>-1.599233e+01</td>
      <td>-8.971479e-01</td>
      <td>-9.884742e-01</td>
      <td>-9.699890e-01</td>
      <td>-8.898078e-01</td>
      <td>-9.893966e-01</td>
      <td>-9.774766e-01</td>
      <td>-9.928536e-01</td>
      <td>...</td>
      <td>-3.840090e+00</td>
      <td>-2.105018e+00</td>
      <td>-1.223666e+00</td>
      <td>-2.935182e+00</td>
      <td>-2.249412e+00</td>
      <td>-1.656110e+00</td>
      <td>-1.996432e+00</td>
      <td>-9.822346e-01</td>
      <td>-3.419341e+00</td>
      <td>-3.542835e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-2.194815e-01</td>
      <td>-1.920751e-01</td>
      <td>-2.581717e-01</td>
      <td>-8.812320e-01</td>
      <td>-9.427214e-01</td>
      <td>-9.175286e-01</td>
      <td>-8.729096e-01</td>
      <td>-9.442036e-01</td>
      <td>-9.218473e-01</td>
      <td>-8.741636e-01</td>
      <td>...</td>
      <td>-6.053818e-01</td>
      <td>-7.334877e-01</td>
      <td>-7.074494e-01</td>
      <td>-3.806759e-01</td>
      <td>-6.496783e-01</td>
      <td>-8.419520e-01</td>
      <td>-7.946339e-01</td>
      <td>-6.225867e-01</td>
      <td>-2.204204e-01</td>
      <td>-2.912801e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.563251e-02</td>
      <td>-3.842552e-03</td>
      <td>1.614739e-02</td>
      <td>-7.448097e-01</td>
      <td>-5.985811e-01</td>
      <td>-5.328190e-01</td>
      <td>-7.473800e-01</td>
      <td>-5.929309e-01</td>
      <td>-5.223275e-01</td>
      <td>-7.499101e-01</td>
      <td>...</td>
      <td>8.108146e-03</td>
      <td>-1.237443e-01</td>
      <td>-2.815708e-01</td>
      <td>-1.942674e-03</td>
      <td>-8.790719e-04</td>
      <td>-1.121840e-02</td>
      <td>-6.359789e-03</td>
      <td>-4.230018e-01</td>
      <td>3.866361e-01</td>
      <td>1.540716e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.950787e-01</td>
      <td>1.438843e-01</td>
      <td>2.277649e-01</td>
      <td>7.990354e-01</td>
      <td>8.981308e-01</td>
      <td>8.174120e-01</td>
      <td>7.963921e-01</td>
      <td>8.996701e-01</td>
      <td>8.228907e-01</td>
      <td>8.100282e-01</td>
      <td>...</td>
      <td>6.767221e-01</td>
      <td>5.784795e-01</td>
      <td>3.959634e-01</td>
      <td>4.087895e-01</td>
      <td>6.561582e-01</td>
      <td>8.668047e-01</td>
      <td>7.827263e-01</td>
      <td>-5.643759e-02</td>
      <td>6.120323e-01</td>
      <td>5.878297e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.668596e+00</td>
      <td>2.423050e+01</td>
      <td>9.068676e+00</td>
      <td>3.650260e+00</td>
      <td>2.948252e+00</td>
      <td>4.026240e+00</td>
      <td>3.938252e+00</td>
      <td>3.092351e+00</td>
      <td>4.079735e+00</td>
      <td>2.716207e+00</td>
      <td>...</td>
      <td>2.892161e+00</td>
      <td>4.033170e+00</td>
      <td>5.074175e+00</td>
      <td>2.855232e+00</td>
      <td>2.190210e+00</td>
      <td>1.590771e+00</td>
      <td>2.041366e+00</td>
      <td>2.922514e+00</td>
      <td>3.020552e+00</td>
      <td>4.012135e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 561 columns</p>
</div>



```python
X_test_stand.describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>551</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>...</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
      <td>7.220000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-3.395253e-16</td>
      <td>3.444459e-17</td>
      <td>-6.642886e-17</td>
      <td>-1.107148e-16</td>
      <td>-1.230164e-17</td>
      <td>1.230164e-18</td>
      <td>1.230164e-18</td>
      <td>-6.150820e-17</td>
      <td>-1.955961e-16</td>
      <td>3.567476e-17</td>
      <td>...</td>
      <td>4.920656e-17</td>
      <td>2.829377e-17</td>
      <td>1.045639e-16</td>
      <td>-1.968262e-17</td>
      <td>3.690492e-17</td>
      <td>1.968262e-17</td>
      <td>-4.920656e-18</td>
      <td>3.567476e-17</td>
      <td>-3.321443e-17</td>
      <td>5.166689e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>...</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
      <td>1.000693e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.105415e+01</td>
      <td>-1.480303e+01</td>
      <td>-4.384599e+00</td>
      <td>-8.679028e-01</td>
      <td>-9.545815e-01</td>
      <td>-9.433283e-01</td>
      <td>-8.604884e-01</td>
      <td>-9.575946e-01</td>
      <td>-9.495121e-01</td>
      <td>-8.763795e-01</td>
      <td>...</td>
      <td>-3.633154e+00</td>
      <td>-1.936066e+00</td>
      <td>-1.185885e+00</td>
      <td>-2.903637e+00</td>
      <td>-2.156165e+00</td>
      <td>-1.762665e+00</td>
      <td>-2.011242e+00</td>
      <td>-1.000798e+00</td>
      <td>-3.215631e+00</td>
      <td>-3.282552e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-9.949983e-02</td>
      <td>-1.459564e-01</td>
      <td>-1.817641e-01</td>
      <td>-8.518483e-01</td>
      <td>-9.082668e-01</td>
      <td>-8.902811e-01</td>
      <td>-8.457977e-01</td>
      <td>-9.095212e-01</td>
      <td>-8.959821e-01</td>
      <td>-8.432647e-01</td>
      <td>...</td>
      <td>-6.022825e-01</td>
      <td>-7.149198e-01</td>
      <td>-7.007901e-01</td>
      <td>-3.670513e-01</td>
      <td>-6.500268e-01</td>
      <td>-7.398856e-01</td>
      <td>-7.590126e-01</td>
      <td>-6.474570e-01</td>
      <td>-2.885209e-01</td>
      <td>-1.860478e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.994966e-02</td>
      <td>3.388159e-02</td>
      <td>-3.166282e-02</td>
      <td>-7.635488e-01</td>
      <td>-6.615906e-01</td>
      <td>-6.086260e-01</td>
      <td>-7.564290e-01</td>
      <td>-6.555679e-01</td>
      <td>-6.002927e-01</td>
      <td>-7.623172e-01</td>
      <td>...</td>
      <td>2.933475e-02</td>
      <td>-1.454774e-01</td>
      <td>-2.721791e-01</td>
      <td>3.052797e-03</td>
      <td>-2.750294e-03</td>
      <td>6.121582e-02</td>
      <td>1.014709e-03</td>
      <td>-4.523286e-01</td>
      <td>4.172759e-01</td>
      <td>2.232202e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.143441e-01</td>
      <td>1.714048e-01</td>
      <td>1.344720e-01</td>
      <td>8.361623e-01</td>
      <td>9.379789e-01</td>
      <td>8.653272e-01</td>
      <td>8.179281e-01</td>
      <td>9.276163e-01</td>
      <td>8.571248e-01</td>
      <td>8.882667e-01</td>
      <td>...</td>
      <td>6.279329e-01</td>
      <td>5.485556e-01</td>
      <td>3.915034e-01</td>
      <td>3.736745e-01</td>
      <td>7.092394e-01</td>
      <td>8.135150e-01</td>
      <td>7.833866e-01</td>
      <td>2.662289e-02</td>
      <td>6.346770e-01</td>
      <td>5.934982e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.040251e+00</td>
      <td>8.737372e+00</td>
      <td>1.510127e+01</td>
      <td>2.709946e+00</td>
      <td>2.670049e+00</td>
      <td>3.325705e+00</td>
      <td>2.861164e+00</td>
      <td>2.664636e+00</td>
      <td>3.072884e+00</td>
      <td>2.127877e+00</td>
      <td>...</td>
      <td>3.027390e+00</td>
      <td>3.549110e+00</td>
      <td>4.522284e+00</td>
      <td>2.979537e+00</td>
      <td>2.161767e+00</td>
      <td>1.537691e+00</td>
      <td>2.006651e+00</td>
      <td>2.671643e+00</td>
      <td>2.756925e+00</td>
      <td>3.701746e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 561 columns</p>
</div>

요약도 마찬가지..


#### 12. AI Model : 분류

- 드디어 위 문제에 관해서 분류 모델링 선택해 진행한다.

- 훈련등은 다음 프로세스에..


- 아래 코드는 f1_score micro형식만 추출해 구분목적으로 코드 작성.


- KNN모델을 예로 첫번째로 진행.

<pre>
KNeighborsClassifier()
</pre>
#### 13. AI모델에서의 .fit // .predict // score (정확도, 정밀도, 재현율, F1 Score)

- 드디어 훈련, 평가, 성능확인 등 진행해본다.

- 일단 대표로 KNN.. 과연....

- 정규화로 진행은 일단 한번 진행하고 나서...

- 훈련 후 평가출력결과는 아래에 표기

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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>STANDING</th>
      <td>1.000000</td>
      <td>0.993333</td>
      <td>0.996656</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>SITTING</th>
      <td>0.872611</td>
      <td>0.907285</td>
      <td>0.889610</td>
      <td>151.000000</td>
    </tr>
    <tr>
      <th>LAYING</th>
      <td>0.864078</td>
      <td>0.809091</td>
      <td>0.835681</td>
      <td>110.000000</td>
    </tr>
    <tr>
      <th>WALKING</th>
      <td>0.930693</td>
      <td>1.000000</td>
      <td>0.964103</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>WALKING_DOWNSTAIRS</th>
      <td>0.990291</td>
      <td>0.944444</td>
      <td>0.966825</td>
      <td>108.000000</td>
    </tr>
    <tr>
      <th>WALKING_UPSTAIRS</th>
      <td>0.990826</td>
      <td>0.990826</td>
      <td>0.990826</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.940443</td>
      <td>0.940443</td>
      <td>0.940443</td>
      <td>0.940443</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.941417</td>
      <td>0.940830</td>
      <td>0.940617</td>
      <td>722.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.940789</td>
      <td>0.940443</td>
      <td>0.940162</td>
      <td>722.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### KNN 분류 결과로...

- STANDING 이 perfect인건 박수칠 만함...

- f1-score 최저값은 LAYING 84%

- 평균은 94%.....

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- KNN 그 외 옵션등으로 성능향상 부분은 나중에..


#### re12, 13. 다음은 의사결정트리

- 의사결정트리로 결과내보기.

- KNN등 다른 모델들과 비교해보면서...

- 성능향상은 다른 모델들 해보고 나서..


```python
# f1-score등 성능확인
dt_res = classification_report(y_test, y_predict, ...)

# ......

df_dt_res
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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>STANDING</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>SITTING</th>
      <td>0.930556</td>
      <td>0.887417</td>
      <td>0.908475</td>
      <td>151.000000</td>
    </tr>
    <tr>
      <th>LAYING</th>
      <td>0.854701</td>
      <td>0.909091</td>
      <td>0.881057</td>
      <td>110.000000</td>
    </tr>
    <tr>
      <th>WALKING</th>
      <td>0.882353</td>
      <td>0.957447</td>
      <td>0.918367</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>WALKING_DOWNSTAIRS</th>
      <td>0.923810</td>
      <td>0.898148</td>
      <td>0.910798</td>
      <td>108.000000</td>
    </tr>
    <tr>
      <th>WALKING_UPSTAIRS</th>
      <td>0.923077</td>
      <td>0.880734</td>
      <td>0.901408</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.923823</td>
      <td>0.923823</td>
      <td>0.923823</td>
      <td>0.923823</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.919083</td>
      <td>0.922140</td>
      <td>0.920018</td>
      <td>722.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.925012</td>
      <td>0.923823</td>
      <td>0.923881</td>
      <td>722.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### 결정트리 분류 결과로...

- STANDING 이 perfect라 박수칠 만함...

- f1-score 최저값은 LAYING 86.7..% KNN보다 괜찮은데?

- 평균은 92% KNN보다는... 아마도 전체적으로 낮아서...

- 그래도 KNN보다 좋아보이는건 사실

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- dt 그 외 옵션등으로 성능향상 부분은 나중에..

- .... 다음은 선형회귀를 할까 했는데 분류가 아닌 예측에 최적이라..

- 로지스틱 회귀로..


#### re12, 13. 다음은 로지스틱회귀

- 로지스틱회귀로 결과내보기.

- KNN등 다른 모델들과 비교해보면서...

- 성능향상은 다른 모델들 해보고 나서..



```python
# f1-score등 성능확인
lg_res = classification_report(y_test, y_predict, ... )

# ......

print(df_lg_res)
```

<pre>
                    precision    recall  f1-score     support
STANDING             1.000000  1.000000  1.000000  150.000000
SITTING              0.959184  0.933775  0.946309  151.000000
LAYING               0.911504  0.936364  0.923767  110.000000
WALKING              0.989362  0.989362  0.989362   94.000000
WALKING_DOWNSTAIRS   0.990741  0.990741  0.990741  108.000000
WALKING_UPSTAIRS     0.990909  1.000000  0.995434  109.000000
accuracy             0.973684  0.973684  0.973684    0.973684
macro avg            0.973617  0.975040  0.974269  722.000000
weighted avg         0.973838  0.973684  0.973697  722.000000
</pre>
#### 로지스틱회귀 분류 결과로...

- STANDING 이 perfect라 박수칠 만함.

- f1-score 최저값은 LAYING 92.3% ...

- 평균은 97% 로지스틱회귀 역대급.. 이었다가 SVM에 밀림. 그래도 역시 이진분류다운...

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- lg 그 외 옵션등으로 성능향상 부분은 나중에..


#### re12, 13. 다음은 랜덤포레스트

- 랜덤포레스트로 결과내보기.

- KNN등 다른 모델들과 비교해보면서...

- 성능향상은 다른 모델들 해보고 나서..


```python
# f1-score등 성능확인
rfc_res = classification_report(y_test, y_predict, ... )

# ......

df_rfc_res = pd.DataFrame(rfc_res).T
df_rfc_res
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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>STANDING</th>
      <td>1.000000</td>
      <td>0.993333</td>
      <td>0.996656</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>SITTING</th>
      <td>0.945946</td>
      <td>0.927152</td>
      <td>0.936455</td>
      <td>151.000000</td>
    </tr>
    <tr>
      <th>LAYING</th>
      <td>0.902655</td>
      <td>0.927273</td>
      <td>0.914798</td>
      <td>110.000000</td>
    </tr>
    <tr>
      <th>WALKING</th>
      <td>0.958763</td>
      <td>0.989362</td>
      <td>0.973822</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>WALKING_DOWNSTAIRS</th>
      <td>0.962264</td>
      <td>0.944444</td>
      <td>0.953271</td>
      <td>108.000000</td>
    </tr>
    <tr>
      <th>WALKING_UPSTAIRS</th>
      <td>0.954128</td>
      <td>0.954128</td>
      <td>0.954128</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.955679</td>
      <td>0.955679</td>
      <td>0.955679</td>
      <td>0.955679</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.953959</td>
      <td>0.955949</td>
      <td>0.954855</td>
      <td>722.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.955925</td>
      <td>0.955679</td>
      <td>0.955711</td>
      <td>722.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### 랜덤포레스트 분류 결과로...

- 원래 n_estimators 기본값은 100 이 중 10이하로 낮은 수치로 사용시 성능이 잘 안나옴.  

그렇기에 평균 92%등 낮게 나오게 됨.

- 반면.. 10 이상 예로 15수치정도로 들어갔음, 성적표는 아래와 같음

- STANDING 이 perfect가 아닌.. 99.6%

- f1-score 최저값은 LAYING 92.0% ...

- 평균은 95.1% .. 의사결정, KNN보단 높게 나옴

- 근데 랜덤포레스트의 경우 좀 더 옵션을 여러개 붙이면 어쩌면...  

실제 다른 분들이 했을 때, 역대 성적이 나온것도 랜덤포레스트 였음.

- 트리모델의 경우 데이터양이 2800정도로는 활약하기 힘든부분도..  

그렇다고 많으면 그 만큼 더 걸리기도함.

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- lg 그 외 옵션등으로 성능향상 부분은 나중에..


#### re12, 13. 다음은 SVM(서포트 벡터 머신)

- 서포트 벡터 머신으로 결과내보기.

- KNN등 다른 모델들과 비교해보면서...

- 성능향상은 다른 모델들 해보고 나서..


```python
# f1-score등 성능확인
svm_res = classification_report(y_test, y_predict, ...)

# ......

df_svm_res = pd.DataFrame(svm_res).T
df_svm_res
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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>STANDING</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>SITTING</th>
      <td>0.953642</td>
      <td>0.953642</td>
      <td>0.953642</td>
      <td>151.000000</td>
    </tr>
    <tr>
      <th>LAYING</th>
      <td>0.936364</td>
      <td>0.936364</td>
      <td>0.936364</td>
      <td>110.000000</td>
    </tr>
    <tr>
      <th>WALKING</th>
      <td>0.989362</td>
      <td>0.989362</td>
      <td>0.989362</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>WALKING_DOWNSTAIRS</th>
      <td>1.000000</td>
      <td>0.990741</td>
      <td>0.995349</td>
      <td>108.000000</td>
    </tr>
    <tr>
      <th>WALKING_UPSTAIRS</th>
      <td>0.990909</td>
      <td>1.000000</td>
      <td>0.995434</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.977839</td>
      <td>0.977839</td>
      <td>0.977839</td>
      <td>0.977839</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.978379</td>
      <td>0.978351</td>
      <td>0.978358</td>
      <td>722.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.977852</td>
      <td>0.977839</td>
      <td>0.977839</td>
      <td>722.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### SVM 분류 결과로...

- STANDING 이 perfect라 박수칠 만함...

- f1-score 최저값은 LAYING 93.6% ...

- 평균은 97.7% 로지스틱회귀 97.3% 에 비해 좀 더 높은 역대급.. 왜 SVM이 잘쓰였는지.

- 기본옵션일 뿐인데..

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- lg 그 외 옵션등으로 성능향상 부분은 나중에..


#### re12, 13. 다음은 나이브 베이지안

- 덤으로 한다는 느낌... 나이브 베이지안으로 결과내보기.

- 나이브 베이즈도 여러 종류모델이 있는데 이 중 가우시안만..

- 가우시안을 택한 이유는 데이터를 기반으로 해야할 행동 분류라면 이 모델이 나아보이며 예시도 있기에

- 다만.. 나이브베이지안 하는 것들을 보니 주의할 점. 라벨들이 독립적이어야 한다.

- 데이터를 보아하니 라벨은 하나뿐이겠다. (실은 여러개 있어야..) 일단 해보는건 어떨까 해서...

- KNN등 다른 모델들과 비교해보면서...

- 성능향상은 다른 모델들 해보고 나서..



```python
# f1-score등 성능확인
gnb_res = classification_report(y_test, y_predict, ...)

# ......

print(df_gnb_res)
```

<pre>
                    precision    recall  f1-score     support
STANDING             1.000000  0.973333  0.986486  150.000000
SITTING              0.880000  0.582781  0.701195  151.000000
LAYING               0.605096  0.863636  0.711610  110.000000
WALKING              0.884615  0.734043  0.802326   94.000000
WALKING_DOWNSTAIRS   0.816327  0.740741  0.776699  108.000000
WALKING_UPSTAIRS     0.720280  0.944954  0.817460  109.000000
accuracy             0.804709  0.804709  0.804709    0.804709
macro avg            0.817720  0.806581  0.799296  722.000000
weighted avg         0.830011  0.804709  0.804066  722.000000
</pre>
#### 나이브 베이지안 분류 결과로...

- f1-score 최저값은 SITTING 70%정도.. 

- 평균은 80.5%정도.. 이 데이터에서는 이 모델 대충옵션을 넣은 것으로 쓰이기엔 애매한 것이구나라고..

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- lg 그 외 옵션등으로 성능향상 부분은 나중에..


#### re12, 13. 다음은 XgBoost

- XgBoost으로 결과내보기. 카글에 은근 쓰이는 모델

- 다만.. 얘가 분류전문인지....

- KNN등 다른 모델들과 비교해보면서...

- 성능향상은 다른 모델들 해보고 나서..


```python
# f1-score등 성능확인
xgb_res = classification_report(y_test, y_predict + 1, ...)

# ......

df_xgb_res
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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>STANDING</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>SITTING</th>
      <td>0.959732</td>
      <td>0.947020</td>
      <td>0.953333</td>
      <td>151.000000</td>
    </tr>
    <tr>
      <th>LAYING</th>
      <td>0.928571</td>
      <td>0.945455</td>
      <td>0.936937</td>
      <td>110.000000</td>
    </tr>
    <tr>
      <th>WALKING</th>
      <td>0.989247</td>
      <td>0.978723</td>
      <td>0.983957</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>WALKING_DOWNSTAIRS</th>
      <td>0.963303</td>
      <td>0.972222</td>
      <td>0.967742</td>
      <td>108.000000</td>
    </tr>
    <tr>
      <th>WALKING_UPSTAIRS</th>
      <td>0.972477</td>
      <td>0.972477</td>
      <td>0.972477</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.969529</td>
      <td>0.969529</td>
      <td>0.969529</td>
      <td>0.969529</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.968888</td>
      <td>0.969316</td>
      <td>0.969074</td>
      <td>722.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.969651</td>
      <td>0.969529</td>
      <td>0.969563</td>
      <td>722.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### XGBoost 분류 결과로...

- XGBoost 답게, 넘사벽의 로지스틱과 SVM만큼은 아니더라도 대충넣은 옵으로 96.9프로 정도

- 다른모델과 다른점은 이 인코딩 라벨을 0부터 시작하지 않으면 삑삑 거림..  

그래서.. y_train - 1, y_predict + 1 이유는 이래서 있는거..

- f1-score 최저값은 LAYING 93.6%정도.

- 평균은 96.9%정도.. XGBoost 모델은 AI모델에서 은근 고급?으로 생각하고 있었는데.. 역시나.

- XGBoost 방식이 랜덤포레스트와 달리 boost 앙상블 같은 다른 방식으로 해결하려하는 모델.  

비슷하거나 경우에 따라선 좋은 결과를 나올 수도 있음

- 다른모델도 이와같이 12, 13 프로세스 반복해 나가면서 진행.

- lg 그 외 옵션등으로 성능향상 부분은 나중에..


##### 정규화 데이터로...

- 이때까지는 정규화 없이 진행 정규화 이후로 어떠한 데이터가 나올지 확인해보자.

- 원래 NN까지 할려고 했는데 생각할게 많아서.. 이 부분은 캡스톤 외에..  

아니면 끝나고도 추후 2-3차적으로 올릴때 보안하는 쪽으로..



```python
models = [
   ...
]
resout1 = []

for model in models:
  #......
  resout1.append(df_outres.copy())
  
  
# ......

# f1-score등 성능확인
out_restxt = classification_report(y_test, y_predict + 1, ...)

# ......
resout1.append(df_outres.copy())
```

#### 14. df_test 에 대한 분류 하여 “activity” 컬럼 작성하여 test_result.csv 파일 만들기



- 'df_test'의 경우 정답 없는 문제지라 AI에게 테스트로 predict로 던져보는... 

- 디코딩해서 만든게 test_result.csv>

- 즉 성능 보고가 아닌.. df_test의 답지를 만들라는거.. 각 모델에 따라..

- 먼저 성능 보고 하고 답지만들고, 답지 출력


<pre>
94.0443
</pre>

```python
# 각 성능지표데이터에 index name 붙이기
resout_name = [
  "knn algorithm",
  "dTree algorithm",
  ......
  "GNB algorithm_norm",
  "XGB algorithm_norm",
]

......
  
pmf1s

# 아래와 같이 성적표 정리를 해둠.
# 참고 및 모델선택은 추후 아래에..
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
      <th>models</th>
      <th>micro f1s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>knn algorithm</td>
      <td>94.0443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dTree algorithm</td>
      <td>92.3823</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LogisR algorithm</td>
      <td>97.3684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RFC algorithm</td>
      <td>95.5679</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SVM algorithm</td>
      <td>97.7839</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GNB algorithm</td>
      <td>80.4709</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGB algorithm</td>
      <td>96.9529</td>
    </tr>
    <tr>
      <th>7</th>
      <td>knn algorithm_norm</td>
      <td>93.6288</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dTree algorithm_norm</td>
      <td>90.8587</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LogisR algorithm_norm</td>
      <td>97.6454</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RFC algorithm_norm</td>
      <td>92.5208</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SVM algorithm_norm</td>
      <td>97.3684</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GNB algorithm_norm</td>
      <td>75.6233</td>
    </tr>
    <tr>
      <th>13</th>
      <td>XGB algorithm_norm</td>
      <td>95.0139</td>
    </tr>
  </tbody>
</table>
</div>


- 다음 위 모델 가지고 result_test.csv 출력

- X_test 지정

- 한꺼번에 처리

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
      <th>knn algorithm</th>
      <th>dTree algorithm</th>
      <th>LogisR algorithm</th>
      <th>RFC algorithm</th>
      <th>SVM algorithm</th>
      <th>GNB algorithm</th>
      <th>XGB algorithm</th>
      <th>knn algorithm_norm</th>
      <th>dTree algorithm_norm</th>
      <th>LogisR algorithm_norm</th>
      <th>RFC algorithm_norm</th>
      <th>SVM algorithm_norm</th>
      <th>GNB algorithm_norm</th>
      <th>XGB algorithm_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1536</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1537</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1538</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1539</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>6</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>1541 rows × 14 columns</p>
</div>



- 디코딩으로 마무리

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
      <th>knn algorithm</th>
      <th>dTree algorithm</th>
      <th>LogisR algorithm</th>
      <th>RFC algorithm</th>
      <th>SVM algorithm</th>
      <th>GNB algorithm</th>
      <th>XGB algorithm</th>
      <th>knn algorithm_norm</th>
      <th>dTree algorithm_norm</th>
      <th>LogisR algorithm_norm</th>
      <th>RFC algorithm_norm</th>
      <th>SVM algorithm_norm</th>
      <th>GNB algorithm_norm</th>
      <th>XGB algorithm_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>SITTING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>SITTING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>SITTING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>STANDING</td>
      <td>SITTING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>SITTING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1536</th>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
    </tr>
    <tr>
      <th>1537</th>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
    <tr>
      <th>1538</th>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
    <tr>
      <th>1539</th>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING_DOWNSTAIRS</td>
      <td>WALKING</td>
      <td>WALKING_DOWNSTAIRS</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
  </tbody>
</table>
<p>1541 rows × 14 columns</p>
</div>


#### 여기까지 정리 및 보고

- 위 각 모델 정리된 pandas데이터를 csv로...

- 지금 기록된 csv와 여기 현재까지 해둔 jupyter로 제출

- 다만 위와 같이 한건 그 당시 전체 출력하면 되겠다는 학구열의 오류.

- 진짜 결론은 아래를 참조


##### 결론!

- 여기서 제일 좋은 모델로 제일 좋은 예측한 답안지를 제출해야함

- 그러면 그래프를 그려본다. 성능 지수는 평균 f1기준..

- 여기까지 기록만 본다면 SVM이 기준.

- 위 svm f1 성능 기록 : 97.78% (아래 지표 및 선택한 근거를 참조)

- SVM기준의 답지등으로 기존 test.csv를 activate에 합쳐서 제출하도록 합시다.


- 아래 데이터는 어느 데이터가 f1s micro 기준으로 제일 나은지를 보여주는 지표


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
      <th>models</th>
      <th>micro f1s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>knn algorithm</td>
      <td>94.0443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dTree algorithm</td>
      <td>92.3823</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LogisR algorithm</td>
      <td>97.3684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RFC algorithm</td>
      <td>95.5679</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SVM algorithm</td>
      <td>97.7839</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GNB algorithm</td>
      <td>80.4709</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGB algorithm</td>
      <td>96.9529</td>
    </tr>
    <tr>
      <th>7</th>
      <td>knn algorithm_norm</td>
      <td>93.6288</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dTree algorithm_norm</td>
      <td>90.8587</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LogisR algorithm_norm</td>
      <td>97.6454</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RFC algorithm_norm</td>
      <td>92.5208</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SVM algorithm_norm</td>
      <td>97.3684</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GNB algorithm_norm</td>
      <td>75.6233</td>
    </tr>
    <tr>
      <th>13</th>
      <td>XGB algorithm_norm</td>
      <td>95.0139</td>
    </tr>
  </tbody>
</table>
</div>


<pre>
97.7839
</pre>
- 위와 같이 최대값을 출력할 때, 97.7839%의 SVM 알고리즘이 최고의 성적표를 제출.

- 정규화했을경우 오히려 살짝 성능 저하를 보여줌

- 위 데이터만의 결론으로 SVM으로 진행.

<pre>
0                 STANDING
1                 STANDING
2                 STANDING
3                 STANDING
4                 STANDING
               ...        
1536    WALKING_DOWNSTAIRS
1537      WALKING_UPSTAIRS
1538      WALKING_UPSTAIRS
1539    WALKING_DOWNSTAIRS
1540      WALKING_UPSTAIRS
Name: activity, Length: 1541, dtype: object
</pre>

```python
from sklearn import svm # support vector machine
# 모델 생성
svm = svm.SVC(kernel = 'linear', ...)

# ......

print(df_svm_res)
```

<pre>
                    precision    recall  f1-score     support
STANDING             1.000000  0.993333  0.996656  150.000000
SITTING              0.948052  0.966887  0.957377  151.000000
LAYING               0.953271  0.927273  0.940092  110.000000
WALKING              1.000000  1.000000  1.000000   94.000000
WALKING_DOWNSTAIRS   0.990826  1.000000  0.995392  108.000000
WALKING_UPSTAIRS     1.000000  1.000000  1.000000  109.000000
accuracy             0.980609  0.980609  0.980609    0.980609
macro avg            0.982025  0.981249  0.981586  722.000000
weighted avg         0.980644  0.980609  0.980574  722.000000
</pre>
- 여기서 98프로 성능으로 나오는건 그외 C, degree등 추가 옵션으로 더 나은 성능으로 데이터를 뽑아낸 것
- 이후 test예시 데이터와 답지를 합쳐서 제출.


```python
....to_csv('./test_result.csv')
```

#### 그외 참조 사이트



- matlip chart 등 확인  

https://pythonspot.com/matplotlib-bar-chart/  

  

- pandas drop확인  

https://fhaktj8-18.tistory.com/entry/drop-%ED%95%A8%EC%88%98-%ED%99%9C%EC%9A%A9%ED%95%B4%EC%84%9C-%ED%96%89-%EC%82%AD%EC%A0%9C%ED%95%98%EA%B8%B0  

  

- 정규화 StandardScaler 기능 및 확인  

https://jimmy-ai.tistory.com/139  




#### 마무리

- 분류모델에 쓰인 온갖 전처리부터, 모델까지 정리하는 유익한 시간.
- 앞에서도 언급했으나 여러모델들을 테스트해서 출력해보는 학구열은 좋았으나 여기서 목적은..
- 고객에게 예시와 함께 판별가능한 모델과 동시에 솔루션을 제공하는게 목적.
- 풀어낸 csv 그리고 성능지표등을 보여주기만 하면 됬었을..
- 사족이 많고 신경망도 추후 개선할 예정이지만 그래도 캡스톤을 완수한 평가는 있음.
