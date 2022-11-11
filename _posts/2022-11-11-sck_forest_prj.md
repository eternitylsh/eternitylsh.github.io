---
layout: single
title:  "sklearn 숲데이터를 끄적여보기"
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


SCK 숲 데이터 셋 다루기 포트폴리오

=============================



## SCK 숲 데이터



***********************



https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes



이 데이터 세트의 샘플은 각 패치의 덮개 유형, 즉 우세한 수종을 예측하는 작업을 위해 수집된 미국의 30×30m 숲 패치에 해당합니다. 7가지 커버타입이 있어 이를 다중 클래스 분류 문제로 만듭니다. 각 샘플에는 데이터세트의 홈페이지 에 설명된 54개의 기능이 있습니다 . 일부 기능은 부울 표시기이고 다른 기능은 이산 또는 연속 측정입니다.



***********************



이러한 설명으로 업로드 되어있는 미국에 있는 숲관련 데이터를 받아

이것저것 분석해본 포트폴리오임을 밝힌다.

<br>

by **E Creator**




```python
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
```


```python
from sklearn.datasets import fetch_covtype

covtype = fetch_covtype()
print(covtype.DESCR)
```

<pre>
.. _covtype_dataset:

Forest covertypes
-----------------

The samples in this dataset correspond to 30×30m patches of forest in the US,
collected for the task of predicting each patch's cover type,
i.e. the dominant species of tree.
There are seven covertypes, making this a multiclass classification problem.
Each sample has 54 features, described on the
`dataset's homepage <https://archive.ics.uci.edu/ml/datasets/Covertype>`__.
Some of the features are boolean indicators,
while others are discrete or continuous measurements.

**Data Set Characteristics:**

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

:func:`sklearn.datasets.fetch_covtype` will load the covertype dataset;
it returns a dictionary-like 'Bunch' object
with the feature matrix in the ``data`` member
and the target values in ``target``. If optional argument 'as_frame' is
set to 'True', it will return ``data`` and ``target`` as pandas
data frame, and there will be an additional member ``frame`` as well.
The dataset will be downloaded from the web if necessary.

</pre>

```python
df = pd.DataFrame(covtype.data, 
                  columns=["x{:02d}".format(i + 1) for i in range(covtype.data.shape[1])],
                  dtype=int)
sy = pd.Series(covtype.target, dtype="category")
df['covtype'] = sy
df.tail()
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
      <th>x01</th>
      <th>x02</th>
      <th>x03</th>
      <th>x04</th>
      <th>x05</th>
      <th>x06</th>
      <th>x07</th>
      <th>x08</th>
      <th>x09</th>
      <th>x10</th>
      <th>...</th>
      <th>x46</th>
      <th>x47</th>
      <th>x48</th>
      <th>x49</th>
      <th>x50</th>
      <th>x51</th>
      <th>x52</th>
      <th>x53</th>
      <th>x54</th>
      <th>covtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>581007</th>
      <td>2396</td>
      <td>153</td>
      <td>20</td>
      <td>85</td>
      <td>17</td>
      <td>108</td>
      <td>240</td>
      <td>237</td>
      <td>118</td>
      <td>837</td>
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
      <td>3</td>
    </tr>
    <tr>
      <th>581008</th>
      <td>2391</td>
      <td>152</td>
      <td>19</td>
      <td>67</td>
      <td>12</td>
      <td>95</td>
      <td>240</td>
      <td>237</td>
      <td>119</td>
      <td>845</td>
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
      <td>3</td>
    </tr>
    <tr>
      <th>581009</th>
      <td>2386</td>
      <td>159</td>
      <td>17</td>
      <td>60</td>
      <td>7</td>
      <td>90</td>
      <td>236</td>
      <td>241</td>
      <td>130</td>
      <td>854</td>
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
      <td>3</td>
    </tr>
    <tr>
      <th>581010</th>
      <td>2384</td>
      <td>170</td>
      <td>15</td>
      <td>60</td>
      <td>5</td>
      <td>90</td>
      <td>230</td>
      <td>245</td>
      <td>143</td>
      <td>864</td>
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
      <td>3</td>
    </tr>
    <tr>
      <th>581011</th>
      <td>2383</td>
      <td>165</td>
      <td>13</td>
      <td>60</td>
      <td>4</td>
      <td>67</td>
      <td>231</td>
      <td>244</td>
      <td>141</td>
      <td>875</td>
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
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



```python
pd.DataFrame(df.nunique()).T
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
      <th>x01</th>
      <th>x02</th>
      <th>x03</th>
      <th>x04</th>
      <th>x05</th>
      <th>x06</th>
      <th>x07</th>
      <th>x08</th>
      <th>x09</th>
      <th>x10</th>
      <th>...</th>
      <th>x46</th>
      <th>x47</th>
      <th>x48</th>
      <th>x49</th>
      <th>x50</th>
      <th>x51</th>
      <th>x52</th>
      <th>x53</th>
      <th>x54</th>
      <th>covtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1978</td>
      <td>361</td>
      <td>67</td>
      <td>551</td>
      <td>700</td>
      <td>5785</td>
      <td>207</td>
      <td>185</td>
      <td>255</td>
      <td>5827</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 55 columns</p>
</div>



```python
df.iloc[:, 10:54] = df.iloc[:, 10:54].astype('category')
```


```python
import seaborn as sns
```


```python
df_count = df.pivot_table(index="covtype", columns="x14", aggfunc="size")
sns.heatmap(df_count, cmap=sns.light_palette("gray", as_cmap=True), annot=True, fmt="0")
plt.show()
```

<img src="data:image/png;base64,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"/>

#### 마무리







- 아직은 여러사정상 숲데이터

