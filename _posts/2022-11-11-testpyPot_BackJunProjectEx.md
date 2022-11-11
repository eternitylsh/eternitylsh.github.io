---
layout: single
title:  "자신나름의 포트폴리오"
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


자신나름의 포트폴리오2

================


### 백준 프로젝트 test1(브루트 포스) : 블랙잭



https://www.acmicpc.net/problem/2798



#### 문제 : 

카지노에서 제일 인기 있는 게임 블랙잭의 규칙은 상당히 쉽다. 카드의 합이 21을 넘지 않는 한도 내에서, 카드의 합을 최대한 크게 만드는 게임이다. 블랙잭은 카지노마다 다양한 규정이 있다.



한국 최고의 블랙잭 고수 김정인은 새로운 블랙잭 규칙을 만들어 상근, 창영이와 게임하려고 한다.



김정인 버전의 블랙잭에서 각 카드에는 양의 정수가 쓰여 있다. 그 다음, 딜러는 N장의 카드를 모두 숫자가 보이도록 바닥에 놓는다. 그런 후에 딜러는 숫자 M을 크게 외친다.



이제 플레이어는 제한된 시간 안에 N장의 카드 중에서 3장의 카드를 골라야 한다. 블랙잭 변형 게임이기 때문에, 플레이어가 고른 카드의 합은 M을 넘지 않으면서 M과 최대한 가깝게 만들어야 한다.



N장의 카드에 써져 있는 숫자가 주어졌을 때, M을 넘지 않으면서 M에 최대한 가까운 카드 3장의 합을 구해 출력하시오.


#### 입력



첫째 줄에 카드의 개수 N(3 ≤ N ≤ 100)과 M(10 ≤ M ≤ 300,000)이 주어진다. 둘째 줄에는 카드에 쓰여 있는 수가 주어지며, 이 값은 100,000을 넘지 않는 양의 정수이다.



합이 M을 넘지 않는 카드 3장을 찾을 수 있는 경우만 입력으로 주어진다.


#### 출력



첫째 줄에 M을 넘지 않으면서 M에 최대한 가까운 카드 3장의 합을 출력한다.


##### 예제 입력 1

5 21 <br>

5 6 7 8 9



##### 예제 출력1

21



##### 예제 입력2

10 500 <br>

93 181 245 214 315 36 185 138 216 295



##### 예제 출력2

497



```python
import random

# rand func.
def getrandnum(min, max):
    return random.randrange(min, max + 1)
```

##### 입력부 구현



```python
# N : maxcardcount
# M : sumans
maxcardcount = getrandnum(3, 100)
sumans = getrandnum(10, 300000)

print( str(maxcardcount) + " " + str(sumans) )

# r_num 초기화는 필요없으나 혹시나 모르니..
def Init():
    safecount = 0
    r_num = 0
    numstr = ""
    cards = []

# 초기 초기화는 어쩔수 없...
safecount = 0
r_num = 0
numstr = ""
cards = []

while(1):
    
    for c in range(maxcardcount):
        r_num = getrandnum(1, 100000)
        if( int(sumans / 3) >= r_num ): safecount += 1

        numstr += ( str(r_num) + " " )
        cards.append( r_num )
    
    if 3 <= safecount: # 이 조건 만족못하면 입력부에서 답 없는 문제를 출시..
        break
    # 다시 하는것이기에 초기화.
    Init()
    
print(numstr)
```

<pre>
34 122757
80069 88443 83544 88379 63076 27733 1591 70400 95584 40375 29487 61732 56811 34377 81038 43500 73060 56612 47350 91256 25558 88968 57690 79169 84097 46506 76424 87710 72774 99977 39001 8891 81987 76115 
</pre>
##### 출력부 구현



```python
### 카드 배열 복사 안하고 하는것이기에 2번이상 반복 실행시.. 제대로 실행안됨.
## 메모리 아낄려고;;; 다시 할려면 전반 입력부 실행하고 할것.
## 만일 같게 할거면 해당 카드 배열을 변수 새로선언 복사해서 해야함.

sel = []
maxoutcardcount = 3 # 출력시 뽑아 계산하는 카드는 3장만..
ans = 300000 # 일부러 최대값 지정.. sumans보다 작거나 같으면 종료.

def cards_pop():
    selnum = max(cards)
    sel.append(selnum)
    cards.remove(selnum)

# 여기서 문제는.. 실제 클경우.. 다음 대안으로.. 어떻게 할것인가...
while( sumans < ans ):
    
    # 선택된 카드가 3장 이상인데 여기로 왔을 경우..
    # 합이 목적보다 많기에.. 제일 큰 수를 빼고 다음 카드의 수를 넣는다.
    if( 3 <= len(sel) ):
        sel.remove(max(sel))
        cards_pop()
    else:
        for c in range(maxoutcardcount):
            cards_pop()
    ans = sum(sel)

print(ans)
```

<pre>
113753
</pre>
### 마무리

블랙잭을 위와 같은 요구사항대로 구현해봄



##### 입력부와 출력부로 나누어 구현.

둘로 나누어 구현하는 것으로 과제에 맞게 답의 가시성을 잘 표현

물론 둘로 나누었기에 따로 실행의 번거로움이 있음.



##### 입력부에서 나온 수들 중 제일 큰 수 순서로 3장 ...

이 보다 더 클시 선택된 수 중에 제일 큰 수를 버리고 다음 큰수를 선택해서 계속 합해서 제일 가까운 수를 구하는 방법으로 구현



##### 아쉬운점

- 쓸데없는 변수가 많을 지도 모르나 줄이기에 시간을 더 들여야 함.

- 입력부에서 답 안나오는 입력결과가 나올시 다시 하는걸로.. 문제는 비용이 너무 들기에 개선책이 필요..

- 수가 적고 아슬아슬 safecount 조건을 만족할때.. 입력 답과 출력 답 오차 최소화도 과제

- 출력부에서도 3장 다 뽑았을때, 너무 클 경우 큰수를 빼고 다음 큰 수를 넣는 과정도 비용이 괜찮은지 검토 필요..



***********************


### 백준 프로젝트 TEST2(집합과 맵) : 숫자 카드 2



https://www.acmicpc.net/problem/10816



#### 문제

숫자 카드는 정수 하나가 적혀져 있는 카드이다. 상근이는 숫자 카드 N개를 가지고 있다.

정수 M개가 주어졌을 때, 이 수가 적혀있는 숫자 카드를 상근이가 몇 개 가지고 있는지 

구하는 프로그램을 작성하시오.


#### 입력



첫째 줄에 상근이가 가지고 있는 숫자 카드의 개수 N(1 ≤ N ≤ 500,000)이 주어진다. 둘째 줄에는 숫자 카드에 적혀있는 정수가 주어진다. 숫자 카드에 적혀있는 수는 -10,000,000보다 크거나 같고, 10,000,000보다 작거나 같다.



셋째 줄에는 M(1 ≤ M ≤ 500,000)이 주어진다. 넷째 줄에는 상근이가 몇 개 가지고 있는 숫자 카드인지 구해야 할 M개의 정수가 주어지며, 이 수는 공백으로 구분되어져 있다. 이 수도 -10,000,000보다 크거나 같고, 10,000,000보다 작거나 같다.


#### 출력



첫째 줄에 입력으로 주어진 M개의 수에 대해서, 각 수가 적힌 숫자 카드를 상근이가 몇 개 가지고 있는지를 공백으로 구분해 출력한다.


##### 예제 입력1



10 <br>

6 3 2 10 10 10 -10 -10 7 3 <br>

8 <br>

10 9 -5 2 3 4 5 -10 <br>



##### 예제 출력1



3 0 0 1 2 0 0 2



```python
# import 및 함수 getrandnum의 경우 전부 test1 import 실행시 받아옴.
# 만일 추가적으로 여기만의 import 및 공통 함수 정의시 여기서 가져오거나 정의
```

##### 입력부 구현



```python
# cs : cards
# N : cs1 : have player card
# M : cs2 : must get cards
card1s = []
card2s = []

r_num = 0
numstr = ""

def genr_card():
#    return getrandnum(-10000000, 10000000)
    return getrandnum(-10, 10)

### card1 입력
# cs1_max = getrandnum(1, 500000)
cs1_max = getrandnum(1, 20)
print(cs1_max)

for i in range(cs1_max):
    r_num = genr_card()
    numstr += ( str(r_num) + " " )
    card1s.append(r_num)

print(numstr)
numstr = ""

### card2 입력
# cs2_max = getrandnum(1, 500000)
cs2_max = getrandnum(1, 20)
print(cs2_max)

for i in range(cs2_max):
    r_num = genr_card()
    numstr += ( str(r_num) + " " )
    card2s.append(r_num)
    
print(numstr)
```

<pre>
13
-9 -9 2 10 7 -5 10 4 7 10 -3 -7 8 
6
-3 0 -3 -6 -9 -4 
</pre>
##### 출력부 구현



```python
numstr = ""
anscount = 0
# cnum : must check num, hnum : player have num
for cnum in card2s:
    for hnum in card1s:
        if( cnum == hnum ): anscount += 1

    numstr += ( str(anscount) + " " )
    anscount = 0

print(numstr)
```

<pre>
1 0 1 0 2 0 
</pre>
### 마무리



문제에 따라 집합과 맵 숫자카드2 구현..



##### 문제에 따라 입력, 출력 나누어 구현

역시 가시성에 맞게.. 물론 2번 실행해야하니..

다만 이번에는 출력부분 여러번 실행해도 문제없... 그럴일이 있나..



##### 아직 미비한 부분이 있음...

2번째 카드 중첩 안되야하는데 중첩안되도록 막는 기능이 추가되어야함..

일단 이게 끝나고 아쉬운 점을 넣고 마무리를 하도록하겠음.


#### 마지막으로..



https://www.acmicpc.net/



해당 프로젝트는 백준 알고리즘 사이트에 나온 문제를 참고해서 나 자신나름 풀어보고 이상 포트폴리오를 작성했음을 밝힙니다.

