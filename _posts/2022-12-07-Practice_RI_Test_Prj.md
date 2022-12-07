---
layout: single
title:  "강화학습으로 연습해보기"
categories: jupyter
tag: [python, blog, jekyll]
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


# Q-learning 통한 강화학습



강화학습은 지도나 비지도학습과는 다른 형태의 학습 데이터를 사용해야 하는 기계학습의 한 종류입니다. 강화학습은 환경을 통해 학습해야 하는 상황에서 명시적으로 사용됩니다. 여러분은 개가 어떻게 재주를 배우는지 생각해 본 적이 있나요? 어떤 방법으로 개를 훈련시킬 수 있는지 생각해 봅시다. 



개는 우리의 언어를 이해하지 못하기 때문에 특정 재주를 부리는 방법을 가르쳐야 합니다. 우리는 개에게 무엇을 하라고 말할 수 없기 때문에 다른 전략을 세워야 합니다. 우리는 개에게 명령이나 신호를 보낼 수 있습니다. 예를 들어 개를 앉히고 싶다면, 바닥을 가리키며 '앉아!'라고 말합니다. 이 시점에서 개는 우리의 명령에 반응할 것입니다. 반응 유형에 따라 반려견에게 보상을 제공합니다. 개가 아무것도 하지 않거나, 움직이면 보상을 하지 않습니다. 개가 앉는 경우에만 우리는 그것에 대한 보상을 합니다. 개는 긍정적인 경험을 통해 무엇을 해야 하는지 배우고 있습니다. 



이제 몇 가지 주요 용어를 살펴보겠습니다.



1. 여기서 에이전트(Agent)는 개입니다.

2. 행동의 결과를 우리가 제공하기 때문에 환경(environment)은 우리 자신입니다.

3. 한 상태에서 다른 상태로 움직이는 것은 개의 행동(action) 입니다.

4. 상태(state)는 개의 움직임 입니다. 예: 앉기, 서기, 걷기

5. 보상(reward)은 개가 알고 있는 받은 간식의 수 입니다. 



이제 강화 학습의 예를 살펴보겠습니다. 다음은 한 위치에서 승객을 태우고 다른 위치에서 내려야 하는 게임입니다. 어떻게 해야 할까요? 몇 가지 라이브러리 가져오기를 먼저 시작하겠습니다.




## 1. 라이브러리 가져오기


라이브러리가 설치되어 있지 않은 경우 터미널에서 다음 단계를 수행하십시오. <br>

pip install gym



```python
%pip install gym
```


```python
%pip install gym[toy_text]
```


```python
%pip install stable_baselines3
```


```python
from collections import defaultdict
import pickle
import random
from IPython.display import clear_output
import numpy as np

import click
import gym
```

`gym.make()` 함수를 사용하여 환경을 만들고 게임을 할 것입니다. 아래 코드를 실행하고 시도하십시오.



```python
env = gym.make("Taxi-v3")
# .env

```

환경이란 무엇입니까?



OpenAI Gym은 강화 학습 알고리즘을 개발하고 비교하기 위한 툴킷입니다. 이것은 표준화된 환경 세트에 접근할 수 있는 gym 오픈 소스 라이브러리입니다.



Open AI Gym은 환경-에이전트로 구성되어 있습니다. 이는 단순히 "환경"에서 특정 작업을 수행할 수 있는 "에이전트"에 대한 접근 권한을 제공한다는 의미입니다. 그 대가로 환경에서 특정한 행동을 수행한 결과로 관찰과 보상을 받습니다.



이것은 Gym 라이브러리를 사용하면 우리가 에이전트를 두고 그것에 대한 행동을 수행할 수 있는 "환경"을 만들 수 있음을 의미합니다.



![Class](resources/class.jpg)



비유하자면 교실 안에서 혼자 있는 자신을 생각해 보십시오. **이 경우 여러분은 에이전트이고 교실은 환경입니다. 그리고 만약 여러분이 책을 가지러 가기로 선택한다면, 그것은 행동입니다.**


현재 작업하고 있는 gym 환경에 대해 조금 더 알아보겠습니다. [이 링크](https://www.gymlibrary.dev/environments/toy_text/taxi/)로 이동하여 환경 소스를 확인하십시오. 이 환경에서 몇 가지 기능을 테스트해 보고 설정을 시작하겠습니다.



환경을 재설정할 수 있는 기능을 찾을 수 있습니까? 아래 코드 블록에서 실행하십시오. 출력은 어떻습니까? 출력은 무엇을 나타냅니까?



학생 스스로 답변을 작성해 보세요.



```python
# Student Answer
env.reset()

# The number represents the state of the environment. 
```

<pre>
326
</pre>
교실에 있는 학생의 예에서 여러분이 책을 가지러 간다면 이는 하나의 행동으로 간주됩니다. 



위의 코드에서 우리는 'Taxi-V3'라는 환경을 만들었습니다. 이러한 환경에서 우리는 택시 시뮬레이션을 진행하려고 합니다. 시뮬레이션은 참고로 아래와 같이 보입니다.



![Taxi](resources/taxi.png)


그러면 이 환경에서 택시가 취할 수 있는 행동(action)은 무엇입니까? 추측해 보세요.




```python
# student answer here
# front, back, right
```

에이전트 택시는 다음과 같이 6가지 행동을 선택할 수 있습니다.



0 = 남쪽(south)  

1 = 북쪽(north)  

2 = 동쪽(east)  

3 = 서쪽(west)  

4 = 픽업(pickup)  

5 = 하차(dropoff)  



환경 속에서 한 단계를 수행할 수 있는 기능이 있습니다.



'env.step()' 함수를 사용하여 작업을 실행할 수 있습니다.아래 답변을 작성해 보세요.



```python
# Student answer
act = 5
env.step(act)

```

<pre>
(326, -10, False, {'prob': 1.0})
</pre>
'env.render()' 함수를 사용하여 환경을 표시해 보세요.



```python
env.render()
```

<pre>
+---------+
|R: | : :[34;1mG[0m|
| : | : : |
| : : : : |
| |[43m [0m: | : |
|[35mY[0m| : |B: |
+---------+
  (Dropoff)
</pre>
### while 루프를 사용하여 게임의 인스턴스를 생성해 보십시오. 



Let us try to make an instance of the game. For this we need to create a while loop first. Do you remember what a while loop is? Try to make it on your own first.

게임의 인스턴스를 만들어 봅시다. 이를 위해 먼저 while 루프를 만들어야 합니다. while 루프가 무엇인지 기억하십니까? 먼저 직접 만들어 보세요.

<br>



```python
# 해당 코드는 설명..
# done = False
# while not done:
    # 환경 렌더링
    # 입력 받기
```

'env.render()' 함수를 사용하여 환경을 렌더링하고, 'input()'을 사용하여 입력을 얻을 수 있습니다. 위의 코드에서 이를 대체해 보십시오.




```python
env.reset()

done = False
while not done:
    env.render() # 이 줄에서 환경 렌더링 함수를 적용합니다.
    i = int(input())
```

입력을 받으면 다음 단계는 무엇입니까? 그것은 실행 단계와 함께 환경을 새로 고치는 것입니다. 그러기 위해서는 환경에서 한 단계를 실행한 다음 출력을 제거하고 환경을 다시 렌더링해야 합니다.




```python
done = False
while not done:
    env.render() # 이 줄에서 환경 렌더링 함수를 적용합니다.
    i = int(input())
    # 단계 실행
    # 출력 제거
```

여기서 `clear_output(wait=True)` 함수를 사용하여 출력을 지우고 `obs,reward,complete,info = env.step(i)`을 사용하여 단계를 실행할 수 있습니다. 단계 함수에서 얻은 변수는 무엇입니까?



```python
done = False
while not done:
    env.render() # 이 줄에서 환경 렌더링 함수를 적용합니다.
    i = int(input())
    obs,reward,complete,info = env.step(i) # 여기에서 환경에 대한 단계를 실행
    clear_output(wait=True)
```

이러한 변수들은 중요합니다. 변수들은 우리에게 환경의 상태를 있는 그대로 알려줍니다. Obs는 택시의 위치와 환경의 다른 부분에 대한 정보를 제공합니다. 보상은 그 행동이 긍정적인 결과를 가져왔는지 여부를 알려줍니다. 완료는 승객을 태우거나 내려주려는 의도한 목표가 달성되었는지 알려줍니다. 마지막으로 정보는 우리에게 다양한 데이터를 제공합니다.



```python
done = False
while not done:
    env.render() # 이 줄에서 환경 렌더링 함수를 적용합니다.
    i = int(input())
    clear_output(wait=True)
    obs,reward,complete,info = env.step(i) # 여기에서 환경에 대한 단계를 실행
    print('Observation = ', obs, '\nreward = ', reward, '\ndone = ', complete, '\ninformation = ', info)
    done = complete
```

##### 실제 위 4개 셀 코드들을 실행해봄.

다만.. 미완성 코드라보니 강제 정지후 오류 코드가 거슬려서 지움


지금까지 우리는 환경과 함께 작업을 진행하였고, 문제를 이해했습니다. 몇가지 용어를 정의해 보겠습니다.



**상태(State)** - 상태는 위 코드에서 변수 'obs'에 의해 제공됩니다. 환경의 상태를 정의합니다.  

**에이전트(Agent)** - 위 예에서는 택시입니다.  

**행동(Action)** - 행동은 수행할 환경에 전달하는 변수입니다. 행동에 따라 에이전트가 작업을 수행합니다.

**보상(Reward)** - 보상은 플레이어가 얼마나 잘하고 있는지 알려주는 숫자입니다. '완료' 상태에 도달하는 단계가 적을수록 좋습니다.


## 2. Q-Learning



본질적으로 Q-learning은 에이전트가 환경의 보상을 사용하여 시간이 지남에 따라 주어진 상태에서 취해야 할 최상의 조치를 학습할 수 있도록 합니다.



AI에 무엇이 효과적 이었나를 기억하기 위해 각 단계의 결과를 **Q-table**이라는 테이블에 저장합니다. 이 테이블에는 (상태, 행동) -> Q-value의 맵이 있습니다. Q-value는 어떤 행동이 유익한지 아닌지를 나타내는 숫자입니다.



다음은 Q-table의 예제입니다.





![qlearning.png](resources/qlearning.png)



##### 참고로 States란..



현재 주어진 칸 25  

에이전트가 판단할 수 있는 종류 5  

택시가 가게 되는 경우의 수 4  

고로.. 계산대로 하게 될 경우  

  

**25 * 5 * 4 = 500**  

  

위 와 같은 결과가 나오기에 위 Q-table의 States는 500경우의 수가 된다.


Q-러닝 알고리즘을 효과적으로 구현하기 위해서는 몇 가지 하이퍼 파라미터가 필요합니다. 학습 과정을 진행하면서 다음을 값을 수정할 수 있습니다.



1. 알파(Alpha) 값. Alpha 값은 0에서 1 사이의 숫자입니다. 학습률의 척도입니다.

2. 감마(Gamma) 값. 이 값은 알고리즘이 얼마나 탐욕스러운지를 측정한 것입니다. 감마 값이 0이면 학습 알고리즘이 더 근시안적입니다.

3. 엡실론(Epsilon) 값. 이 변수는 훈련이 이전 데이터에 얼마나 의존해야 하고, 새로운 데이터에 얼마나 의존해야 하는지를 설정합니다.


이러한 매개변수 중 몇 가지를 더 자세히 살펴보겠습니다.



**Alpha**



알파 값은 모델이 학습하는 속도를 나타냅니다. 따라서 학습률이 높으면 모델은 무언가를 학습하는 데 한단계를 거치지만 학습률이 낮으면 모델은 학습하는 데 더 많은 단계를 수행합니다. 이것은 무엇을 의미할까요?



학습률이 너무 낮으면 학습하는 데 너무 많은 시간이 걸리기 때문에 학습률은 매우 중요합니다. 너무 높은 학습률은 우리에게 최적의 결과를 주지 못합니다. 따라서 올바른 학습률을 선택하는 것이 중요합니다. 아래 학습률에 대한 실행 속도 및 정확도의 예를 볼 수 있습니다. 종종 학습률은 시행착오의 게임입니다.



![lr](resources/lr.png)



**Gamma**



감마 값은 모델이 학습하는 방법을 결정하는 데 중요합니다. 감마가 너무 높으면 모델은 멀리서 크게 보고, 감마가 낮으면 너무 가깝게 자세히 봅니다. 시험 공부를 하는 학생의 예를 들어보겠습니다. 시험을 준비하기 위해 학생은 근시안적으로 계획성 없이 매일 무작으로 주제를 선정하여 집중적으로 공부할 수 있습니다. 또는 학생이 장기적으로 계획을 세웠지만 낮 시간에는 집중적으로 공부하지 않을 수 있습니다. 학생으로서 여러분 중 일부는 이러한 예들 중 하나와 관련이 있다고 확신합니다. 우리 중 어떤 사람들은 공부할 계획을 세우고 목표를 가지고 준비하지만 지금 당장은 공부하지 않습니다. 우리 중 일부는 현재 공부는 하고있지만 장기적인 비전과 계획이 부족합니다. 감마 값은 이러한 난제를 나타냅니다. 핵심은 장기와 단기 목표 균형에 초점을 맞추는 것처럼 적절한 감마 값을 갖는 것입니다.





![lr](resources/lr1.jpg)



**Epsilon**



우리가 과거의 실패로부터 더 많은 것을 배울 수 있는 모델을 원할 때 엡실론 값을 높일 수 있습니다. 이것이 우리 모델에 어떤 의미가 있을까요? 일부 모델은 다른 모델보다 과거의 경험으로부터 더 많은 혜택을 받습니다. 그리고 다시 올바른 값을 선택하는 것은 시행착오의 과정입니다. 다음 실습에서는 새로운 학습보다 오래된 학습에 더 집중하기를 원하기 때문에 0.1의 엡실론 값을 제공하였습니다.



```python
# 하이퍼 파라미터
alpha = 0.1
gamma = 0.6
epsilon = 0.1

NUM_EPISODES = 100000
```

### 에피소드 수는?



한 에피소드는 성공적인 택시 픽업 및 하차를 수행하는 한 번의 시도입니다. 따라서 한 에피소드 내에서 모델이 실패하거나 성공할 때까지 작업을 반복합니다.



다음 단계는 q-table을 만드는 것입니다. 위의 Q-table 이미지를 참조하여 축을 확인하십시오. 표의 x축에는 6개의 값이 있고, y축에는 500개의 값이 있습니다. 하지만 이것들을 수동으로 입력할 필요는 없습니다. 아래에서 q-table을 만드는 코드를 찾을 수 있습니다. np.zeros는 모든 값이 0인 표를 만듭니다. [여기](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html) 문서를 살펴보세요.




```python
q_table = np.zeros([env.observation_space.n, env.action_space.n])
```

#### Task: python에서 다음 함수를 풀어보세요.



$$

Q(state, action)  \leftarrow (1 -  \alpha ) *Q (state,action) +  \alpha (reward +  \gamma  \max Q(next state, all  Actions))

$$





가장 먼저 필요한 것은 상태, 행동에 대한 Q-value입니다. 이것을 어떻게 얻습니까? Q-table을 `q_table[state, action]`과 같이 참조하기만 하면 됩니다.



```python
# 완성되지 않은 코드 실행하면 에러..
# old_value = q_table[state, action]
# next_max = np.max(q_table[next_state])

# Student answer
# add line here

# q_table[state, action] = new_value
```

이제 모델을 훈련시키도록 합시다. 어떻게 시작할 수 있습니까? 먼저 기본 루프부터 다시 시작하겠습니다.



```python
# 이 코드 블록은 아직 완성되지 않아서 실행하면 에러가 발생합니다.
# for i in range(1, NUM_EPISODES+1):
    # code here
print("Training finished. yet not started;; \n")
```

<pre>
Training finished. yet not started;; 

</pre>
이 루프 안에 환경을 단계적으로 반복하는 것을 목표로 하는 또 다른 루프를 추가해야 합니다. 우리는 전에 이것을 했습니다. 코드를 복사해 보겠습니다.



```python
# 이 코드 블록은 아직 완성되지 않아서 실행하면 에러가 발생합니다.
# for i in range(1, NUM_EPISODES+1):
#     done = False
    
#     while not done:
        
#         next_state, reward, done, info = env.step(action) # 다음 단계를 수행합니다.
        
print("Training finished. yet not started;; \n")
```

<pre>
Training finished. yet not started;; 

</pre>
이제 단계를 수행할 수 없는 `action` 변수를 가져와야 합니다. 위의 Q-table을 사용하여 추천된 행동을 취할 수 있는 방법은 무엇입니까?

답은 'q_table[state]'를 사용하는 것이다.



```python
# 이 코드 블록은 아직 완성되지 않아서 실행하면 에러가 발생합니다.
# for i in range(1, NUM_EPISODES+1):
#     done = False
    
#     while not done:
#         action = np.argmax(q_table[state])
#         next_state, reward, done, info = env.step(action) # 다음 단계를 수행합니다.
        
print("Training finished. ......\n")
```

<pre>
Training finished. ......

</pre>
다음 단계는 엡실론 값을 포함하는 것입니다. 새로운 공간을 탐험할 확률이 10%라는 사실을 기억하시나요? 우리는 이것을 다음과 같이 코딩할 수 있습니다.



```python
# 이 코드 블록은 아직 완성되지 않아서 실행하면 에러가 발생합니다.
# for i in range(1, NUM_EPISODES+1):
#     done = False
    
#     while not done:
#         if random.uniform(0, 1) < epsilon: # 이전 지식을 사용하는 대신 새로운 행동을 탐구할 확률이 10%입니다.
#             action = env.action_space.sample() # 작업 공간 탐색
#         else:
#             action = np.argmax(q_table[state]) # 학습된 값 이용
            
#         next_state, reward, done, info = env.step(action) # 다음 단계를 수행합니다.
        
print("Training finished. .....\n")
```

<pre>
Training finished. .....

</pre>
다음 단계는 Q-table을 계산하고 업데이트하는 것입니다. 어떻게 하면 될까요? q-table의 새로운 값을 찾는 데 사용한 공식을 기억하십니까? 그 코드를 다시 사용하십시오.



```python
# state.....

# for i in range(1, NUM_EPISODES+1):
#     done = False
    
#     while not done:
#         if random.uniform(0, 1) < epsilon: # 이전 지식을 사용하는 대신 새로운 행동을 탐구할 확률이 10%입니다.
#             action = env.action_space.sample() # 작업 공간 탐색
#         else:
#             action = np.argmax(q_table[state]) # 학습된 값 이용
            
#         next_state, reward, done, info = env.step(action) # 다음 단계를 수행합니다.
        
        # 할일: 위의 공식을 사용하여 여기에 코드를 입력하세요.
        
print("Training finished. .....\n")
```

<pre>
Training finished. .....

</pre>
이제 결과를 출력하고 중요한 데이터를 저장하는 코드를 더 추가해 보겠습니다.



```python
all_epochs = []
all_penalties = []

for i in range(1, NUM_EPISODES+1):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon: # 이전 지식을 사용하는 대신 새로운 행동을 탐구할 확률이 10%입니다.
            action = env.action_space.sample() # 작업 공간 탐색
        else:
            action = np.argmax(q_table[state]) # 학습된 값 이용

        next_state, reward, done, info = env.step(action) # 다음 단계를 수행합니다.
        
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
```

<pre>
Episode: 100000
Training finished.

</pre>
축하합니다!!! 

Q-러닝 모델을 성공적으로 학습했습니다. 지도 및 비지도 학습 모델에서 우리는 모델 객체에 모델을 저장했지만, 강화학습의 경우는 어떻습니까? 

이 경우 모델이 무엇이며 어떻게 저장되는지 대답할 수 있습니까?



```python
# Student answer
# q_table : 훈련한 모델의 데이터가 저장. 상태, 활동값
```

## 3. 평가



이제 Q-table을 평가해 보겠습니다. 어떻게 하면 될까요? 우리는 Q-table을 업데이트하기 위해 공식을 추가하지 않는다는 점을 제외하고는 동일한 훈련 알고리즘을 사용합니다. 직접 해보십시오.



```python

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
```

<pre>
Results after 100 episodes:
Average timesteps per episode: 12.82
Average penalties per episode: 0.0
</pre>
모델 평가 완료를 축하합니다. 평가 결과는 무엇을 나타낼까요? 이 후에 강화 학습의 기본과 이를 위한 모델을 구축하는 방법을 쉽게 이해해야 합니다.


#### 파라미터를 조정하면서 최선의 결과는?



파라미터 조정을 해보아서 최선의 결과를 내보자.



```python
# 하이퍼 파라미터
alpha = 0.05
gamma = 0.8
epsilon = 0.5 

NUM_EPISODES = 100000
```

#### step1: 훈련



```python
all_epochs = []
all_penalties = []

for i in range(1, NUM_EPISODES+1):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon: # 이전 지식을 사용하는 대신 새로운 행동을 탐구할 확률이 10%입니다.
            action = env.action_space.sample() # 작업 공간 탐색
        else:
            action = np.argmax(q_table[state]) # 학습된 값 이용

        next_state, reward, done, info = env.step(action) # 다음 단계를 수행합니다.
        
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
```

<pre>
Episode: 100000
Training finished.

</pre>
#### step2 : 평가



```python

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
```

<pre>
Results after 100 episodes:
Average timesteps per episode: 13.17
Average penalties per episode: 0.0
</pre>
#### 마무리

- 강화학습은 역시 보상에 쫓는 다는점에서 비지도와 다른점을 보임.

- 강화학습의 대표적 예로 무인배달, 자동운전 등등..

- 위 최단결과만 볼때... 아쉽게도 12.3 이하를 못 내는건 아쉬움.

- 강화학습도 더 응용한 부분이 있을거라 생각.. 요즘은 딮뉴런(깊은 신경망)을 응용한 부분이 대표적예로..

(물론.. 위 사항은 지금까지 일수도 아닐 수도 있음.)

- 스스로 답을 찾는 머신러닝 강화학습의 좋은 면을 볼 수 있어 좋았다.

- 여기에 GAN이라는 적대적 생성 신경망의 경우 비지도와 강화학습과는 뭔가 다른 면을 보이기도 하고..


#### 주의사항



본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.

