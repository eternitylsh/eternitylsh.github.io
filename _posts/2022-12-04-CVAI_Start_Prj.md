---
layout: single
title:  "컴퓨터 비전 준비운동하기"
categories: jupyter
tag: [python, blog, jupyter, opencv]
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


# 컴퓨터 비전 소개


문서의 위아래로 탐색하려면 키보드의 방향키 키를 사용할 수 있습니다.<br />
이 노트북의 코드를 실행하려면 코드 블록을 선택하고 **Shift+Enter**를 누르십시오. <br />
코드 블록을 편집하려면 Enter 키를 누릅니다.

#### 노트북의 코드는 누적되어 사용됩니다.(정의된 변수는 노트북이 닫힐 때까지 계속 사용 가능) <br />
따라서 실행 순서를 위에서부터 아래로 진행하세요. 과정을 뛰어넘어 진행한다면 에러가 발생할 수 있습니다!


쥬피터 노트북 사용에 대한 추가 도움말을 보려면 위의 메뉴에서 도움말(Help) > 사용자 인터페이스 둘러보기(User Interface Tour)를 클릭하세요. <br />
또는 다음 사이트를 방문하여 살펴보세요. https://jupyter-notebook.readthedocs.io/en/stable/ui_components.html

여러분의 아이디어를 실험하고 테스트해보세요. 그것이 학습에 있어서 가장 빠른 방법 중 하나이기 때문입니다!

## 1. 컴퓨터는 어떻게 볼수 있을까요?

중고등 학생들은 학교의 모든 학생들을 알고 지낼수는 없지만 거리에서 교복을 입은 모습을 보고 같은 학교 학생이라고 구별할 수 있습니다. 학생들은 보기 위해 눈을 사용했고, 뇌는 그 정보를 처리했습니다.

컴퓨터가 비슷한 일을 하도록 할 수 있습니까?
<br />
어디서 부터 시작할 수 있을까요?

우리가 같은 학교 학생을 구별해 내는 2 단계와 동일하게 처리할 수 있습니다.<br />
1) 눈으로 보기<br />
2) 우리가 보는 것을 이해하기(거리에서 학생들이 입은 교복이 우리학교 교복이라는 것을 인식)

### 코드 시작하기

이 세션에서는 Python 및 OpenCV용 인텔® 배포판을 사용합니다.

또 다른 유용한 Python 라이브러리는 배열/행렬을 빠르게 처리할 수 있는른 Numpy 라이브러리입니다. 
이미지는 실제로 픽셀의 배열/행렬로 구성되므로 Numpy 라이브러리를 사용하여 더 빠르게 이미지 처리를 수행할 수 있습니다.

Python 및 Numpy에 대한 사용 경험이 없는 경우<br />
https://www.datacamp.com/courses/intro-to-python-for-data-science 에서 더 많은 정보를 확인하실 수 있습니다.

아래 코드 블록을 실행하려면 블록을 선택하고 **Shift+Enter**를 누르십시오. <br />
실행 결과는 코드 블록 바로 아래에 출력됩니다. 첫 코드 블록은 여러분의 컴퓨터에 설치된 OpenCV 및 Python 버전을 표시합니다.

### 라이브러리 가져오기


```python
# library 설치해보자.
!pip install opencv-python
```

    Collecting opencv-python
      Using cached opencv_python-4.6.0.66-cp36-abi3-win_amd64.whl (35.6 MB)
    Requirement already satisfied: numpy>=1.17.3 in c:\users\user\.conda\envs\myai\lib\site-packages (from opencv-python) (1.23.4)
    Installing collected packages: opencv-python
    Successfully installed opencv-python-4.6.0.66
    


```python
import cv2              # OpenCV 라이브러리 가져오기
import numpy as np      # Numpy 라이브러리 가져오기
import sys

print ("You have successfully installed OpenCV version "+cv2.__version__) 
print ("Your version of Python is " + sys.version)
```

    You have successfully installed OpenCV version 4.6.0
    Your version of Python is 3.9.13 | packaged by conda-forge | (main, May 27 2022, 16:51:29) [MSC v.1929 64 bit (AMD64)]
    


```python
import matplotlib.pyplot as plt
%matplotlib inline

# 단일 mat 그리기 함수
def matdraw(img):
    plt.axis('off') # 창에있는 x축 y축 제거
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
# 다수 mat 그리기 함수
def matsdraw(imgs, row, col):
    for idx in range(len(imgs)):
        plt.subplot(row, col, idx + 1)
        plt.axis('off') # 창에있는 x축 y축 제거
        plt.imshow(cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB))
    plt.show()
```

### 1.1 보기. 첫번째 사진을 화면에 표시해 봅시다!


```python
# 나중에 현재 jupyter cell 내에 실행할 수 있는 방법 찾아보는중;;;
# 참고로 아래 waitkey, destroyallwindow.. 이 처리 안하면.. 창이..

# - flags
#         - cv2.IMREAD_UNCHANGED : 원본 사용(-1), alpha channel까지 포함해읽음(png파일)
#         - cv2.IMREAD_GRAYSCALE : 그레이스케일로 읽음(0), 1 채널
#         - cv2.IMREAD_COLOR : COLOR로 읽음(1) , 3 채널, BGR 이미지 사용어올 파일명

img = cv2.imread("[Dataset] Module 20 images/image001.png", cv2.IMREAD_UNCHANGED)   # 이미지 파일을 메모리에 읽기
cv2.imshow("Image", img)                  # 해당 이미지 표시

cv2.waitKey(0)                            # 아무키나 누르면 이미지 표시창이 종료됩니다.
cv2.destroyAllWindows()

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(imgRGB)
plt.show()
```



![png](../assets/images/CVAI_Start_Prj_8_0.png)
    



```python
# plt로 띄워보자..
# cv2 format > plt format.. BGR > RGB
img = cv2.imread("[Dataset] Module 20 images/image001.png", cv2.IMREAD_GRAYSCALE)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(imgRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_9_0.png)
    



```python
# 마지막으로 cv2 color로..
img = cv2.imread("[Dataset] Module 20 images/image001.png", cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(imgRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_10_0.png)
    


#### 위의 코드 블록을 실행한 후 창에서 이미지가 생성되었는지 확인하세요!

이 이미지의 크기를 확인해 봅시다!


```python
print (img[:,:,2].shape)
```

    (600, 800)
    


```python
# khs code..
# img 3번째 요소를 보면..
print (img[:,:,:].shape)
print (img[:,:,0].shape)
print (img[:,:,1].shape)
```

    (600, 800, 3)
    (600, 800)
    (600, 800)
    

#### 그러면 나머지 600, 800 등 1,2번째 요소는..

말할 것 없이 x좌표 해상도, y좌표 해상도 크기라고 하겠다.<br>
크기로 표현한다고만 하겠다.<br>
실제.. : 대신에 수치를 넣게되면 단순히.. 3, ... 형태의 출력만 나오게 되어서;;

잘하셨습니다! 여러분은 OpenCV를 사용하여 이미지를 읽고 새로운 창에 표시해 보았습니다.

다른 라이브러리를 사용하여 새로운 창이 아닌 노트북에 이미지를 표시할 수 있습니다.

여러분이 code를 편집하여 이미지를 노트북에 표시해 보세요.

### 작업 1: matplotlib 라이브러리를 가져오고 이 노트북에 이미지를 표시합니다.


```python
# your code here
# 위에 이미 작성되어있는;;;;
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(imgRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_16_0.png)
    


표시한 이미지의 색상이 올바른지 확인해 보세요. 그렇지 않다면 색 공간을 변경해서 다시 표시해야 합니다.

이 단계에서 컴퓨터가 이미지의 내용을 이해하고 있습니까?

### 1.1b 사진 대신 웹캠을 사용해 봅시다.


```python
# 이 내용은 패스하겠음.. 일단.. 원격 컴이 연결되어야 하는데...... 나중에...
# 직접 웹캠 연결해서 작업;;
camera = cv2.VideoCapture(0) #' 첫 번째' 카메라(웹캠)로 VideoCapture 객체 생성

while(True):
    ret, frame = camera.read()                              # 프레임 단위로 캡처
    cv2.imshow('Press Spacebar to Exit',frame)              # 프레임 표시
    
    if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스바가 감지되면 중지
        break

camera.release()                           # 스페이스바가 감지된 후 창을 종료
cv2.destroyAllWindows()
```

축하합니다! 새로운 창에서 여러분의 얼굴을 보았습니까?

#### 1.1까지에서..
- 기본적으로 opencv에서 이미지처리가 어떻게 흘러가는지 알게 되는 것 같다.
- 다음에도 나오겠다만 opencv와 matlib의 색상 처리가 다르다.
> - opencv는 RGB, matlib은 BGR.. 만일 그대로 출력을 할경우 이상한 색상으로 표현된다.
> - 고로 변환을해서 그대로 출력이 필요 이 분야는 다시 나오면 언급
- 웹캠 처리방식도 기본적으로 어떻게 흘러가는지 알아두자.

### 1.2 보는 것을 이해하기

눈이 보는 것과 같이 컴퓨터가 볼 수 있도록 하는 방법을 학습하였습니다. 이제 컴퓨터는 보이는 것을 이해해야 합니다.

이전 노트북에서 Numpy 라이브러리를 사용해 보았습니다. OpenCV 라이브러리를 사용하면 이미지를 Numpy Array로 저장합니다.

기본적으로 이미지를 빠르게 분석하는 데 사용할 수 있는 메서드가 내장되어 있습니다.<br />
예를 들어 **.shape**는 저장된 이미지의 Numpy 배열 크기(높이, 너비, 채널)를 알려줍니다.<br />
이 외에도 배열을 다룰 수 있는 다양한 고급 기술이 있지만 우선은 간단한 메서드만 사용하겠습니다.

이미지는 기본적으로 R(빨간색), G(녹색), B(파란색)의 3개 채널로 구성되어 있으며, 각 채널별 픽셀 강도를 갖고 있습니다. 
이것은 OpenCV에서 사용하는 기본 색 공간입니다.

- image001.png 이미지의 크기를 확인해 보세요.
- 이미지의 여러 부분에 색상을 확인해 보세요.
- 이미지의 색상 강도를 표현하는 방법을 확인해 보세요.

![Drawing](../assets/images/image001.jpg)
<!-- <p><img src="../assets/images/image001.png" alt="Drawing" style="width: 400px; border:1px solid; float:left;"/></p> -->
<div style="clear: both;"></div>

아래 이미지를 이용하여 직접 확인해 보세요!

#### 이 이미지의 크기는 얼마입니까?


```python
print(img.shape)          # 이 이미지의 크기는 얼마입니까?
                          # 너비, 높이, 채널은 수는?
                          # 힌트: 이미지는 Numpy 배열에서 (높이, 너비, 채널)로 표시됩니다.
```

    (600, 800, 3)
    

#### 위 문제 이미지 크기 관련 답.
- 이미지 크기 : 600, 800
- 너비 : 600, 높이 : 800, 채널(RGB) 3

#### 이미지의 왼쪽 상단 모서리의 색상은 무엇입니까?


```python
print(img[0,0])           # 배열의 인덱싱은 왼쪽위부터 (0, 0)으로 시작시작합니다.
                          # 힌트: OpenCV에서 채널의 순서는 BGR 입니다.
```

    [  0 255   0]
    

#### 위 문제 왼쪽 상단 모서리의 색상 답

- 표기된 색상은 R(0) G(255) B(0) 으로 나온다.

### 작업 2: 이미지의 오른쪽 상단 모서리의 색상을 찾습니다.

이미지의 오른쪽 상단 모서리의 색상은 무엇입니까?


```python
# your code here
print(img[0,799]) # 힌트: 맨 오른쪽 픽셀은 800이 아니라 799입니다. Numpy 배열 인덱싱은 0부터 시작합니다.
```

    [255   0   0]
    

#### 위 문제 우측 상단 모서리 색상 답
- 여기서 답을 하기전에 위 결과를 보면 Red쪽에서 255가 나온다. 실은 Blue(255)다. 왜일까?
> - 먼저... 아래 코드를 보자. 위에 matplotlib로 출력을 위해 참조한 코드중 하나다.<br>
> **imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)**
> - 위 코드를 보면.. cv2 에서 mat로 출력하는데.. BGR에서 RGB로 바꿔야 화면이 제대로 나온다고 한다.
> - 즉.. 위 사실로 추론해보면, cv2의 이미지 출력순서가 Blue, Red, Green임을 알 수 있다.
- 그러하기에 위 사항을 근거로 행렬도 Blue, Green, Red순으로 나옴.
- 그렇기에 위 출력결과처럼 맨 앞에 255가 나오는대는 다 cv2의 특징이라 하겠다.

### 작업 3: 이미지 중간의 색상을 찾습니다.

이미지 중간의 색상은 무엇입니까?


```python
# your code here
print(img[300,400])
```

    [  0   0 255]
    

#### 여기서는 3번째 요소가 255로 나온다.

참고로 위 이미지에서는 빨강.. 위 설명했던 근거로 Red가 3번째 요소이니까 당연한 결과다.

아직 컴퓨터는 샘플 이미지에 사각형과 원이 있다는 것을 이해하지 못합니다.<br/>
3가지 채널(RGB)에서 0에서 255 사이의 값을 갖는 픽셀 정보만을 알고 있습니다. 

3가지 채널에서 0에서 255까지의 픽셀 강도가 의미하는 것은 기본적으로 특정 색상이 얼마나 많이 존재하는지입니다. <br />
0은 색상이 하나도 없음을 의미하고, 255는 해당 색상으로 가득 차 있음을 의미합니다.

따라서 (0,0,0)은 검은색이고 (255,255,255)는 흰색이 됩니다. 
파란색, 녹색 또는 빨간색의 픽셀 강도를 어떻게 표현할 수 있을까요?

## 2. 이미지 처리

우리는 컴퓨터가 이미지를 픽셀 강도의 배열로 인식한다는 것을 알았습니다. <br />
이제 그 이미지를 이해하는 것은 컴퓨터 비전 개발자인 여러분에게 달려 있습니다.

유용하게 사용할 수 있는 몇 가지 일반적인 이미지 처리 기술을 살펴보겠습니다. <br />
여러분이 더 자세한 내용을 깊게 이해하실 수 있도록 몇 개의 링크를 제공합니다.

그 전에 오늘날 컴퓨터 비전이 실제 세계에서 어떻게 사용되고 있는지 몇 가지 예를 생각해 볼 수 있나요? 
(이후 수업에서 논의할 예정이므로 메모를 해두세요.)

### 2.1 색 공간/ 색 구성

앞서 이미지 처리에서 파란색, 녹색,  빨간색의 색 공간을 사용했습니다.

모든 색상에 대한 정보가 필요하지는 않고, 이미지가 얼마나 밝거나 어두운지를 알아야 한다면 어떻게 할수 있을까요? 

사진을 회색조(Grayscale)로 변환할 수 있습니다.

회색조 이미지를 Numpy에서 어떻게 표현할 수 있는지 확인해 보세요.


```python
img = cv2.imread("[Dataset] Module 20 images/image001.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 색상을 BGR에서 회색조로 변환
cv2.imshow("Grey",grey)

cv2.waitKey(0)                                # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

maskRGB = cv2.cvtColor(grey, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_39_0.png)
    


### 작업 4: 이제 이 노트북에 회색조 그림을 표시합니다.

이전에 사용한 img 변수가 아닌 새로운 변수를 사용하십시오.


```python
# your code here
imgGray = cv2.imread("[Dataset] Module 20 images/image001.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grey",imgGray)

cv2.waitKey(0)                                # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

maskRGB = cv2.cvtColor(imgGray, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_41_0.png)
    


#### 경고!

위 결과에서 보듯 grayscale로 이미지로 불러오는거와 <br>
**grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)**<br>
위 코드 처럼 변환해서 나타난 이미지 그레이스케일 값이 미묘하게 다른점을 상기!

### 작업 5: 회색조 배열의 크기를 표시합니다.

회색조 이미지 배열의 크기가 어떻게 표시 될지 예상해 보세요.


```python
#your code here                      # 출력된 크기가 이전의 img.shape와 다른가요?
print(imgGray.shape)
```

    (600, 800)
    

회색조 이미지는 원래 배열의 1/3 크기이므로 메모리도 적게 사용하고, 이미지 처리도 더 빠를 것이라 예상됩니다.

### 작업 6: 이미지의 왼쪽 상단 모서리 색상을 찾습니다.

이미지의 왼쪽 상단 모서리의 색상은 무엇입니까?


```python
#your code here           # 이미지의 왼쪽 상단 모서리의 색상은 무엇입니까? 배열 인덱싱은 0부터 시작합니다.
                          # 원본 이미지의 결과와 무엇이 다른가요?
print(imgGray[0,0])
```

    200
    

### 작업 7: 이미지의 오른쪽 상단 모서리의 색상을 찾습니다.

이미지의 오른쪽 상단 모서리의 색상은 무엇입니까?


```python
#your code here          # 이미지의 오른쪽 상단 모서리의 색상은 무엇입니까?
print(imgGray[0,799])      # 원본 이미지의 결과와 무엇이 다른가요?
```

    95
    

일부 색상은 회색조로 변환할 때 실제로 다른 색상보다 더 어둡게 나타날수 있습니까?
https://docs.opencv.org/4.0.0/de/d25/imgproc_color_conversions.html 에서 다양한 색상 공간에 대해 자세히 알아볼 수 있습니다.

색 공간은 RBG 이외에도 표현하는 방식이 많이 있습니다. 그러나 우리는 더 깊게 들어가지는 않을 것이며, 여러분이 관심이 있다면 위의 링크를 읽어보십시오. 더 많은 정보를 원하면 인터넷을 이용하여 정보를 찾아보세요.

### 2.2 임계값, 마스킹 및 관심 영역

앞서 우리는 일부 색상이 다른 색상보다 얼마나 어두운지 확인해 보았습니다. 매우 어둡거나 매우 밝은 그림의 일부 영역에만 관심이 있다면 어떨까요? 화면 오른쪽 상단에 있는 사각형만 찾아낼 수 있습니까?

**기술 1: 회색조(Greyscale) 강도**


```python
print('grey', grey)
```

    grey [[150 150 150 ...  29  29  29]
     [150 150 150 ...  29  29  29]
     [150 150 150 ...  29  29  29]
     ...
     [255 255 255 ... 255 255 255]
     [255 255 255 ... 255 255 255]
     [255 255 255 ... 255 255 255]]
    


```python
# 오른쪽 상단에 있는 사각형의 픽셀 강도가 29임을 기억하십시오.
# 이제 29보다 큰 값을 가진 모든 것은 255(흰색)로 표시합니다.
# 이것은 임계값을 29로 설정하고 있음을 의미합니다.

ret,thresholded = cv2.threshold(grey,29,255,cv2.THRESH_BINARY)  
cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

오른쪽 상단의 사각형은 검은색으로 나타나고, 이미지의 다른 부분은 흰색으로 나타납니다. 
이제 추가적인 이미지 처리를 위해 이 영역에 집중할 수 있습니다.

우리가 집중하고자 하는 영역은 일반적으로 관심 영역(ROI)이라고 이야기합니다.

thresholding type은 아래와 같습니다.<br>
cv2.THRESH_BINARY<br>
cv2.THRESH_BINARY_INV<br>
cv2.THRESH_TRUNC<br>
cv2.THRESH_TOZERO<br>
cv2.THRESH_TOZERO_INV<br>

#### graysclae로 불러온걸로 위와 같이 한다면..

어떻게 될까?


```python
ret,thresholded = cv2.threshold(grey,29,255,cv2.THRESH_BINARY)  
cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

#### 위 잠깐 결론..

기존 29 수치로 할 경우 아무것도 감지가 안된다. grayscale 각 픽셀 수치중에서 29에 해당하는 수치가 없기때문..<br>
95로 할경우 아래 글자와 같이 우측상자와 나오게 되더라..<br>
하지만.. 우측 상자만 나오게 한다면.. 역시 grey같이 이미지 불러올때도 신경을 써야...

### 작업 8: 텍스트, 원, 가운데, 오른쪽 상자를 캡처하려면(검은색으로 표시) 어떻게 합니까?

마스크작업을 슬슬..


```python
#your code here 
ret,thresholded = cv2.threshold(grey,29,255,cv2.THRESH_BINARY)  
cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

maskRGB = cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_59_0.png)
    


임계값 29로 계속 작업해 보겠습니다.

일반적으로 관심 영역(ROI)은 흰색으로, 다른 영역은 검은색으로 하고 싶습니다. 다음 코드를 실행해 보세요:


```python
ret,thresholded = cv2.threshold(grey,29,255,cv2.THRESH_BINARY_INV)    #we use cv2.THRESH_BINARY_INV instead of cv2.THRESH_BINARY
cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

maskRGB = cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_61_0.png)
    


### 작업 9: 텍스트, 원, 가운데, 오른쪽 상자가 ROI(흰색으로 표시됨)가 되도록 하려면 어떻게 해야 합니까?


```python
#your code here 
ret,thresholded = cv2.threshold(grey,100,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

maskRGB = cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_63_0.png)
    


#### 임계점에 관해서..

결론으로 보자면 해당 threshold 2번째 인수에 따라 그 이하는 다 검정 그 외에는 흰색.

여러분은 위의 예제가 어디에 사용되는 것인지 궁금해 하실 수 있습니다.

관심 영역(ROI)을 임계값으로 지정하면 이를 마스크로 사용하여 원본 이미지에 표시할 수 있습니다.

### 마스크?

**그러면 마스크는 무엇입니까?**

아래 그림을 살펴봅시다.

![Drawing](../assets/images/image001_masking.jpg)
<!-- <img src="../assets/images/image001_masking.jpg" /> -->

위의 이미지(가운데)에서 오른쪽 상단 모서리에 파란색 사각형에 대한 마스크를 볼 수 있습니다. 해당 마스크(가운데 이미지)를 원본 이미지(왼쪽 이미지)에 적용하면 마스크된 이미지(오른쪽 이미지)에 파란색 사각형만 남습니다.

마스크 레이어는 이미지에서 관심 영역을 강조 표시하는 데 도움이 됩니다. 마스크를 이미지에 적용하면 관심 있는 부분(마스크의 흰색 영역)만 유지되고 나머지 부분(검은색 영역)은 삭제됩니다.

참고: 여러분은 이 개념을 Adobe Photoshop과 같은 인기 있는 이미지 편집 소프트웨어에서도 볼 수 있는데, 이 소프트웨어에서는 "clipping masks" 기능이라고 합니다.


```python
ret,thresholded = cv2.threshold(grey,29,255,cv2.THRESH_BINARY_INV)  

masked = cv2.bitwise_and(img, img, mask = thresholded) 
cv2.imshow("Masked", masked)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

파란색 사각형을 필터링 하였습니까?

이제 여러분이 임계값에 추가하고 싶은 것이 무엇인지 실험하고 확인해 볼 시간입니다. 

- 중앙에 있는 원만 임계값으로 지정할 수 있습니까? 
- 회색조 이미지에서는 어떻게 할 수 있을까요?

회색조 이미지는 작업하기에 가장 좋은 이미지는 아닐 것입니다. 원본 이미지가 있음을 기억하십시오.

![Drawing](../assets/images/image001.jpg)
<!-- <img src="../assets/images/image001.png" alt="Drawing" style="width: 400px; border:1px solid; float:left;"/> -->
<div style="clear: both;"></div>

**기술 2: 색상.**<br />

이미지가 numpy 배열이라는 것을 기억하십니까? Numpy 배열은 고급 필터로 쉽게 필터링할 수 있습니다.

우리는 이미지 처리를 좀 더 쉽게 진행하기 위해 배경의 흰색을 대신 검정색으로 바꾸고 싶을 수도 있습니다.


```python
mask = img.copy()                         # 우리가 만들 마스크 이미지. 초기 이미지의 복사본으로 초기화합니다.
(b,g,r) = cv2.split(img)                  # BGR 이미지를  각 채널별로 분할하여 별도로 작업할 수 있습니다.
mask[(b==255)&(g==255)&(r==255)] = 0     # 흰색 배경(BGR 채널이 모두 255인 경우)을 0(검정색)으로 변경합니다.

cv2.imshow("Mask",mask)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

마스크의 다른 레이어가 어떻게 보이는지 살펴보겠습니다. 각각 0, 1 및 2 채널 입니다.


```python
cv2.imshow("Blue Mask",mask[:,:,0])       # 단어가 어떻게 파란색인지 주목하십시오.
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```


```python
cv2.imshow("Green Mask",mask[:,:,1])
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```


```python
cv2.imshow("Red Mask",mask[:,:,2])
cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

**기술 3: 위치 기반의 빠른 Numpy 배열 조작**

녹색 채널의 결과에서 단어 주변에 노이즈가 있을을 확인할 수 있습니다. 이것을 어떻게 깔끔하게 만들 수 있을까요?


```python
mask[300:,:,1]=0                          # 이미지는 행렬임을 기억하십시오. 아래쪽 절반을 검은색(0)으로 만듭니다.
cv2.imshow("Green Mask",mask[:,:,1])
cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

이미지의 크기가 기억나지 않는 경우 어떤 명령을 사용하여 알 수 있습니까?


```python
mask.shape
```




    (600, 800, 3)



이제 빨간색 채널의 글자 주변의 노이즈를 확인해 봅시다.


```python
cv2.imshow("Red Mask",mask[:,:,2])
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

##### mat로 어떻게 노이즈 제거가 비교되는지 출력해보기


```python
maskRGB = cv2.cvtColor(mask[:,:,2], cv2.COLOR_BGR2RGB)

plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_81_0.png)
    


빨간색 채널 단어 주변에도 일부 노이즈를 확인할 수 있습니다. 이것을 어떻게 깔끔하게 만들 수 있을까요?

### 작업 10: 빨간색 채널 단어 주변의 노이즈를 제거합니다.


```python
#your code here 
mask[400:,:,2]=0                          # 이미지는 행렬임을 기억하십시오. 아래쪽 절반을 검은색(0)으로 만듭니다.
cv2.imshow("Red Mask",mask[:,:,2])

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

##### mat로 어떻게 노이즈 제거가 비교되는지 출력해보기


```python
maskRGB = cv2.cvtColor(mask[:,:,2], cv2.COLOR_BGR2RGB)

plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_85_0.png)
    


##### 잠깐의 결론..

노이즈는 제거되 보인다.

이제 색상을 기반으로 이미지에서 객체(원, 사각형)를 간단히 얻을 수 있습니다.


```python
# 사전작업.
maskRGB = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
drawimg = [maskRGB]
```


```python
# 두번째 채널이 빨간색입니다. OpenCV에서 색공간은 (B,G,R)입니다.
masked = cv2.bitwise_and(img,img,mask=mask[:,:,2])
cv2.imshow("Circle",masked)                   

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

drawimg.append(cv2.cvtColor(masked.copy(), cv2.COLOR_BGR2RGB))
```


```python
# 첫번째 채널이 녹색입니다. OpenCV에서 색공간은 (B,G,R)입니다.
masked = cv2.bitwise_and(img,img,mask=mask[:,:,1])
cv2.imshow("Left Green Rectangle",masked)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

drawimg.append(cv2.cvtColor(masked.copy(), cv2.COLOR_BGR2RGB))
```


```python
# 세번째 채널이 파란색입니다. OpenCV에서 색공간은 (B,G,R)입니다.
masked = cv2.bitwise_and(img,img,mask=mask[:,:,0])
cv2.imshow("Right Blue Rectangle",masked)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

drawimg.append(cv2.cvtColor(masked.copy(), cv2.COLOR_BGR2RGB))
```

#### 위 채널들을 한꺼번에 mat로 출력해보기

딱 한번에 볼 수 있으면 좋으니까..


```python
# 이 때는 함수로 정립하기 이전..
for idx in range(4):
    plt.subplot(2, 2, idx + 1)
    plt.axis('off') # 창에있는 x축 y축 제거
    plt.imshow(drawimg[idx])
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_93_0.png)
    


#### 파랑색에 단어가 남아있는데..

엣헴.. 바로 다음 공정이기에;;;

파란색 채널의 결과 사각형뿐만 아니라 단어도 표시되는 것을 확인할 수 있습니다. 단어도 파란색이기 때문입니다!

단어가 표시되지 않도록 하려면 다음과 같이 "지울 수 있습니다".

### 작업 11: 단어를 지우세요!


```python
#your code here 

mask[300:,:,0] = 0
masked = cv2.bitwise_and(img, img, mask=mask[:,:,0])
cv2.imshow("Right Blue Rectangle",masked)
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

우리는 마스킹, 임계값 및 관심 영역에 상당한 시간을 할애했습니다. 색상, 픽셀 강도, Numpy Array 조작(예: 이미지의 일부에 접근 및 수정 하기)에 따라 사용할 수 있는 다양한 접근 방식이 있습니다. 시간을 내어 이러한 기술을 연습하고 다른 이미지로도 실험해 보십시오.

컴퓨터 비전도 인생과 같이 동일한 목표에 도달할 수 있는 여러 가지 방법이 있습니다. 관심 영역을 얻는 더 효율적인 방법을 생각해 볼 수 있습니다.

##### mat로 출력


```python
maskRGB = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_99_0.png)
    


### 2.3 기하학적 변환(크기 조정 및 자르기)

이미지가 너무 크거나 너무 작은 경우 어떻게 크기를 조정할 수 있습니까?

##### 기하학적 변환

### 작업 12: 800x600 이미지를 400x300 이미지로 만들기


```python
# 이미지 다시 불러오기
img = cv2.imread("[Dataset] Module 20 images/image001.png")
img.shape
```




    (600, 800, 3)




```python
#your code here 
resized = cv2.resize(img,(400, 300))           # 두 번째 매개변수는 원하는 모양(너비, 높이)입니다.
cv2.imshow("img",img)
cv2.imshow("Resized",resized)

cv2.waitKey(0)                                 # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

이제 이미지의 크기는 얼마입니까?


```python
#your code here 
resized.shape
```




    (300, 400, 3)



##### mat으로 출력해보기


```python
maskRGB = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_106_0.png)
    


##### mat에서는 사이즈 상관없이.. 인지도..

그렇다보니 티나게 하게하기 위해 일부러 축제거를 없앰

가로, 세로 비율이 다른 경우 크기 조정 기능을 사용하여 이미지를 확장할 수도 있습니다.

### 작업 13: 800x600 이미지를 200x300 이미지로 늘이기


```python
#your code here 
dst = cv2.resize(img,(1000, 800))
cv2.imshow("img",img)
cv2.imshow("Resized",dst)

cv2.waitKey(0)                                 # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

#### 비율로 크기 조정

### <span style='color:blue'>@khs - cv2.resize()
- Image resize : 이미지가 큰 경우 더 많은 데이터가 필요함. 이런 고해상도 이미지는 매우 디테일하지만, 컴퓨터 비전/이미지 처리 관점에서 볼 때 이미지의 구조적 구성 요소가 더 중요하기 때문에 고해상도의 이미지를 다운샘플링하여 더 빠르고 정확하게 실행될 수 있도록 한다.cv2.resize를 이용하며 종횡비를 계산하여 왜곡되지 않도록 합니다
이미지의 비율을 변경하면 존재하지 않는 영역에 새로운 픽셀값을 매핑하거나 존재하는 픽셀들을 압축해서 새로운 값을 할당해야 합니다. 따라서 추정하는 픽셀은 interpolation 이용
- cv2.resize(img, dsize=(0,0), fx=0.3, fy=0.7, interpolation = cv2.INTER_LINEAR)
    - dsize : 절대크기 ( 너비, 높이 )
    - fx, fy : 상태크기- dsize(절대크기)를 (0,0)으로 정하고 x,y 방향 스케일비율(상태크기)의 값을 할당
    - 사이즈를 축소할 경우(다운샘플링): cv2.INTER_AREA(영역보간),
    - 사이즈를 확대할 경우(업샘플링): cv2.INTER_CUBIC(바이큐빅보간), cv2.INTER_LINEAR(쌍 선형 보간)
    - 다운 또는 업샘플링할때 기본값 : cv2.INTER_LINEAR


```python
# khs Code
# dst = cv2.resize(img, None, fx=0.5, fy = 0.5)  # x, y비율 정의(0.5배로 축소)
dst = cv2.resize(img, None, fx=2, fy = 2)  # x, y비율 정의(2배로 축소)
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 보간법으로 크기 조정

이미지를 변경할 때 자연스럽게 하는 방법

- cv2.INTER_AREA : 크기 줄일 때 사용
- cv2.INTER_CUBIC : 크기 늘일 때 사용 (속도 느림, 퀄리티 좋음)
- cv2.Inter_LINEAR : 크기 늘릴 때 사용 (기본값)


```python
# 이번 이미지 예는 고양이 이미지로.. 먼저 축소할때
img = cv2.imread('[Dataset] Module 20 images/img.jpg')
dst = cv2.resize(img, None, fx=0.5, fy = 0.5 , interpolation = cv2.INTER_AREA )  # x, y비율 정의(0.5배로 축소)
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 축소된 이미지 아래에 출력
maskRGB = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_114_0.png)
    



```python
# 확대에서는..

dst = cv2.resize(img, None, fx=1.5, fy = 1.5 , interpolation = cv2.INTER_CUBIC )  # x, y비율 정의(0.5배로 축소)
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 확대된 이미지 아래에 출력
maskRGB = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_115_0.png)
    


#### 영상재생으로 조정해보자.

영상을 불러와서, 위 방법처럼 사이즈 조정 플레이 해보자.<br>
영상은 아쉽게도.. 코드만;;


```python
# 적당한 사이즈로 할때.
cap = cv2.VideoCapture('[Dataset] Module 20 images/video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_resize = cv2.resize(frame, (400, 500))
    cv2.imshow('video', frame_resize)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```


```python
# 크기 늘리고, CUBIC으로 보정한다면...
cap = cv2.VideoCapture('[Dataset] Module 20 images/video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_resize = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC )
    cv2.imshow('video', frame_resize)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### 자르기

### 작업 14: 이미지를 잘라서 위쪽 절반만 얻기:


```python
#your code here 
img = cv2.imread('[Dataset] Module 20 images/img.jpg')
cv2.imshow("", img[:300,:, :])

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

# 자른 이미지를 아래에 출력
maskRGB = cv2.cvtColor(img[:300,:, :], cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_120_0.png)
    


### Task 15: 이미지를 잘라서 오른쪽 반만 얻기


```python
#your code here 
img = cv2.imread('[Dataset] Module 20 images/img.jpg')
cv2.imshow("", img[:,300:, :])

cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

# 자른 이미지를 아래에 출력
maskRGB = cv2.cvtColor(img[:,300:, :], cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_122_0.png)
    


관심 영역을 중심으로 이미지를 잘라볼 수 있습니다. 관심 영역(ROI)을 추출하는 또 다른 유용한 방법을 살펴보겠습니다.

#### 응용문제 : 일부 고양이 얼굴만 복붙 원이미지에 붙이는 방법은?

crop명칭으로 일부 이미지 배열영역을 복사 덮어씌우면 된다.


```python
# img.shape : (390, 640, 3)  => 이 이미지보다 작게 자름
crop = img[100:200, 200:400] # 세로기준 100:200, 가로기준 300:400
cv2.imshow('img', img)
cv2.imshow('crop', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


```python
# crop image 만 출력
maskRGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_126_0.png)
    



```python
# 덮어씌워서 하나의 이미지로..
cropimg = img.copy()
cropimg[150:250, 350:550] = crop

cv2.imshow('crop', cropimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


```python
# crop image mat 출력
maskRGB = cv2.cvtColor(cropimg, cv2.COLOR_BGR2RGB)
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(maskRGB)
plt.show()
```


    
![png](../assets/images/CVAI_Start_Prj_128_0.png)
    


### 2.4 윤곽 감지(Contour Detection)

일반적으로 임계값 마스크를 사용하여 관심 영역을 찾습니다.

**그러면 윤곽선이란 무엇입니까?**

윤곽선은 [경계를 따라 그린 곡선](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html) 으로 생각할 수 있습니다.

단순하게 흑백 마스크를 생각하면, 경계 지역에서는 색상이 급격하게 변화될 것입니다. 윤곽선은 이 경계를 따라 그려진 곡선입니다.

윤곽 감지는 기본적으로 이러한 서로 다른 그룹을 윤곽으로 찾아 값을 반환합니다.

예를 들어 아래 이미지에 흰색 영역이 몇 개 있다고 생각하십니까?

![](../assets/images/image001_3contours.jpg)
<!-- <img src="../assets/images/image001_3contours.png" style="width:400px; float:left;" /> -->
<div style="clear: both;"></div>

3개의 윤곽선이 감지되었다고 생각하셨습니까? 그러면 해당 이미지를 로드하고 윤곽선을 그려 보겠습니다.

https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a<br> 아래 링크


```python
greytest = cv2.imread("[Dataset] Module 20 images/image001_3contours.png",0)    # 해당 이미지 읽기
contouroutlines = np.zeros(greytest.shape,dtype="uint8")    # 감지된 윤곽을 그리기 위한 빈 캔버스 만들기

# 윤곽선을 찾아봅시다! 
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
(cnts,_) = cv2.findContours(greytest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# i > 윤곽선 idx, c: 윤곽선 점
for (i, c) in enumerate(cnts):
    cv2.drawContours(contouroutlines, [c], -1, 255, 1)  # 각 윤곽선에 대해 윤곽 영역의 바깥선만 그립니다.
                                                        # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
cv2.imshow("Contour Outlines",contouroutlines)          # 결과 표시
cv2.waitKey(0)                                          # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

print("There are "+str(len(cnts))+" contours!")         # 감지된 윤곽 수를 출력합니다.
```

    There are 3 contours!
    

#### 위 윤곽선 검출에 관해서..

프로세스 순서대로 보자면..
> 1. 회색이미지를 불러와서
> 2. 윤곽을 그릴 빈 캔퍼스를 만든다.(예로.. 600*800크기의 값이 모두 0인 배열)
> 3. 회색이미지에 opencv2의 findcontours로 윤곽선에 해당할 모든 점을 옵션에 따라 찾는다.
> 4. 찾은 점 따라서 빈 캔퍼스에 흰선으로 그리기
> 5. 결과 표시, 찾은 윤곽선 갯수도 표시

3개의 윤곽선이 있는 간단한 예제입니다.

원본 이미지에서는 몇 개의 윤곽선이 감지될 것으로 예상하십니까?

![Drawing](../assets/images/image001_allcontours.jpg)
<!-- <img src="../assets/images/image001_allcontours.png" style="width:400px; float:left;" /> -->
<div style="clear:both;"></div>


먼저 임계값을 이용하여 이미지를 만듭니다.


```python
img = cv2.imread("[Dataset] Module 20 images/image001.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 임계값을 적용합니다
(T, thresholded) = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
```

이제 이미지의 윤곽을 찾아보겠습니다. 윤곽이 몇 개나 될 것 같습니까? 이미지를 잘 살펴보세요!


```python
# 윤곽선을 찾아보자!
(cnts,_) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(img.shape,dtype="uint8")  # 감지된 윤곽을 그리기 위한 빈 캔버스 만들기
for (i, c) in enumerate(cnts):    
    cv2.drawContours(mask, [c], -1, (0,255,0), 1) 
    
cv2.imshow("Mask",mask)  
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()


print("There are "+str(len(cnts))+" contours!")
```

    There are 78 contours!
    

78개의 윤곽선이 있는 이유는 무엇입니까? 그 이유는 바로 글자 때문입니다.

실제로 계산되는 것을 시각화할 수 있도록 윤곽선에 라벨을 지정해 보겠습니다.

아래에서 각 문자가 1개의 윤곽선을 형성하는 경향을 볼 수 있습니다. 그러나 "i" 문자의 경우 상단과 하단이 연결되어 있지 않기 때문에 일부 문자가 실제로 2개의 윤곽선으로 계산되고 있습니다. 느낌표도 마찬가지입니다.

아래 코드는 주석 처리된 코드가 포함되어 있어서 더 길어 보일 수 있습니다. 섹션 2.5에서 코드에 대한 내용을 더 자세하게 다루기 때문에 지금은 이 코드에 대해 걱정하지는 마십시오. 
코드를 실행하고 윤곽선이 어떻게 계산되는지 확인해 보십시오.
각 "윤곽선" 주위에 그려진 빨간색 경계 상자를 주목해 보세요.


```python
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 임계값을 적용합니다
(T, thresholded) = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Thresholded",thresholded)

# 윤곽선을 찾아보자!
(cnts,_) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda cnts: cv2.boundingRect(cnts)[1])  # 윤곽선을 위에서 아래로 정렬합니다.

mask = cv2.merge([thresholded,thresholded,thresholded])  # 감지된 윤곽을 그리기 위한 캔버스 만들기
for (i, c) in enumerate(cnts):                           # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html  
    cv2.drawContours(mask, [c], -1, (255,255,255), -1) 
    (x, y, w, h) = cv2.boundingRect(c)                   # 윤곽 경계 상자의 x,y 좌표를 가져옵니다.
    cv2.rectangle(mask, (x,y), (x+w,y+h), (0,0,255))     # 경계 상자를 빨간색으로 그립니다.

    cv2.putText(mask, ""+str(i+1), (x,y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,255,0), 1)
    
cv2.imshow("Mask",mask)  
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()


print("There are "+str(len(cnts))+" contours!")

# 역시 mat로 한번에 볼 수 있도록 해보자.
drawimg = [
    cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB), 
    cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
]

for idx in range(2):
    plt.subplot(1, 2, idx + 1)
    plt.axis('off') # 창에있는 x축 y축 제거
    plt.imshow(drawimg[idx])
plt.show()
```

    There are 78 contours!
    


    
![png](../assets/images/CVAI_Start_Prj_138_1.png)
    


윤곽선에 대해 [사이트] https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html 을 방문하여 더 자세히 알아볼 수 있습니다. 더 많은 내용들을 인터넷에서 검색해 보세요. 코드를 작성하고, 에러를 처리하고, 라이브러리에 대하여 더 많은 것을 알아보는데 매우 유용할 것입니다!

팁: 위의 예에서 우리는 cv2.RETR_EXTERNAL을 사용하여 외부 윤곽을 얻었습니다. 다른 유형의 윤곽선을 얻기 위해 지정할 수 있는 다른 옵션도 있습니다. 예를 들어, cv2.RETR_LIST는 외부 윤곽뿐만 아니라 모든 윤곽을 나태냅니다.

**윤곽선을 이미지 마스크로 사용하기**

우리는 앞 부분에서 이미지 마스크에 대하여 배웠습니다. 윤곽선을 이용하여 마스크를 만들 수도 있습니다!

drawContour 함수의 마지막 매개변수를 -1로 설정하여 윤곽선 대신 윤곽 영역을 만들어 이를 마스크로 사용할 수 있습니다!


```python
(T, thresholded) = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
thresholded[410:,:]=0                     # 이미지 하단에 있는 텍스트를 제거합니다.
#cv2.imshow("Thresholded",thresholded)

# 윤곽선이 몇개인지 확인하기.
(cnts,_) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(thresholded.shape,dtype="uint8")
for (i, c) in enumerate(cnts):    
    cv2.drawContours(mask, [c], -1, 255, -1)  # 마지막 매개변수는 윤곽선 두께를 정의합니다. -1은 윤곽선 내부를 채웁니다.

mask1 = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("Mask",mask)
cv2.imshow("Masked Image", mask1)  
cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()


print("There are "+str(len(cnts))+" contours!")

# 역시 mat로 한번에 볼 수 있도록 해보자.
drawimg = [
    cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), 
    cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB),
]

for idx in range(2):
    plt.subplot(1, 2, idx + 1)
    plt.axis('off') # 창에있는 x축 y축 제거
    plt.imshow(drawimg[idx])
plt.show()
```

    There are 3 contours!
    


    
![png](../assets/images/CVAI_Start_Prj_141_1.png)
    


임계값을 이용한 마스크 이미지 생성보다, 윤곽선을 이용하여 마스크를 생성하는 것이 더 간단합니다.

#### 윤곽선에 대한 추가 예시 및 설명에 관해서..

### <span style="color:blue">@khs - cv2.findContours() : 이진화 이미지에서 윤곽선(컨투어)를 검색
>https://bkshin.tistory.com/entry/OpenCV-22-%EC%BB%A8%ED%88%AC%EC%96%B4Contour
- contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
- 윤곽선, 계층구조 = cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)
    - 검색 방법
        - cv2.RETR_EXTERNAL : 외곽 윤곽선만 검출하며, 계층 구조를 구성하지 않습니다.
        - cv2.RETR_LIST : 모든 윤곽선을 검출하며, 계층 구조를 구성하지 않습니다.
        - cv2.RETR_CCOMP : 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
        - cv2.RETR_TREE : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성합니다. (Tree 구조)
    - 근사화 방법
        - cv2.CHAIN_APPROX_NONE : 윤곽점들의 모든 점을 반환합니다.
        - cv2.CHAIN_APPROX_SIMPLE : 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
        - cv2.CHAIN_APPROX_TC89_L1 : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
        - cv2.CHAIN_APPROX_TC89_KCOS : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
    - contours : Numpy 구조의 배열로 검출된 윤곽선의 지점들이 담겨있음
    - hierarchy : 윤곽선의 계층 구조를 의미. 윤곽선의 포함 관계 여부 나타냄 (외곽/내곽/같은 게층구조)윤곽선에 해당하는 속정 정보 포함함

### <span style="color:blue">@khs - cv2.drawContours(): 검출된 윤곽선 그리기
- cv.drawContours(	image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]	) ->	image
- cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형타입)
    - 윤곽선 : 검출된 윤곽선들이 저장된 Numpy 배열
    - 윤곽선 인덱스: 검출된 윤곽선 배열에서 몇 번째 인덱스의 윤곽선을 그릴지를 의미합니다.
        - 윤곽선 인덱스를 0으로 사용할 경우 0 번째 인덱스의 윤곽선을 그리게 됩니다. 하지만, 윤곽선 인수를 대괄호로 다시 묶을 경우, 0 번째 인덱스가 최댓값인 배열로 변경됩니다.
        - 동일한 방식으로 [윤곽선], 0과 윤곽선, -1은 동일한 의미를 갖습니다. (-1은 윤곽선 배열 모두를 의미)
        - 예시 [ 2 -1 1 -1]  : [다음 윤곽선, 이전 윤곽선, 내곽 윤곽선, 외곽 윤곽선]에 대한 인덱스 정보를 포함
            - 인덱스 0의 윤곽선의 다음 윤곽선은 인덱스 2의 윤곽선을 의미하며 이전 윤곽선은 존재하지 않다는 것을 의미
            - 내곽 윤곽선은 인덱스 1에 해당하는 윤곽선을 자식 윤곽선으로 두고 있다는 의미입니다.즉, 인덱스 0 윤곽선 내부에 인덱스 1의 윤곽선이 포함되어 있습니다.
            - 외곽 윤곽선은 -1의 값을 갖고 있으므로 외곽 윤곽선은 존재하지 않습니다.

### 2.5 선 그리기 및 텍스트 쓰기

drawContour를 사용하여 윤곽선을 이용한 마스크 이미지를 생성해 보았습니다. 이미지에 글자와 선을 추가하는 방법을 살펴보겠습니다. 2.4의 예를 다시 살펴보고 윤곽선에 라벨을 추가해 보겠습니다!

코드에서 변경된 3줄만 설명합니다. 
코드의 다른 줄은 2.4의 예제와 유사하며 해당 예제를 참조하여 해당 코드 줄이 수행하는 작업을 확인할 수 있습니다.

먼저 각 윤곽선에 대한 경계 상자를 가져오고 그 주위에 직사각형을 그린 다음 각 윤곽선에 라벨을 지정할 텍스트를 추가합니다.


```python
greytest = cv2.imread("[Dataset] Module 20 images/image001_3contours.png",0)
contouroutlines = np.zeros(greytest.shape,dtype="uint8")

(cnts,_) = cv2.findContours(greytest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for (i, c) in enumerate(cnts):    
    cv2.drawContours(contouroutlines, [c], -1, 255, 1)

    # 각 윤곽의 경계 상자 가져오기
    (x, y, w, h) = cv2.boundingRect(c)
    
    # 각 윤곽 주위에 직사각형 그리기 (즉, 경계 상자를 그립니다)
    cv2.rectangle(contouroutlines, (x, y), (x+w, y+h), (255,255,0), 2) 
    
    # 각 윤곽에 "COUNTOUR <>"라는 텍스트 추가
    cv2.putText(contouroutlines, "Contour "+str(i+1), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    
cv2.imshow("Contour Outlines",contouroutlines)          
cv2.waitKey(0)                                          
cv2.destroyAllWindows()

print("There are "+str(len(cnts))+" contours!")

matdraw(contouroutlines)
```

    There are 3 contours!
    


    
![png](../assets/images/CVAI_Start_Prj_146_1.png)
    


화면에 텍스트를 입력하고 사각형, 원과 같은 모양을 그리는 것에 대한 자세한 내용은 https://docs.opencv.org/4.0.0/dc/da5/tutorial_py_drawing_functions.html 을 참조하세요.

나중에 객체 감지용 응용 프로그램을 만들 경우 이 방법을 사용하여 실제로 감지한 항목에 주석을 달 수 있습니다. 
또는 코드만으로 자신만의 예술 작품과 이미지를 만들 수도 있습니다!

처음부터 무언가를 그려 봅시다.

**ACCESS DENIED**


```python
# 빈 캔버스 생성(높이, 너비, 채널) - 3개의 색상 채널, 너비 400, 높이 300
canvas_accessdenied = np.zeros((600,800,3),dtype="uint8")      

# 좌표 (x=100,y=230)에 색상 (255,255,0), 선 두께가 2인  직사각형 추가
cv2.rectangle(canvas_accessdenied, (100, 230), (700, 370), (255,255,0), 2)  

# 좌표 (x=150,y=320)에 색상(100,100,255), 글꼴 크기 2, 선 두께 5인 텍스트 추가
cv2.putText(canvas_accessdenied, "ACCESS DENIED", (150,320), cv2.FONT_HERSHEY_SIMPLEX, 2, (100,100,255), 5)

cv2.imshow("Canvas Access Denied",canvas_accessdenied)  
cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

matdraw(canvas_accessdenied)
```


    
![png](../assets/images/CVAI_Start_Prj_148_0.png)
    


"Access Granted" 텍스트를 그려보세요.

**ACCESS GRANTED**


```python
# 빈 캔버스 생성(높이, 너비, 채널) - 3개의 색상 채널, 너비 400, 높이 300
canvas_accessgranted = np.zeros((600,800,3),dtype="uint8")      

# 좌표 (x=100,y=230)에 색상 (255,255,0), 선 두께가 2인 직사각형 추가
cv2.rectangle(canvas_accessgranted, (100, 230), (700, 370), (255,255,0), 2)  

# 좌표 (x=130,y=320)에 색상 (255,100,100), 글꼴 크기 2, 선 두께 5인 텍스트 추가
cv2.putText(canvas_accessgranted, "ACCESS GRANTED", (130,320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,100), 5)

cv2.imshow("Canvas Access Granted",canvas_accessgranted)  
cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

matdraw(canvas_accessgranted)
```


    
![png](../assets/images/CVAI_Start_Prj_150_0.png)
    


### 작업 16: 2개의 상자가 있는 캔버스를 만듭니다. 첫 번째 상자에는 "ACCESS GRANTED"라고 표시되고 두 번째 상자에는 "PLEASE PROCEED"라고 표시됩니다.


```python
# 빈 캔버스 생성(높이, 너비, 채널) - 3개의 색상 채널, 너비 400, 높이 300
canvas_accesspaint = np.zeros((600,800,3),dtype="uint8")      

# 좌표 (x=100,y=230)에 색상 (255,255,0), 선 두께가 2인 직사각형 추가
cv2.rectangle(canvas_accesspaint, (100, 130), (700, 270), (255,255,0), 2)  

# 좌표 (x=130,y=320)에 색상 (255,100,100), 글꼴 크기 2, 선 두께 5인 텍스트 추가
cv2.putText(canvas_accesspaint, "ACCESS GRANTED", (130,220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,100), 5)

# 좌표 (x=100,y=230)에 색상 (255,255,0), 선 두께가 2인  직사각형 추가
cv2.rectangle(canvas_accesspaint, (100, 330), (700, 470), (255,255,0), 2)  

# 좌표 (x=150,y=320)에 색상(100,100,255), 글꼴 크기 2, 선 두께 5인 텍스트 추가
cv2.putText(canvas_accesspaint, "PLEASE PROCEED", (130,420), cv2.FONT_HERSHEY_SIMPLEX, 2, (100,100,255), 5)


cv2.imshow("Canvas Access Paint",canvas_accesspaint)  
cv2.waitKey(0)                             # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

matdraw(canvas_accesspaint)
```


    
![png](../assets/images/CVAI_Start_Prj_152_0.png)
    


## 축하합니다!

## 이제 여러분이 더 흥미롭운 콘텐츠를 만들어 볼 때입니다!

도전 과제를 진행하면서 도움이 필요하거나 openCv 함수의 구문을 알아야 하는 경우 <br />
https://docs.opencv.org/4.0.0/d2/d96/tutorial_py_table_of_contents_imgproc.html 에서 찾아보십시오.

해결 방안을 찾을 수 없다면 다른 사람들이 여러분을 도와줄 수도 있습니다. 인터넷을 활용하여 더 많은 정보를 알아보세요!

여러분의 질문에 대한 해결 방법을 계속해서 찾고, 세상의 많은 도전 과제에 도움을 줄 수 있는 좋은 자료를 계속 만드세요!

### 도전과제 1: 섹션 1.1b의 비디오 예제에서 비디오 크기를 800x600으로 조정하고 회색조로 표시


```python
#your code here
#your code here
camera = cv2.VideoCapture(0) #' 첫 번째' 카메라(웹캠)로 VideoCapture 객체 생성

while(True):
    ret, frame = camera.read()                             # 프레임 단위로 캡처
    frame = cv2.resize(frame, (800, 600), interpolation = cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Press Spacebar to Exit',frame)              # 프레임 표시
    
    if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스바가 감지되면 중지
        break

camera.release()                           # 스페이스바가 감지된 후 창을 종료
cv2.destroyAllWindows()
```

### 도전과제 2: Python  time 라이브러리를 사용하여 웹캠 비디오 화면에 타임스탬프 추가하기


```python
import numpy as np
```


```python
# 타임스탬프를 얻기 위한 샘플 코드. 필요한 경우 온라인에서 더 많은 옵션을 검색하십시오.
from datetime import datetime
print (datetime.now())
```

    2022-11-23 14:50:48.456778
    


```python
#your code here
#your code here
COLOR = (255, 255, 255) # 흰색
THICKNESS = 1     # 두께
SCALE = 1         # 크기

camera = cv2.VideoCapture(0) #' 첫 번째' 카메라(웹캠)로 VideoCapture 객체 생성

while(True):
    ret, frame = camera.read()                             # 프레임 단위로 캡처
    
    cv2.putText(frame, str(datetime.now()), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
    cv2.imshow('Press Spacebar to Exit',frame)              # 프레임 표시
    
    if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스바가 감지되면 중지
        break

camera.release()                           # 스페이스바가 감지된 후 창을 종료
cv2.destroyAllWindows()
```

### 도전과제 3: 빨간색과 녹색 마커를 사용하여 다양한 색상의 카드를 만들고, 특정 색상의 카드가 제시될 때마다 컴퓨터가 인식하도록 할 수 있습니까?

힌트: 먼저 제작한 카드의 사진을 몇 장 찍은 다음 간단한 이미지 처리를 사용하여 카드의 색상 패턴을 분석합니다. 필요한 경우 다양한 색상 공간을 탐색합니다. 이 시스템이 조명 조건의 변화에 어떻게 영향을 받는지 살펴보십시오.


```python
#your code here 여기는 따로....
```

### 도전과제 4: 카메라 앞에 나타날 때마다 "ACCESS GRANTED"를 표시하는 응용 프로그램 만들기

여러분의 창의력을 발휘하십시오. 
유일한 규칙은 키보드 입력은 허용되지 않으며, 비디오 피드를 처리하는 카메라를 사용해야 한다는 것입니다.

추가적으로, 여러분만 접근 권한을 부여하고 다른 친구들의 접근을 불허하는 시스템을 만들어 보세요.
먼저 시스템에 접근하는 방법을 친구에게 보여주고, 친구가 시도하였을 때 3분 이내에 접근 권한을 얻지 못하면 여러분의 승리입니다!


```python
def ViewAccessGranted(frame):
    # 좌표 (x=100,y=230)에 색상 (255,255,0), 선 두께가 2인 직사각형 추가
    cv2.rectangle(frame, (55, 180), (580, 280), (255,255,0), 2)  

    # 좌표 (x=130,y=320)에 색상 (255,100,100), 글꼴 크기 2, 선 두께 5인 텍스트 추가
    cv2.putText(frame, "ACCESS GRANTED", (160,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,100,100), 5)
    
    return frame

def ViewAccessDenied(frame):
    # 좌표 (x=100,y=230)에 색상 (255,255,0), 선 두께가 2인  직사각형 추가
    cv2.rectangle(frame, (55, 180), (580, 280), (50,255,255), 2)  

    # 좌표 (x=150,y=320)에 색상(100,100,255), 글꼴 크기 2, 선 두께 5인 텍스트 추가
    cv2.putText(frame, "ACCESS DENIED", (170,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100,100,255), 5)
    
    return frame
```

### 얼굴인식 프로젝트-기초

- CascadeClassifier는, 단순히 얼굴 검출기라기보다는, 유사-하르 필터를 이용하여 유사도로 특정 객체를 검출해내는 검출기로, 어떤 객체를 검출할지를 바로 이 xml파일에 설정함으로써 변경 가능하고, load 메소드를 통해 미리 훈련된 분류기 정보를 가져올수 있습니다.
- 미리 훈련된 분류기 XML 파일은 OpenCV에서 제공

**참고 다른 주피터파일에서 가져옴**


```python
# 적용 가능 기술: 특정 색상 셔츠 이용하기, 특정 색상의 종이 이용하기
# 학생들은 모든 가능성을 자유롭게 생각하고 상대 팀이 본인이 무엇을 하려고 하는지 알아채는 것을 어렵게 만들 수 있습니다.
#your code here

# 유사-하르 필터를 이용한 얼굴 검출기 CascadeClassifier을 사용하여 분류기를 face_classifier 로 지정하기
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ① VideoCapture() 를 이용하여 카메라로부터 실시간 촬영 frame을 받아 'cap'에 저장하기
# your code here
camera = cv2.VideoCapture(0)

while True :   
    
    # ② cap 을 읽어  반환된 두 개의 값을 두 변수 (ret, frame ) 지정하기
    # your code here
    ret, frame = camera.read()
       
    # ③ cv2.cvtColor()사용하여 frame 컬러를 회색조(cv2.COLOR_BGR2GRAY)로 변경하여 'gray'에 저장하기
    # your code here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 검출기를 사용하여 검출하기
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # 검출된 부분의 좌표를 사용하여 검출된 부분에 사각 박스를 그리고 'face'라고 쓰기
    for (x, y, w, h) in faces:
        # ④ frame 에 녹색 사각형을 그리기 
        # your code here
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 2) 
        
        # ⑤ frame 에 'face'라고 cv2.FONT_HERSHEY_SIMPLEX 폰트로 파란색으로 텍스트 쓰기
        # your code here
        cv2.putText(frame, "face", (x+w-50, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,100), 5)
    
    if( 0 < len(faces) ):
        frame = ViewAccessGranted(frame)
    
    # ⑥ frame 을 'Face'라는 윈도운 창 이름에  보여주기
    # your code here
    cv2.imshow('Face',frame) 
           
    # ⑦ 키보드에서 'q' 를 입력받으면 화면 종료(break)하기
    # your code here 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q키로 종료
        break
    # your code here 2

# ⑧ open한 cap 객체 해제하기        
# your code here 
camera.release()
# ⑨ 열린 모든 창 닫기
# your code here
cv2.destroyAllWindows()
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In [9], line 23
         20 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         22 # 검출기를 사용하여 검출하기
    ---> 23 faces = face_classifier.detectMultiScale(
         24     gray,
         25     scaleFactor=1.1,
         26     minNeighbors=5,
         27     minSize=(30, 30),
         28 )
         30 # 검출된 부분의 좌표를 사용하여 검출된 부분에 사각 박스를 그리고 'face'라고 쓰기
         31 for (x, y, w, h) in faces:
         32     # ④ frame 에 녹색 사각형을 그리기 
         33     # your code here
    

    KeyboardInterrupt: 


#### 여기서..
위 KeyboardInterrupt 만해도, 오류는 아님.  
오류보단 영상으로 나오다가 중단시키다 보니 위와 같은 결과가 나옴.

#### 위와 같은 결과로...

물론.. 인식되자마자 grandted등 가능은 한데...<br>
위 문제에서 봐듯이 인식하고 낮선사람이면 DENIED, 익숙한 사람이면 GRANTED를 띄우는게 중요함.<br>

**그렇다면.. 자기 및 아군 인식이 중요**

##### 아래 코드는 일단.. 직접 해보고... 일단 기록만


```python
# 유사-하르 필터를 이용한 얼굴 검출기 CascadeClassifier을 사용하여 분류기를 face_classifier 로 지정하기
#face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 전체 사진에서 얼굴 부위만 추출하는 함수
def face_extractor(img):
    # 흑백처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 찾기
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    cropped_face = None
    for (x, y, w, h) in faces:
        # 해당 얼굴 크기만큼 cropped_face에 잘라 넣기
        # 근데... 얼굴이 2개 이상 감지되면?? 가장 마지막의 얼굴만 남을 듯
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face
# 카메라 실행
cap = cv2.VideoCapture(0)
# 저장할 이미지 카운트 변수
count = 0
while True:
    ret, frame = cap.read()
    # 얼굴 감지 하여 얼굴만 가져오기
    if face_extractor(frame) is not None:
        count += 1
        # 얼굴 이미지 크기를 200x200으로 조정
        face = cv2.resize(face_extractor(frame), (200, 200))
        # 조정된 이미지를 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # ex > faces/user0.jpg   faces/user1.jpg ....
        file_name_path = 'faces/user' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)  # faces폴더에 jpg파일로 저장
        # 화면에 얼굴과 현재 저장 개수 표시
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        # print("Face not Found")
        pass
    if cv2.waitKey(1) == ord('q') or count == 100:
        break
cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In [11], line 38
         35     else:
         36         # print("Face not Found")
         37         pass
    ---> 38     if cv2.waitKey(1) == ord('q') or count == 100:
         39         break
         40 cap.release()
    

    KeyboardInterrupt: 


#### 일단 훈련시켜보자

#### 만일.. face못찾는다고 나온다면..
> !python -m pip install --upgrade pip

> !pip3 install scikit-build<br>
> !pip install cmake

> !pip install opencv-contrib-python

위 명령어를 콘솔이든 jupyter등 시도해본다.



```python
!python -m pip install --upgrade pip
```

    Requirement already satisfied: pip in c:\users\user\.conda\envs\myai\lib\site-packages (22.3.1)
    


```python
!pip3 install scikit-build
!pip install cmake
```

    Collecting scikit-build
      Downloading scikit_build-0.16.2-py3-none-any.whl (78 kB)
         -------------------------------------- 78.1/78.1 kB 722.4 kB/s eta 0:00:00
    Collecting distro
      Downloading distro-1.8.0-py3-none-any.whl (20 kB)
    Requirement already satisfied: wheel>=0.32.0 in c:\users\user\.conda\envs\myai\lib\site-packages (from scikit-build) (0.38.4)
    Requirement already satisfied: setuptools>=42.0.0 in c:\users\user\.conda\envs\myai\lib\site-packages (from scikit-build) (65.5.1)
    Requirement already satisfied: packaging in c:\users\user\.conda\envs\myai\lib\site-packages (from scikit-build) (21.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\user\.conda\envs\myai\lib\site-packages (from packaging->scikit-build) (3.0.9)
    Installing collected packages: distro, scikit-build
    Successfully installed distro-1.8.0 scikit-build-0.16.2
    Collecting cmake
      Downloading cmake-3.25.0-py2.py3-none-win_amd64.whl (32.6 MB)
         --------------------------------------- 32.6/32.6 MB 14.2 MB/s eta 0:00:00
    Installing collected packages: cmake
    Successfully installed cmake-3.25.0
    


```python
from os import listdir
from os.path import isfile, join

data_path = 'faces/'
#faces폴더에 있는 파일 리스트 얻기 
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

#데이터와 매칭될 라벨 변수 
Training_Data, Labels = [], []

#파일 개수 만큼 반복하기
for i, files in enumerate(onlyfiles):    
    image_path = data_path + onlyfiles[i]
     
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #이미지 파일이 아니거나 못 읽어 왔다면 무시
    if images is None:
        continue    
    
    #Training_Data 리스트에 이미지를 바이트 배열로 추가 
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    #Labels 리스트엔 카운트 번호 추가 
    Labels.append(i)

#훈련할 데이터가 없다면 종료.
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

#Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)

#모델 생성 
model = cv2.face.LBPHFaceRecognizer_create()

#학습 시작 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("모델 훈련이 완료되었습니다!!!!!")
```

    모델 훈련이 완료되었습니다!!!!!
    

##### 마지막으로 훈련된 모델을 바탕으로 탐지!


```python
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is () :
        return img,[]
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


#카메라 열기
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    # 얼굴 검출 시도
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        #위에서 학습한 모델로 예측하기
        result = model.predict(face)
        
        # result[1]은 신뢰도 값으로 0에 가까울수록 자신과 같다는 의미로 신뢰도가 높음
        if result[1] < 50:
            
            confidence = int(100*(1-(result[1])/300))  #신뢰도를 %로 나타냄
            display_string = str(confidence)+'% Confidence it is user'  # 유사도 화면에 표시
        
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        
        #75 보다 크면 동일 인물로 간주해 접근허가!
        if confidence > 75:
            cv2.putText(image, "ACCESS GRANTED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:
           #75 이하면 타인.. 접근불가!!
            cv2.putText(image, "ACCESS GRANTED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:
        #얼굴 검출 안됨
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass
    
    if cv2.waitKey(1)== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### 도전과제4 마무리
- 대략적인 얼굴인식 및 페이스 인증에 관해서 익숙해지는 유익한 시간
- 대략적으로 어떠한 원리인지 파악할 기회.
- 응용에 대해서는 고민중...

### 도전과제 5: https://www.youtube.com/watch?v=xyfSUOfFI_E 비디오 시청 

컴퓨터 비전의 능력을 이용하여 여러분이 만들어 보고 싶은 것에 대한 아이디어를 작성해 보세요. <br />

##### .... 고민중

### 도전과제 6: 현실 세계에서 본 Computer Vision 응용 프로그램의 예를 3개 이상 작성하기

##### 현실에서 적용된 Computer Vision 예 중에서..

1. vFlat등으로 책 내용 스마트폰 카메라로 스캔  
https://youtu.be/d6BMiCrPMqk

2. 컴퓨터 비전 기반 무인 버스 운행시스템  
https://koreascience.kr/article/CFKO201735553776688.pdf

3. 아마존 매장의 스마트 매장 시스템  
https://brunch.co.kr/@spacecontext/87

### 도전과제 7: 작업 4에서 만든 시스템의 몇 가지 제한 사항 작성하기

제약 사항을 어떻게 개선할 수 있을까요?


```python
##### 만일 회색조로 
```

#### 그 외 참조한 사이트들.

- 도전과제6 제외 (이미 사이트 링크..)

#### 마무리

- 주로 OpenCV로 어떻게 어디까지, 그리고 어떻게 할까? 에 관한 고민의 기회.
- 아쉽게도 도전과제 5,7을 다 못하고 올리는건 아쉬움.
(여러모로 고민할게 한 둘이 아닌.. 되는대로 계속 갱신예정.)
- 막상 얼굴추적 확인 시스템이 되는걸 볼때, 역시 opencv가 괜히 유명해지지 않는가? 생각을..
- 하지만 이 기반으로 더 현재 잘 쓰이는건 yolov5이상의 버전인데 앞으로 어떨지..

#### 주의사항

본 포트폴리오는 **인텔 AI** 저작권에 따라 그대로 사용하기에 상업적으로 사용하기 어려움을 말씀드립니다.  
