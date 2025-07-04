# 10일차
## matplotlib 활용 예제 5개 풀이
[활용 예제 5개](https://github.com/audalsgh/20250704/blob/main/0704_python_matplotlib_%ED%95%A8%EC%88%98%EC%98%88%EC%A0%9C5%EA%B0%9C.ipynb)

## 인공지능 개요 짧게 수업
**AI > ML > DL 순서.**
- 지도학습 : 정답이 있는 데이터(라벨링, 규칙이 있음)로 학습하고, 입력->출력 관계가 존재한다.
- 비지도학습 : 학습할 데이터만 있고 (라벨링, 규칙은 없음)정답은 없다. 비정상적인 상황을 감지하는데 쓰임.
- 강화학습 : 시행착오를 통한 학습으로, 보상을 최대화하는 전략을 사용한다. ex) 최단거리

- 딥러닝 : (CNN / RNN) 으로 나뉘고, 각각 (영상처리 이미지처리 / 시간이 있는 데이터 처리)에 쓰인다. <br>
우리는 자율주행이 목적이므로 주로 CNN을 다룰것.

-> 지도학습과 자율주행이 무슨 관계가 있는지 조사하고, 코드같은게 있다면 50줄정도 코랩에서 테스트해보고 제출하기.

## 지도학습(Supervised Learning)과 자율주행의 관계
1. 개요<br>
- 지도학습은 입력 데이터를 바탕으로 정답(레이블)을 학습하여 새로운 입력에 대한 예측을 수행하는 기계학습 기법.<br>
- 자율주행 시스템에서 지도학습은 주행 환경 인식(차선 인식, 객체 탐지, 신호등 인식 등)과 제어 명령 예측(스티어링 각도, 가속/제동 명령 예측 등)에 필수적으로 활용됨.

3. 활용 사례<br>
- 차선 감지(Lane Detection): 도로 영상을 입력으로 차선 위치를 학습하고, 실시간으로 차선을 검출하여 차로 유지 제어에 활용
- 객체 검출(Object Detection): 보행자·차량·장애물 등을 학습된 모델로 탐지하여 충돌 방지 및 경로 계획에 기여
- 신호등/표지판 인식: 교통 신호등과 표지판을 분류하여 정지·감속 등의 정책 결정에 반영
- 행동 명령 학습(End-to-End Learning): 전방 카메라 영상을 입력으로 스티어링 각도를 직접 예측하는 방식

4. 데이터 준비 및 전처리<br>
- 데이터 수집: 전방 카메라 영상(image)과 차량의 스티어링 각도, 속도 같은 주행 로그(record) 동기화 데이터 확보
- 레이블링: 각 영상 프레임마다 스티어링 각도, 제동 여부 등을 레이블로 매핑
- 전처리:<br>
 해상도 조정 및 정규화<br>
 데이터 증강(Augmentation): 회전, 이동, 밝기 변화 등<br>
 학습/검증/테스트 셋 분리

**5. gpt로 조사한 자율주행에 쓰이는 짧은 예시 코드 (30줄 내외)**
```python
import os  #tf 라이브러리, keras 환경 설치
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 디렉토리 설정
train_dir = 'data/traffic_signs/train'
val_dir   = 'data/traffic_signs/val'

# 1) 데이터 증강 및 전처리
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
).flow_from_directory(
    train_dir, target_size=(64,64), batch_size=32, class_mode='categorical'
)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(64,64), batch_size=32, class_mode='categorical'
)

# 2) 모델 정의, 여기서 지도학습의 정답이 되는 부분을 설정하는것
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# 3) 컴파일 및 학습, 지도학습 진행
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# 4) 지도학습 결과를 평가 및 저장하기.
loss, acc = model.evaluate(val_gen)
print(f"Validation accuracy: {acc*100:.2f}%")
model.save('traffic_sign_classifier.h5')
```
