#회귀모형
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from sklearn.datasets import load_boston // 윤리적 문제로 제공 안됨
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

print(datasets.feature_names)  #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

# 속성 정보:
# - 블럭 그룹의 중위수 소득
# - 블럭 그룹의 주택 연령 중위수
# - 가구당 평균 객실 수
# - 평균 가구당 침실 수
# - 모집단 블럭 그룹 모집단
# - 평균 가구원수
# - Latitude 블록 그룹 위도
# - 경도 블록 그룹 경도
print(x.shape)   #(20640, 8)
print(y.shape)   #(20640,)

x_train, x_test , y_train , y_test = train_test_split(
    x,y, train_size=0.7, random_state=100, shuffle=True # 트레인 테스트 사이즈 배정
)

print(x_train.shape)  #(14447,8)
print(y_train.shape)  #(14447,)
print(y_test)

# 2. 모델구성

model = Sequential()
model.add(Dense(100,input_dim=8))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일 , 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train , epochs=1000, batch_size=100)

# 4. 평가 , 예측

loss = model.evaluate(x_test, y_test)   # loss :  0.6231313347816467
print('loss : ', loss)

y_predict = model.predict(x_test) # x 전체값 넣어주는거보단 나음!
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)    # r2 스코어 : 0.5297377856444552