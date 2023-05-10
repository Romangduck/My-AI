# 1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1))   # 입력층
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(50))                # 은닉층 * 히든레이어
model.add(Dense(1))                 # 출력층 


# 3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)


# 4. 예측,평가
loss = model.evaluate(x, y)
print('loss :', loss)              #loss :0.00028251923504285514

result = model.predict([4])
print('4의 예측값 :', result)       #4의 예측값 : [[3.9665813]]

#여기서 나온 숫자 중 가장 좋은 w 값을 선택하여 사용