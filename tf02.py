import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,5,4,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))


model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mae',optimizer='adam') # w값이 -일때 'mae' 사용
model.fit(x,y, epochs=1000,batch_size=3)

#4. 예측 , 평가
loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([6])
print('6의 예측값 :', result)
