import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#멀티미디어 퍼셉트론

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,2,1,1,2,1.1,1.2,1.4,1.5,1.6],
              [10,9,8,7,6,5,4,3,2,1,]])

y = np.array([11,12,13,14,15,16,17,18,19,20])

# 모델구성부터 평가예측까지 완성하시오
# 예측 [[10,1.6,1]]

print(x.shape) #(2, 10)  -> shape은 행과열을 표시해줌
print(y.shape) #(10, )

x = x.transpose()   #x = x.T()
print(x.shape) #(10,2)
print(x)

# 2. 모델구성

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))

model.add(Dense(1))

# 3. 컴파일 , 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=5)

# 4. 예측 , 평가

loss = model.evaluate(x,y)      #loss: 3.640897148216027e-06
print('loss:', loss)

result = model.predict([[10,1.6,1]])
print('[10],[1.6],[1]의 예측값 : ', result )    #[10],[1.6],[1]의 예측값 :  [[20.001509]]

