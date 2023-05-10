#[실습]
# 1. R2 를 음수가 아닌 0.5 이하로 만들어 보세요
# 2. 데이터는 건드리지 마세요
# 3. 레이어는 인풋 , 아웃풋 포함 7개 이상(히든레이어 5개이상)으로 만들어 주세요
# 4. batch_size=1
# 5. 히든레이어의 노드 갯수는 10개이상 100개 이하
# 6. train_size = 0.7 
# 7. epochos = 100 이상
# [실습시작]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x_train, x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.3,train_size=0.7, random_state=0, shuffle=True
)

# # 2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(45))
model.add(Dense(100))
model.add(Dense(14))
model.add(Dense(100))
model.add(Dense(59))
model.add(Dense(100))
model.add(Dense(11))
model.add(Dense(100))
model.add(Dense(45))
model.add(Dense(100))
model.add(Dense(14))
model.add(Dense(100))
model.add(Dense(59))
model.add(Dense(100))
model.add(Dense(11))
model.add(Dense(100))
model.add(Dense(45))
model.add(Dense(100))
model.add(Dense(59))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))




model.add(Dense(1))

# # 3. 컴파일 , 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=100,batch_size=1)

# # 4. 평가 , 예측 
loss= model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict = model.predict(x)

### R2score

from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

