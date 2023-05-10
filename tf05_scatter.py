import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
# 1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x_train, x_test, y_train,y_test = train_test_split(
    x, y, test_size=0.3,train_size=0.7, random_state=72, shuffle=True
) # x , y 데이터 , test의 사이즈 보통 30% , train 사이즈 보통 70%
  # random_state 는 데이터를 난수값에 의해 추출한다는 의미이며 , 중요한파라미터임
  # shuffle=True 는 데이터를 섞어서 가지고 올 것인지를 정함 
#x = np.array(range(1,21))
#y = np.array(range(1,21))


# [ 실습 ] x와 y 데이터를 파이썬 리스트 스플릿으로 분리하세요


# print(x.shape)
# print(y.shape)
# print(x)

# x_train = np.array(x[0:14])
# y_train = np.array(y[0:14])

# x_test = np.array(x[14:20])
# y_test = np.array(y[14:20])


# # 2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(50))
model.add(Dense(1))

# # 3. 컴파일 , 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=100,batch_size=1)

# # 4. 평가 , 예측 
loss= model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict = model.predict(x)

######## scatter 시각화
import matplotlib.pyplot as plt
plt.scatter(x,y) # 산점도 그리기
plt.plot(x,y_predict,color='blue')
plt.show()

