import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import time

# 1. 데이터

datasets= load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
# :속성 정보:
# - cm 단위의 세팔 길이
# - 단면 너비(cm)
# - 꽃잎 길이(cm)
# - 꽃잎 너비(cm)
# - 클래스:
# - 이리스세토사
# - 아이리스-베르시컬러
# - 아이리스-버지니아 주
# 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x= datasets.data     #(150,4)
y= datasets.target   #(150,)
print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.7,random_state=100,shuffle=True
)

print(x_train.shape, y_train.shape)  #(105,4) (105,)
print(x_test.shape,y_test.shape)     #(45,4)  (45,)
print(y_test)

# 2. 모델구성

model=Sequential()
model.add(Dense(105,activation='linear', input_dim=4))
model.add(Dense(52, activation= 'relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(13))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일 훈련

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
              metrics=['mse','accuracy'])

start_time = time.time()
model.fit(x_train,y_train, epochs=500, batch_size=100, verbose=0)
end_time= time.time() - start_time

print('걸린시간 : ',end_time)

# 4. 평가 , 예측

loss, acc ,mse= model.evaluate(x_test, y_test)
print('loss: ', loss)   # loss:  0.01778794452548027
print('acc: ', acc)     # acc:  1.4724591970443726
print('mse: ',mse)      # mse : 1.0
