#[실습] loss = ' sparse_categorical_crossentropy ' 를 사용하여 분석

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import time

# 1. 데이터

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)
# :속성 정보:
# - 술
# - 말산
# - 재
# - 회분의 염도
# - 마그네슘
# - 페놀 총량
# - 플라바노이드
# - 비플라바노이드 페놀
# - 프로안토시아닌
# - 색강도
# - 색조
# - OD280/OD315 희석 와인
# - 프롤린
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data    #(178,13)
y = datasets.target  #(178,)
print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.7,random_state=100,shuffle=True
)
print(x_train.shape,y_train.shape)  #(124,13) (124,)
print(x_test.shape,y_test.shape)    #(54,13)  (54,)
print(y_test)

# 2. 모델구성

model= Sequential()
model.add(Dense(124,activation='linear',input_dim=13))
model.add(Dense(62,activation='relu'))
model.add(Dense(31,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 컴파일 훈련

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam'
              , metrics='accuracy')

start_time = time.time()
model.fit(x_train,y_train,epochs=100,batch_size=100,verbose=0)
end_time= time.time() - start_time

print('걸린시간 : ', end_time)

# 4. 평가 , 예측 

loss,acc = model.evaluate(x_test,y_test)
print('loss : ', loss)   #loss: 0.8117342591285706
print('acc : ',acc)      #acc :  0.6111111044883728

### argmax 로 accuracy score 구하기
y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)
y_test = y_test.argmax(axis=1)
argmax_acc = accuracy_score(y_test, y_predict)
print('argmax_acc', argmax_acc)