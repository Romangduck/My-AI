import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import time

# 1. 데이터

datasets = load_diabetes()
print(datasets.DESCR)
print(datasets.feature_names)
#   속성 정보:
# - 연령
# - 성별의
# - 체질량 지수
# - bp 평균 혈압
# - s1 tc, 총 혈청 콜레스테롤
# - s2 ldl, 저밀도 지질단백질
# - s3 hdl, 고밀도 지질단백질
# - sch, 총 콜레스테롤 / HDL
# - s5 ltg, 혈청 트리글리세리드 수치 기록 가능
# - s6 글루, 혈당 수치
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x = datasets.data
y = datasets.target
print(x.shape,y.shape) #(442,10) (442,)

x_train,x_test,y_train,y_test= train_test_split(
    x,y,train_size=0.7,random_state=100,shuffle=True
)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(y_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(221,activation="linear", input_dim=10))
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

# 3. 컴파일 훈련

model.compile(loss='binary_crossentropy',optimizer='adam'
              ,metrics='accuracy')
start_time = time.time()
model.fit(x_train,y_train,epochs=100,batch_size=200,verbose=0)
end_time = time.time() - start_time

# 4. 평가 예측

# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('acc : ', acc)

loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
y_predict=np.round(y_predict)
acc = accuracy_score(y_test,y_predict)
print('loss : ',loss)
print('acc: ',acc)