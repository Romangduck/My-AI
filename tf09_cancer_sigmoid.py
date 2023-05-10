#이진분류

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score # r2 는 회귀,accuracy는 분류 분석
from sklearn.datasets import load_breast_cancer
import time 

# 1. 데이터

datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
# 속성 정보:
# - 반지름(주변의 중심에서 점까지의 거리 범위)
# - 텍스처(회색 척도 값의 표준 편차)
# - 주변의
# - 지역
# - 평활도(반경 길이의 국부적 변동)
# - 컴팩트도(θ^2 / 면적 - 1.0)
# - 오목한 부분(윤곽선의 오목한 부분)
# - 오목한 점(윤곽선의 오목한 부분 수)
# - 대칭성
# - 프랙탈 차원("평균 근사" - 1)

#  'mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
# #  'worst concave points' 'worst symmetry' 'worst fractal dimension'
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569,30) , (569,)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.7,random_state=100,shuffle=True
)

# 2. 모델구성
model = Sequential()
model.add(Dense(100,activation="linear", input_dim=30)) 
model.add(Dense(100,activation="relu"))
model.add(Dense(80,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(1,activation="sigmoid")) #이진분류는 무조건 아웃풋 레이어의
                                         #활성화 함수를 'sigmoid'로 해줘야한다


# 3. 컴파일 훈련

model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy' ,'mse']) #이진분류에서는 binary_crossentrophy써야함 , mse도 보고싶어서 metrice로 추가
start_time = time.time()
model.fit(x_train,y_train,epochs=100,batch_size=200, verbose=2) #verbose = 0 은 불필요한 정보 없이 간결하고 간략한 답변
end_time = time.time() - start_time
# 4. 평가 예측

loss = model.evaluate(x_test,y_test)
y_predict= model.predict(x_test)

# score 값내기 

# y_predict = np.where(y_predict > 0.5, 1 , 0) # y_predice 값이 0.5 보다 크면 1 , 아니면 0 으로 출력
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)
print('loss: ',loss)             #loss:  [0.1706417053937912, 0.05060313642024994] -> loss값이 두개인 이유는 binary와 mse값 둘다 나온거임.
print('acc: ',acc)               #acc:  0.935672514619883
print('걸린 시간:', end_time)
#[실습]accuracy_score를 출력하라
