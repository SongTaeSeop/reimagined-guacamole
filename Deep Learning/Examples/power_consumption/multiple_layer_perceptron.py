import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolite_percentage_error(y_true, y_pred):
    y_trure, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred) / y_true)) * 100

data = pd.read_csv("power_consumption_dataset.csv", engine="python")
# print(data.head())

# 2016년 1/18 18:00부터 2/18 17:00까지
train_data = data.iloc[168:840, 1:12]
# 2016년 2/18 18:00부터 5/27 18:00까지
test_data = data.iloc[840:, 1:12]

## 데이터 정규화
X_Data = train_data.iloc[:,:-1]
scaler = StandardScaler().fit(X_Data)
X_Data = scaler.transform(X_Data)
Y_Data = train_data.iloc[:,-1]

X_Test = test_data.iloc[:, :-1]
X_Test = scaler.transform(X_Test)
Y_Test = test_data.iloc[:,-1]

## 다층 퍼셉트론 구성
# model1: 은닉층 1층, 7개의 노드
model1 = MLPRegressor(hidden_layer_sizes=(7), activation="relu", solver="adam", alpha=0.0001, batch_size=28, learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=2000, random_state=42)

# model2: 은닉층 2층, 7, 7개의 노드
model2 = MLPRegressor(hidden_layer_sizes=(7,7), activation="relu", solver="adam", alpha=0.0001, batch_size=28, learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=2000, random_state=42)

model3 = MLPRegressor(hidden_layer_sizes=(7,7,7), activation="relu", solver="adam", alpha=0.0001, batch_size=28, learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=2000, random_state=42)

model4 = MLPRegressor(hidden_layer_sizes=(7,7,7,7), activation="relu", solver="adam", alpha=0.0001, batch_size=28, learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=2000, random_state=42)

model5 = MLPRegressor(hidden_layer_sizes=(7,7,7,7,7), activation="relu", solver="adam", alpha=0.0001, batch_size=28, learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=2000, random_state=42)

## 5가지 모델로 예측
result = []
# model1
model1.fit(X_Data, Y_Data.values.ravel())
Y_Pred = model1.predict(X_Test)
Y_True = Y_Test
rmse = math.sqrt(mean_squared_error(Y_Pred, Y_True))
mae = mean_absolute_error(Y_Pred, Y_True)
mape = mean_absolite_percentage_error(Y_Pred, Y_True)
Y_Pred = pd.DataFrame(Y_Pred, columns=["Prediction"])
Y_Pred.to_csv("ANN1.csv", header=True, index = False)

result.append(["HL=1",rmse, mae, mape])

# model2
model2.fit(X_Data, Y_Data.values.ravel())
Y_Pred = model2.predict(X_Test)
Y_True = Y_Test
rmse = math.sqrt(mean_squared_error(Y_Pred, Y_True))
mae = mean_absolute_error(Y_Pred, Y_True)
mape = mean_absolite_percentage_error(Y_Pred, Y_True)
Y_Pred = pd.DataFrame(Y_Pred, columns=["Prediction"])
Y_Pred.to_csv("ANN2.csv", header=True, index = False)

result.append(["HL=2",rmse, mae, mape])

# model3
model3.fit(X_Data, Y_Data.values.ravel())
Y_Pred = model3.predict(X_Test)
Y_True = Y_Test
rmse = math.sqrt(mean_squared_error(Y_Pred, Y_True))
mae = mean_absolute_error(Y_Pred, Y_True)
mape = mean_absolite_percentage_error(Y_Pred, Y_True)
Y_Pred = pd.DataFrame(Y_Pred, columns=["Prediction"])
Y_Pred.to_csv("ANN3.csv", header=True, index = False)

result.append(["HL=3",rmse, mae, mape])

# model4
model4.fit(X_Data, Y_Data.values.ravel())
Y_Pred = model4.predict(X_Test)
Y_True = Y_Test
rmse = math.sqrt(mean_squared_error(Y_Pred, Y_True))
mae = mean_absolute_error(Y_Pred, Y_True)
mape = mean_absolite_percentage_error(Y_Pred, Y_True)
Y_Pred = pd.DataFrame(Y_Pred, columns=["Prediction"])
Y_Pred.to_csv("ANN4.csv", header=True, index = False)

result.append(["HL=4",rmse, mae, mape])

# model5
model5.fit(X_Data, Y_Data.values.ravel())
Y_Pred = model5.predict(X_Test)
Y_True = Y_Test
rmse = math.sqrt(mean_squared_error(Y_Pred, Y_True))
mae = mean_absolute_error(Y_Pred, Y_True)
mape = mean_absolite_percentage_error(Y_Pred, Y_True)
Y_Pred = pd.DataFrame(Y_Pred, columns=["Prediction"])
Y_Pred.to_csv("ANN5.csv", header=True, index = False)

result.append(["HL=5",rmse, mae, mape])

## 결과

result = np.asarray(result)
result = pd.DataFrame(result, columns=["Model", "RMSE", "MAE", "MAPE"])
result.to_csv("Results.csv", header=True, index=False)

print(result)