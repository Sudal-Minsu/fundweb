# -*- coding: utf-8 -*-

#prophet이 설치되지 않은 경우 이 코드부터 작성
!pip install --upgrade cmdstanpy
!pip install prophet

import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

# ✅ 1. 주식 데이터 가져오기 (Yahoo Finance 사용)
def get_stock_data(symbol, start="2024-01-01", end="2025-01-01"):
    stock = yf.download(symbol, start=start, end=end)
    stock.reset_index(inplace=True)
    stock = stock[["Date", "Close"]]
    stock.columns = ["ds", "y"]  # Prophet 모델의 입력 컬럼명 (필수)
    return stock

# 삼성전자 (005930.KQ) 데이터 가져오기
df = get_stock_data("005930.KQ")

# ✅ 2. Prophet 모델 초기화 및 학습
model = Prophet()
model.fit(df)

# ✅ 3. 미래 데이터 프레임 생성 (60일 예측 -> 2025년 3월까지)
future = model.make_future_dataframe(periods=60)  # 60일 예측
forecast = model.predict(future)

# ✅ 4. 예측 결과 시각화 (2025년 3월 초까지 표시)
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], label="Actual Prices", color="blue")
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted Prices", color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Prophet Stock Price Prediction")
plt.grid()
plt.show()

# ✅ 5. 예측 결과 출력 (2025년 1월~3월 초)
forecast_filtered = forecast[(forecast["ds"] >= "2025-01-01") & (forecast["ds"] <= "2025-03-05")]
print(forecast_filtered[["ds", "yhat", "yhat_lower", "yhat_upper"]])

"""Prophet + LSTM"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ✅ 1. 주가 데이터 가져오기
def get_stock_data(symbol, start="2023-01-01", end="2025-01-01"):
    stock = yf.download(symbol, start=start, end=end)
    stock.reset_index(inplace=True)
    stock = stock[["Date", "Close"]]
    stock.columns = ["ds", "y"]  # Prophet 모델용 컬럼명
    return stock

df = get_stock_data("AAPL")  # 애플 주가 데이터

# ✅ 2. Prophet 모델 학습
model = Prophet()
model.fit(df)

# ✅ 3. 미래 데이터 프레임 생성 (90일 예측)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# ✅ 4. Prophet 예측값을 LSTM 입력으로 변환
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(forecast[["yhat"]].values)

# ✅ 5. LSTM 학습 데이터 생성
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ✅ 6. LSTM 모델 생성 및 학습
lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.fit(X, Y, epochs=20, batch_size=16, verbose=1)

# ✅ 7. 최종 예측 수행 (Prophet + LSTM 조합)
future_data = scaled_data[-look_back:].reshape(1, look_back, 1)
lstm_forecast = []

for _ in range(90):  # 90일 예측
    pred = lstm_model.predict(future_data)
    lstm_forecast.append(pred[0, 0])
    new_input = np.append(future_data[0, 1:], pred).reshape(1, look_back, 1)
    future_data = new_input

# ✅ 8. 정규화 해제 및 결합
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
final_forecast = (forecast["yhat"].values[-90:] + lstm_forecast) / 2  # Prophet + LSTM 평균 조합

# ✅ 9. 시각화
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], label="Actual Prices", color="blue")
plt.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast", linestyle="dashed", color="red")
plt.plot(forecast["ds"][-90:], lstm_forecast, label="LSTM Forecast", linestyle="dotted", color="green")
plt.plot(forecast["ds"][-90:], final_forecast, label="Hybrid Forecast (Prophet+LSTM)", linestyle="dashdot", color="purple")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Prophet + LSTM Stock Price Prediction")
plt.grid()
plt.show()

# ✅ 10. 최종 예측 결과 출력 (90일)
final_df = pd.DataFrame({
    "Date": forecast["ds"][-90:],
    "Prophet": forecast["yhat"].values[-90:],
    "LSTM": lstm_forecast,
    "Final Prediction": final_forecast
})

# 결과 출력 (마지막 30일만 보기)
print(final_df.tail(30))

"""Prophet + 감성 분석 + LSTM

감성분석 코드 필요
"""

!pip install yfinance pandas numpy matplotlib prophet tensorflow scikit-learn vaderSentiment

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ 1. 주가 데이터 가져오기
def get_stock_data(symbol, start="2023-01-01", end="2025-01-01"):
    stock = yf.download(symbol, start=start, end=end)
    stock.reset_index(inplace=True)
    stock = stock[["Date", "Close"]]
    stock.columns = ["ds", "y"]  # Prophet 모델용 컬럼명
    return stock

df = get_stock_data("AAPL")  # 애플 주가 데이터

# ✅ 2. 뉴스 감성 분석 추가 (VADER 사용)
analyzer = SentimentIntensityAnalyzer()

# 감성 점수 임의 생성 (실제 뉴스 크롤링 필요)
df["sentiment"] = [analyzer.polarity_scores("Apple stock is doing well.")["compound"] if x % 2 == 0 else
                    analyzer.polarity_scores("Apple stock is dropping.")["compound"] for x in range(len(df))]

# ✅ 3. Prophet 모델 학습 (감성 점수 추가)
model = Prophet()
model.add_regressor("sentiment")  # 감성 점수 반영
model.fit(df)

# ✅ 4. 미래 데이터 프레임 생성 (90일 예측)
future = model.make_future_dataframe(periods=90)
future["sentiment"] = 0.1  # 미래 감성 점수는 평균값 사용
forecast = model.predict(future)

# ✅ 5. Prophet 예측값을 LSTM 입력으로 변환
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(forecast[["yhat"]].values)

# ✅ 6. LSTM 학습 데이터 생성
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ✅ 7. LSTM 모델 생성 및 학습
lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.fit(X, Y, epochs=20, batch_size=16, verbose=1)

# ✅ 8. 최종 예측 수행 (Prophet + LSTM 조합)
future_data = scaled_data[-look_back:].reshape(1, look_back, 1)
lstm_forecast = []

for _ in range(90):  # 90일 예측
    pred = lstm_model.predict(future_data)
    lstm_forecast.append(pred[0, 0])
    new_input = np.append(future_data[0, 1:], pred).reshape(1, look_back, 1)
    future_data = new_input

# ✅ 9. 정규화 해제 및 결합
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
final_forecast = (forecast["yhat"].values[-90:] + lstm_forecast) / 2  # Prophet + LSTM 평균 조합

# ✅ 10. 시각화
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], label="Actual Prices", color="blue")
plt.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast", linestyle="dashed", color="red")
plt.plot(forecast["ds"][-90:], lstm_forecast, label="LSTM Forecast", linestyle="dotted", color="green")
plt.plot(forecast["ds"][-90:], final_forecast, label="Hybrid Forecast (Prophet+LSTM)", linestyle="dashdot", color="purple")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Prophet + Sentiment Analysis + LSTM Stock Price Prediction")
plt.grid()
plt.show()

# ✅ 11. 최종 예측 결과 출력 (90일)
final_df = pd.DataFrame({
    "Date": forecast["ds"][-90:],
    "Prophet": forecast["yhat"].values[-90:],
    "LSTM": lstm_forecast,
    "Final Prediction": final_forecast
})

# 결과 출력 (마지막 30일만 보기)
print(final_df.tail(30))
