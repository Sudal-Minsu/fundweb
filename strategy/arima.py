"""ARIMA 모델 단일 예측"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# 주식 데이터 다운로드
ticker = "AAPL"  # 예제: 애플 주식
start_date = "2020-01-01"
end_date = "2024-03-20"

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]  # 종가만 사용

# ARIMA 모델 설정 및 훈련
p, d, q = 5, 1, 0  # ARIMA(5,1,0) 예제, 최적의 값은 튜닝 필요
model = ARIMA(data['Close'], order=(p, d, q))
model_fit = model.fit()

# 미래 예측 (30일)
future_days = 30
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
forecast = model_fit.forecast(steps=future_days)

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label="Actual Price")
plt.plot(future_dates, forecast, label="Predicted Price", linestyle='dashed', color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction (ARIMA)")
plt.legend()
plt.show()

"""ARIMA 모델 + GARCH 모델"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from datetime import timedelta

# 📌 주식 데이터 다운로드
ticker = "AAPL"  # 예제: 애플 주식
start_date = "2020-01-01"
end_date = "2024-03-20"

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]  # 종가만 사용
returns = 100 * data['Close'].pct_change().dropna()  # 로그 수익률 계산

# 📌 ARIMA 모델 적용 (평균 수준 예측)
p, d, q = 5, 1, 0  # 하이퍼파라미터는 조정 가능
arima_model = ARIMA(data['Close'], order=(p, d, q))
arima_fit = arima_model.fit()

# 📌 GARCH 모델 적용 (변동성 예측)
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# 📌 미래 예측 (30일)
future_days = 30
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

# ARIMA 예측 (평균값 예측)
arima_forecast = arima_fit.forecast(steps=future_days)

# 📌 GARCH 예측 수정
garch_forecast = garch_fit.forecast(horizon=future_days, reindex=False)  # reindex=False 추가
volatility_forecast = np.sqrt(garch_forecast.variance.values[-1])  # 변동성 예측 수정

# ARIMA-GARCH 조합
np.random.seed(42)
simulated_returns = np.random.normal(loc=0, scale=volatility_forecast)
predicted_prices = arima_forecast.values + simulated_returns  # 변동성을 반영한 예측값

# 📌 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label="Actual Price", color='blue')
plt.plot(future_dates, arima_forecast, label="ARIMA Predicted Price", linestyle='dashed', color='red')
plt.plot(future_dates, predicted_prices, label="ARIMA-GARCH Predicted Price", linestyle='dotted', color='green')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction (ARIMA-GARCH)")
plt.legend()
plt.show()

"""ARIMA + LSTM"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# 📌 주식 데이터 다운로드
ticker = "AAPL"  # 예제: 애플 주식
start_date = "2020-01-01"
end_date = "2024-03-20"

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]  # 종가만 사용

# 📌 1️⃣ ARIMA 모델 적용 (장기 트렌드 예측)
p, d, q = 5, 1, 0  # ARIMA 하이퍼파라미터 (튜닝 가능)
arima_model = ARIMA(data['Close'], order=(p, d, q))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=30)  # 30일 예측

# 📌 데이터 정규화 (LSTM 학습을 위해)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 📌 LSTM 학습 데이터 준비
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # 과거 60일 데이터 기반 예측
X, Y = create_dataset(scaled_data, look_back)

# 📌 데이터 차원 변환 (LSTM 입력 형식: [samples, time steps, features])
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 📌 2️⃣ LSTM 모델 생성 및 학습 (비선형 패턴 예측)
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=20, batch_size=16, verbose=1)

# 📌 미래 데이터 예측 (30일)
future_data = scaled_data[-look_back:].reshape(1, look_back, 1)  # 가장 최근 데이터 사용
lstm_forecast = []

for _ in range(30):
    pred = model.predict(future_data)
    lstm_forecast.append(pred[0, 0])
    new_input = np.append(future_data[0, 1:], pred).reshape(1, look_back, 1)
    future_data = new_input  # 새로운 데이터 추가

# 📌 정규화 해제 (원래 주가 범위로 변환)
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

# 📌 3️⃣ ARIMA + LSTM 결합 (단순 평균)
final_forecast = (arima_forecast.values + lstm_forecast) / 2

# 📌 미래 날짜 생성
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]

# 📌 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label="Actual Price", color='blue')
plt.plot(future_dates, arima_forecast, label="ARIMA Forecast", linestyle='dashed', color='red')
plt.plot(future_dates, lstm_forecast, label="LSTM Forecast", linestyle='dotted', color='green')
plt.plot(future_dates, final_forecast, label="Hybrid Forecast (ARIMA+LSTM)", linestyle='dashdot', color='purple')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction (ARIMA + LSTM)")
plt.legend()
plt.show()
