# -*- coding: utf-8 -*-
"""
High risk
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ====================
# CONFIG
# ====================
tickers = {
    "Apple": "AAPL",
    "Samsung": "005930.KS",
    "SK": "034730.KQ"
}
train_start = "2020-01-01"
train_end = "2024-10-31"
test_start = "2024-11-01"
test_end = "2024-12-31"

# ====================
# 기술적 지표 함수
# ====================
def add_technical_indicators(df):
    df['rsi'] = compute_rsi(df['y'])
    df['macd'], df['macd_signal'] = compute_macd(df['y'])
    df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger(df['y'])
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_bollinger(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

# ====================
# GRU 모델 정의
# ====================
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

def prepare_gru_data(series, seq_len=60):
    data = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

def train_gru(X, y, epochs=100):
    model = GRUModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def predict_gru(model, series, future_days, seq_len=60, scaler=None):
    data = series.values.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    input_seq = torch.tensor(data_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
    preds = []
    for _ in range(future_days):
        with torch.no_grad():
            pred = model(input_seq)
        preds.append(pred.item())
        new_input = torch.cat([input_seq[:, 1:, :], pred.view(1, 1, 1)], dim=1)
        input_seq = new_input
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ====================
# 기존 모델 학습 함수
# ====================
def download_data(ticker):
    df = yf.download(ticker, start=train_start, end=test_end)[['Close']].dropna()
    df.columns = ['y']
    df['ds'] = df.index
    return df

def train_prophet(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df[['ds', 'y']])
    return model

def train_xgboost(df):
    df['day'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    X = df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']].fillna(0)
    y = df['y']
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    return model

def train_arima(df):
    model = ARIMA(df['y'], order=(5, 1, 0))
    return model.fit()

# ====================
# 예측 앙상블
# ====================
def ensemble_predict(prophet_model, xgb_model, arima_model, gru_model, df, test_df, scaler):
    future_df = test_df[['ds']].copy()
    future_days = len(test_df)

    prophet_pred = prophet_model.predict(future_df)[['ds', 'yhat']]

    future_df['day'] = future_df['ds'].dt.dayofyear
    future_df['year'] = future_df['ds'].dt.year
    for col in ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']:
        future_df[col] = df[col].iloc[-1]  # 최근값 유지
    xgb_pred = xgb_model.predict(future_df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']])

    arima_pred = arima_model.forecast(steps=future_days)

    gru_pred = predict_gru(gru_model, df['y'], future_days, scaler=scaler)

    # 앙상블: Prophet(0.2) + XGBoost(0.3) + ARIMA(0.1) + GRU(0.4)
    final_df = future_df.copy()
    final_df['yhat'] = (
        0.2 * prophet_pred['yhat'].values +
        0.3 * xgb_pred +
        0.1 * arima_pred.values +
        0.4 * gru_pred
    )
    return final_df

# ====================
# 실행 루프
# ====================
for name, ticker in tickers.items():
    print(f"\n▶ {name} 예측 중...")

    df = download_data(ticker)
    df = add_technical_indicators(df)

    train_df = df[df['ds'] < test_start]
    test_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].reset_index(drop=True)

    # 모델 학습
    prophet_model = train_prophet(train_df)
    xgb_model = train_xgboost(train_df)
    arima_model = train_arima(train_df)

    X_gru, y_gru, gru_scaler = prepare_gru_data(train_df['y'])
    gru_model = train_gru(X_gru, y_gru)

    # 예측
    predictions = ensemble_predict(prophet_model, xgb_model, arima_model, gru_model, train_df, test_df, gru_scaler)

    # 평가
    merged = pd.merge(test_df[['ds', 'y']], predictions[['ds', 'yhat']], on='ds', how='inner')
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    print(f"{name} MAE: {round(mae, 2)}")

    # 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(merged['ds'], merged['y'], label='Actual')
    plt.plot(merged['ds'], merged['yhat'], label='Predicted')
    plt.title(f"{name} Test")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ========== 기술 지표 함수 (NaN 제거 없이 보완 처리) ==========
def add_technical_indicators(df):
    df['rsi'] = compute_rsi(df['y'])
    df['macd'], df['macd_signal'] = compute_macd(df['y'])
    upper, lower = compute_bollinger(df['y'])
    df['bollinger_upper'] = upper
    df['bollinger_lower'] = lower
    df = df.fillna(method='bfill').fillna(method='ffill')  # NaN 보완
    return df

# ========== 수정된 XGBoost 학습 함수 ==========
def train_xgboost(df):
    df['day'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    X = df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']].fillna(0)
    y = df['y']
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    return model

# ========== 미래 7일 예측 실행 ==========
for name, ticker in tickers.items():
    print(f"\n▶ {name} 미래 7일 예측 중...")

    today = datetime.today().date()
    df = download_data(ticker, start="2020-01-01", end=today)
    df = add_technical_indicators(df)

    full_df = df.copy()

    # 모델 학습
    prophet_model = train_prophet(full_df)
    xgb_model = train_xgboost(full_df)
    arima_model = train_arima(full_df)
    X_gru, y_gru, gru_scaler = prepare_gru_data(full_df['y'])
    gru_model = train_gru(X_gru, y_gru)

    # 예측할 future df 생성: 오늘부터 7 영업일
    future_dates = pd.date_range(start=pd.Timestamp(today), periods=7, freq='B')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['day'] = future_df['ds'].dt.dayofyear
    future_df['year'] = future_df['ds'].dt.year

    # 기술 지표는 가장 최근 값으로 유지
    for col in ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']:
        future_df[col] = full_df[col].iloc[-1]

    # 개별 모델 예측
    prophet_pred = prophet_model.predict(future_df)[['ds', 'yhat']]
    xgb_pred = xgb_model.predict(future_df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']])
    arima_pred = arima_model.forecast(steps=7)
    gru_pred = predict_gru(gru_model, full_df['y'], future_days=7, scaler=gru_scaler)

    # 앙상블 결합
    future_df['yhat'] = (
        0.2 * prophet_pred['yhat'].values +
        0.3 * xgb_pred +
        0.1 * arima_pred.values +
        0.4 * gru_pred
    )

    # 예측 그래프 출력
    plt.figure(figsize=(10, 4))
    plt.plot(future_df['ds'], future_df['yhat'], label='Predicted')
    plt.title(f"{name} Predict")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

"""Low risk"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ====================
# CONFIG
# ====================
tickers = {
    "Apple": "AAPL",
    "Samsung": "005930.KS",
    "SK": "034730.KQ"
}
train_start = "2020-01-01"
train_end = "2024-10-31"
test_start = "2024-11-01"
test_end = "2024-12-31"

# ====================
# 기술적 지표 함수
# ====================
def add_technical_indicators(df):
    df['rsi'] = compute_rsi(df['y'])
    df['macd'], df['macd_signal'] = compute_macd(df['y'])
    df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger(df['y'])
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_bollinger(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

# ====================
# GRU 모델 정의
# ====================
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

def prepare_gru_data(series, seq_len=60):
    data = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

def train_gru(X, y, epochs=100):
    model = GRUModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def predict_gru(model, series, future_days, seq_len=60, scaler=None):
    data = series.values.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    input_seq = torch.tensor(data_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
    preds = []
    for _ in range(future_days):
        with torch.no_grad():
            pred = model(input_seq)
        preds.append(pred.item())
        new_input = torch.cat([input_seq[:, 1:, :], pred.view(1, 1, 1)], dim=1)
        input_seq = new_input
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ====================
# 기존 모델 학습 함수
# ====================
def download_data(ticker):
    df = yf.download(ticker, start=train_start, end=test_end)[['Close']].dropna()
    df.columns = ['y']
    df['ds'] = df.index
    return df

def train_prophet(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df[['ds', 'y']])
    return model

def train_xgboost(df):
    df['day'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    X = df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']].fillna(0)
    y = df['y']
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    return model

def train_arima(df):
    model = ARIMA(df['y'], order=(5, 1, 0))
    return model.fit()

# ====================
# 예측 앙상블
# ====================
def ensemble_predict(prophet_model, xgb_model, arima_model, gru_model, df, test_df, scaler):
    future_df = test_df[['ds']].copy()
    future_days = len(test_df)

    prophet_pred = prophet_model.predict(future_df)[['ds', 'yhat']]

    future_df['day'] = future_df['ds'].dt.dayofyear
    future_df['year'] = future_df['ds'].dt.year
    for col in ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']:
        future_df[col] = df[col].iloc[-1]  # 최근값 유지
    xgb_pred = xgb_model.predict(future_df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']])

    arima_pred = arima_model.forecast(steps=future_days)

    gru_pred = predict_gru(gru_model, df['y'], future_days, scaler=scaler)

    # 앙상블: Prophet(0.2) + XGBoost(0.3) + ARIMA(0.1) + GRU(0.4)
    final_df = future_df.copy()
    final_df['yhat'] = (
        0.2 * prophet_pred['yhat'].values +
        0.3 * xgb_pred +
        0.1 * arima_pred.values +
        0.4 * gru_pred
    )
    return final_df

# ====================
# 실행 루프
# ====================
for name, ticker in tickers.items():
    print(f"\n▶ {name} 예측 중...")

    df = download_data(ticker)
    df = add_technical_indicators(df)

    train_df = df[df['ds'] < test_start]
    test_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].reset_index(drop=True)

    # 모델 학습
    prophet_model = train_prophet(train_df)
    xgb_model = train_xgboost(train_df)
    arima_model = train_arima(train_df)

    X_gru, y_gru, gru_scaler = prepare_gru_data(train_df['y'])
    gru_model = train_gru(X_gru, y_gru)

    # 예측
    predictions = ensemble_predict(prophet_model, xgb_model, arima_model, gru_model, train_df, test_df, gru_scaler)

    # 평가
    merged = pd.merge(test_df[['ds', 'y']], predictions[['ds', 'yhat']], on='ds', how='inner')
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    print(f"{name} MAE: {round(mae, 2)}")

    # 저장
    predictions.to_csv(f'predictions_{name}.csv', index=False)
    print(f"{name} 예측 결과 저장됨: predictions_{name}.csv")

    # 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(merged['ds'], merged['y'], label='Actual')
    plt.plot(merged['ds'], merged['yhat'], label='Predicted')
    plt.title(f"{name} 주가 예측 (강화 앙상블 모델)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ====================
# 미래 7일 예측 블럭
# ====================
for name, ticker in tickers.items():
    print(f"\n▶ {name} 미래 7일 예측 중...")

    today = datetime.today().date()
    df = yf.download(ticker, start="2020-01-01", end=today)[['Close']].dropna()
    df.columns = ['y']
    df['ds'] = df.index
    df = add_technical_indicators(df)

    full_df = df.copy()

    # 모델 학습
    prophet_model = train_prophet(full_df)
    xgb_model = train_xgboost(full_df)
    arima_model = train_arima(full_df)
    X_gru, y_gru, gru_scaler = prepare_gru_data(full_df['y'])
    gru_model = train_gru(X_gru, y_gru)

    # 예측용 future_df 생성: 오늘부터 7 영업일
    future_dates = pd.date_range(start=pd.Timestamp(today), periods=7, freq='B')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['day'] = future_df['ds'].dt.dayofyear
    future_df['year'] = future_df['ds'].dt.year

    # 기술 지표는 최근 값으로 유지
    for col in ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']:
        future_df[col] = full_df[col].iloc[-1]

    # 개별 모델 예측
    prophet_pred = prophet_model.predict(future_df)[['ds', 'yhat']]
    xgb_pred = xgb_model.predict(future_df[['day', 'year', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']])
    arima_pred = arima_model.forecast(steps=7)
    gru_pred = predict_gru(gru_model, full_df['y'], future_days=7, scaler=gru_scaler)

    # 앙상블 결합
    future_df['yhat'] = (
        0.2 * prophet_pred['yhat'].values +
        0.3 * xgb_pred +
        0.1 * arima_pred.values +
        0.4 * gru_pred
    )

    # 예측 그래프 출력
    plt.figure(figsize=(10, 4))
    plt.plot(future_df['ds'], future_df['yhat'], label='Predicted')
    plt.title(f"{name} predict")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
