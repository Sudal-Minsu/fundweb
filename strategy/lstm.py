# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17Dp28zhY9YHUx03D072uYnwFeE7yY0-k

CNN+LSTM
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from datetime import timedelta

# 1. Yahoo Finance에서 데이터 다운로드
def fetch_stock_data(ticker, start='2020-01-01', end='2025-01-01'):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data[['Close']]  # 종가만 사용

# 2. 데이터 전처리 (정규화 및 시퀀스 변환)
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])

    return np.array(X), np.array(y), scaler

# 3. CNN + LSTM 모델 구축
def create_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Dropout(0.2),

        # LSTM이 입력을 3D로 받아야 하므로 차원 조정
        Reshape((input_shape[0] - 4, 128)),  # (timesteps, features)

        LSTM(100, return_sequences=True),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 4. 모델 학습 및 예측 (미래 예측 포함)
def train_and_predict(ticker, future_days=30):
    # 데이터 불러오기
    stock_data = fetch_stock_data(ticker)

    # 날짜 인덱스 저장
    dates = stock_data.index

    # 데이터 전처리
    sequence_length = 60
    X, y, scaler = preprocess_data(stock_data, sequence_length)

    # 훈련 및 검증 데이터 분리
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # 테스트 데이터의 날짜 가져오기 (y_test에 대응하는 날짜)
    test_dates = dates[split + sequence_length:]

    # 모델 생성
    model = create_cnn_lstm_model((sequence_length, 1))

    # 학습
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # 예측 (기존 데이터)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # 실제값 변환
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 5. 미래 예측
    future_predictions = []
    last_sequence = X_test[-1]  # 마지막 시퀀스 가져오기

    future_dates = [test_dates[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

    for _ in range(future_days):
        next_prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
        future_predictions.append(next_prediction[0, 0])

        # 시퀀스 업데이트 (가장 오래된 값 제거 후 새로운 예측값 추가)
        last_sequence = np.roll(last_sequence, shift=-1, axis=0)
        last_sequence[-1] = next_prediction

    # 미래 예측값 변환 (스케일링 원복)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # 6. 결과 시각화
    plt.figure(figsize=(12,6))
    plt.plot(test_dates, actual_prices, label="Actual Prices", color='blue')
    plt.plot(test_dates, predictions, label="Predicted Prices", color='red')
    plt.plot(future_dates, future_predictions, label="Future Predictions", color='green', linestyle="dashed")
    plt.xlabel("Date")  # x축 라벨 추가
    plt.ylabel("Stock Price (KRW)")
    plt.legend()
    plt.title(f"Samsung Electronics (005930.KS) Stock Price Prediction")
    plt.xticks(rotation=45)  # 날짜 가독성을 위해 회전
    plt.show()

# 실행 (삼성전자 005930.KS, 미래 30일 예측)
train_and_predict('005930.KS', future_days=30)