import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import yfinance as yf
import os

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def fetch_stock_data(symbol, start="2018-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError(f"{symbol} 종목의 데이터를 가져올 수 없습니다.")
    return df["Close"].dropna().values.reshape(-1, 1)


def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


# 모델 정의
class MLP(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = x.squeeze(-1)
        return self.model(x)


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class RNNTransformerHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 64, batch_first=True)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), 1)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.trans(x)
        return self.fc(x[:, -1])


class DeepAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc_mu(out[:, -1])


class EnsembleModel(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.mlp = MLP(seq_len)
        self.rnn = RNNModel()
        self.rnn_trans = RNNTransformerHybrid()
        self.deepar = DeepAR()
    def forward(self, x):
        preds = [
            self.mlp(x),
            self.rnn(x),
            self.rnn_trans(x),
            self.deepar(x)
        ]
        return torch.stack(preds, dim=0).mean(0)


def train_model(symbol='005930.KS', epochs=100, window_size=60, batch_size=32, save_plot_path=None):
    data = fetch_stock_data(symbol)
    if len(data) < window_size + 100:
        raise ValueError("데이터가 충분하지 않습니다.")

    # 학습/전체 스케일링
    train_size = int(len(data) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(data[:train_size])
    scaled_all = scaler.transform(data)

    X_all, y_all = create_sequences(scaled_all, window_size)
    X_train = torch.tensor(X_all[:train_size - window_size], dtype=torch.float32)
    y_train = torch.tensor(y_all[:train_size - window_size], dtype=torch.float32)
    X_test = torch.tensor(X_all[train_size - window_size:], dtype=torch.float32)
    y_test = torch.tensor(y_all[train_size - window_size:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # 모델 및 학습 설정
    model = EnsembleModel(window_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # EarlyStopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"📌 Early stopping at epoch {epoch + 1}")
                break

    model.eval()
    # ✅ 과거 14일 예측 (X_test 기준)
    past_X = X_test[-14:]
    with torch.no_grad():
        past_pred = model(past_X).cpu().numpy()
    past_pred_inversed = scaler.inverse_transform(past_pred)

    # ✅ 미래 14일 예측 (마지막 X 기준 autoregressive)
    last_X = scaled_all[-window_size:]
    current_input = torch.tensor(last_X.reshape(1, window_size, 1), dtype=torch.float32)
    future_preds = []
    with torch.no_grad():
        for _ in range(14):
            pred = model(current_input).item()
            future_preds.append([pred])
            next_input = torch.cat([current_input[:, 1:, :], torch.tensor(pred).view(1, 1, 1)], dim=1)
            current_input = next_input
    pred_future_inversed = scaler.inverse_transform(np.array(future_preds))

    # ✅ 실제값 (과거 14일) — 그래프 및 평가용
    real_segment = data[-14:]

    # ✅ 평가 지표 (과거 예측 기준)
    mae = mean_absolute_error(real_segment, past_pred_inversed)
    rmse = math.sqrt(mean_squared_error(real_segment, past_pred_inversed))
    mape = np.mean(np.abs((real_segment - past_pred_inversed) / real_segment)) * 100
    acc = round((1 - mae / real_segment.mean()) * 100, 2)

    # ✅ 시각화
    today = datetime.today()
    past_dates = [today - timedelta(days=13 - i) for i in range(14)]
    future_dates = [today + timedelta(days=i) for i in range(14)]

    if save_plot_path:
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        ax.set_facecolor('#2f2f2f')
        plt.plot(past_dates, real_segment, label="실제", color='yellow')
        plt.plot(past_dates, past_pred_inversed, label="과거 예측", color='white', linestyle='--')
        plt.plot(future_dates, pred_future_inversed, label="미래 예측", color='mediumpurple')
        plt.title(f"{symbol} 주가 예측", color='white')
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        legend = plt.legend(loc='upper left', facecolor='#2f2f2f', edgecolor='gray')
        for text in legend.get_texts():
            text.set_color("white")
        plt.grid(True, color='gray', alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, facecolor='#3a3a3a')
        plt.close()

    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "accuracy": acc,
        "plot_path": save_plot_path,
        "symbol": symbol
    }