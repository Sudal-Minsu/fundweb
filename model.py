import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
import hashlib, struct
from datetime import datetime, timedelta

SEQ_LEN = 5
PERCENT = 2
VAL_LOSS_THRESHOLD = 0.693
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def code_to_seed(code: str, base=42):
    h = hashlib.md5(code.encode('utf-8')).digest()
    return base + (struct.unpack_from(">I", h)[0] % 10_000_000)

def load_stock_data(code, engine=None):
    code = code.strip().upper()
    if code.isdigit():
        if len(code) == 6:
            code += ".KS"
        elif len(code) == 8:
            code += ".KQ"
    elif len(code) <= 5 and code.isalpha():
        pass

    try:
        df = yf.download(code, start="2010-01-01", progress=False)
        df.reset_index(inplace=True)
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        })
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Code"] = code
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load stock data from yfinance for {code}: {e}")
        return pd.DataFrame()

STANDARD_COLS = ['Close_RET', 'USD_KRW_RET', 'KOSPI_RET', 'KOSDAQ_RET', 'Momentum_3']
MINMAX_COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'TradingValue']
FEATURE_COLUMNS = STANDARD_COLS + MINMAX_COLS

def engineer_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df['Close_RET'] = df['Close'].pct_change().fillna(0)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df['TradingValue'] = df['Close'] * df['Volume']

    df[['KOSPI', 'KOSDAQ', 'USD_KRW', 'US_RATE']] = df[['KOSPI', 'KOSDAQ', 'USD_KRW', 'US_RATE']].ffill().fillna(0)
    df['USD_KRW_RET'] = df['USD_KRW'].pct_change().fillna(0)
    df['KOSPI_RET'] = df['KOSPI'].pct_change().fillna(0)
    df['KOSDAQ_RET'] = df['KOSDAQ'].pct_change().fillna(0)
    df['Momentum_3'] = df['Close'] / df['Close'].shift(3) - 1

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    return df

def scale_features(df, scalers):
    parts = []
    if 'minmax' in scalers:
        parts.append(scalers['minmax'].transform(df[MINMAX_COLS]))
    if 'standard' in scalers:
        parts.append(scalers['standard'].transform(df[STANDARD_COLS]))
    return np.concatenate(parts, axis=1)

class StockModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(32, 16)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.norm(out[:, -1, :])
        out = self.act1(self.fc1(out))
        out = self.act2(self.fc2(out))
        out = self.drop(out)
        return self.out(out)

def prepare_sequences(features, closes):
    X, y = [], []
    for i in range(len(features) - SEQ_LEN - 1):
        window = features[i:i+SEQ_LEN]
        base = closes[i + SEQ_LEN - 1]
        future = closes[i + SEQ_LEN]
        if future >= base * (1 + PERCENT / 100):
            label = 0
        elif future <= base * (1 - PERCENT / 100):
            label = 1
        else:
            continue
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)

def compute_confidence_threshold(model, val_loader, percentile=PERCENT):
    confidences = []
    model.eval()
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1)
            confidences.extend(probs.max(dim=1)[0].cpu().numpy())
    if not confidences:
        return 1.0
    return np.percentile(confidences, 100 - percentile)

def train_model(df, code, epochs, batch_size):
    seed = code_to_seed(code)
    torch.manual_seed(seed)
    np.random.seed(seed)

    split_idx = int(len(df) * 0.7)

    scalers = {
        "minmax": MinMaxScaler().fit(df[MINMAX_COLS].iloc[:split_idx]),
        "standard": StandardScaler().fit(df[STANDARD_COLS].iloc[:split_idx])
    }

    features = scale_features(df, scalers)
    closes = df['Close'].values
    X, y = prepare_sequences(features, closes)

    if len(y) < 100:
        return None, None, None, None

    split = int(len(X) * 0.7)
    train_ds = TensorDataset(torch.tensor(X[:split], dtype=torch.float32),
                             torch.tensor(y[:split], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X[split:], dtype=torch.float32),
                           torch.tensor(y[split:], dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = StockModel(X.shape[2], dropout=0.1).to(device)
    opt = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_loss = float('inf')
    best_model = None
    patience = 3
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                total += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = total / len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_model is None:
        return None, None, None, None

    model.load_state_dict(best_model)
    alpha = compute_confidence_threshold(model, val_loader, percentile=PERCENT)

    return model, scalers, best_loss, alpha

def run_model_and_return_data_last_month(symbol, epochs=30, input_sequence_length=5, batch_size=32):
    global SEQ_LEN
    SEQ_LEN = input_sequence_length

    stock = load_stock_data(symbol)
    if stock.empty:
        raise ValueError(f"{symbol}에 대한 주가 데이터가 없습니다.")
    
    df = engineer_features(stock)
    df = df.sort_values("Date")

    if df.empty:
        raise ValueError("결과 데이터가 없습니다.")

    end_date = df['Date'].max() - timedelta(days=1)
    start_date = end_date - timedelta(days=40)
    df_test = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

    if len(df_test) < SEQ_LEN + 2:
        raise ValueError(f"예측 구간 데이터가 부족합니다. 현재 {len(df_test)}개, 최소 {SEQ_LEN + 2}개 필요")

    df_train = df[df['Date'] < start_date].copy()

    model, scalers, val_loss, alpha = train_model(df_train, code=symbol, epochs=epochs, batch_size=batch_size)
    if model is None:
        raise ValueError(f"{symbol} 학습 불가")

    preds, trues, dates, probs, corrects = [], [], [], [], []
    for i in range(SEQ_LEN, len(df_test) - 1):
        window_x = df_test.iloc[i - SEQ_LEN:i]
        target_day = df_test.iloc[i + 1]

        if window_x.shape[0] != SEQ_LEN:
            continue

        x = scale_features(window_x, scalers)
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.softmax(model(xt), dim=1).cpu().numpy()[0]
        confidence = prob.max()

        if confidence < alpha:
            preds.append(-1)
            trues.append(int(target_day['Close'] > df_test.iloc[i]['Close']))
            corrects.append(0)
            probs.append(float(confidence))
            dates.append(str(target_day['Date'].date()))
            continue

        pred_label = int(prob[0] > prob[1]) ^ 1
        true_label = int(target_day['Close'] < df_test.iloc[i]['Close'])
        correct = int(pred_label == true_label)

        preds.append(pred_label)
        trues.append(true_label)
        corrects.append(correct)
        probs.append(float(confidence))
        dates.append(str(target_day['Date'].date()))

    return {
        "dates": dates,
        "preds": preds,
        "trues": trues,
        "probs": probs,
        "corrects": corrects
    }