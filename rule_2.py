import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Parallel, delayed
import multiprocessing as mp
import hashlib, struct
from config import DB_CONFIG

# ------------------- 설정 -------------------
TRAIN_YEARS = 12
TARGET_PERCENT = 0.02
BACKTEST_START_DATE = pd.to_datetime("2024-07-11")
TEST_PERIOD_DAYS = 300
SEQ_LEN = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 20
VAL_LOSS_THRESHOLD = 0.68
PROB_THRESHOLD = 0.65

# 병렬 설정
N_CORE = max(1, mp.cpu_count() - 1) # N_CORE = 5

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 출력 폴더 설정
OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 병렬 환경의 비결정성 방지
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# PyTorch 결정적 알고리즘 사용
torch.use_deterministic_algorithms(True)

# 종목 코드별로 항상 같은 시드를 생성하기 위한 해시 기반 시드 함수
def code_to_seed(code: str, base=42):
    # MD5 해시 → int → % 10_000_000
    h = hashlib.md5(code.encode('utf-8')).digest()
    return base + (struct.unpack_from(">I", h)[0] % 10_000_000)

# 고정 시드 설정
random.seed(42)
np.random.seed(42)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ------------------- 피처 정의 -------------------
STANDARD_COLS = [
    # 수익률
    'Close_RET', 
    # 모멘텀
    'Momentum_5',
    'Momentum_10',
    'Momentum_20', 
    # Disparity
    'Disparity_5',
    'Disparity_5_Slope',
    'Disparity_10',
    'Disparity_20', 
    # MA 기울기 & 갭
    'MA_5_slope',
    'MA_20_slope',
    'MA_20_slope_norm',
    'MA5_MA20_gap',
    'MA_std_5_10_20', 
    # 변동성
    'Volatility_5',
    'Volatility_ratio_5_20', 
    # MACD
    'MACD_Histogram',
    'MACD_Histogram_Slope', 
    # RSI
    'RSI_Slope', 
    # 거시 지표 변화량
    'USD_KRW_RET',
    'USD_KRW_Momentum',
    'USD_KRW_DEVIATION',
    'KOSPI_RET',
    'KOSPI_VOLATILITY',
    'KOSDAQ_RET',
    'KOSDAQ_VOLATILITY',
    # 시가/고저 활용
    'Intraday_Range',
    # 최근 고점/저점 대비
    'Price_vs_high_20d',
    'Price_vs_low_20d',
    # 시장 간 스프레드
    'KOSPI_vs_KOSDAQ', 
]

MINMAX_COLS = [
    # 레벨 값
    'Close',
    'LogVolume',
    'TradingValue',
    'TradingValue_Ratio', 
    # 수준 지표
    'RSI_n',
    'KOSPI',
    'KOSDAQ',
    'USD_KRW',
    'USD_KRW_MA_5', 
    # 밴드/확률형 지표 (0~1 or 0~100)
    'BB_PCT',
    'Stoch_K',
    'Stoch_D', 
    # 종가의 고저 내 상대 위치 (0~1)
    'Close_to_High',
]

BINARY_COLS = [
    # RSI 상태
    'RSI_oversold',
    'RSI_overbought',
    'RSI_normal_zone', 
    # 단기 급락, 거래량 급증, 변동성 스파이크
    'ShortTerm_Drop',
    'Volume_jump',
    'Volatility_spike',
    # 저가권 근접
    'Price_near_low_20d', 
    # MA 돌파
    'Breaks_MA_20',
    'Breaks_MA_5',
    'MA20_above_MA60',
    'Strong_Trend', 
    # 환율↓+MACD↑, 거래량+MACD 콤보, 복합 매수 시그널
    'USD_down_MACD_up',
    'MACD_cross_up',
    'MACD_cross_down',
    'MACD_bullish_cross',
    'Volume_MACD_Combo',
    'Composite_Bullish_Signal', 
    # 갭업/갭다운
    'Gap_Up',
    'Gap_Down',
    # 캔들 패턴
    'Bullish_Candle',
    'Bearish_Candle',
    # 추가 MACD, 반등 감지
    'MACD_histogram_above_zero', 
    'Rebound_from_low', 
]
CONTINUOUS_COLS = STANDARD_COLS + MINMAX_COLS 
FEATURE_COLUMNS = CONTINUOUS_COLS + BINARY_COLS

# ------------------- DB & 데이터 로딩 -------------------
def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def load_stock_data(code, engine):
    q = f"SELECT Date, Open, High, Low, Close, Volume FROM stock_data WHERE Code='{code}' ORDER BY Date"
    df = pd.read_sql(q, engine, parse_dates=['Date'])
    df['Code'] = code
    return df

def load_market_index(engine):
    q = "SELECT Date, IndexType, Close FROM market_index"
    df = pd.read_sql(q, engine, parse_dates=['Date'])
    return df.pivot(index='Date', columns='IndexType', values='Close').reset_index()

# ------------------- 지표 계산 -------------------
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return (100 - 100/(1+rs)).fillna(0)

# ------------------- 전처리 공통 -------------------
def engineer_features(df):
    df = df.sort_values('Date').reset_index(drop=True)
    # --- 기본 정리 ---
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df['LogVolume'] = np.log1p(df['Volume'])
    df['TradingValue'] = df['Close'] * df['Volume']
    df['Close'] = df['Close']
    df['Close_RET'] = df['Close'].pct_change().fillna(0)

    # --- 거시 경제 변수 전처리 ---
    df[['KOSPI','KOSDAQ','USD_KRW','US_RATE']] = df[['KOSPI','KOSDAQ','USD_KRW','US_RATE']].ffill().fillna(0)
    df['USD_KRW_RET'] = df['USD_KRW'].pct_change().fillna(0)
    df['USD_KRW_MA_5'] = df['USD_KRW'].rolling(5).mean().fillna(0)
    df['USD_KRW_DEVIATION'] = (df['USD_KRW'] - df['USD_KRW_MA_5']) / df['USD_KRW_MA_5']
    df['USD_KRW_Momentum'] = df['USD_KRW'].pct_change(3).fillna(0)
    df['KOSPI_RET'] = df['KOSPI'].pct_change().fillna(0)
    df['KOSDAQ_RET'] = df['KOSDAQ'].pct_change().fillna(0)
    df['KOSPI_VOLATILITY'] = df['KOSPI'].rolling(5).std().fillna(0)
    df['KOSDAQ_VOLATILITY'] = df['KOSDAQ'].rolling(5).std().fillna(0)
    df['KOSPI_vs_KOSDAQ'] = df['KOSPI_RET'] - df['KOSDAQ_RET']

    # --- 거래량 기반 ---
    df['TradingValue_Ratio'] = df['TradingValue'] / df['TradingValue'].rolling(5).mean()
    df['Volume_jump'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)

    # --- 모멘텀 ---
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

    # --- 단기 낙폭 및 저가 근접 ---
    df['ShortTerm_Drop'] = (df['Close'].pct_change(1) < -0.02).astype(int)
    df['Price_near_low_20d'] = (df['Close'] < df['Close'].rolling(20).min() * 1.05).astype(int)
    df['Price_vs_high_20d'] = df['Close'] / df['Close'].rolling(20).max() - 1
    df['Price_vs_low_20d'] = df['Close'] / df['Close'].rolling(20).min() - 1

    # --- 변동성 ---
    df['Volatility_5'] = df['Close'].rolling(5).std().fillna(0)
    df['Volatility_spike'] = (df['Volatility_5'] > df['Volatility_5'].rolling(30).mean()).astype(int)
    df['Volatility_ratio_5_20'] = df['Volatility_5'] / df['Close'].rolling(20).std()

    # --- 이동 평균 / 추세 ---
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_60'] = df['Close'].rolling(60).mean()
    df['Disparity_5'] = df['Close'] / df['Close'].rolling(5).mean() - 1
    df['Disparity_10'] = df['Close'] / df['Close'].rolling(10).mean() - 1
    df['Disparity_20'] = df['Close'] / df['Close'].rolling(20).mean() - 1
    df['Disparity_5_Slope'] = df['Disparity_5'].diff().fillna(0)
    df['MA_20_slope'] = df['MA_20'].diff().fillna(0)
    df['MA_5_slope'] = df['Close'].rolling(5).mean().diff().fillna(0)
    df['MA_20_slope_norm'] = df['MA_20_slope'] / df['MA_20'].shift(1)
    df['MA5_MA20_gap'] = df['Close'].rolling(5).mean() / df['MA_20'] - 1
    df['MA_std_5_10_20'] = df[['MA_5', 'MA_10', 'MA_20']].std(axis=1)
    df['Breaks_MA_20'] = (df['Close'] > df['MA_20']).astype(int)
    df['Breaks_MA_5'] = (df['Close'] > df['Close'].rolling(5).mean()).astype(int)
    df['MA20_above_MA60'] = (df['MA_20'] > df['MA_60']).astype(int)
    df['Strong_Trend'] = ((df['MA_5_slope'] > 0) & (df['MA20_above_MA60'] == 1)).astype(int)

    # --- RSI ---
    df['RSI_n'] = compute_rsi(df['Close'], 2)
    df['RSI_Slope'] = df['RSI_n'].diff().fillna(0)
    df['RSI_normal_zone'] = ((df['RSI_n'] >= 30) & (df['RSI_n'] <= 70)).astype(int)
    df['RSI_oversold'] = (df['RSI_n'] < 30).astype(int)
    df['RSI_overbought'] = (df['RSI_n'] > 70).astype(int)

    # --- MACD ---
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']
    df['MACD_Histogram_Slope'] = df['MACD_Histogram'].diff().fillna(0)
    df['MACD_cross_up'] = ((df['MACD_Histogram'].shift(1) < 0) & (df['MACD_Histogram'] > 0)).astype(int)
    df['MACD_cross_down'] = ((df['MACD_Histogram'].shift(1) > 0) & (df['MACD_Histogram'] < 0)).astype(int)
    df['MACD_bullish_cross'] = ((df['MACD_Line'].shift(1) < df['Signal_Line'].shift(1)) & (df['MACD_Line'] > df['Signal_Line'])).astype(int)
    df['MACD_histogram_above_zero'] = (df['MACD_Histogram'] > 0).astype(int)

    # --- 복합 시그널 ---
    df['USD_down_MACD_up'] = ((df['USD_KRW_DEVIATION'] < -0.002) & (df['MACD_cross_up'] == 1)).astype(int)
    df['Volume_MACD_Combo'] = ((df['Volume_jump'] == 1) & (df['MACD_bullish_cross'] == 1)).astype(int)
    df['Composite_Bullish_Signal'] = (
        df[['USD_down_MACD_up', 'Strong_Trend', 'Volume_MACD_Combo']].sum(axis=1) >= 2
    ).astype(int)

    # --- Bollinger Bands ---
    df['BB_MID'] = df['Close'].rolling(20).mean()
    df['BB_STD'] = df['Close'].rolling(20).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    df['BB_PCT'] = (df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'] + 1e-5)

    # --- Stochastic Oscillator ---
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-5)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # --- 시가/고저 기반 피처 ---
    df['Gap_Up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02).astype(int)
    df['Gap_Down'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) < -0.02).astype(int)
    df['Close_to_High'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-5)
    df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
    df['Bullish_Candle'] = ((df['Close'] > df['Open']) & (df['Close_to_High'] > 0.7)).astype(int)
    df['Bearish_Candle'] = ((df['Close'] < df['Open']) & (df['Close_to_High'] < 0.3)).astype(int)

    # --- 반등 시나리오 ---
    df['Rebound_from_low'] = ((df['Price_near_low_20d'] == 1) & (df['Close_RET'] > 0)).astype(int)
    
    # 결측값 제거
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    # 날짜 기준으로 오름차순 정렬
    df.sort_values('Date', inplace=True)
    return df

# ------------------- 모델 정의 -------------------
class StockModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.global_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.gelu1 = nn.GELU()
        self.fc_middle = nn.Linear(32, 16)  
        self.gelu2 = nn.GELU()
        self.fc2 = nn.Linear(16, 2)        

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        gru_out, _ = self.gru(x)         # gru_out: (batch, seq_len, hidden_size)
        out = gru_out[:, -1, :]          # 마지막 시점의 output 선택 (batch, hidden_size)

        out = self.layernorm(out)
        out = self.fc1(out)
        out = self.gelu1(out)
        out = self.fc_middle(out)        
        out = self.gelu2(out)
        out = self.global_dropout(out)
        out = self.fc2(out)
        return out

# ------------------- 시퀀스 생성 함수 -------------------
def prepare_sequences(features, close_prices):
    sequences, targets = [], []
    max_i = len(features) - SEQ_LEN 
    for i in range(max_i):
        window = features[i: i + SEQ_LEN]
        base_price = close_prices[i + SEQ_LEN - 1] # 1일 전
        future_price = close_prices[i + SEQ_LEN] # 오늘
        if future_price >= base_price * (1 + TARGET_PERCENT):
            label = 0  # 상승
        elif future_price <= base_price * (1 - TARGET_PERCENT):
            label = 1  # 하락
        else:
            continue
        sequences.append(window)
        targets.append(label)
    return np.array(sequences), np.array(targets)

# ------------------- 스케일링 -------------------
def scale_features(df, scalers): 
    if df.empty:
        raise ValueError("[scale_features] 입력 데이터프레임이 비어 있습니다.")
    parts = []
    if 'minmax' in scalers:
        parts.append(scalers['minmax'].transform(df[MINMAX_COLS]))
    if 'standard' in scalers:
        parts.append(scalers['standard'].transform(df[STANDARD_COLS]))
    if BINARY_COLS:
        parts.append(df[BINARY_COLS].values)
    return np.concatenate(parts, axis=1)

# ------------------- 학습 함수 -------------------
def train_model(df_train, code=None):
    
    # 종목별 시드 설정 (같은 종목이면 항상 같은 시드)
    seed = code_to_seed(code)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1) 검증 누수 방지: 학습 데이터 앞 70%만으로 스케일러 fit ---
    split_idx = int(len(df_train) * 0.7)
    df_train_feat = df_train.iloc[:split_idx]  # scaler 학습에 사용할 데이터

    scalers = {}
    if MINMAX_COLS:
        scalers['minmax'] = MinMaxScaler().fit(df_train_feat[MINMAX_COLS])
    if STANDARD_COLS:
        scalers['standard'] = StandardScaler().fit(df_train_feat[STANDARD_COLS])

    # 전체 df_train에 스케일러 적용
    features = scale_features(df_train, scalers)

    # 시퀀스 및 레이블 생성
    X_seq, y = prepare_sequences(features, df_train['Close'].values)
    if len(y) < 100:
        return None, scalers

    # 2) train/val split (앞 70%: 학습, 뒤 30%: 검증) ---
    split_idx = int(len(y) * 0.7)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    g = torch.Generator()
    g.manual_seed(seed + 12345)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = StockModel(input_size=X_seq.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 3) 학습 루프
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    early_stopping_patience = 3
    
    for epoch in range(EPOCHS):
        # train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        # scheduler step
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scalers, best_val_loss 

# ------------------- 학습, 예측, 평가, 매수 후보 -------------------
def process_code_for_date(args):
    code, data, date = args
    df = data['df'] # 전처리된 주가 데이터
    
    # 1) 학습용 기간 설정
    train_end = date # 예측 당일 전날까지만 학습에 사용
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    df_train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]

    # 2) 모델 학습
    model, scalers, best_val = train_model(df_train, code=code)
    if model is None or best_val is None or best_val >= VAL_LOSS_THRESHOLD:
        return code, {}

    # 3) 최신 시퀀스로 예측
    window = df[df['Date'] < date].tail(SEQ_LEN) # 입력용 데이터 뽑기
    inp = scale_features(window, scalers) # 정규화(SEQ_LEN, feature_dim)
    inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device) # PyTorch 텐서로 변환, 배치 차원 추가 (1, SEQ_LEN, feature_dim)
    # 예측([prob_up, prob_down])
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(inp_tensor), dim=1).cpu().numpy()[0]

    subset = df[df['Date'] < date] # date 하루 전까지 데이터

    # 4) 예측 결과 처리 
    result = {}
    if date in df['Date'].values:
        future_price = df.loc[df['Date'] == date, 'Close'].iloc[0]
        base_price = subset.iloc[-1]['Close'] # 1일 전 종가
        true_label = None
        # 정답 라벨
        if future_price > base_price:
            true_label = 0
        elif future_price < base_price:
            true_label = 1

        if true_label is not None:
            pred_class = None # 확신 없는 예측은 무시
            # 예측 threshold로 판단
            if prob[0] >= PROB_THRESHOLD:
                pred_class = 0
            elif prob[1] >= PROB_THRESHOLD:
                pred_class = 1
            result = {'true_label': true_label, 'pred_class': pred_class}

    return code, result

# ------------------- 테스트 -------------------
def run_test(preproc_dfs, test_dates):
    # 각 종목 상태 저장
    stock_models = {code: {
        'df': df,
        'true': [],
        'pred': []
    } for code, df in preproc_dfs.items()}

    for date in tqdm(test_dates, desc='테스트'):
        codes = list(stock_models.keys())
        # 종목별로 바로 병렬 실행
        parallel_input = [(code, stock_models[code], date) for code in codes]
        results = Parallel(
            n_jobs=N_CORE,
            backend='loky',
            prefer='processes'
        )(
            delayed(process_code_for_date)(args) for args in parallel_input
        )

        for code, res in results:
            if not res:
                continue
            data = stock_models[code]
            if 'true_label' in res:
                data['true'].append(res['true_label'])
                data['pred'].append(res.get('pred_class'))

    return stock_models

# 테스트 기간 Confusion Matrix
def plot_score(stock_models, filename="confusion_matrix.png"):
    all_true, all_pred = [], []
    for data in stock_models.values():
        true_list = data.get('true', [])
        pred_list = data.get('pred', [])
        for t, p in zip(true_list, pred_list):
            # t, p 모두 0 또는 1만 허용 
            if t is not None and p in [0, 1]:
                all_true.append(t)
                all_pred.append(p)

    if not all_true:
        print("유효한 예측 결과가 없어 confusion matrix를 생성하지 않습니다.")
        return

    label_names = ["상승(0)", "하락(1)"]
    pred_names = ["상승(0)", "하락(1)"]

    # 2x2 행렬
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    # 합계 추가된 3x3 확장 행렬
    cm_extended = np.zeros((3, 3), dtype=int)
    cm_extended[:2, :2] = cm
    cm_extended[2, :2] = cm.sum(axis=0)    # 예측 클래스별 합계
    cm_extended[:2, 2] = cm.sum(axis=1)    # 실제 클래스별 합계
    cm_extended[2, 2] = cm.sum()           # 전체 합계

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.Blues

    # 셀 그리기
    for i in range(3):
        for j in range(3):
            value = cm_extended[i, j]
            if i < 2 and j < 2:
                percent = (cm[i, j] / cm.sum() * 100) if cm.sum() > 0 else 0.0
                scaled = 0.2 + 0.8 * (cm[i, j] / cm.sum()) if cm.sum() > 0 else 0.2
                color = cmap(scaled)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color))
                ax.text(j + 0.5, i + 0.5, f"{value}\n({percent:.1f}%)",
                        va='center', ha='center', fontsize=12, color='black')
            else:
                # 합계 행/열
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
                ax.text(j + 0.5, i + 0.5, f"{value}",
                        va='center', ha='center', fontsize=12, fontweight='bold')

    # 축 라벨
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(pred_names + ['합계'], fontsize=13)
    ax.set_yticklabels(label_names + ['합계'], fontsize=13)
    ax.set_xlabel('예측 라벨', fontsize=14)
    ax.set_ylabel('실제 라벨', fontsize=14)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.invert_yaxis()
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Confusion matrix 저장 완료: {os.path.join(OUTPUT_DIR, filename)}")
    
# 오늘 매수후보 리스트 생성
def predict(engine=None):
    today_date = pd.Timestamp.today().normalize()
    output_path = os.path.join(OUTPUT_DIR, "buy_list.csv")   # 파일명 고정

    if engine is None:
        engine = get_engine()
    
    limit = 200
    codes = pd.read_sql(f"SELECT DISTINCT Code FROM stock_data LIMIT {limit}", engine)['Code'].tolist()
    market_idx_df = load_market_index(engine)

    buy_candidates = []
    for code in tqdm(codes, desc="실시간 매수 후보 예측"):
        try:
            df_stock = load_stock_data(code, engine)
            df_full = df_stock.merge(market_idx_df, on='Date', how='left')
            df = engineer_features(df_full)

            train_end = today_date
            train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
            df_train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]

            model, scalers, best_val = train_model(df_train, code=code)
            if model is None or best_val is None or best_val >= VAL_LOSS_THRESHOLD:
                continue

            window = df[df['Date'] < today_date].tail(SEQ_LEN)
            if len(window) < SEQ_LEN:
                continue

            inp = scale_features(window, scalers)
            inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                prob = torch.softmax(model(inp_tensor), dim=1).cpu().numpy()[0]

            if prob[0] >= PROB_THRESHOLD:
                buy_candidates.append({
                    '종목코드': code,
                    '상승확률': round(prob[0], 3)   # 소수점 셋째 자리 반올림
                })

        except Exception as e:
            print(f"[{code}] 예측 실패: {e}")
            continue

    # 확률 기준으로 정렬
    buy_candidates_sorted = sorted(buy_candidates, key=lambda x: x['상승확률'], reverse=True)

    # 무조건 CSV 저장
    if buy_candidates_sorted:
        df_out = pd.DataFrame(buy_candidates_sorted)
        df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[predict_today_candidates] 매수 후보 {len(df_out)}개를 CSV로 저장: {output_path}")
    else:
        print(f"[predict_today_candidates] 매수 후보 없음. CSV 파일 저장하지 않음.")

    return buy_candidates_sorted
        
# ------------------- 메인 -------------------
def main():
    # 1) db 연결
    eng = get_engine()
    
    # 2) 종목 리스트 로딩
    codes = pd.read_sql(f"SELECT DISTINCT Code FROM stock_data LIMIT {200}", eng)['Code'].tolist()
    
    # 3) 전체 종목 데이터 로딩 및 전처리 
    market_idx_df = load_market_index(eng)
    preproc_dfs = {}
    for code in codes:
        df_stock = load_stock_data(code, eng)
        df_full = df_stock.merge(market_idx_df, on='Date', how='left')
        preproc_dfs[code] = engineer_features(df_full)
    
    # 4) 백테스트 날짜 구간 설정
    dates_df = pd.read_sql(f"SELECT Date FROM stock_data WHERE Code='{codes[0]}' ORDER BY Date", eng, parse_dates=['Date'])
    start_idx = dates_df[dates_df['Date'] >= BACKTEST_START_DATE].index[0]
    test_dates = dates_df['Date'].iloc[start_idx : start_idx + TEST_PERIOD_DAYS]

    # 5) 백테스트 실행
    stock_models = run_test(preproc_dfs, test_dates)

    # 6) 결과 저장
    plot_score(stock_models)
    
    print("테스트 및 후처리 완료")

if __name__ == '__main__':
    main()
