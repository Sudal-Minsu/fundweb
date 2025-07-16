import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from tqdm import tqdm
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
import multiprocessing as mp
from config import DB_CONFIG

# ------------------- 설정 -------------------
TRAIN_YEARS = 12
TARGET_PERCENT = 0.02
BACKTEST_START_DATE = pd.to_datetime("2024-07-11")
TEST_PERIOD_DAYS = 300
SEQ_LEN = 5
TRAIN_YEARS = 12
TEST_PERIOD_DAYS = 60
TARGET_PERCENT = 0.01
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32
INITIAL_CAPITAL = 100_000_000
BUY_PROB_THRESHOLD = 0.55
SELL_PROB_THRESHOLD = 0.55
TOP_N_FOR_BUY = 3
BOTTOM_N_FOR_SELL_RANK = 190
FEE_RATE = 0.000140527
TAX_RATE = 0.0015
BACKTEST_START_DATE = pd.to_datetime("2024-07-11")
STOCK_NUMBER = 200

# 병렬 설정
N_CORE = max(1, mp.cpu_count() - 1) # N_CORE = 5
CHUNK = 20  # 워커당 묶을 종목 수

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 출력 폴더 설정
OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 고정 시드 설정
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
# ------------------- 시퀀스 생성 함수 -------------------
def prepare_sequences(features, close_prices, target_percent):
    sequences, targets = [], []
    max_i = len(features) - SEQ_LEN 
    for i in range(max_i):
        window = features[i: i + SEQ_LEN]
        base_price = close_prices[i + SEQ_LEN - 1] # 1일 전
        future_price = close_prices[i + SEQ_LEN]   # 오늘
        if future_price >= base_price * (1 + target_percent):
            label = 0  # 상승
        elif future_price <= base_price * (1 - target_percent):
            label = 1  # 하락
        else:
            continue
        sequences.append(window)
        targets.append(label)
    return np.array(sequences), np.array(targets)


# ------------------- 모델 정의 -------------------
class StockModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.3, drop_features=None, mask_scale=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = SEQ_LEN

        self.drop_features = drop_features if drop_features is not None else []
        self.mask_scale = mask_scale

        self.global_dropout = nn.Dropout(p=dropout)

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size)  # LayerNorm 추가

        self.fc1 = nn.Linear(hidden_size, 32)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)

        if self.drop_features:
            x = x.clone()
            for idx in self.drop_features:
                x[:, :, idx] = x[:, :, idx] * self.mask_scale

        out, _ = self.gru(x)             # (batch, seq_len, hidden_size)
        out = out[:, -1, :]              # 마지막 시점만
        out = self.layernorm(out)        # LayerNorm 적용
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.global_dropout(out)
        out = self.fc2(out)
        return out

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
def train_model(df_train, early_stopping_patience=3):
    # --- 1. 검증 누수 방지: 학습 데이터 앞 70%만으로 스케일러 fit ---
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
    X_seq, y = prepare_sequences(features, df_train['Close'].values, TARGET_PERCENT)
    if len(y) < 100:
        return None, scalers

    # --- 2. train/val split (앞 70%: 학습, 뒤 30%: 검증) ---
    split_idx = int(len(y) * 0.7)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    bad_feature_names = []
    bad_feature_idxs = [FEATURE_COLUMNS.index(name) for name in bad_feature_names if name in FEATURE_COLUMNS]

    model = StockModel(input_size=X_seq.shape[2], drop_features=bad_feature_idxs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    weight_tensor = torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # --- train ---
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # --- validation ---
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
    return model, scalers

# ------------------- 학습, 예측, 평가, 매수 후보 -------------------
def process_code_for_date(args):
    code, data, date = args
    df = data['df'] # 전처리된 주가 데이터
    
    # 1) 학습용 기간 설정
    train_end = date  # 예측 당일 전날까지만 학습에 사용
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS) 
    df_train = df[(df['Date']>=train_start) & (df['Date'] < train_end)]

    # 2) 모델 학습
    model, scalers = train_model(df_train)
    if model is None:
        return code, {}

    # 3) 최신 시퀀스로 예측
    window = df[df['Date'] < date].tail(SEQ_LEN) # 입력용 데이터 뽑기
    inp = scale_features(window, scalers) # 정규화 (SEQ_LEN, feature_dim)
    inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device) # PyTorch 텐서로 변환, 배치 차원 추가 (1, SEQ_LEN, feature_dim)
    # 예측 ([prob_up, prob_down])
    with torch.no_grad():
        prob = torch.softmax(model(inp), dim=1).cpu().numpy()[0]

    result = {
        'model': model,
        'scalers': scalers,
        'last_pred_prob': prob
    }

    # 4) 정답 레이블 
    subset = df[df['Date'] < date]  # date 하루 전까지 데이터
    if len(subset) >= 1:
        base_price = subset.iloc[-1]['Close'] # 1일 전 종가 
        future_price = df.loc[df['Date'] == date, 'Close'].iloc[0] if date in df['Date'].values else None # 오늘 종가
        if future_price is not None:
            # 정답 라벨
            if future_price >= base_price * (1 + TARGET_PERCENT):
                true_label = 0
            elif future_price <= base_price * (1 - TARGET_PERCENT):
                true_label = 1
            else:
                true_label = None

            if true_label is not None:
                # 예측 threshold로 판단 
                if prob[0] >= BUY_PROB_THRESHOLD:
                    pred_class = 0
                elif prob[1] >= SELL_PROB_THRESHOLD:
                    pred_class = 1
                else:
                    pred_class = None  # 확신 없는 예측은 무시
                # 2일 전 → 1일 전 변화율, 3일 전 → 2일 전 변화율
                prev1 = base_price
                prev2 = subset.iloc[-2]['Close']
                prev3 = subset.iloc[-3]['Close'] 
                
                pct_change   = prev1 / prev2 - 1
                pct_change_2 = prev2 / prev3 - 1

                result.update({
                    'true_label': true_label,
                    'pred_class':  pred_class,
                    'pct_change':  pct_change,
                    'pct_change_2': pct_change_2
                })
                    
    # 매수 후보
    if result.get('last_pred_prob') is not None and result.get('pct_change') is not None:
        if (result['last_pred_prob'][0] >= BUY_PROB_THRESHOLD):
            result['buy_candidate'] = {'price': window['Close'].iloc[-1], 'pct_change': result['pct_change']}

    return code, result

# ------------------- 청크 처리 -------------------
def _process_chunk(codes_chunk, stock_models, date):
    chunk_results = {}
    for code in codes_chunk:
        res_code, res = process_code_for_date((code, stock_models[code], date))
        if res: # 결과가 있으면 chunk_results에 저장
            chunk_results[res_code] = res
    return chunk_results

# ------------------- 백테스트 -------------------
def run_backtest(preproc_dfs, test_dates):
    # 각 종목 상태 저장
    stock_models = {code: {'df': df, 'model': None, 'scalers': None,
                            'shares': 0, 'buy_price': None,
                            'true': [], 'pred': [], 'trade_history': []}
                    for code, df in preproc_dfs.items()}
    total_cash = INITIAL_CAPITAL
    portfolio_values = []

    for idx, date in enumerate(tqdm(test_dates, desc='백테스트')):
        # 종목 리스트를 CHUNK 단위로 나눔
        codes = list(stock_models.keys())
        chunks = [codes[i:i+CHUNK] for i in range(0, len(codes), CHUNK)]

        # 각 청크에 대해 joblib으로 병렬 실행
        results_list = Parallel(
            n_jobs=N_CORE,      # 사용할 프로세스 개수
            backend='loky',     # 프로세스 기반 병렬 처리
            prefer='processes', # 프로세스를 선호
            batch_size=1        # 각 프로세스가 청크 하나씩만 처리
        )(delayed(_process_chunk)(chunk, stock_models, date) for chunk in chunks)

        # 결과 병합
        buy_candidates = {}
        for res_dict in results_list:
            for code, res in res_dict.items():
                # 복사
                data = stock_models[code]
                # 모델 & 스케일러
                data['model'] = res['model']
                data['scalers'] = res['scalers']
                data['last_pred_prob'] = res['last_pred_prob']
                # 정답 & 예측
                if 'true_label' in res:
                    data['true'].append(res['true_label'])
                    data['pred'].append(res['pred_class'])
                    if 'pred_dates' not in data:
                        data['pred_dates'] = []
                    data['pred_dates'].append(date)
                    if 'probs' not in data:
                        data['probs'] = []
                    data['probs'].append(res['last_pred_prob'])                   
                # buy 후보
                if 'buy_candidate' in res:
                    buy_candidates[code] = res['buy_candidate']

        # 1) 매도 시뮬레이션
        day_max = total_cash * 0.1
        per_max = day_max * 0.3

        for code, data in stock_models.items():
            shares = data.get('shares', 0)
            if shares <= 0 or data.get('buy_price') is None:
                continue
            # 매도 판단을 하는 오늘 날짜 전까지의 데이터를 df_hist에 저장(매도 판단)
            df_hist = data['df'][data['df']['Date'] < date]
            if df_hist.empty:
                continue
            last_price = df_hist['Close'].iloc[-1]
            buy_price = data['buy_price']
            pred_prob = data.get('last_pred_prob')
            # 초기화
            sell_pct = 0; reason = None
            
            if last_price > buy_price:
                sell_pct = 100; reason = '이익'                
            elif last_price < buy_price:
                sell_pct = 100; reason = '손해'          
                  
            # 실제 매도
            if sell_pct > 0:
                # 매도 수량
                qty = int(shares * sell_pct / 100)
                if qty > 0:
                    # 매도 금액
                    proceeds = qty * last_price * (1 - FEE_RATE - TAX_RATE)
                    total_cash += proceeds
                    data['shares'] -= qty
                    profit_pct = (last_price * (1 - FEE_RATE - TAX_RATE)) / (buy_price * (1 + FEE_RATE)) - 1
                    # 매도 기록 업데이트
                    for trade in data['trade_history']:
                        # 아직 매도하지 않은 거래 찾기
                        if trade.get('sell_date') is None:
                            # 매도 수량>=보유 수량: 전량 매도 처리
                            if qty >= trade['qty']:
                                trade['sell_date'] = date
                                trade['sell_price'] = last_price
                                trade['profit_pct'] = profit_pct
                                trade['sell_reason'] = reason
                                break
                            else: # 매도 수량<보유 수량: 부분 매도 처리
                                trade['qty'] -= qty
                                data['trade_history'].append({
                                    'buy_date': trade['buy_date'],
                                    'buy_price': trade['buy_price'],
                                    'qty': qty,
                                    'sell_date': date,
                                    'sell_price': last_price,
                                    'profit_pct': profit_pct,
                                    'sell_reason': reason
                                })
                                break
        # 2) 매수 시뮬레이션
        if buy_candidates:
            # 코드 리스트를 확률 순으로 정렬 (높은 확률부터)
            sorted_codes = sorted(
                buy_candidates.keys(),
                key=lambda c: stock_models[c]['last_pred_prob'][0],
                reverse=True
            )
            # 상위 N개 종목만 뽑아서 dict 생성
            top_buy = {code: buy_candidates[code] for code in sorted_codes[:TOP_N_FOR_BUY]}
        else:
            top_buy = {}
        for code, info in top_buy.items():
            # 매수 시점 가격(전날 종가)
            price = info["price"] 
            # 매수 수량 
            qty = int(per_max // (price * (1 + FEE_RATE))) 
            if qty <= 0:
                continue
            # 실제 총 매수 비용 계산 (수수료 포함)
            cost = qty * price * (1 + FEE_RATE) 
            if cost > total_cash: # 자금 부족 시 총 자본금 한도 내에서 다시 수량 계산
                qty = int(total_cash // (price * (1 + FEE_RATE)))
                cost = qty * price * (1 + FEE_RATE)
            if qty <= 0:
                continue
            total_cash -= cost
            data = stock_models[code] # 해당 종목 데이터 가져오기
            # 이미 보유 중이라면 평균 단가 새로 계산
            if data['shares'] > 0 and data['buy_price'] is not None:
                prev_s, prev_p = data['shares'], data['buy_price']
                data['buy_price'] = (prev_p * prev_s + price * qty) / (prev_s + qty)
            else: # 신규 매수라면 그냥 현재 매수 가격으로 설정
                data['buy_price'] = price
            data['shares'] += qty
            data['trade_history'].append({
                'buy_date': date,
                'buy_price': price,
                'qty': qty,
                'sell_date': None,
                'sell_price': None,
                'profit_pct': None,
                'sell_reason': None
            })
        
        # 3) 강제 청산 (마지막 날짜)
        if idx == len(test_dates) - 1:
            for code, data in stock_models.items():
                shares = data.get('shares', 0)
                buy_price = data.get('buy_price')
                # 보유 중일 경우 청산 시작
                if shares > 0 and buy_price is not None:
                    # 마지막 날짜 종가
                    price = data['df'][data['df']['Date'] <= date]['Close'].iloc[-1]
                    # 전량 매도 시 수익금, 수익률 계산
                    proceeds = shares * price * (1 - FEE_RATE - TAX_RATE)
                    total_cash += proceeds
                    profit_pct = (price * (1 - FEE_RATE - TAX_RATE)) / (buy_price * (1 + FEE_RATE)) - 1
                    for trade in data['trade_history']:
                        if trade.get('sell_date') is None:
                            trade['sell_date'] = date
                            trade['sell_price'] = price
                            trade['profit_pct'] = profit_pct
                            trade['sell_reason'] = '강제청산'
                    data['shares'] = 0
                    data['buy_price'] = None
                    
        # 4) 포트폴리오 가치 업데이트
        current_value = total_cash
        for code, data in stock_models.items():
            shares = data.get('shares', 0)
            if shares > 0:
                price = data['df'][data['df']['Date'] <= date]['Close'].iloc[-1] # 현재가
                current_value += shares * price # 평가금액(보유 수량 * 현재가) 누적
        portfolio_values.append(current_value)

    return stock_models, portfolio_values

# ------------------- 결과 시각화 및 저장 -------------------
# 백테스트 결과 시각화
def plot_backtest_results(dates, values):
    plt.figure(figsize=(14,6))
    plt.plot(dates, values, label='총 자산')
    plt.title('백테스트 기간의 총 자산')
    plt.xlabel('날짜')
    plt.ylabel('총 자산 (원)')
    formatter = ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '총 자산.png'), dpi=300)
    plt.close()

# 매매 로그 CSV 저장
def trade_log_csv(stock_models, filename="매매 로그.csv"):
    records = []
    for code, data in stock_models.items():
        for t in data.get('trade_history', []):
            records.append({
                'code': code,
                'buy_date': t.get('buy_date'),
                'buy_price': t.get('buy_price'),
                'sell_date': t.get('sell_date'),
                'sell_price': t.get('sell_price'),
                'profit_pct': round(t.get('profit_pct', 0) * 100, 2) if t.get('profit_pct') is not None else None,
                'sell_reason': t.get('sell_reason')
            })
    df = pd.DataFrame(records)
    if not df.empty and 'buy_date' in df.columns:
        df = df.sort_values(by='buy_date')  # buy_date 기준으로 오름차순 정렬
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, encoding='utf-8-sig')
    
# 백테스트 기간 Confusion Matrix
def plot_score(stock_models, filename="backtest_matrix.png"):
    all_true, all_pred = [], []
    for data in stock_models.values():
        true_list = data.get('true', [])
        pred_list = data.get('pred', [])
        filtered = [(t, p) for t, p in zip(true_list, pred_list) if t is not None and p is not None]
        for t, p in filtered:
            all_true.append(t)
            all_pred.append(p)

    if not all_true:
        print("유효한 예측 결과가 없어 confusion matrix를 생성하지 않습니다.")
        return

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    labels = ["상승(0)", "하락(1)"]
    total = cm.sum()

    # 확장된 행렬: 3x3 (예측 2 + 합계) x (실제 2 + 합계)
    cm_extended = np.zeros((3, 3), dtype=int)
    cm_extended[:2, :2] = cm
    cm_extended[2, :2] = cm.sum(axis=0)         # 열 합계 (실제 기준)
    cm_extended[:2, 2] = cm.sum(axis=1)         # 행 합계 (예측 기준)
    cm_extended[2, 2] = total                   # 전체 합계

    # 퍼센트 계산
    cm_percent = cm_extended / total

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.Blues

    for i in range(3):
        for j in range(3):
            value = cm_extended[i, j]
            if i < 2 and j < 2:  # Confusion matrix 부분
                percent = cm_percent[i, j] * 100
                color = cmap(cm_percent[i, j])
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color))
                ax.text(j + 0.5, i + 0.5, f"{value} ({percent:.1f}%)", 
                        va='center', ha='center', fontsize=11, color='black')
            else:  # 합계 영역
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
                ax.text(j + 0.5, i + 0.5, f"{value}", 
                        va='center', ha='center', fontsize=11, fontweight='bold')

    # 축 설정
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(labels + ['합계'], fontsize=11)
    ax.set_yticklabels(labels + ['합계'], fontsize=11)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.invert_yaxis()
    ax.set_title('백테스트 기간 Confusion Matrix', fontsize=14)
    ax.set_xlabel('실제 라벨', fontsize=12)
    ax.set_ylabel('예측 라벨', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

# 오늘 매수후보 리스트 생성
# ------------------- [predict] -------------------
def predict(engine=None):
    """
    오늘 매수 후보 CSV를 생성하고 저장한다.
    """
    today_date = pd.Timestamp.today().normalize()
    output_path = os.path.join(OUTPUT_DIR, "buy_list.csv")  

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

            model, scalers = train_model(df_train)
            if model is None:
                continue

            window = df[df['Date'] < today_date].tail(SEQ_LEN)
            if len(window) < SEQ_LEN:
                continue

            inp = scale_features(window, scalers)
            inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.softmax(model(inp_tensor), dim=1).cpu().numpy()[0]

            if prob[0] >= BUY_PROB_THRESHOLD:
                buy_candidates.append({
                    '종목코드': str(code).zfill(6),
                    '상승확률': round(prob[0], 3),
                    'prob_down': round(prob[1], 3),
                    'price': window['Close'].iloc[-1]
                })

        except Exception as e:
            print(f"❌ [predict] {code} 처리 중 오류: {e}")
            continue

    # ✅ 확률 기준 정렬 후 상위 TOP_N
    top_candidates = sorted(buy_candidates, key=lambda x: x['상승확률'], reverse=True)
    top_candidates = top_candidates[:TOP_N_FOR_BUY]

    # ✅ 항상 CSV 저장
    df_out = pd.DataFrame(top_candidates)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"📝 [predict] 매수 후보 CSV 저장 완료: {output_path} (총 {len(df_out)}건)")

    return top_candidates

def get_today_candidates(engine=None):
    """
    ✅ buy_list.csv를 안전하게 읽어서 종목코드가 항상 문자열 6자리로 유지되도록 한다.
    ✅ 매수제안 컬럼이 '매수'인 것만 필터링한다.
    """
    df = pd.read_csv(
        "rule_2_결과/buy_list.csv",
        dtype={'종목코드': str}   # ← 핵심! 종목코드를 문자열로 강제!
    )
    # 만약 buy_list.csv에 '매수제안' 컬럼이 있다면
    if '매수제안' in df.columns:
        df = df[df['매수제안'] == '매수']  # ✔️ '매수'만 추출

    print(f"✅ [get_today_candidates] 불러온 후보 수: {len(df)}")
    print(df[['종목코드', '상승확률']])  # 디버깅 출력
    return df.to_dict(orient="records")
# ------------------- [main] -------------------
def main():
    eng = get_engine()

    # 종목 로딩, 데이터 전처리 등
    codes = pd.read_sql(f"SELECT DISTINCT Code FROM stock_data LIMIT {STOCK_NUMBER}", eng)['Code'].tolist()
    market_idx_df = load_market_index(eng)
    preproc_dfs = {}
    for code in codes:
        df_stock = load_stock_data(code, eng)
        df_full = df_stock.merge(market_idx_df, on='Date', how='left')
        preproc_dfs[code] = engineer_features(df_full)

    # 백테스트 구간
    dates_df = pd.read_sql(f"SELECT Date FROM stock_data WHERE Code='{codes[0]}' ORDER BY Date", eng, parse_dates=['Date'])
    start_idx = dates_df[dates_df['Date'] >= BACKTEST_START_DATE].index[0]
    test_dates = dates_df['Date'].iloc[start_idx : start_idx + TEST_PERIOD_DAYS]

    # 백테스트
    stock_models, portfolio_values = run_backtest(preproc_dfs, test_dates)

    # 결과 저장
    plot_backtest_results(test_dates, portfolio_values)
    plot_score(stock_models)
    trade_log_csv(stock_models)

    print("✅ 백테스트 및 후처리 완료!")

    # ✅ 백테스트 후 실시간 후보 생성까지
    print("\n🧠 실시간 매수 후보 예측 중...")
    predict(eng)
    print("✅ [main] buy_list.csv 최신화 완료!")

if __name__ == '__main__':
    print("⚡ rule_2.py 직접 실행됨: main() 시작!")
    main()
