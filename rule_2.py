import os
import random
import platform
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
from config_choi import DB_CONFIG

# ------------------- 설정 -------------------
TRAIN_YEARS = 10
BACKTEST_START_DATE = pd.to_datetime("2024-07-01")
TEST_PERIOD_DAYS = 300
SEQ_LEN = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 20
VAL_LOSS_THRESHOLD = 0.693
PERCENT = 5

# 병렬 설정
N_CORE = max(1, mp.cpu_count() - 1) # N_CORE = 5

# OS에 따라 폰트 경로 설정
if platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == 'Linux':
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
else:
    font_path = None

# 폰트 설정
if font_path and os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("지정된 폰트를 찾을 수 없습니다.")

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
    # 변화량
    'Close_RET',
    'KOSPI_RET',
    'KOSDAQ_RET',
]

MINMAX_COLS = [
    # 기본값
    'Close', 
    'Open',
    'High',
    'Low',
    'Volume',
]

FEATURE_COLUMNS = STANDARD_COLS + MINMAX_COLS

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

# ------------------- 전처리 공통 -------------------
def engineer_features(df):
    df = df.sort_values('Date').reset_index(drop=True)
    # --- 기본 정리 ---
    df['Close_RET'] = df['Close'].pct_change().fillna(0)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    
    # --- 거시 경제 변수 전처리 ---
    df[['KOSPI','KOSDAQ','USD_KRW']] = df[['KOSPI','KOSDAQ','USD_KRW']].ffill().fillna(0)
    df['KOSPI_RET'] = df['KOSPI'].pct_change().fillna(0)
    df['KOSDAQ_RET'] = df['KOSDAQ'].pct_change().fillna(0)
    
    # 결측값 제거
    df = df.replace([np.inf, -np.inf], np.nan)
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
def prepare_sequences(features, close_prices, target_percent=0.02):
    sequences, targets = [], []
    max_i = len(features) - SEQ_LEN 
    for i in range(max_i):
        window = features[i: i + SEQ_LEN]
        base_price = close_prices[i + SEQ_LEN - 1]
        future_price = close_prices[i + SEQ_LEN]
        if future_price >= base_price * (1 + target_percent):
            label = 0  # 상승
        elif future_price <= base_price * (1 - target_percent):
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
    return np.concatenate(parts, axis=1)

# ------------------- 신뢰도 임계값 계산 -------------------
def compute_confidence_threshold(model, val_loader, percentile=PERCENT):
    confidences = [] # 검증셋의 각 샘플별 모델의 최대 확신도(softmax 최대값)”를 저장
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1)
            max_conf = probs.max(dim=1)[0].cpu().numpy()
            confidences.extend(max_conf)
    if not confidences:
        return 1.0  # fallback, 집계 불가 상황
    target_percentile = 100 - percentile
    return np.percentile(confidences, target_percentile)

# ------------------- 학습 함수 -------------------
def train_model(df_train, code=None, conf_percentile=PERCENT):
    
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
    # α 계산
    alpha = compute_confidence_threshold(model, val_loader, percentile=conf_percentile)
    return model, scalers, best_val_loss, alpha

# ------------------- 학습, 예측, 평가, 매수 후보 -------------------
def process_code_for_date(args):
    code, data, date, conf_percentile = args
    df = data['df'] # 전처리된 주가 데이터
    
    # 1) 학습용 기간 설정
    train_end = date # 예측 당일 전날까지만 학습에 사용
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    df_train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]

    # 2) 모델 학습
    model, scalers, best_val, alpha = train_model(df_train, code=code, conf_percentile=conf_percentile)
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
            pred_class = None
            # threshold(α) 이상일 때만 예측 인정
            if max(prob) >= alpha:
                if prob[0] > prob[1]:
                    pred_class = 0
                    result = {'true_label': true_label, 'pred_class': pred_class, 'prob_up': float(prob[0])}
                elif prob[1] > prob[0]:
                    pred_class = 1
                    result = {'true_label': true_label, 'pred_class': pred_class, 'prob_up': float(prob[0])}
                else:
                    result = {}
            else:
                result = {}

    return code, result

# ------------------- 테스트 -------------------
def run_test(preproc_dfs, test_dates, conf_percentile=PERCENT):
    # 각 종목 상태 저장
    stock_models = {code: {
        'df': df,
        'true': [],
        'pred': [],
        'date': [],
        'prob_up': []
    } for code, df in preproc_dfs.items()}

    for date in tqdm(test_dates, desc='테스트'):
        codes = list(stock_models.keys())
        # 종목별로 바로 병렬 실행
        parallel_input = [(code, stock_models[code], date, conf_percentile) for code in codes]
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
                data['date'].append(date)
                data['prob_up'].append(res.get('prob_up'))
                
    return stock_models

# 테스트 기간 Confusion Matrix
def plot_score(stock_models, filename="confusion_matrix.png", csv_filename="raw.csv"):
    # 집계용(전체), raw 저장용(상승예측만) 따로 누적
    all_true, all_pred, all_code, all_date, all_prob_up = [], [], [], [], []
    all_true_all, all_pred_all = [], []  # 집계용

    for code, data in stock_models.items():
        true_list = data.get('true', [])
        pred_list = data.get('pred', [])
        date_list = data.get('date', [])
        prob_up_list = data.get('prob_up', [])
        for t, p, d, prob_up in zip(true_list, pred_list, date_list, prob_up_list):
            # 1. confusion matrix 용 전체 예측 누적
            if t is not None and p in [0, 1]:
                all_true_all.append(t)
                all_pred_all.append(p)
            # 2. raw 저장용: 상승 예측만
            if t is not None and p == 0:
                all_code.append(code)
                all_date.append(str(d)[:10])
                all_true.append(t)
                all_pred.append(p)
                all_prob_up.append(round(prob_up, 3)) 

    # 1. raw.csv (상승 예측만)
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    df_new = pd.DataFrame({
        "code": all_code,
        "date": all_date,
        "true_label": all_true,
        "pred_label": all_pred,
        "prob_up": all_prob_up,
    })
    # 만약 기존에 파일이 있다면, 읽어서 병합 후 중복 제거
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, dtype={"code": str, "date": str})
        df_merge = pd.concat([df_old, df_new], ignore_index=True)
        df_merge.drop_duplicates(subset=['code', 'date'], keep='last', inplace=True)
        df_merge.sort_values(by=['date', 'prob_up'], ascending=[False, False], inplace=True)
        df_merge.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df_new.sort_values(by=['date', 'prob_up'], ascending=[False, False], inplace=True)
        df_new.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Confusion matrix raw 데이터 저장 완료: {csv_path}")

    # Confusion Matrix(전체 예측)
    if not all_true_all:
        print("유효한 예측 결과가 없어 confusion matrix를 생성하지 않습니다.")
        return

    label_names = ["상승(0)", "하락(1)"]
    pred_names = ["상승(0)", "하락(1)"]

    # 2x2 행렬
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(all_true_all, all_pred_all):
        cm[t, p] += 1

    # Confusion Matrix 통계 CSV 저장
    cm_stats_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    rows = []
    total = cm.sum()
    for t in range(2):
        for p in range(2):
            count = cm[t, p]
            percent = (count / total * 100) if total > 0 else 0.0
            rows.append({
                "실제 라벨": t,
                "예측 라벨": p,
                "건수": count,
                "비율(%)": round(percent, 2)
            })
    df_cm_stats = pd.DataFrame(rows, columns=["실제 라벨", "예측 라벨", "건수", "비율(%)"])
    df_cm_stats.to_csv(cm_stats_path, index=False, encoding="utf-8-sig")
    print(f"Confusion matrix 칸별 집계 저장 완료: {cm_stats_path}")

    # 아래는 시각화 부분 (생략 없음)
    cm_extended = np.zeros((3, 3), dtype=int)
    cm_extended[:2, :2] = cm
    cm_extended[2, :2] = cm.sum(axis=0)
    cm_extended[:2, 2] = cm.sum(axis=1)
    cm_extended[2, 2] = cm.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.Blues

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
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
                ax.text(j + 0.5, i + 0.5, f"{value}",
                        va='center', ha='center', fontsize=12, fontweight='bold')

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
    stock_models = run_test(preproc_dfs, test_dates, conf_percentile=PERCENT)

    # 6) 결과 저장
    plot_score(stock_models)
    
    print("테스트 및 후처리 완료")

if __name__ == '__main__':
    main()