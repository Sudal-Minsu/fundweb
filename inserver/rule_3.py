# gru_triplebarrier_reco.py
# -----------------------------------------
# 공유 GRU로 오늘 추천 종목만 산출/프린트 (백테스트 제거)
# -----------------------------------------
import os, argparse, warnings, random
import pandas as pd, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sqlalchemy import create_engine
from collections import Counter
from pandas.tseries.offsets import BDay
warnings.filterwarnings("ignore")

import json


# MySQL 연결 설정
DB_CONFIG = {    
    "host": "127.0.0.1",
    "user": "visitor",
    "password": "1234567890",
    "port": 3306,
    "database": "news_db",
    "charset": "utf8mb4"
}


# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50156249",  # 계좌번호 앞 8지리
    "ACNT_PRDT_CD": "01",  # 계좌번호 뒤 2자리
}



def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = 'PSYoLcVh9J1P6c7kOGfQlasCL66dG2iy3kky'
    app_secret = '/6Ur3JrcY1rjljdcXAtmukAqnXlhu+hPgbkIwfllhTzJ0/FPrJ3Lih8/v5hotpbJNzXbYHr3jFvnGy1mq9QgaTD+5Hf5MCUKAY9Z/8gVFjfk2uc3o/d+ixKU8uE35Tu2B6JR3FuwYh4U42+MkrbvVWKMJ7+DYpfgGA9l5wESVspGi9h3Vyo='
    print("app_key:", app_key)
    print("app_secret:", app_secret)
    return app_key, app_secret

# ===== 하이퍼파라미터 =====
SEQ_LEN    = 30
HIDDEN     = 32
BATCH_SIZE = 128
LR         = 0.002188011423490845
EPOCHS     = 9
P_THRESHOLD = 0.53
K_MAX       = 4
TP_PCT      = 0.04
SL_PCT      = 0.02
HORIZON     = 6
COMMISSION  = 0.003
ROLL_NORM   = 60
VAL_SPLIT_DAYS = 60

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")




# ===== DB =====
def get_engine():
    uri = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
          f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(uri)

def load_codes(eng, limit=None):
    q = "SELECT DISTINCT Code FROM stock_data ORDER BY Code"
    if limit: q += f" LIMIT {int(limit)}"
    return pd.read_sql(q, eng)['Code'].tolist()

def load_df(eng, code):
    q = f"SELECT Date, Open, High, Low, Close, Volume FROM stock_data WHERE Code='{code}' ORDER BY Date"
    df = pd.read_sql(q, eng, parse_dates=['Date'])
    df['Code'] = code
    return df.dropna().sort_values('Date').reset_index(drop=True)

# ===== 피처 =====
def add_features(df):
    for n in [1,3,5,20]:
        df[f"ret_{n}"] = df['Close'].pct_change(n)
    tr = np.maximum(df['High']-df['Low'],
                    np.maximum((df['High']-df['Close'].shift()).abs(),
                               (df['Low'] -df['Close'].shift()).abs()))
    df['ATR14'] = tr.rolling(14).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['gap_sma5'] = (df['Close'] - df['SMA5']) / df['SMA5']
    df['gap_sma20'] = (df['Close'] - df['SMA20']) / df['SMA20']
    df['Turnover'] = df['Close'] * df['Volume']
    df['vol_std20'] = df['Close'].pct_change().rolling(20).std()

    feat_cols = ['ret_1','ret_3','ret_5','ret_20','ATR14','gap_sma5','gap_sma20','Turnover','vol_std20','Volume']
    for c in feat_cols:
        mu = df[c].rolling(ROLL_NORM).mean()
        sd = df[c].rolling(ROLL_NORM).std()
        df[c+'_z'] = (df[c]-mu)/sd
    df['liquid'] = df['Turnover'].rolling(20).mean()
    df['uptrend'] = (df['Close'] > df['SMA20']).astype(int)

    z_cols = [c+'_z' for c in feat_cols]
    df = df.dropna(subset=z_cols).reset_index(drop=True)
    return df, z_cols

# ===== 트리플 배리어 라벨 =====
def triple_barrier_label(df, tp=TP_PCT, sl=SL_PCT, h=HORIZON):
    n = len(df)
    labels = np.full(n, np.nan)
    entry_dates = [pd.NaT]*n
    exit_dates = [pd.NaT]*n
    exit_prices = np.full(n, np.nan)
    reasons = np.array(['']*n, dtype=object)

    for i in range(n-(h+1)):
        entry_idx = i+1
        entry_date = df.at[entry_idx, 'Date']
        entry_px   = df.at[entry_idx, 'Open']
        tp_px = entry_px*(1+tp)
        sl_px = entry_px*(1-sl)

        fut = df[(df['Date']>=entry_date) & (df['Date']<= entry_date+pd.Timedelta(days=h))]
        lab, ex_price, ex_date, reason = None, None, None, None
        for _, r in fut.iterrows():
            if r['High'] >= tp_px:
                lab, ex_price, ex_date, reason = 1, tp_px, r['Date'], 'TP'
                break
            if r['Low'] <= sl_px:
                lab, ex_price, ex_date, reason = 0, sl_px, r['Date'], 'SL'
                break
        if lab is None:
            r = fut.iloc[-1]
            lab = 1 if r['Close'] >= entry_px else 0
            ex_price, ex_date, reason = r['Close'], r['Date'], 'TIME'

        labels[i] = lab
        entry_dates[i] = entry_date
        exit_dates[i] = ex_date
        exit_prices[i] = ex_price
        reasons[i] = reason

    return pd.DataFrame({
        'Date': df['Date'],
        'label': labels,
        'entry_date': entry_dates,
        'exit_date': exit_dates,
        'exit_px': exit_prices,
        'reason': reasons
    })

# ===== 시퀀스 =====
def build_sequences(df_feat, z_cols, lab_df):
    X_list, y_list, meta = [], [], []
    for i in range(SEQ_LEN-1, len(df_feat)):
        y = lab_df.at[i, 'label']
        if np.isnan(y): 
            continue
        seq = df_feat[z_cols].iloc[i-SEQ_LEN+1:i+1].values.astype(np.float32)
        X_list.append(seq)
        y_list.append(int(y))
        meta.append({
            'code': df_feat.at[i,'Code'],
            'asof_date': df_feat.at[i,'Date'],
            'entry_date': lab_df.at[i,'entry_date'],
            'exit_date': lab_df.at[i,'exit_date'],
        })
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df

def build_sequences_infer(df_feat, z_cols):
    """
    라벨 없이도 최신 시퀀스 1개를 만들어 내일(entry_date=다음 영업일) 예측에 사용.
    - asof_date: df_feat 마지막 Date
    - entry_date: asof_date의 다음 영업일
    """
    if len(df_feat) < SEQ_LEN:
        return None, None
    seq = df_feat[z_cols].iloc[-SEQ_LEN:].values.astype(np.float32)
    X = np.expand_dims(seq, axis=0)  # (1, SEQ_LEN, n_feat)
    meta = pd.DataFrame([{
        'asof_date': df_feat['Date'].iloc[-1],
        'entry_date': (df_feat['Date'].iloc[-1].normalize() + BDay(1)).normalize()
    }])
    return X, meta


# ===== GRU =====
class MiniGRU(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train_gru(Xtr, ytr, Xva, yva, input_size):
    model = MiniGRU(input_size).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    cnt = Counter(ytr.tolist())
    n0, n1 = cnt.get(0,0), cnt.get(1,0)
    if n0 == 0 or n1 == 0:
        crit = nn.CrossEntropyLoss()
    else:
        w0 = min(max(len(ytr)/(2.0*n0), 0.25), 4.0)
        w1 = min(max(len(ytr)/(2.0*n1), 0.25), 4.0)
        weight = torch.tensor([w0,w1], dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=weight)

    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    va_ds = TensorDataset(torch.tensor(Xva), torch.tensor(yva))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_state, best_loss = None, 1e9
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # val
        model.eval(); vloss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss += crit(model(xb), yb).item() * len(yb); n += len(yb)
        vloss /= max(n,1)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss

# ===== 오늘 추천만 선택 =====
def select_today_recos(pred_df, code2df, k=K_MAX, p_thr=P_THRESHOLD):
    if pred_df.empty:
        return pred_df.iloc[0:0]

    df = pred_df.copy()
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    target_entry_date = df['entry_date'].max().normalize()
    df = df[df['entry_date'].dt.normalize() == target_entry_date]
    if df.empty:
        return df

    # 동일 종목 중복 제거
    df = df.sort_values('score', ascending=False).drop_duplicates(subset=['code'], keep='first')

    # (옵션) 유동성/추세 필터 유지
    def _turn20(row):
        c = row['code']; base = row['entry_date']
        dp = code2df.get(c)
        if dp is None: return np.nan
        win = dp[dp['Date'] <= base].tail(20)
        if len(win) < 20: return np.nan
        return float((win['Close'] * win['Volume']).mean())

    def _uptrend(row):
        c = row['code']; base = row['entry_date']
        dp = code2df.get(c)
        if dp is None: return False
        win = dp[dp['Date'] <= base].tail(20)
        if len(win) < 20: return False
        return float(win['Close'].iloc[-1]) > float(win['Close'].mean())

    df['turn20'] = df.apply(_turn20, axis=1)
    df = df.dropna(subset=['turn20'])
    if df.empty:
        return df
    thr = df['turn20'].quantile(0.5)
    df = df[df['turn20'] >= thr]
    df = df[df.apply(_uptrend, axis=1)]

    # p 임계 + score>0 → 없으면 완화
    cands = df[(df['p'] >= p_thr) & (df['score'] > 0)]
    if cands.empty:
        cands = df

    picks = cands.sort_values(['p','score'], ascending=False).head(k).reset_index(drop=True)
    return picks



# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit_codes', type=int, default=50)
    args = ap.parse_args()

    eng = get_engine()
    codes = load_codes(eng, limit=args.limit_codes)

    all_X, all_y, all_meta = [], [], []
    code2df = {}
    for code in tqdm(codes, desc="피처/라벨"):
        df = load_df(eng, code)
        if len(df) < 200: 
            continue
        code2df[code] = df.copy()
        df_feat, z_cols = add_features(df.copy())
        lab_df = triple_barrier_label(df_feat, TP_PCT, SL_PCT, HORIZON)
        X, y, meta = build_sequences(df_feat, z_cols, lab_df)
        if len(y) == 0:
            continue
        all_X.append(X); all_y.append(y); meta['Code']=code; all_meta.append(meta)

    if len(all_y) == 0:
        print("유효 데이터가 없습니다.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)

    # 시간 분리 (최근 VAL_SPLIT_DAYS → 추천 계산 구간)
    cutoff = meta['entry_date'].max() - pd.Timedelta(days=VAL_SPLIT_DAYS)
    tr_idx = meta['entry_date'] <= cutoff
    te_idx = meta['entry_date'] > cutoff

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte      = X[te_idx]
    met_te   = meta[te_idx].reset_index(drop=True)

    # 내부 val 분리
    ntr = len(Xtr)
    split = int(ntr*0.85) if ntr > 10 else max(1, ntr-1)
    Xtrain, ytrain = Xtr[:split], ytr[:split]
    Xval,   yval   = (Xtr[split:], ytr[split:]) if ntr > 1 else (Xtr[:1], ytr[:1])

    # ===== 학습 완료 =====                                                                                                                                                                                                                                                                                                                                                     
    model, _ = train_gru(Xtrain, ytrain, Xval, yval, input_size=X.shape[2])

    # ===== 운영용 최신(내일 진입) inference: 전 종목에서 1개 시퀀스씩 =====
    infer_X_list, infer_meta_rows = [], []
    for code, df_full in code2df.items():
        # 피처 재계산 (주의: df_full은 원시 데이터)
        df_feat_i, z_cols_i = add_features(df_full.copy())
        Xi, metai = build_sequences_infer(df_feat_i, z_cols_i)
        if Xi is None:
            continue
        infer_X_list.append(Xi)
        r = metai.iloc[0].to_dict()
        r['code'] = code
        infer_meta_rows.append(r)

    if len(infer_X_list) == 0:
        print("[TODAY] 최신 구간 inference 대상이 없습니다. (데이터 길이 부족?)")
        return

    X_infer = np.concatenate(infer_X_list, axis=0)
    meta_infer = pd.DataFrame(infer_meta_rows)
    meta_infer['asof_date'] = pd.to_datetime(meta_infer['asof_date'])
    meta_infer['entry_date'] = pd.to_datetime(meta_infer['entry_date'])

    with torch.no_grad():
        p_infer = torch.softmax(model(torch.tensor(X_infer)), dim=1).numpy()[:,1]
    meta_infer['p'] = p_infer

    # 점수 (entry_px는 내일 시가라 아직 DB에 없을 수 있으므로 여기선 계산/필터에 사용하지 않음)
    meta_infer['score'] = meta_infer['p'] * TP_PCT - (1 - meta_infer['p']) * SL_PCT - COMMISSION
    meta_infer['entry_px'] = np.nan  # buy 봇이 아침 '현재가'로 대체해 사용

    # 오늘 추천용 pred_df는 inference 결과를 사용 (최근 구간 전체가 같은 entry_date임)
    pred_df = meta_infer[['asof_date','entry_date','code','p','score','entry_px']].copy()


    # 오늘 추천 선택
    picks = select_today_recos(pred_df, code2df, k=K_MAX, p_thr=P_THRESHOLD)

    # 저장 & 프린트
    os.makedirs('results_3', exist_ok=True)
    pred_df.to_csv('results_3/preds.csv', index=False)
    picks.to_csv('results_3/today_recos.csv', index=False)

    if picks.empty:
        print("[TODAY] 오늘자 추천이 없습니다. (데이터 최신 여부 확인)")
    else:
        print("\n[TODAY] 추천 종목 (상위 K)")
        # 보기 좋은 출력
        view = picks[['code','p','score','entry_date','entry_px']].copy()
        # 소수 자리 정리
        view['p'] = (view['p']*100).round(2)  # %
        view['score'] = view['score'].round(4)
        view['entry_px'] = view['entry_px'].round(2)
        # 프린트
        print(view.to_string(index=False))
        print("\n[SAVE] results_3/today_recos.csv")

    print(f"[DEBUG] preds entry_date range: {pred_df['entry_date'].min().date()} ~ {pred_df['entry_date'].max().date()}")
    print(f"[DEBUG] DB last date (참고): {max(dp['Date'].max() for dp in code2df.values() if len(dp)).date()}")

if __name__ == "__main__":
    main()
