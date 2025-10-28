# gru_triplebarrier.py
# -----------------------------------------
# 공유 GRU로 진입 후보 선정 + 트리플 배리어(익절/손절/시간)로 청산 백테스트
# -----------------------------------------
import os, argparse, math, warnings, random
import pandas as pd, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings("ignore")

import json
import keyring

# MySQL 연결 설정
DB_CONFIG = {"host": "127.0.0.1", 
             "user": "stockuser", 
             "password": "stockpass123!", 
             "port": 3306, 
             "database": "news_db", 
             "charset": "utf8mb4", }

# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50150860",  # 계좌번호 앞 8지리
    "ACNT_PRDT_CD": "01",  # 계좌번호 뒤 2자리
}

keyring.set_password('mock_app_key', '고민수', 'PSXtsebcvZLq1ZKGsppEYYxCd0RoOd48INlF')
keyring.set_password('mock_app_secret', '고민수', 'pnPjHI+nULtuBz3jTzPhvBQY+9VKfMCql6lN3btyp19EGhi1hALeHrPjhsFj016eaGqACCcDWdZ3ivhNOIVhBZRATrHdiTk8L8uCxVNQn3qpWSk+54SQ/XMCyJvVpUSaPiRBf+n0iSu7blyUjBxQgt9zBMUvBB23ylyMg8yrWCDJZpgQXM4=')

def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '고민수')
    app_secret = keyring.get_password('mock_app_secret', '고민수')
    print("app_key:", app_key)
    print("app_secret:", app_secret)
    return app_key, app_secret


# ✅ NEW: 평가 지표용 추가 임포트
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ✅ NEW: 결과 파일 경로
RESULT_DIR = "results"
METRICS_CSV = os.path.join(RESULT_DIR, "model_metrics.csv")         # 모델/백테스트 지표 누적
PREDICTIONS_CSV = os.path.join(RESULT_DIR, "model_predictions.csv")  # 검증/테스트 예측 덤프 누적
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========
# 하이퍼파라미터 (대회용 공격 프리셋)
# ==========
SEQ_LEN    = 30
HIDDEN     = 32
BATCH_SIZE = 128
LR         = 0.002188011423490845
EPOCHS     = 9
P_THRESHOLD = 0.53
K_MAX       = 4
TP_PCT      = 0.03
SL_PCT      = 0.03
HORIZON     = 6
COMMISSION = 0.003         # 왕복 수수료 (0.3%)
ROLL_NORM = 60             # 롤링 Z-정규화 윈도우
VAL_SPLIT_DAYS = 60        # 최근 60영업일은 테스트/백테스트 구간
SEED = 42                  # 랜덤 시드 (재현성)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")  # 실행 디바이스 (CPU 고정)

# ==========
# CSV 유틸
# ==========
# ✅ NEW: 누적 저장 헬퍼
def _append_csv(path: str, row_dict: dict):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=list(row_dict.keys()))
    # 누락 컬럼 자동 보정
    for c in row_dict.keys():
        if c not in df.columns:
            df[c] = pd.NA
    df = pd.concat([df, pd.DataFrame([row_dict])[df.columns]], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")

# ✅ NEW: 스플릿 평가 헬퍼
def _evaluate_split(y_true: np.ndarray, p: np.ndarray):
    if len(p) == 0:
        return dict(acc=np.nan, pre=np.nan, rec=np.nan, f1=np.nan, auc=np.nan, brier=np.nan,
                    tn=np.nan, fp=np.nan, fn=np.nan, tp=np.nan)
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(y_true, yhat)
    pre = precision_score(y_true, yhat, zero_division=0)
    rec = recall_score(y_true, yhat, zero_division=0)
    f1  = f1_score(y_true, yhat, zero_division=0)
    try:
        auc = roc_auc_score(y_true, p) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    try:
        brier = brier_score_loss(y_true, p)
    except Exception:
        brier = float("nan")
    if len(set(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    else:
        tn = fp = fn = tp = np.nan
    return dict(acc=acc, pre=pre, rec=rec, f1=f1, auc=auc, brier=brier, tn=tn, fp=fp, fn=fn, tp=tp)

# ==========
# DB
# ==========
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

# ==========
# 피처
# ==========
def add_features(df):
    # 기본 수익률
    for n in [1,3,5,20]:
        df[f"ret_{n}"] = df['Close'].pct_change(n)
    # 변동성/ATR
    tr = np.maximum(df['High']-df['Low'], 
                    np.maximum((df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()))
    df['ATR14'] = tr.rolling(14).mean()
    # 추세/괴리
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['gap_sma5'] = (df['Close'] - df['SMA5']) / df['SMA5']
    df['gap_sma20'] = (df['Close'] - df['SMA20']) / df['SMA20']
    # 거래대금/볼륨
    df['Turnover'] = df['Close'] * df['Volume']
    df['vol_std20'] = df['Close'].pct_change().rolling(20).std()
    # 롤링 Z-정규화
    feat_cols = ['ret_1','ret_3','ret_5','ret_20','ATR14','gap_sma5','gap_sma20','Turnover','vol_std20','Volume']
    for c in feat_cols:
        mu = df[c].rolling(ROLL_NORM).mean()
        sd = df[c].rolling(ROLL_NORM).std()
        df[c+'_z'] = (df[c]-mu)/sd
    # 후보 필터용 간단 추세/유동성
    df['liquid'] = df['Turnover'].rolling(20).mean()
    df['uptrend'] = (df['Close'] > df['SMA20']).astype(int)
    # 유효 행만
    z_cols = [c+'_z' for c in feat_cols]
    df = df.dropna(subset=z_cols).reset_index(drop=True)
    return df, z_cols

# ==========
# 트리플 배리어 라벨
# ==========
def triple_barrier_label(df, tp=TP_PCT, sl=SL_PCT, h=HORIZON):
    """
    각 시점 t에 대해: t+1 시가에 진입한다고 가정.
    h일 내 High가 TP 도달하면 1, Low가 SL 도달하면 0,
    둘 다 아니면 h일째 종가 기준으로 +면 1, 아니면 0.
    """
    n = len(df)
    labels = np.full(n, np.nan)
    entry_dates = [pd.NaT]*n
    exit_dates = [pd.NaT]*n
    exit_prices = np.full(n, np.nan)
    reasons = np.array(['']*n, dtype=object)

    for i in range(n- (h+1)):              # i+1이 존재하고 h일 뒤까지 있어야 함
        entry_idx = i+1
        entry_date = df.at[entry_idx, 'Date']
        entry_px = df.at[entry_idx, 'Open']
        tp_px = entry_px*(1+tp)
        sl_px = entry_px*(1-sl)

        # 미래 구간: entry_date ~ entry_date+h일
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
            # 시간배리어: h일째 종가
            r = fut.iloc[-1]
            lab = 1 if r['Close'] >= entry_px else 0
            ex_price, ex_date, reason = r['Close'], r['Date'], 'TIME'

        labels[i] = lab
        entry_dates[i] = entry_date
        exit_dates[i] = ex_date
        exit_prices[i] = ex_price
        reasons[i] = reason

    lab_df = pd.DataFrame({
        'Date': df['Date'],
        'label': labels,
        'entry_date': entry_dates,
        'exit_date': exit_dates,
        'exit_px': exit_prices,
        'reason': reasons
    })
    return lab_df

# ==========
# 시퀀스 데이터셋 생성(공유모델)
# ==========
def build_sequences(df_feat, z_cols, lab_df):
    """
    시퀀스의 마지막 시점 = i, 라벨은 i의 label (진입은 i+1)
    """
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
            'asof_date': df_feat.at[i,'Date'],   # 신호 계산 기준일
            'entry_date': lab_df.at[i,'entry_date'],
            'exit_date': lab_df.at[i,'exit_date'],
        })
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df

# ==========
# GRU 모델(공유)
# ==========
class MiniGRU(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train_gru(Xtr, ytr, Xva, yva, input_size):
    model = MiniGRU(input_size).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    # --- 클래스 가중치 계산 (불균형 보정) ---
    cnt = Counter(ytr.tolist())
    n0 = cnt.get(0, 0)
    n1 = cnt.get(1, 0)

    if n0 == 0 or n1 == 0:
        print("[WARN] One class is missing in training set; using unweighted loss.")
        crit = nn.CrossEntropyLoss()
    else:
        # inverse frequency: 각 클래스 샘플이 전체 손실에 비슷한 기여를 하도록
        w0 = len(ytr) / (2.0 * n0)
        w1 = len(ytr) / (2.0 * n1)
        # 과도한 가중치 방지(클램프, 선택사항)
        w0 = min(max(w0, 0.25), 4.0)
        w1 = min(max(w1, 0.25), 4.0)
        weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=weight)
        print(f"[CLASS-WEIGHT] n0={n0}, n1={n1}, w0={w0:.3f}, w1={w1:.3f}")

    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    va_ds = TensorDataset(torch.tensor(Xva), torch.tensor(yva))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_state, best_loss = None, 1e9
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # validation
        model.eval(); vloss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = crit(out, yb).item()
                vloss += loss * len(yb); n += len(yb)
        vloss /= max(n, 1)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # 검증 지표
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(Xva)), dim=1).numpy()[:, 1]
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else np.nan
        brier = brier_score_loss(yva, p)
    except Exception:
        auc, brier = np.nan, np.nan

    return model, {'val_loss': best_loss, 'val_auc': auc, 'val_brier': brier}


# ==========
# 백테스트 엔진(동시 보유 K, 다음날 시가 진입, 트리플 배리어 청산)
# ==========
def compute_exit(df, entry_date, entry_px, tp=TP_PCT, sl=SL_PCT, h=HORIZON):
    tp_px = entry_px*(1+tp); sl_px = entry_px*(1-sl)
    fut = df[(df['Date']>=entry_date) & (df['Date']<= entry_date+pd.Timedelta(days=h))]
    for _, r in fut.iterrows():
        if r['High'] >= tp_px:   return r['Date'], tp_px, 'TP'
        if r['Low']  <= sl_px:   return r['Date'], sl_px, 'SL'
    r = fut.iloc[-1]
    return r['Date'], r['Close'], 'TIME'

def backtest(pred_df, code2df):
    """
    pred_df: columns=[entry_date, code, p, score, entry_px]
    code2df: dict(code -> price df)
    """
    all_dates = sorted(pred_df['entry_date'].dropna().unique())
    cash = 1.0
    positions = []  # list of dict: code, entry_date, entry_px, exit_date, exit_px, reason, cash_alloc
    equity_curve = []
    trade_log = []

    for d in all_dates:
        # 1) 오늘 종료되는 포지션 먼저 정산(종료일 종가 시점 가정)
        still_open = []
        for pos in positions:
            if pos['exit_date'] <= d:
                # 실현
                ret_gross = (pos['exit_px']/pos['entry_px']) - 1.0
                ret_net = ret_gross - COMMISSION
                realized = pos['cash_alloc']*(1+ret_net)
                cash += realized
                trade_log.append({
                    'code': pos['code'],
                    'entry_date': pos['entry_date'],
                    'exit_date': pos['exit_date'],
                    'entry_px': pos['entry_px'],
                    'exit_px': pos['exit_px'],
                    'ret_net': ret_net,
                    'reason': pos['reason']
                })
            else:
                still_open.append(pos)
        positions = still_open

        # 2) 오늘 진입 후보 선정(빈 슬롯 채우기)
        slots = K_MAX - len(positions)
        if slots > 0:
            today_full = pred_df[pred_df['entry_date'] == d].copy()

            # 컬럼 중복 제거(예외 방지)
            if today_full.columns.duplicated().any():
                today_full = today_full.loc[:, ~today_full.columns.duplicated()]

            # 동일 code 중복 행 제거 (점수 높은 행만 유지)
            if 'code' in today_full.columns:
                today_full = today_full.sort_values('score', ascending=False)
                today_full = today_full.drop_duplicates(subset=['code'], keep='first')
            
            # --- NEW: 유동성(20일 평균 거래대금 상위 50%) & 추세(SMA20 위) 필터 ---
            def _turn20(row):
                code = row['code']
                dfp = code2df[code]
                win = dfp[dfp['Date'] <= d].tail(20)
                if len(win) < 20: return np.nan
                return float((win['Close'] * win['Volume']).mean())
            today_full['turn20'] = today_full.apply(_turn20, axis=1)
            # 유동성 하위 절반 컷
            thr = today_full['turn20'].quantile(0.5)
            today_full = today_full[today_full['turn20'] >= thr]

            def _uptrend(row):
                code = row['code']
                dfp = code2df[code]
                win = dfp[dfp['Date'] <= d].tail(20)
                if len(win) < 20: return False
                sma20 = win['Close'].mean()
                return float(win['Close'].iloc[-1]) > sma20
            today_full = today_full[today_full.apply(_uptrend, axis=1)]

            # 1차 후보: score>0
            cands = today_full[today_full['score'] > 0].copy()

            # 이미 보유중인 종목 제외
            held = set([p['code'] for p in positions])
            if 'code' in cands.columns:
                cands = cands[~cands['code'].astype(str).isin(held)]

            # p 임계값 적용 + 폴백
            picks = pd.DataFrame(columns=today_full.columns)
            if not cands.empty:
                main = cands[cands['p'] >= P_THRESHOLD].head(slots)
                remain = slots - len(main)
                if remain > 0:
                    # 폴백: p 임계 미달이지만 score>0 상위로 채움
                    extra = cands.loc[~cands.index.isin(main.index)].head(remain)
                    picks = pd.concat([main, extra])
                else:
                    picks = main

            # (선택) 그래도 비었으면 보수적으로 p>=0.50 상위로 채움
            if len(picks) == 0 and not today_full.empty:
                safe = today_full[today_full['p'] >= 0.50]
                if safe.empty:
                    safe = today_full
                picks = safe.sort_values('score', ascending=False).head(slots)
                picks = picks[~picks['code'].astype(str).isin(held)]

            # 최종 컬럼 중복 방어
            if len(picks) > 0 and picks.columns.duplicated().any():
                picks = picks.loc[:, ~picks.columns.duplicated()]

            # 체결
            if len(picks) > 0 and cash > 0:
                alloc_each = cash / len(picks)
                for _, r in picks.iterrows():
                    code = str(r['code'])
                    entry_px = float(r['entry_px'])
                    df = code2df[code]
                    exit_date, exit_px, reason = compute_exit(df, r['entry_date'], entry_px)
                    positions.append({
                        'code': code,
                        'entry_date': r['entry_date'],
                        'entry_px': entry_px,
                        'exit_date': exit_date,
                        'exit_px': exit_px,
                        'reason': reason,
                        'cash_alloc': alloc_each
                    })
                    cash -= alloc_each

        # 3) 마감 후 에쿼티(보유 포지션은 종가 기준 평가)
        nav = cash
        for pos in positions:
            df = code2df[pos['code']]
            # 현재 날짜 d의 종가로 평가(없으면 직전가)
            row = df[df['Date']==d]
            if len(row)==0:
                # 이전 영업일 종가로 평가
                row = df[df['Date']<d].iloc[-1:]
            cur_close = float(row['Close'].values[0])
            nav += pos['cash_alloc'] * (cur_close/pos['entry_px'])
        equity_curve.append({'date': d, 'equity': nav})

    eq_df = pd.DataFrame(equity_curve)
    tr_df = pd.DataFrame(trade_log)
    return eq_df, tr_df

# ==========
# 메인 파이프라인
# ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit_codes', type=int, default=50, help='유니버스 상한(거래대금/유동성 필터는 아래에서)')
    ap.add_argument('--min_turnover', type=float, default=None, help='최소 일평균 거래대금(원). None이면 자동 상위 50선정에 맡김')
    ap.add_argument('--model_name', type=str, default="GRU Trader", help='지표 저장 시 표시할 모델명')  # ✅ NEW
    args = ap.parse_args()

    eng = get_engine()
    codes = load_codes(eng, limit=args.limit_codes)

    # 1) 각 종목 DF + 피처/라벨 생성
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
        if len(y)==0:
            continue
        all_X.append(X); all_y.append(y); meta['Code']=code; all_meta.append(meta)

    if len(all_y)==0:
        print("유효 데이터가 없습니다.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)

    # 2) 학습/검증 분리(최근 VAL_SPLIT_DAYS는 테스트/백테스트)
    cutoff = meta['entry_date'].max() - pd.Timedelta(days=VAL_SPLIT_DAYS)
    tr_idx = meta['entry_date'] <= cutoff
    te_idx = meta['entry_date'] > cutoff

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    met_te = meta[te_idx].reset_index(drop=True)

    # 검증을 위해 학습 내부에서 일부를 val로 사용(시간 순서 유지)
    # 여기선 간단히 앞 85%/뒤 15%
    ntr = len(Xtr)
    split = int(ntr*0.85)
    Xtrain, ytrain = Xtr[:split], ytr[:split]
    Xval,   yval   = Xtr[split:], ytr[split:]
    met_tr = meta[tr_idx].reset_index(drop=True)   # ✅ NEW: train/val 메타
    mval = met_tr.iloc[split:].reset_index(drop=True) if len(met_tr) > split else pd.DataFrame()

    # 3) 학습
    model, vstats = train_gru(Xtrain, ytrain, Xval, yval, input_size=X.shape[2])
    print(f"[VAL] loss={vstats['val_loss']:.4f}, AUC={vstats['val_auc']:.3f}, Brier={vstats['val_brier']:.4f}")

    # ✅ NEW: 검증 예측/지표
    with torch.no_grad():
        p_val = torch.softmax(model(torch.tensor(Xval)), dim=1).numpy()[:,1] if len(Xval)>0 else np.array([])
    val_eval = _evaluate_split(yval, p_val)

    # 4) 테스트 구간 예측 확률
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(Xte)), dim=1).numpy()[:,1]
    met_te['p'] = p

    # --- 분포/충족률 찍기 ---
    q = np.quantile(p, [0.1, 0.25, 0.5, 0.75, 0.9])
    hit = float((p >= P_THRESHOLD).mean())
    print(f"[P-DIST] q10={q[0]:.3f}, q25={q[1]:.3f}, q50={q[2]:.3f}, q75={q[3]:.3f}, q90={q[4]:.3f}")
    print(f"[P-HIT ] share(p >= {P_THRESHOLD:.2f}) = {hit*100:.1f}%")

    # 5) 점수(score) 계산 + 엔트리 가격(entry_px) 계산 + pred_df 구성
    met_te = met_te.copy()

    # 점수 계산 (기대값 근사: p*TP - (1-p)*SL - 비용)
    met_te['score'] = met_te['p'] * TP_PCT - (1 - met_te['p']) * SL_PCT - COMMISSION

    # 엔트리 가격: entry_date의 시가, 없으면 '그 이후 첫 거래일' 시가
    def _entry_px(df_map, code_col, entry_date):
        dfp = df_map[code_col]
        r = dfp[dfp['Date'] >= entry_date].head(1)
        return float(r['Open'].iloc[0]) if len(r) else np.nan

    met_te['entry_px'] = met_te.apply(lambda r: _entry_px(code2df, r['Code'], r['entry_date']), axis=1)
    met_te = met_te.dropna(subset=['entry_px'])

    # pred_df 최종 구성 (여기서 score가 반드시 포함되게!)
    pred_df = met_te[['entry_date', 'Code', 'p', 'score', 'entry_px']].rename(columns={'Code':'code'}).reset_index(drop=True)

    # 방어적 체크
    required = {'entry_date','code','p','score','entry_px'}
    missing = required - set(pred_df.columns)
    if missing:
        print("[ERROR] pred_df missing columns:", missing)
        print("pred_df columns:", list(pred_df.columns))
        return

    # 6) 백테스트(동시 보유 K_MAX, 트리플 배리어 청산)
    eq, tr = backtest(pred_df[['entry_date','code','p','score','entry_px']], code2df)

    # 7) 결과 저장/출력 (기존 파일 유지)
    eq.to_csv(os.path.join(RESULT_DIR, 'equity_curve.csv'), index=False)
    tr.to_csv(os.path.join(RESULT_DIR, 'trades.csv'), index=False)
    met_te[['asof_date','entry_date','Code','p','score','entry_px']].rename(columns={'Code':'code'}) \
         .to_csv(os.path.join(RESULT_DIR, 'preds.csv'), index=False)

    # ✅ NEW: 테스트 지표 계산/저장
    te_eval = _evaluate_split(yte, met_te['p'].values[:len(yte)])
    final_eq = float(eq['equity'].iloc[-1])
    ret_total = final_eq - 1.0
    roll_max = eq['equity'].cummax()
    dd = (eq['equity'] / roll_max - 1.0)
    mdd = float(dd.min())

    # 콘솔 출력 유지
    print(f"[RESULT] Final Equity: {final_eq:.3f}  (Total Return {ret_total*100:.2f}%)  MDD {mdd*100:.2f}%")
    try:
        auc = roc_auc_score(yte, met_te['p'][:len(yte)])
        print(f"[TEST] AUC={auc:.3f}")
    except Exception:
        pass

    # ✅ NEW: model_metrics.csv에 한 줄 누적
    now_ts = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    metrics_row = {
        "ts": now_ts,
        "model_name": args.model_name,
        "seq_len": SEQ_LEN, "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR,
        # 검증
        "val_loss": vstats.get("val_loss"), "val_auc": vstats.get("val_auc"), "val_brier": vstats.get("val_brier"),
        "val_acc": val_eval["acc"], "val_pre": val_eval["pre"], "val_rec": val_eval["rec"], "val_f1": val_eval["f1"],
        "val_tn": val_eval["tn"], "val_fp": val_eval["fp"], "val_fn": val_eval["fn"], "val_tp": val_eval["tp"],
        # 테스트
        "te_acc": te_eval["acc"], "te_pre": te_eval["pre"], "te_rec": te_eval["rec"], "te_f1": te_eval["f1"],
        "te_auc": te_eval["auc"], "te_brier": te_eval["brier"],
        "te_tn": te_eval["tn"], "te_fp": te_eval["fp"], "te_fn": te_eval["fn"], "te_tp": te_eval["tp"],
        "n_train": int(len(Xtrain)), "n_val": int(len(Xval)), "n_test": int(len(Xte)),
        # 백테스트 결과
        "bt_final_equity": final_eq, "bt_total_return": ret_total, "bt_mdd": mdd,
        "k_max": K_MAX, "tp_pct": TP_PCT, "sl_pct": SL_PCT, "horizon": HORIZON, "commission": COMMISSION,
        "p_threshold": P_THRESHOLD
    }
    _append_csv(METRICS_CSV, metrics_row)

    # ✅ NEW: model_predictions.csv에 검증/테스트 예측 전부 누적 저장
    # 검증 덤프
    if len(p_val) > 0 and not mval.empty:
        mval_dump = mval.copy()
        mval_dump["split"] = "val"
        mval_dump["p"] = p_val
        mval_dump["y_true"] = yval
        mval_dump["score"] = mval_dump["p"] - 0.5
        # entry_px (val 구간)
        def _entry_px_val(row):
            dfp = code2df[row["code"] if "code" in row else row["Code"]]
            r = dfp[dfp["Date"] >= row["entry_date"]].head(1)
            return float(r["Open"].iloc[0]) if len(r) else np.nan
        if "code" not in mval_dump.columns and "Code" in mval_dump.columns:
            mval_dump = mval_dump.rename(columns={"Code":"code"})
        mval_dump["entry_px"] = mval_dump.apply(_entry_px_val, axis=1)
        cols = ["split","asof_date","entry_date","code","p","score","y_true","entry_px"]
        for _, r in mval_dump[cols].iterrows():
            _append_csv(PREDICTIONS_CSV, dict(ts=now_ts, **r.to_dict()))

    # 테스트 덤프
    mte_dump = met_te.copy()
    mte_dump["split"] = "test"
    mte_dump["y_true"] = yte[:len(mte_dump)]
    cols = ["split","asof_date","entry_date","Code","p","score","y_true","entry_px"]
    mte_dump = mte_dump.rename(columns={"Code":"code"})
    cols = ["split","asof_date","entry_date","code","p","score","y_true","entry_px"]
    for _, r in mte_dump[cols].iterrows():
        _append_csv(PREDICTIONS_CSV, dict(ts=now_ts, **r.to_dict()))

    # 그래프
    plt.figure(figsize=(8,4))
    plt.plot(pd.to_datetime(eq['date']), eq['equity'])
    plt.title("Equity Curve (K=%d, TP=%.1f%%, SL=%.1f%%, H=%d)" % (K_MAX, TP_PCT*100, SL_PCT*100, HORIZON))
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR, "equity_curve.png")); plt.close()

if __name__ == "__main__":
    main()
