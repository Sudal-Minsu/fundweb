# -*- coding: utf-8 -*-
"""
full_kelly_backtest_line_only.py
- Bar(Compare) 모드 완전 제거
- 기간 연속 백테스트(라인 차트)만 수행
- 실제 거래일만 사용, 휴장일 자동 스킵
- CSV/PNG 저장: data/results/kelly_backtest_{START}_{END}.{csv,png}
"""

import os
import sys
import random
from io import StringIO
from urllib.parse import urljoin
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
START_DATE = "2025-10-24"  # inclusive
END_DATE   = "2025-10-28"  # inclusive

INIT_EQUITY           = 100_000_000
EQUITY_UTILIZATION    = 1.0
PER_STOCK_CAP_PCT     = 0.34
MIN_ORDER_VALUE       = 1_000_000

STOP_LOSS_PCT         = 0.025
TAKE_PROFIT_PCT       = 0.05
R                     = TAKE_PROFIT_PCT / STOP_LOSS_PCT  # 2.0
TRIGGER_PRIORITY      = 'SL_FIRST'   # 'TP_FIRST' | 'SL_FIRST' | 'NEUTRAL'

USE_RANDOM_UNIVERSE   = False
RANDOM_MIN_CODES      = 3
RANDOM_MAX_CODES      = 5
RANDOM_SEED           = 42
RANDOM_UNIVERSE_POOL  = [
    "005930","000660","035420","051910","207940",
    "068270","005380","035720","000270","005490",
    "028260","012330","105560","055550","006400",
]

FETCH_START_BUFFER = "2025-08-15"
FETCH_END_BUFFER   = "2025-10-31"

DEBUG_SPAN         = True
DEBUG_ALLOCATION   = False

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(BASE_DIR, "data", "ohlcv"),
    os.path.join(os.path.dirname(BASE_DIR), "data", "ohlcv"),
    os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "fundweb", "data", "ohlcv"),
]
DATA_DIR = None
for d in CANDIDATES:
    if "fundweb" in d.replace("\\", "/") and d.endswith(os.path.join("data", "ohlcv")):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = CANDIDATES[1]

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")
BUYLIST_PATH = os.path.join(OUTPUT_DIR, "buy_list.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("📂 DATA_DIR =", DATA_DIR)
print("📂 OUTPUT_DIR =", OUTPUT_DIR)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _to_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def daterange(start: str, end: str):
    d0, d1 = _to_date(start), _to_date(end)
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)

def adjust_price_to_tick(price: int) -> int:
    if price < 1000: tick = 1
    elif price < 5000: tick = 5
    elif price < 10000: tick = 10
    elif price < 50000: tick = 50
    elif price < 100000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
    return int(price - (price % tick))

REQUIRED_COLS = ["date","open","high","low","close"]

def load_daily_csv(code: str) -> pd.DataFrame | None:
    p = Path(DATA_DIR) / f"{code}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if not set(REQUIRED_COLS).issubset(df.columns):
        raise ValueError(f"CSV columns missing for {code}: need {REQUIRED_COLS}")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df

# ─────────────────────────────────────────────
# NAVER FETCHER
# ─────────────────────────────────────────────
def fetch_ohlcv_naver(code: str, start: str, end: str, max_pages: int = 160) -> str:
    base = "https://finance.naver.com"
    url  = f"/item/sise_day.nhn?code={code}"
    frames = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for page in range(1, max_pages + 1):
        u = urljoin(base, url + f"&page={page}")
        resp = session.get(u, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            break
        df = tables[0].dropna().copy()
        if df.empty:
            break
        df.columns = ["date","close","diff","open","high","low","volume"]
        df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d", errors="coerce")
        df = df[df["date"].notna()].copy()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        frames.append(df[["date","open","high","low","close","volume"]])
        if frames[-1]["date"].min() <= start:
            break

    if not frames:
        raise RuntimeError(f"No data scraped for {code}")

    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    mask = (full["date"] >= start) & (full["date"] <= end)
    out = full.loc[mask].copy()
    if out.empty:
        out = full.copy()

    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    csv_path = os.path.join(DATA_DIR, f"{code}.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path

def ensure_ohlcv_for(codes, start: str, end: str):
    """Ensure CSV exists *and* covers [start..end]. If not, refetch and overwrite."""
    ok = []
    for c in codes:
        p = os.path.join(DATA_DIR, f"{c}.csv")
        needs_fetch = False
        if not os.path.exists(p):
            needs_fetch = True
        else:
            try:
                df = pd.read_csv(p)
                if not set(REQUIRED_COLS).issubset(df.columns):
                    needs_fetch = True
                else:
                    dmin = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
                    dmax = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                    if dmax < end or dmin > start:
                        needs_fetch = True
            except Exception:
                needs_fetch = True
        try:
            if needs_fetch:
                print(f"⬇️ Fetching {c} (ensure coverage {start}~{end}) ...")
                saved = fetch_ohlcv_naver(c, FETCH_START_BUFFER, FETCH_END_BUFFER)
                print("   ->", saved)
            else:
                print(f"✅ Exists {p}")
            ok.append(c)
        except Exception as e:
            print(f"⚠️ Fetch failed for {c}:", e)
    return ok

# ─────────────────────────────────────────────
# Kelly helpers
# ─────────────────────────────────────────────
def compute_p_from_history(df: pd.DataFrame, ref_date: str, lookback: int = 15,
                           default_hit: float = 0.55) -> float:
    hist = df[df['date'] < ref_date].tail(max(lookback+1, 20))
    if len(hist) < lookback+1:
        return default_hit
    closes = hist['close'].astype(float).to_numpy()

    def last_n_return(arr, n):
        if len(arr) < n+1 or arr[-(n+1)] <= 0: return 0.0
        return float(arr[-1]/arr[-(n+1)] - 1.0)

    ret5  = last_n_return(closes, 5)
    ret15 = last_n_return(closes, 15)
    z = 0.6*ret5 + 0.4*ret15
    p_raw = 1.0/(1.0+np.exp(-6.0*z))
    p = float(np.clip(0.7*p_raw + 0.3*default_hit, 0.50, 0.80))
    return p

def kelly_fraction(p: float, R: float) -> float:
    q = 1.0 - p
    return p - (q / R)

# ─────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────
def pick_universe() -> list[str]:
    if USE_RANDOM_UNIVERSE:
        pool = RANDOM_UNIVERSE_POOL or []
        if not pool:
            pool = ["005930","000660","035720"]
        rng = random.Random(RANDOM_SEED)
        n = min(rng.randint(RANDOM_MIN_CODES, RANDOM_MAX_CODES), len(pool))
        return sorted(rng.sample(pool, n))
    if not os.path.exists(BUYLIST_PATH):
        return ["000660","005930","012330","051910","207940"]
    df = pd.read_csv(BUYLIST_PATH, dtype={'종목코드': str, 'code': str})
    codes = []
    for _, row in df.iterrows():
        code = (row.get('종목코드') or row.get('code') or '').zfill(6)
        if code:
            codes.append(code)
    codes = sorted(list(dict.fromkeys(codes)))
    if not codes:
        return ["000660","005930","012330","051910","207940"]
    return codes

# ─────────────────────────────────────────────
# Day simulation (open→TP/SL→close)
# ─────────────────────────────────────────────
def simulate_day_open_tp_sl(open_px: float, high: float, low: float,
                            sl_pct: float, tp_pct: float,
                            priority: str = 'SL_FIRST', close_px: float | None = None) -> float:
    tp_price = adjust_price_to_tick(int(open_px * (1+tp_pct)))
    sl_price = adjust_price_to_tick(int(open_px * (1-sl_pct)))

    hit_tp = high >= tp_price
    hit_sl = low  <= sl_price

    if hit_tp and hit_sl:
        if priority.upper() == 'SL_FIRST':
            return -sl_pct
        elif priority.upper() == 'TP_FIRST':
            return tp_pct
        else:
            return 0.5*(tp_pct - sl_pct)
    if hit_tp:
        return tp_pct
    if hit_sl:
        return -sl_pct
    if close_px is None:
        return 0.0
    return (close_px / open_px) - 1.0

# ─────────────────────────────────────────────
# RANGE backtest (only)
# ─────────────────────────────────────────────
def run_backtest_range(start: str, end: str, codes: list[str]) -> pd.DataFrame:
    data = {}
    for c in codes:
        df = load_daily_csv(c)
        if df is None:
            print(f"⚠️ CSV not found for {c}. Fetching...")
            fetch_ohlcv_naver(c, FETCH_START_BUFFER, FETCH_END_BUFFER)
            df = load_daily_csv(c)
        if df is not None:
            data[c] = df
    codes = [c for c in codes if c in data]
    if not codes:
        raise SystemExit("No data available for selected codes.")

    if DEBUG_SPAN:
        for c, df in data.items():
            print(f"  {c} span: {df['date'].min()} ~ {df['date'].max()}")

    equity = INIT_EQUITY
    curve = []
    # pre-start anchor for plotting
    curve.append({'date': (pd.to_datetime(start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                  'equity': int(equity)})

    # build union of trading days in window
    date_union = set()
    for c in codes:
        s = data[c]
        m = (s['date'] >= start) & (s['date'] <= end)
        date_union |= set(s.loc[m, 'date'].tolist())
    trading_days = sorted(date_union)

    for d in trading_days:
        # rows for the day
        day_rows = {}
        for c in codes:
            row = data[c][data[c]['date'] == d]
            if row.empty:
                continue
            day_rows[c] = row.iloc[0]
        if not day_rows:
            continue

        # Kelly selection
        klist = []
        for c in list(day_rows.keys()):
            p = compute_p_from_history(data[c], d)
            f = kelly_fraction(p, R)
            if f <= 0:
                del day_rows[c]
                continue
            klist.append({'code': c, 'p': p, 'f': f})

        if not klist:
            curve.append({'date': d, 'equity': int(equity)})
            continue

        # allocate
        budget = equity * EQUITY_UTILIZATION
        sum_f = sum(x['f'] for x in klist)
        used = 0.0
        alloc = {}
        for x in sorted(klist, key=lambda z: z['f'], reverse=True):
            tgt = budget * (x['f'] / sum_f)
            cap = budget * PER_STOCK_CAP_PCT
            tgt = min(tgt, cap)
            remain = budget - used
            if remain <= 0:
                qty = 0
            else:
                tgt = min(tgt, remain)
                o = float(day_rows[x['code']]['open'])
                qty = 0 if tgt < MIN_ORDER_VALUE else int(tgt // o)
            alloc[x['code']] = qty
            used += qty * float(day_rows[x['code']]['open'])

        # simulate
        day_pnl = 0.0
        for c, qty in alloc.items():
            if qty <= 0:
                continue
            r = day_rows[c]
            o, h, l, cl = float(r['open']), float(r['high']), float(r['low']), float(r['close'])
            rc = simulate_day_open_tp_sl(o, h, l, STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRIGGER_PRIORITY, close_px=cl)
            pos_value = qty * o
            day_pnl += pos_value * rc
        equity += day_pnl
        curve.append({'date': d, 'equity': int(equity)})

    return pd.DataFrame(curve)

# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Universe
    def pick_universe():
        if USE_RANDOM_UNIVERSE:
            pool = RANDOM_UNIVERSE_POOL or []
            if not pool:
                pool = ["005930","000660","035720"]
            rng = random.Random(RANDOM_SEED)
            n = min(rng.randint(RANDOM_MIN_CODES, RANDOM_MAX_CODES), len(pool))
            return sorted(rng.sample(pool, n))
        if not os.path.exists(BUYLIST_PATH):
            return ["000660","005930","012330","051910","207940"]
        df = pd.read_csv(BUYLIST_PATH, dtype={'종목코드': str, 'code': str})
        codes = []
        for _, row in df.iterrows():
            code = (row.get('종목코드') or row.get('code') or '').zfill(6)
            if code:
                codes.append(code)
        codes = sorted(list(dict.fromkeys(codes)))
        if not codes:
            return ["000660","005930","012330","051910","207940"]
        return codes

    codes = pick_universe()
    print("Universe:", codes)

    # 2) Ensure OHLCV exists/covered
    codes = ensure_ohlcv_for(codes, start=START_DATE, end=END_DATE)
    if not codes:
        print("⚠️ No codes available after fetch.")
        sys.exit(0)

    # 3) Run RANGE backtest
    df_curve = run_backtest_range(START_DATE, END_DATE, codes)
    if df_curve.empty:
        print("⚠️ No trading days or data in the selected window.")
        sys.exit(0)

    # 4) Save equity curve & line chart (Return % vs INIT_EQUITY)
    out_csv = Path(OUTPUT_DIR) / f"kelly_backtest_{START_DATE}_{END_DATE}.csv"
    out_png = Path(OUTPUT_DIR) / f"kelly_backtest_{START_DATE}_{END_DATE}.png"
    df_curve.to_csv(out_csv, index=False, encoding='utf-8-sig')

    base = float(INIT_EQUITY)
    ret_series = (df_curve['equity'].astype(float) / base - 1.0) * 100.0

    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,6))
    plt.plot(df_curve['date'].tolist(), ret_series.tolist(), label='Return (%)')
    plt.xticks(rotation=45)
    plt.title(f"Kelly Backtest {START_DATE}~{END_DATE}")
    plt.xlabel("Date"); plt.ylabel("Return (%)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    total_ret = (df_curve['equity'].iloc[-1] / base - 1.0) * 100.0
    print(f"Final Return: {total_ret:.2f}% from {START_DATE} to {END_DATE}")
    print(f"✅ Saved: {out_csv}")
    print(f"✅ Saved: {out_png}")
