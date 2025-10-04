# -*- coding: utf-8 -*-
"""
All-day runner:
- Intraday: continuously track & SELL when volatility outside band
- 15:19: horizon-based SELL once (regardless of volatility)
- 15:40: log/plot & exit
"""

import os
import time
import math
import csv
import traceback
import datetime as dt
from datetime import datetime, timedelta
from collections import deque, defaultdict
from zoneinfo import ZoneInfo
import pandas as pd

from config_ko import DB_CONFIG, ACCOUNT_INFO  # noqa
from functions import (
    get_auth_info,
    check_account,
    get_current_price,
    execute_order,
    log_account,
    plot_equity_and_return_from_csv,
)

# ===== Settings =====
TZ = ZoneInfo("Asia/Seoul")
MARKET_OPEN  = dt.time(9, 0)
MARKET_CLOSE = dt.time(15, 30)

FORCE_HORIZON_SELL_TIME = dt.time(15, 19)
POST_CLOSE_TASK_TIME    = dt.time(15, 40)

POLL_SEC = 60
REFRESH_PORTFOLIO_SEC = 60  # 보유종목 추적을 위해 1분마다 업데이트 권장
BUY_FILE = "results_3/today_recos.csv"

# Position sizing (네가 쓰던 방식)
POS_DENOM = 4
FEE_BUFFER = 0.98

# Horizon
HORIZON_DAYS = 6
HOLDINGS_FILE = "holdings.csv"  # code, buy_time, qty, buy_price ...

# Volatility tracking
VOL_WINDOW_MIN  = 30      # 최근 30분
VOL_BAND_LOWER  = 0.010   # 1.0% 이하(너무 잔잔) → 장중 매도하지 않음
VOL_BAND_UPPER  = 0.035   # 3.5% 이상(충분히 요동) → 장중 매도 실행
# 측도: (window_max - window_min) / mid, mid=(max+min)/2

# ===== Utils =====
def now_kst() -> datetime:
    return datetime.now(TZ)

def is_time_between(now: datetime, t0: dt.time, t1: dt.time) -> bool:
    return t0 <= now.timetz().replace(tzinfo=None) <= t1

def read_recos(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"code": str, "종목코드": str})
        if "code" not in df.columns and "종목코드" in df.columns:
            df = df.rename(columns={"종목코드": "code"})
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return pd.DataFrame()

def try_buy_recos(app_key, app_secret, access_token, portfolio_df, account_dict) -> int:
    df = read_recos(BUY_FILE)
    if df.empty or "code" not in df.columns:
        return 0

    held = set()
    if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
        code_col = "종목코드" if "종목코드" in portfolio_df.columns else ("code" if "code" in portfolio_df.columns else None)
        if code_col:
            held = set(portfolio_df[code_col].astype(str).str.zfill(6).tolist())

    tot_evlu = int(account_dict.get("tot_evlu_amt", 0))
    scts_evlu = int(account_dict.get("scts_evlu_amt", 0))
    cash_per_pick = int((tot_evlu - scts_evlu) / POS_DENOM / FEE_BUFFER)

    bought = 0
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        if code in held:
            continue
        try:
            price = int(get_current_price(app_key, app_secret, access_token, code))
            if price <= 0:
                continue
            qty = max(cash_per_pick // price, 0)
            if qty <= 0:
                continue
            execute_order(app_key, app_secret, access_token, code, qty, "매수", "시장가")
            time.sleep(1)
            bought += 1
        except Exception as e:
            print(f"[BUY-ERR] {code}: {e}")
    return bought

# ===== Volatility tracking =====
price_hist = defaultdict(lambda: deque(maxlen=VOL_WINDOW_MIN + 5))

def update_price_history(app_key, app_secret, access_token, codes):
    ts = now_kst()
    for code in codes:
        try:
            p = int(get_current_price(app_key, app_secret, access_token, code))
            if p > 0:
                price_hist[code].append((ts, p))
        except Exception as e:
            print(f"[PRICE-ERR] {code}: {e}")

def window_range_vol(code: str, window_min: int = VOL_WINDOW_MIN) -> float:
    ts_now = now_kst()
    dq = price_hist.get(code)
    if not dq:
        return 0.0
    cutoff = ts_now - timedelta(minutes=window_min)
    prices = [p for (t, p) in dq if t >= cutoff]
    if len(prices) < max(3, window_min // 3):
        return 0.0
    hi, lo = max(prices), min(prices)
    mid = (hi + lo) / 2.0 if (hi + lo) else 0.0
    if mid <= 0:
        return 0.0
    return (hi - lo) / mid

# ===== Horizon helpers (read from holdings.csv) =====
def read_holdings_for_horizon() -> pd.DataFrame:
    if not os.path.exists(HOLDINGS_FILE):
        return pd.DataFrame(columns=["code", "buy_time", "qty"])
    df = pd.read_csv(HOLDINGS_FILE, dtype={"code": str})
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "buy_time" in df.columns:
        df["buy_time"] = pd.to_datetime(df["buy_time"], errors="coerce")
    else:
        df["buy_time"] = pd.NaT
    if "qty" not in df.columns:
        df["qty"] = 0
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    return df

def horizon_candidates(df_hold: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    if df_hold.empty:
        return df_hold
    now = now_kst()
    mask = df_hold["buy_time"].notna() & ((now - df_hold["buy_time"]).dt.days >= horizon_days)
    return df_hold.loc[mask, ["code", "qty"]].query("qty > 0")

# ===== Intraday sell by volatility (continuous) =====
def intraday_volatility_sell(app_key, app_secret, access_token, portfolio_df):
    """
    장중 계속 호출:
    - 각 보유종목 변동성 범위를 계산
    - 변동성이 밴드 '밖'(<=LOWER 또는 >=UPPER)일 때 전량 매도 실행
    """
    if portfolio_df is None or portfolio_df.empty:
        return 0
    code_col = "종목코드" if "종목코드" in portfolio_df.columns else ("code" if "code" in portfolio_df.columns else None)
    qty_col  = "보유수량" if "보유수량" in portfolio_df.columns else ("qty" if "qty" in portfolio_df.columns else None)
    if not code_col or not qty_col:
        return 0

    # 가격 히스토리 업데이트
    codes = portfolio_df[code_col].astype(str).str.zfill(6).unique().tolist()
    if codes:
        update_price_history(app_key, app_secret, access_token, codes)

    sells = 0
    for _, row in portfolio_df.iterrows():
        code = str(row[code_col]).zfill(6)
        qty  = int(row[qty_col])
        if qty <= 0:
            continue
        rng_vol = window_range_vol(code, VOL_WINDOW_MIN)
        if rng_vol <= VOL_BAND_LOWER:
            # 너무 잔잔 → 매도 보류(추적만)
            continue
        if rng_vol >= VOL_BAND_UPPER:
            try:
                execute_order(app_key, app_secret, access_token, code, qty, "매도", "시장가")
                time.sleep(1)
                sells += 1
                print(f"[INTRADAY-SELL] {code} x {qty}, vol={rng_vol:.4f} (>= {VOL_BAND_UPPER:.4f})")
            except Exception as e:
                print(f"[INTRADAY-SELL-ERR] {code}: {e}")
    return sells

# ===== Horizon sell at 15:19 (once, no volatility gate) =====
def horizon_sell_once(app_key, app_secret, access_token):
    holds = read_holdings_for_horizon()
    cands = horizon_candidates(holds, HORIZON_DAYS)
    if cands.empty:
        print("[HORIZON] No candidates.")
        return 0
    sold = 0
    for _, row in cands.iterrows():
        code = row["code"]
        qty  = int(row["qty"])
        if qty <= 0:
            continue
        try:
            execute_order(app_key, app_secret, access_token, code, qty, "매도", "시장가")
            time.sleep(1)
            sold += 1
            print(f"[HORIZON-SELL] {code} x {qty}")
        except Exception as e:
            print(f"[HORIZON-SELL-ERR] {code}: {e}")
    return sold

# ===== Main loop =====
def main():
    app_key, app_secret, access_token = get_auth_info()
    print(f"[{now_kst()}] Auth OK")

    last_refresh = None
    forced_horizon_done = False
    post_close_done = False

    # 초기 스냅샷
    portfolio, account = check_account(app_key, app_secret, access_token)
    print(f"[{now_kst()}] Initial portfolio/account loaded")

    while True:
        now = now_kst()
        try:
            # 주기적 업데이트
            if (last_refresh is None) or ((now - last_refresh).total_seconds() >= REFRESH_PORTFOLIO_SEC):
                portfolio, account = check_account(app_key, app_secret, access_token)
                last_refresh = now
                print(f"[{now}] Refreshed portfolio/account")

            if is_time_between(now, MARKET_OPEN, MARKET_CLOSE):
                # 1) 장중 매수
                try_buy_recos(app_key, app_secret, access_token, portfolio, account)
                # 2) 장중 변동성 기반 매도(지속 추적)
                intraday_volatility_sell(app_key, app_secret, access_token, portfolio)
            else:
                print(f"[{now}] Market closed window. Standing by.")

            # 15:19 Horizon 매도(1회, 변동성 무시)
            if now.time().hour == FORCE_HORIZON_SELL_TIME.hour and \
               now.time().minute == FORCE_HORIZON_SELL_TIME.minute and not forced_horizon_done:
                print(f"[{now}] Horizon sell @ {FORCE_HORIZON_SELL_TIME}")
                horizon_sell_once(app_key, app_secret, access_token)
                forced_horizon_done = True

            # 15:40 로그/그래프 후 종료
            if now.time().hour == POST_CLOSE_TASK_TIME.hour and \
               now.time().minute == POST_CLOSE_TASK_TIME.minute and not post_close_done:
                try:
                    log_account(account)
                    plot_equity_and_return_from_csv()
                    post_close_done = True
                    print(f"[{now}] Post-close tasks done. Shutting down for the day.")
                    break
                except Exception as e:
                    print(f"[POST-CLOSE ERR] {e}")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Shutting down gracefully...")
            break
        except Exception as e:
            print(f"[LOOP ERR] {type(e).__name__}: {e}")
            traceback.print_exc()

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
