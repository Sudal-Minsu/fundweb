# -*- coding: utf-8 -*-
"""
full_kelly.py (inserver-only, standardized)
- Uses local config.py (same folder) for credentials:
    get_account("acc2") -> {"APP_KEY","APP_SECRET","ACCOUNT_INFO": {"CANO","ACNT_PRDT_CD"}}
- Reads buy_list.csv from inserver/data/results/
- Writes ALL outputs to inserver/data/results with ENGLISH filenames/columns
- DRY_RUN=1 (default) prevents real orders
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Paths fixed to inserver/
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))        # .../fundweb/inserver
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BUYLIST_PATH   = os.path.join(OUTPUT_DIR, "buy_list.csv")
TRADE_LOG_FILE = Path(OUTPUT_DIR) / "trade_log.csv"
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portfolio.json")

# ─────────────────────────────────────────────
# Config (MUST be in the same folder: inserver/config.py)
# ─────────────────────────────────────────────
try:
    from config import get_account
except Exception as e:
    print("❌ Failed to import config.py in the same folder (inserver).")
    print("   It must define get_account(name) returning dict with APP_KEY/APP_SECRET/ACCOUNT_INFO.")
    print(f"Error: {e}")
    raise SystemExit(1)

account     = get_account("acc2")
app_key     = account["APP_KEY"]
app_secret  = account["APP_SECRET"]
ACCOUNT_INFO = account["ACCOUNT_INFO"]
if not app_key or not app_secret:
    print("❌ Empty APP_KEY/APP_SECRET from config.get_account('acc2').")
    raise SystemExit(1)

# ─────────────────────────────────────────────
# Switches / Strategy params
# ─────────────────────────────────────────────
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"  # default ON for safety

# Total per-loop buy budget and per-symbol cap (KRW)
TOTAL_BUY_BUDGET_ALL = 100_000_000
MAX_BUY_BUDGET       = 10_000_000
ENFORCE_TOTAL_BUDGET_CAP = True

# Stop/Take percentages and derived Kelly R
STOP_LOSS_PCT   = 0.025   # -2.5%
TAKE_PROFIT_PCT = 0.05    # +5%
R = TAKE_PROFIT_PCT / STOP_LOSS_PCT

# Position housekeeping
SELL_NON_CANDIDATES    = True
CHECK_TP_SL_EVERY_LOOP = True

# Paper trading base URL
url_base = "https://openapivts.koreainvestment.com:29443"

# ───────────── Utilities ─────────────
def adjust_price_to_tick(price: int) -> int:
    if price < 1000:
        tick = 1
    elif price < 5000:
        tick = 5
    elif price < 10000:
        tick = 10
    elif price < 50000:
        tick = 50
    elif price < 100000:
        tick = 100
    elif price < 500000:
        tick = 500
    else:
        tick = 1000
    return int(price - (price % tick))

def issue_access_token() -> str:
    res = requests.post(
        f"{url_base}/oauth2/tokenP",
        headers={"content-type": "application/json"},
        data=json.dumps({
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret
        })
    )
    try:
        j = res.json()
    except Exception:
        print("❌ Token response JSON parse failed:", res.text)
        raise SystemExit(1)
    token = j.get("access_token", "")
    if not token:
        print("❌ Failed to issue access token:", j)
        raise SystemExit(1)
    print("🔐 Access token issued.\n", flush=True)
    return token

access_token = issue_access_token()

# ───────────── Quote / Balance / Order APIs ─────────────
def get_current_price(stock_code: str) -> int | None:
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100",
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200:
        return None
    try:
        pr = int(res.json()['output']['stck_prpr'])
        return adjust_price_to_tick(pr)
    except Exception:
        return None

def get_real_balance_qty(stock_code: str) -> int:
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",
    }
    params = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200:
        return 0
    try:
        for item in res.json().get('output1', []):
            if item.get('pdno') == stock_code:
                return int(item.get('hldg_qty', 0))
    except Exception:
        pass
    return 0

def get_hashkey(data: dict) -> str:
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json().get("HASH", "")
    except Exception:
        return ""

def send_order(stock_code: str, price: int, qty: int, order_type: str = "BUY") -> dict:
    """order_type: 'BUY' or 'SELL'"""
    if DRY_RUN:
        return {"rt_cd": "0", "msg1": "DRY_RUN: no order sent"}
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type.upper() == "BUY" else "VTTC0801U"
    adjusted_price = adjust_price_to_tick(int(price))
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(adjusted_price)),
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "hashkey": get_hashkey(data),
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json()
    except Exception:
        return {"rt_cd": "-1", "msg1": "INVALID_JSON"}

# ───────────── Portfolio & Logging ─────────────
def load_portfolio() -> dict:
    p = Path(PORTFOLIO_PATH)
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}

def save_portfolio(data: dict):
    json.dump(data, open(PORTFOLIO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def log_trade(ts, code, price, p, R, fstar, qty, side, order_result):
    ts_str = ts if isinstance(ts, str) else ts.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": ts_str,
        "code": code,
        "price": int(price),
        "p_pct": round(float(p) * 100, 2),
        "R": round(float(R), 3),
        "kelly_f": round(float(fstar), 4),
        "qty": int(qty),
        "side": side,  # "BUY"/"SELL"
        "result_msg": order_result.get("msg1", "NO_RESPONSE"),
    }
    if TRADE_LOG_FILE.exists():
        df = pd.read_csv(TRADE_LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(TRADE_LOG_FILE, index=False, encoding="utf-8-sig")

# ───────────── Housekeeping ─────────────
def wait_until_all_non_candidate_sold(portfolio: dict, current_buy_codes: set[str]):
    if not SELL_NON_CANDIDATES:
        return
    changed = True
    while changed:
        changed = False
        for code in list(portfolio.keys()):
            if code not in current_buy_codes:
                real_shares = get_real_balance_qty(code)
                if real_shares > 0:
                    last = get_current_price(code) or 0
                    res = send_order(code, last, qty=real_shares, order_type="SELL")
                    print(f"🔁 [Trim non-candidate] {code}: -{real_shares} → {res}", flush=True)
                    log_trade(datetime.now(), code, last,
                              portfolio[code].get("p", 0),
                              portfolio[code].get("R", 0),
                              portfolio[code].get("fstar", 0),
                              real_shares, "SELL", res)
                if code in portfolio:
                    del portfolio[code]
                    changed = True
        if changed:
            print("Waiting fills for non-candidates... sleep 10s", flush=True)
            time.sleep(10)

def check_takeprofit_stoploss(portfolio: dict):
    if not CHECK_TP_SL_EVERY_LOOP:
        return
    for code, pos in list(portfolio.items()):
        shares = int(pos.get('qty', 0))
        if shares <= 0:
            continue
        last = get_current_price(code)
        if last is None:
            continue
        tp = pos.get('tp_price')
        sl = pos.get('sl_price')
        if tp and last >= tp:
            res = send_order(code, last, qty=shares, order_type="SELL")
            print(f"🎯 [Take Profit] {code} {shares} @ {last} → {res}", flush=True)
            log_trade(datetime.now(), code, last, pos.get("p", 0), pos.get("R", 0), pos.get("fstar", 0), shares, "SELL", res)
            if res.get("rt_cd") == "0":
                del portfolio[code]
                continue
        if sl and last <= sl:
            res = send_order(code, last, qty=shares, order_type="SELL")
            print(f"🛑 [Stop Loss] {code} {shares} @ {last} → {res}", flush=True)
            log_trade(datetime.now(), code, last, pos.get("p", 0), pos.get("R", 0), pos.get("fstar", 0), shares, "SELL", res)
            if res.get("rt_cd") == "0":
                del portfolio[code]

# ───────────── Kelly helpers ─────────────
def extract_prob_from_row(row: dict) -> float:
    keys = ["prob_up", "p", "prob", "상승확률", "확률"]
    for k in row.keys():
        if str(k).lower() in [x.lower() for x in keys]:
            try:
                p = float(row[k])
                if p > 1.0:  # if given in %
                    p = p / 100.0
                return max(0.0, min(1.0, p))
            except Exception:
                pass
    return 0.55

def compute_kelly_fraction(p: float, R: float) -> float:
    q = 1.0 - p
    return p - (q / R)

# ───────────── Main ─────────────
if __name__ == "__main__":
    print("📄 Loading buy_list.csv ...", flush=True)
    if not os.path.exists(BUYLIST_PATH):
        print("❌ buy_list.csv not found:", os.path.abspath(BUYLIST_PATH))
        raise SystemExit(1)

    df_cand = pd.read_csv(BUYLIST_PATH, dtype={'종목코드': str, 'code': str})
    rows = []
    for _, row in df_cand.iterrows():
        d = row.to_dict()
        code = (d.get('종목코드') or d.get('code') or '').zfill(6)
        if not code:
            continue
        d['종목코드'] = code
        rows.append(d)
    if not rows:
        print("❌ No valid candidates.")
        raise SystemExit(1)

    current_buy_codes = set([r['종목코드'] for r in rows])
    print(f"✅ Candidates loaded: {len(rows)}", flush=True)

    loop_count = 1
    portfolio = load_portfolio()
    equity_curve = []   # [{time,total_value,cum_return}]
    INITIAL_ASSET = 100_000_000  # if you want a cash baseline for cum_return

    try:
        while True:
            print(f"\n[LOOP {loop_count}] {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

            # Housekeeping
            if SELL_NON_CANDIDATES:
                wait_until_all_non_candidate_sold(portfolio, current_buy_codes)
                save_portfolio(portfolio)
            if CHECK_TP_SL_EVERY_LOOP:
                check_takeprofit_stoploss(portfolio)
                save_portfolio(portfolio)

            # 1) Kelly stats
            kelly_list = []
            for d in rows:
                code = d['종목코드']
                price = get_current_price(code)
                if not price or price <= 0:
                    print(f"❌ Failed to fetch last price: {code}", flush=True)
                    continue
                p = extract_prob_from_row(d)
                fstar = compute_kelly_fraction(p, R)
                if fstar <= 0:
                    continue
                kelly_list.append({'code': code, 'price': price, 'p': p, 'R': R, 'fstar': fstar})

            if not kelly_list:
                print("⚠️ No positive Kelly names. Sleeping…", flush=True)
                loop_count += 1
                time.sleep(600)
                continue

            # 2) Budget proportional to f*
            sum_f = sum(x['fstar'] for x in kelly_list)
            allocated_total = 0
            for x in sorted(kelly_list, key=lambda z: z['fstar'], reverse=True):
                target_value = TOTAL_BUY_BUDGET_ALL * (x['fstar'] / sum_f)
                target_value = min(target_value, MAX_BUY_BUDGET)
                if ENFORCE_TOTAL_BUDGET_CAP:
                    remain = TOTAL_BUY_BUDGET_ALL - allocated_total
                    target_value = 0 if remain <= 0 else min(target_value, remain)
                qty_target = int(target_value // x['price'])
                x['target_value'] = int(target_value)
                x['target_qty']   = max(0, qty_target)
                allocated_total  += x['target_value']

            # 3) Rebalance
            for x in kelly_list:
                code = x['code']
                price = x['price']
                p     = x['p']
                fstar = x['fstar']
                target_qty = x['target_qty']
                cur_qty    = int(portfolio.get(code, {}).get('qty', 0))

                if target_qty > cur_qty:
                    add = target_qty - cur_qty
                    if add > 0:
                        res = send_order(code, price, qty=add, order_type="BUY")
                        print(f"✅ BUY {code}: +{add} @ {price} → {res}", flush=True)
                        log_trade(datetime.now(), code, price, p, R, fstar, add, "BUY", res)
                        if res.get("rt_cd") == "0":
                            buy_price = price
                            tp_price = adjust_price_to_tick(int(buy_price * (1 + TAKE_PROFIT_PCT)))
                            sl_price = adjust_price_to_tick(int(buy_price * (1 - STOP_LOSS_PCT)))
                            if code in portfolio:
                                portfolio[code]['qty'] += add
                                portfolio[code]['tp_price'] = portfolio[code].get('tp_price', tp_price)
                                portfolio[code]['sl_price'] = portfolio[code].get('sl_price', sl_price)
                                portfolio[code]['p'] = p; portfolio[code]['R'] = R; portfolio[code]['fstar'] = fstar
                            else:
                                portfolio[code] = {
                                    'buy_price': buy_price,
                                    'qty': add,
                                    'tp_price': tp_price,
                                    'sl_price': sl_price,
                                    'p': p, 'R': R, 'fstar': fstar
                                }

                elif target_qty < cur_qty:
                    sell = cur_qty - target_qty
                    if sell > 0:
                        res = send_order(code, price, qty=sell, order_type="SELL")
                        print(f"↘️ SELL {code}: -{sell} @ {price} → {res}", flush=True)
                        log_trade(datetime.now(), code, price, p, R, fstar, sell, "SELL", res)
                        if res.get("rt_cd") == "0":
                            portfolio[code]['qty'] -= sell
                            if portfolio[code]['qty'] <= 0:
                                del portfolio[code]
                else:
                    print(f"[HOLD] {code} qty {cur_qty}", flush=True)

            save_portfolio(portfolio)

            # 4) Mark-to-market & equity curve
            # If you want a cash baseline, treat unused cash as (INITIAL_ASSET - cost basis).
            # Otherwise, you can just sum MTM of positions.
            total_value = 0
            for code, pos in portfolio.items():
                shares = int(pos.get('qty', 0))
                if shares > 0:
                    last = get_current_price(code)
                    if last:
                        total_value += shares * last

            # If you want to include "cash" approximation:
            cost_basis = sum(int(pos.get('qty', 0)) * int(pos.get('buy_price', 0)) for pos in portfolio.values())
            cash_approx = max(0, INITIAL_ASSET - cost_basis)
            total_value += cash_approx

            cum_return = (total_value / INITIAL_ASSET) - 1.0
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            equity_curve.append({"time": now_str, "total_value": int(total_value), "cum_return": round(cum_return, 6)})

            csv_path = Path(OUTPUT_DIR) / "equity_curve.csv"
            pd.DataFrame(equity_curve).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 equity_curve saved: {csv_path}", flush=True)
            print(f"[Loop {loop_count}] total_value: {total_value:,.0f}  cum_return: {cum_return*100:.2f}%", flush=True)

            loop_count += 1
            time.sleep(600)  # 10 minutes

    except KeyboardInterrupt:
        print("Interrupted! Saving final CSV/plot...", flush=True)

    finally:
        try:
            if equity_curve:
                csv_path = Path(OUTPUT_DIR) / "equity_curve.csv"
                pd.DataFrame(equity_curve).to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ Final equity CSV saved: {csv_path}", flush=True)
        except Exception as e:
            print(f"CSV save error: {e}", flush=True)

        try:
            if equity_curve:
                plt.rcParams['axes.unicode_minus'] = False
                plt.figure(figsize=(10, 6))
                plt.plot([x['total_value'] for x in equity_curve], label="Cumulative Portfolio Value")
                plt.title("Cumulative Return")
                plt.xlabel("Loop")
                plt.ylabel("Portfolio Value")
                plt.grid(True); plt.legend(); plt.tight_layout()
                out_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
                plt.savefig(out_path, dpi=300)
                print(f"✅ Equity curve saved: {out_path}", flush=True)
            else:
                print("No data to plot.", flush=True)
        except Exception as e:
            print(f"Plot save error: {e}", flush=True)
