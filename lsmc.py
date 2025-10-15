# -*- coding: utf-8 -*-
"""
lsmc.py (inserver + config_Jin)
- 루트(fundweb)/config_Jin.py만 사용 (상위 폴더 의존 허용)
- 모든 산출물/로그: inserver/data/results, inserver/logs (영문 파일명)
- buy_list.csv는 inserver/data/results에서 탐색
- DRY_RUN(기본 1)일 때 주문은 실제로 보내지 않음
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# ─────────────────────────────
# 경로 고정 (inserver 기준)
# ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))     # .../fundweb/inserver
REPO_ROOT  = os.path.dirname(BASE_DIR)                      # .../fundweb
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BUYLIST_PATH   = os.path.join(OUTPUT_DIR, "buy_list.csv")
LOG_FILE       = os.path.join(OUTPUT_DIR, "trade_log.csv")
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portfolio.json")

# ─────────────────────────────
# 설정/비밀: 루트의 config_Jin.py 사용
# ─────────────────────────────
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from config_Jin import DB_CONFIG, get_api_keys, ACCOUNT_INFO
    print("✅ using repo-root/config_Jin.py")
except Exception as e:
    print("❌ Failed to import config_Jin.py from repo root (fundweb/).")
    print("   Required: DB_CONFIG, get_api_keys(), ACCOUNT_INFO")
    print(f"Error: {e}")
    raise SystemExit(1)

# ─────────────────────────────
# 전략 파라미터
# ─────────────────────────────
TOTAL_RISK_BUDGET_ALL = 500_000_000   # total portfolio risk budget (KRW)
MAX_BUY_BUDGET        = 10_000_000    # per-stock max budget cap (KRW)

# 실주문 방지 스위치 (기본 ON)
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"

# 모의투자 베이스 URL
url_base = "https://openapivts.koreainvestment.com:29443"

# ─────────────────────────────
# 유틸/공통
# ─────────────────────────────
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

def issue_access_token():
    app_key, app_secret = get_api_keys()
    if not app_key or not app_secret:
        print("❌ app_key/app_secret missing in config_Jin.get_api_keys().")
        raise SystemExit(1)
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
        print("❌ Failed to parse token response JSON.")
        raise SystemExit(1)
    token = j.get("access_token", "")
    if not token:
        print("❌ Access token issue failed:", j)
        raise SystemExit(1)
    print("🔐 Access token issued", flush=True)
    return token, app_key, app_secret

access_token, app_key, app_secret = issue_access_token()

# ─────────────────────────────
# 시세/잔고/주문 API
# ─────────────────────────────
def get_historical_prices_api(stock_code, start_date="20220101", end_date="20240101"):
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010400"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code,
        "fid_org_adj_prc": "0",
        "fid_period_div_code": "D",
        "fid_begin_date": start_date,
        "fid_end_date": end_date
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200:
        return None
    j = res.json()
    out = j.get("output")
    if not out:
        return None
    df = pd.DataFrame(out)
    df = df[df['stck_clpr'] != '']
    return df['stck_clpr'].astype(float).values[::-1]

def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": app_key, "appSecret": app_secret}
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json().get("HASH", "")
    except Exception:
        return ""

def get_current_price(stock_code):
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100"
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

def get_real_balance_qty(stock_code):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R"
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
        "CTX_AREA_NK100": ""
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

def send_order(stock_code, price, qty, order_type="매수"):
    if DRY_RUN:
        return {"rt_cd": "0", "msg1": "DRY_RUN: no order sent"}
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(adjust_price_to_tick(int(price))))
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "hashkey": get_hashkey(data)
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json()
    except Exception:
        return {"rt_cd": "-1", "msg1": "INVALID_JSON"}

# ─────────────────────────────
# 포트폴리오/로깅
# ─────────────────────────────
def load_portfolio():
    p = Path(PORTFOLIO_PATH)
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}

def save_portfolio(data):
    json.dump(data, open(PORTFOLIO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def log_trade(ts, code, price, prob_up, exp_profit, exp_loss, rr_ratio, qty, side, order_result):
    entry = {
        "timestamp": str(ts),
        "code": code,
        "price": int(price),
        "prob_up_pct": round(float(prob_up) * 100, 2),
        "expected_profit": round(float(exp_profit), 2),
        "expected_loss": round(float(exp_loss), 2),
        "rr_ratio": round(float(rr_ratio), 2),
        "qty": int(qty),
        "side": side,
        "result_msg": order_result.get("msg1", "NO_RESPONSE")
    }
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

# ─────────────────────────────
# LSMC 시뮬레이션
# ─────────────────────────────
def simulate_future_prices(current_price, days=10, paths=100, past_returns=None):
    simulated = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated[i, j] = price
    return simulated

def lsmc_expected_profit_and_risk_with_prob(stock_code, current_price):
    prices = get_historical_prices_api(stock_code)
    if prices is None or len(prices) < 20:
        return {'expected_profit': 0, 'expected_loss': 0, 'rr_ratio': 0, 'prob_up': 0}
    returns = np.diff(np.log(prices))
    sim = simulate_future_prices(current_price, past_returns=returns)
    max_profits = np.maximum(sim.max(axis=1) - current_price, 0)
    max_losses  = np.maximum(current_price - sim.min(axis=1), 0)
    exp_profit  = float(np.mean(max_profits))
    exp_loss    = float(np.mean(max_losses))
    rr_ratio    = exp_profit / exp_loss if exp_loss > 0 else 0.0
    prob_up     = float(np.mean(max_profits > 0))
    return {'expected_profit': exp_profit, 'expected_loss': exp_loss, 'rr_ratio': rr_ratio, 'prob_up': prob_up}

# ─────────────────────────────
# 메인
# ─────────────────────────────
if __name__ == "__main__":
    print("Working directory:", os.getcwd(), flush=True)

    # 1) buy_list.csv 로드
    if not os.path.exists(BUYLIST_PATH):
        print("❌ buy_list.csv not found:", os.path.abspath(BUYLIST_PATH))
        raise SystemExit(1)

    try:
        df = pd.read_csv(BUYLIST_PATH, dtype={'종목코드': str, 'code': str})
    except Exception as e:
        print("❌ Failed to read buy_list.csv:", e)
        raise SystemExit(1)

    candidates = []
    for _, row in df.iterrows():
        code = (row.get('종목코드') or row.get('code') or '').zfill(6)
        if code:
            candidates.append(code)
    candidates = list(dict.fromkeys(candidates))  # dedup
    if not candidates:
        print("❌ No valid candidates")
        raise SystemExit(1)
    print(f"✅ Candidates: {len(candidates)}", flush=True)

    portfolio = load_portfolio()
    portfolio_values = []
    loop_count = 1

    try:
        while True:
            print(f"\n[LOOP {loop_count}] {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

            # 비후보 전량 정리
            current_set = set(candidates)
            changed = True
            while changed:
                changed = False
                for code in list(portfolio.keys()):
                    if code not in current_set:
                        q = get_real_balance_qty(code)
                        if q > 0:
                            last = get_current_price(code) or 0
                            res = send_order(code, last, q, order_type="매도")
                            print(f"🔁 [Non-candidate SELL] {code}: {q} → {res}", flush=True)
                            log_trade(datetime.now(), code, last, 0, 0, 0, 0, q, "매도", res)
                        del portfolio[code]
                        changed = True
            save_portfolio(portfolio)

            # 후보별 기대수익/손실/손익비
            results = []
            rr_sum = 0.0
            for code in candidates:
                price = get_current_price(code)
                if not price:
                    print(f"❌ Failed to get price: {code}", flush=True)
                    continue
                r = lsmc_expected_profit_and_risk_with_prob(code, price)
                r.update({'code': code, 'price': price})
                results.append(r)
                rr_sum += r['rr_ratio']

            # 상위 10개 by expected_profit
            results_sorted = sorted(results, key=lambda x: x['expected_profit'], reverse=True)[:10]

            # 수량 계산
            for r in results_sorted:
                if rr_sum > 0 and r['rr_ratio'] > 0 and r['expected_loss'] > 0:
                    max_loss_allowed = TOTAL_RISK_BUDGET_ALL * (r['rr_ratio'] / rr_sum)
                    exp_loss = max(r['expected_loss'], 100.0)   # lower bound for stability
                    qty_by_risk = max_loss_allowed / exp_loss
                    budget_limited_qty = MAX_BUY_BUDGET // r['price']
                    r['optimal_qty'] = int(min(qty_by_risk, budget_limited_qty))
                else:
                    r['optimal_qty'] = 0
                print(f"[{r['code']}] price:{r['price']} RR:{r['rr_ratio']:.2f} "
                      f"expP:{r['expected_profit']:.2f} Qty:{r['optimal_qty']}", flush=True)

            # 후보 통계 CSV 저장
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path    = os.path.join(OUTPUT_DIR, f"candidates_stats_{ts}.csv")
                latest_path = os.path.join(OUTPUT_DIR, "latest_candidates_stats.csv")
                rows = []
                for r in results_sorted:
                    rows.append({
                        "code": r["code"],
                        "price": int(r["price"]),
                        "expected_profit": round(float(r["expected_profit"]), 2),
                        "expected_loss": round(float(r["expected_loss"]), 2),
                        "rr_ratio": round(float(r["rr_ratio"]), 2),
                        "prob_up_pct": round(float(r["prob_up"]) * 100, 2),
                        "optimal_qty": int(r["optimal_qty"]),
                        "loop": loop_count,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                if rows:
                    df_stats = pd.DataFrame(rows)
                    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    df_stats.to_csv(latest_path, index=False, encoding="utf-8-sig")
                    print(f"📄 candidates saved: {csv_path}", flush=True)
                else:
                    print("⚠️ No candidate stats to save", flush=True)
            except Exception as e:
                print("CSV save error:", e, flush=True)

            # 리밸런싱
            for r in results_sorted:
                code = r['code']
                price = r['price']
                target = r['optimal_qty']
                cur = int(portfolio.get(code, {}).get('qty', 0))
                if target > cur:
                    add = target - cur
                    if add > 0:
                        res = send_order(code, price, add, order_type="매수")
                        print(f"✅ BUY {code}: +{add} @{price} → {res}", flush=True)
                        log_trade(datetime.now(), code, price, r['prob_up'],
                                  r['expected_profit'], r['expected_loss'], r['rr_ratio'],
                                  add, "매수", res)
                        if res.get("rt_cd") == "0":
                            if code in portfolio:
                                portfolio[code]['qty'] += add
                            else:
                                portfolio[code] = {'buy_price': price, 'qty': add}
                elif target < cur:
                    sell = cur - target
                    if sell > 0:
                        res = send_order(code, price, sell, order_type="매도")
                        print(f"↘️ SELL {code}: -{sell} @{price} → {res}", flush=True)
                        log_trade(datetime.now(), code, price, r['prob_up'],
                                  r['expected_profit'], r['expected_loss'], r['rr_ratio'],
                                  sell, "매도", res)
                        if res.get("rt_cd") == "0":
                            portfolio[code]['qty'] -= sell
                            if portfolio[code]['qty'] <= 0:
                                del portfolio[code]
                else:
                    print(f"[HOLD] {code} qty {cur}", flush=True)

            save_portfolio(portfolio)

            # 평가금액 기록 & 그래프
            total_value = 0
            for code, pos in portfolio.items():
                q = int(pos.get('qty', 0))
                if q > 0:
                    last = get_current_price(code)
                    if last:
                        total_value += q * last
            portfolio_values.append(total_value)
            print(f"[Loop {loop_count}] total value: {total_value:,.0f}", flush=True)

            # 루프 주기
            loop_count += 1
            time.sleep(600)

    except KeyboardInterrupt:
        print("Interrupted! Saving equity curve...", flush=True)

    finally:
        try:
            if portfolio_values:
                plt.rcParams['axes.unicode_minus'] = False
                plt.figure(figsize=(10, 6))
                plt.plot(portfolio_values, label="Cumulative Portfolio Value")
                plt.title("Cumulative Return")
                plt.xlabel("Loop")
                plt.ylabel("Portfolio Value")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
                plt.savefig(out_path, dpi=300)
                print(f"✅ equity curve saved: {out_path}", flush=True)
            else:
                print("No data to save", flush=True)
        except Exception as e:
            print("Plot save error:", e, flush=True)
