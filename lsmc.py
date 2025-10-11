# -*- coding: utf-8 -*-
"""
auto_pipeline_single.py (inserver 표준 버전)
- config_Jin.py: 상위 폴더에서 import 가능하도록 sys.path 보정
- 모든 산출물은 inserver/data/results 로 저장 (영문 파일명)
- buy_list.csv 탐색도 inserver/data/results 중심
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

# ───────────── 경로 표준화 ─────────────
# 이 파일이 inserver 안에 있다는 가정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../fundweb/inserver
REPO_ROOT = os.path.dirname(BASE_DIR)                                 # .../fundweb
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")                # .../fundweb/inserver/data/results
LOG_DIR = os.path.join(BASE_DIR, "logs")                              # .../fundweb/inserver/logs
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 상위 폴더 모듈 import 가능하도록 sys.path 보정 (config_Jin.py 등)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 🔐 상위 폴더에 있는 config_Jin.py 사용
try:
    from config_Jin import DB_CONFIG, get_api_keys, ACCOUNT_INFO
except Exception as e:
    print("❌ config_Jin.py 불러오기 실패. 상위 폴더( fundweb/ )에 존재하는지 확인하세요.")
    print(f"오류: {e}")
    sys.exit(1)

# ───────────── 출력 폴더/파일명(영문) ─────────────
# NOTE: rule_2 결과 요약도 같은 OUTPUT_DIR로 통일 (요구사항 2)
RULE2_OUTPUT_DIR = OUTPUT_DIR
TOTAL_RISK_BUDGET_ALL = 5_000_000_00
MAX_BUY_BUDGET = 10_000_000

# 매매 로그 파일 (영문)
LOG_FILE = os.path.join(OUTPUT_DIR, "trade_log.csv")

# API 베이스 (모의투자)
url_base = "https://openapivts.koreainvestment.com:29443"

# ───────────── 공통 유틸 ─────────────
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

# ───────────── 토큰 발급 ─────────────
def issue_access_token():
    app_key, app_secret = get_api_keys()
    if not app_key or not app_secret:
        print("❌ app_key/app_secret 비어있음. config_Jin.py의 keyring 설정 확인.")
        sys.exit(1)

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
        print("❌ 토큰 응답 JSON 파싱 실패:", res.text)
        sys.exit(1)

    access_token = j.get("access_token", "")
    if not access_token:
        print("❌ 액세스 토큰 발급 실패:", j)
        sys.exit(1)
    print("🔐 액세스 토큰 발급 성공", flush=True)
    return access_token, app_key, app_secret

# ───────────── 데이터/주문 API ─────────────
def get_historical_prices_api(access_token, app_key, app_secret, stock_code, start_date="20220101", end_date="20240101"):
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
    if 'output' not in j:
        return None
    df = pd.DataFrame(j['output'])
    df = df[df['stck_clpr'] != '']
    return df['stck_clpr'].astype(float).values[::-1]

def get_hashkey(app_key, app_secret, data):
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

def get_current_price(access_token, app_key, app_secret, stock_code):
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200:
        return None
    j = res.json()
    if 'output' not in j or not j['output'].get('stck_prpr'):
        return None
    return adjust_price_to_tick(int(j['output']['stck_prpr']))

def get_real_balance_qty(access_token, app_key, app_secret, stock_code):
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
    j = res.json()
    if 'output1' not in j:
        return 0
    for item in j['output1']:
        if item.get('pdno') == stock_code:
            try:
                return int(item.get('hldg_qty', 0))
            except Exception:
                return 0
    return 0

def send_order(access_token, app_key, app_secret, stock_code, price, qty, order_type="매수"):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"
    adjusted_price = adjust_price_to_tick(int(price))
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(adjusted_price))
    }
    hashkey = get_hashkey(app_key, app_secret, data)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "hashkey": hashkey
    }
    time.sleep(1.2)
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json()
    except Exception:
        return {"rt_cd": "-1", "msg1": "INVALID_JSON"}

# ───────────── 포트폴리오/로그 ─────────────
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portfolio.json")

def load_portfolio():
    path = Path(PORTFOLIO_PATH)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_portfolio(data):
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def log_trade(timestamp, stock_code, price, prob_up, exp_profit, exp_loss, rr_ratio, qty, order_type, order_result):
    log_entry = {
        "timestamp": str(timestamp),
        "code": stock_code,
        "price": int(price),
        "prob_up_pct": round(float(prob_up) * 100, 2),
        "expected_profit": round(float(exp_profit), 2),
        "expected_loss": round(float(exp_loss), 2),
        "rr_ratio": round(float(rr_ratio), 2),
        "qty": int(qty),
        "side": order_type,
        "result_msg": order_result.get("msg1", "NO_RESPONSE")
    }
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

def wait_until_all_non_candidate_sold(access_token, app_key, app_secret, portfolio, current_buy_codes):
    has_non_candidates = True
    while has_non_candidates:
        has_non_candidates = False
        for stock_code in list(portfolio.keys()):
            if stock_code not in current_buy_codes:
                real_shares = get_real_balance_qty(access_token, app_key, app_secret, stock_code)
                if real_shares > 0:
                    last_price = get_current_price(access_token, app_key, app_secret, stock_code)
                    order_result = send_order(access_token, app_key, app_secret, stock_code, last_price, qty=real_shares, order_type="매도")
                    print(f"🔁 [비후보 매도] {stock_code}: {real_shares}주 → {order_result}", flush=True)
                    log_trade(datetime.now(), stock_code, last_price, 0, 0, 0, 0, real_shares, "매도", order_result)
                    if order_result.get("rt_cd") == "0" or order_result.get("msg_cd") == "40240000":
                        del portfolio[stock_code]
                        has_non_candidates = True
                else:
                    del portfolio[stock_code]
                    has_non_candidates = True
        if has_non_candidates:
            print("비후보 종목 매도 체결 대기중... 10초 대기", flush=True)
            time.sleep(10)

# ───────────── LSMC 관련 ─────────────
def simulate_future_prices(current_price, days=10, paths=100, past_returns=None):
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated_prices[i, j] = price
    return simulated_prices

def lsmc_expected_profit_and_risk_with_prob(access_token, app_key, app_secret, stock_code, current_price):
    prices = get_historical_prices_api(access_token, app_key, app_secret, stock_code)
    if prices is None or len(prices) < 20:
        return {'expected_profit': 0, 'expected_loss': 0, 'rr_ratio': 0, 'optimal_qty': 0, 'prob_up': 0}
    returns = np.diff(np.log(prices))
    simulated = simulate_future_prices(current_price, past_returns=returns)
    max_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    max_losses = np.maximum(current_price - simulated.min(axis=1), 0)
    expected_profit = float(np.mean(max_profits))
    expected_loss = float(np.mean(max_losses))
    rr_ratio = expected_profit / expected_loss if expected_loss > 0 else 0.0
    prob_up = float(np.mean(max_profits > 0))
    return {
        'expected_profit': expected_profit,
        'expected_loss': expected_loss,
        'rr_ratio': rr_ratio,
        'prob_up': prob_up
    }

# ───────────── 메인 루프 ─────────────
if __name__ == "__main__":
    # 토큰/키
    access_token, app_key, app_secret = issue_access_token()

    print("현재 작업 디렉토리:", os.getcwd(), flush=True)

    # 1) buy_list.csv 탐색: inserver/data/results 중심
    #   (rule_2 산출도 같은 곳으로 표준화)
    BUYLIST_DIRS = [OUTPUT_DIR]
    buy_list_path = None
    for d in BUYLIST_DIRS:
        p = os.path.join(d, "buy_list.csv")
        if os.path.exists(p):
            buy_list_path = p
            break

    if not buy_list_path:
        print("❌ buy_list.csv 파일이 없습니다.", flush=True)
        print("   아래 경로 중 하나에 파일을 두세요:")
        for d in BUYLIST_DIRS:
            print("   -", os.path.abspath(os.path.join(d, "buy_list.csv")), flush=True)
        sys.exit(1)

    print("📄 사용 buy_list 경로:", os.path.abspath(buy_list_path), flush=True)

    # 2) CSV 로드
    try:
        top_candidates_df = pd.read_csv(buy_list_path, dtype={'종목코드': str})
    except Exception as e:
        print(f"❌ buy_list.csv 읽기 실패: {e}", flush=True)
        sys.exit(1)

    # 3) 6자리 종목코드 정규화
    top_candidates = [
        {**row, '종목코드': row['종목코드'].zfill(6)} for _, row in top_candidates_df.iterrows()
    ]
    current_buy_codes = set([c['종목코드'] for c in top_candidates])
    print(f"✅ [get_today_candidates] 불러온 후보 수: {len(top_candidates)}", flush=True)

    loop_count = 1
    portfolio = load_portfolio()
    portfolio_values = []

    try:
        while True:
            print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

            # 비후보 종목 정리
            wait_until_all_non_candidate_sold(access_token, app_key, app_secret, portfolio, current_buy_codes)
            save_portfolio(portfolio)

            results = []
            rr_total = 0.0

            # 1) 후보별 기대수익/손실/손익비 추정
            for candidate in top_candidates:
                stock_code = candidate['종목코드']
                print(f"종목 코드: {stock_code}", flush=True)
                price = get_current_price(access_token, app_key, app_secret, stock_code)
                if not price:
                    print(f"❌ 현재가 조회 실패: {stock_code}", flush=True)
                    continue
                result = lsmc_expected_profit_and_risk_with_prob(access_token, app_key, app_secret, stock_code, price)
                rr_total += result['rr_ratio']
                result.update({'code': stock_code, 'price': price})
                results.append(result)

            # 2) 기대수익 상위 10개
            results_sorted = sorted(results, key=lambda x: x['expected_profit'], reverse=True)[:10]

            # 3) 주문 수량 계산
            for r in results_sorted:
                rr = r['rr_ratio']
                if rr_total > 0 and rr > 0 and r['expected_loss'] > 0:
                    max_loss_allowed = TOTAL_RISK_BUDGET_ALL * (rr / rr_total)
                    if r['expected_loss'] < 100:  # 손실 추정 하한
                        r['expected_loss'] = 100.0
                    qty_by_risk = max_loss_allowed / r['expected_loss']
                    budget_limited_qty = MAX_BUY_BUDGET // r['price']
                    r['optimal_qty'] = int(min(qty_by_risk, budget_limited_qty))
                else:
                    r['optimal_qty'] = 0
                print(f"[{r['code']}] price:{r['price']} RR:{rr:.2f} expP:{r['expected_profit']:.2f} Qty:{r['optimal_qty']}", flush=True)

            # 3.5) 후보 통계 CSV (영문 파일명, OUTPUT_DIR)
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(OUTPUT_DIR, f"candidates_stats_{ts}.csv")
                latest_csv_path = os.path.join(OUTPUT_DIR, "latest_candidates_stats.csv")
                export_rows = []
                for r in results_sorted:
                    export_rows.append({
                        'code': r['code'],
                        'price': int(r['price']),
                        'expected_profit': round(float(r['expected_profit']), 2),
                        'expected_loss': round(float(r['expected_loss']), 2),
                        'rr_ratio': round(float(r['rr_ratio']), 2),
                        'prob_up_pct': round(float(r['prob_up']) * 100, 2),
                        'optimal_qty': int(r.get('optimal_qty', 0)),
                        'loop': loop_count,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                if export_rows:
                    df_stats = pd.DataFrame(export_rows)
                    df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    df_stats.to_csv(latest_csv_path, index=False, encoding='utf-8-sig')
                    print(f"📄 후보 통계 CSV 저장: {csv_path}", flush=True)
                else:
                    print("⚠️ 저장할 후보 통계가 없습니다.", flush=True)
            except Exception as e:
                print(f"CSV 저장 중 오류: {e}", flush=True)

            # 3.6) 요약 지표 CSV (영문 파일명, RULE2_OUTPUT_DIR=OUTPUT_DIR)
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                metrics_rows = []
                for r in results_sorted:
                    metrics_rows.append({
                        'code': r['code'],
                        'price': int(r['price']),
                        'expected_profit': round(float(r['expected_profit']), 2),
                        'expected_loss': round(float(r['expected_loss']), 2),
                        'rr_ratio': round(float(r['rr_ratio']), 2),
                        'loop': loop_count,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    })
                if metrics_rows:
                    df_metrics = pd.DataFrame(metrics_rows)
                    metrics_csv_path = os.path.join(RULE2_OUTPUT_DIR, f"lsmc_metrics_{ts}.csv")
                    latest_metrics_csv_path = os.path.join(RULE2_OUTPUT_DIR, "latest_lsmc_metrics.csv")
                    df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
                    df_metrics.to_csv(latest_metrics_csv_path, index=False, encoding='utf-8-sig')
                    print(f"📄 요약 CSV 저장: {metrics_csv_path}", flush=True)
                else:
                    print("⚠️ 저장할 요약 지표가 없습니다.", flush=True)
            except Exception as e:
                print(f"요약 CSV 저장 중 오류: {e}", flush=True)

            # 4) 리밸런싱 주문
            for r in results_sorted:
                stock_code = r['code']
                price = r['price']
                optimal_qty = r['optimal_qty']
                current_qty = portfolio.get(stock_code, {}).get('qty', 0)

                if optimal_qty > current_qty:
                    add_qty = optimal_qty - current_qty
                    if add_qty > 0:
                        order_result = send_order(access_token, app_key, app_secret, stock_code, price, qty=add_qty, order_type="매수")
                        print(f"✅ 추가 매수 요청 결과: {order_result}", flush=True)
                        log_trade(datetime.now(), stock_code, price, r['prob_up'],
                                  r['expected_profit'], r['expected_loss'], r['rr_ratio'],
                                  add_qty, "매수", order_result)
                        if order_result.get("rt_cd") == "0":
                            if stock_code in portfolio:
                                portfolio[stock_code]['qty'] += add_qty
                            else:
                                portfolio[stock_code] = {'buy_price': price, 'qty': add_qty}

                elif optimal_qty < current_qty:
                    sell_qty = current_qty - optimal_qty
                    if sell_qty > 0:
                        order_result = send_order(access_token, app_key, app_secret, stock_code, price, qty=sell_qty, order_type="매도")
                        print(f"부분 매도 요청 결과: {order_result}", flush=True)
                        log_trade(datetime.now(), stock_code, price, r['prob_up'],
                                  r['expected_profit'], r['expected_loss'], r['rr_ratio'],
                                  sell_qty, "매도", order_result)
                        if order_result.get("rt_cd") == "0":
                            portfolio[stock_code]['qty'] -= sell_qty
                            if portfolio[stock_code]['qty'] <= 0:
                                del portfolio[stock_code]
                else:
                    print(f"[유지] {stock_code} 현재 수량 유지", flush=True)

            save_portfolio(portfolio)

            # 5) 평가금액 기록
            total_value = 0
            for stock_code, pos in portfolio.items():
                shares = pos.get('qty', 0)
                if shares > 0:
                    last_price = get_current_price(access_token, app_key, app_secret, stock_code)
                    total_value += shares * last_price
            portfolio_values.append(total_value)
            print(f"[Loop {loop_count}] 평가금액: {total_value:,.0f}", flush=True)

            loop_count += 1
            time.sleep(600)

    except KeyboardInterrupt:
        print("사용자 중단! 누적 수익률 그래프 저장…", flush=True)

    finally:
        if portfolio_values:
            # 그래프 파일명 영문화
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
            print(f"✅ equity curve 저장 완료: {out_path}", flush=True)
        else:
            print("저장할 데이터가 없습니다.", flush=True)
