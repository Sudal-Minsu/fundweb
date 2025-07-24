import os
import sys
import time
import json
import keyring
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from config import DB_CONFIG, ACCOUNT_INFO, get_api_keys

# ───────────── 설정 ─────────────
OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_RISK_BUDGET_ALL = 5_000_000_00
MAX_BUY_BUDGET = 10_000_000
app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"

# ───────────── 토큰 발급 ─────────────
res = requests.post(f"{url_base}/oauth2/tokenP",
                    headers={"content-type": "application/json"},
                    data=json.dumps({
                        "grant_type": "client_credentials",
                        "appkey": app_key,
                        "appsecret": app_secret
                    }))
access_token = res.json().get("access_token", "")
if not access_token:
    print("❌ 액세스 토큰 발급 실패:", res.json())
    sys.exit()
print(f"액세스 토큰: {access_token}\n")

# ───────────── 공통 API 함수 ─────────────
def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    return res.json().get("HASH", "")

def get_current_price(stock_code):
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
    if res.status_code != 200 or 'output' not in res.json():
        return None
    return int(res.json()['output']['stck_prpr'])

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
    if res.status_code != 200 or 'output' not in res.json():
        return None
    df = pd.DataFrame(res.json()['output'])
    df = df[df['stck_clpr'] != '']
    return df['stck_clpr'].astype(float).values[::-1]

# ───────────── LSMC 시뮬 함수 ─────────────
def simulate_future_prices(current_price, days=10, paths=100, past_returns=None):
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated_prices[i, j] = price
    return simulated_prices

def lsmc_expected_profit_and_risk_with_prob(stock_code, current_price):
    prices = get_historical_prices_api(stock_code)
    if prices is None or len(prices) < 20:
        return {'expected_profit': 0, 'expected_loss': 0, 'rr_ratio': 0, 'optimal_qty': 0, 'prob_up': 0}
    returns = np.diff(np.log(prices))
    simulated = simulate_future_prices(current_price, past_returns=returns)
    max_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    max_losses = np.maximum(current_price - simulated.min(axis=1), 0)
    expected_profit = np.mean(max_profits)
    expected_loss = np.mean(max_losses)
    rr_ratio = expected_profit / expected_loss if expected_loss > 0 else 0
    prob_up = np.mean(max_profits > 0)
    return {
        'expected_profit': expected_profit,
        'expected_loss': expected_loss,
        'rr_ratio': rr_ratio,
        'prob_up': prob_up
    }

# ───────────── 주문 함수 ─────────────
def send_order(stock_code, price, qty, order_type="머수"):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "머수" else "VTTC0801U"
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(price)
    }
    hashkey = get_hashkey(data)
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
    return res.json()

# ───────────── 로그 기록 ─────────────
def log_trade(timestamp, stock_code, price, prob_up, exp_profit, exp_loss, rr_ratio, qty, order_type, order_result):
    log_file = Path("trade_log.csv")
    log_entry = {
        "거래시간": timestamp,
        "종목코드": stock_code,
        "현재가": price,
        "상증확률(%)": round(prob_up * 100, 2),
        "기대수익": round(exp_profit, 2),
        "예상손실": round(exp_loss, 2),
        "손익비": round(rr_ratio, 2),
        "주문수량": qty,
        "주문종류": order_type,
        "주문결과": order_result.get("msg1", "NO_RESPONSE")
    }
    if log_file.exists():
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(log_file, index=False, encoding='utf-8-sig')

# ───────────── 메인 실행 ─────────────
if __name__ == "__main__":
    print("buy_list.csv에서 매수 호불 불러오는 중...")
    buy_list_path = os.path.join(OUTPUT_DIR, "buy_list.csv")
    if not os.path.exists(buy_list_path):
        print("❌ buy_list.csv 파일이 존재하지 않습니다.")
        sys.exit()

    top_candidates = pd.read_csv(buy_list_path, dtype={'종목코드': str})
    top_candidates = [
        {**row, '종목코드': row['종목코드'].zfill(6)} for _, row in top_candidates.iterrows()
    ]
    current_buy_codes = set([c['종목코드'] for c in top_candidates])
    print(f"[get_today_candidates] 불러온 호불 수: {len(top_candidates)}")

    loop_count = 1
    portfolio = {}
    portfolio_values = []

    try:
        while True:
            print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            results = []
            rr_total = 0

            # 비호불 종목 전략 매도
            for stock_code in list(portfolio.keys()):
                if stock_code not in current_buy_codes:
                    shares = portfolio[stock_code]['qty']
                    if shares > 0:
                        last_price = get_current_price(stock_code)
                        order_result = send_order(stock_code, last_price, qty=shares, order_type="매도")
                        print(f"[비호불 종목 매도] {stock_code}: {shares}주 → {order_result}")
                        log_trade(datetime.now(), stock_code, last_price, 0, 0, 0, 0, shares, "매도", order_result)
                        del portfolio[stock_code]

            for candidate in top_candidates[:3]:
                stock_code = candidate['종목코드']
                print(f"종목 코드: {stock_code}")
                price = get_current_price(stock_code)
                if not price:
                    print(f"❌ 현재가 조회 실패: {stock_code}")
                    continue
                result = lsmc_expected_profit_and_risk_with_prob(stock_code, price)
                rr_total += result['rr_ratio']
                result.update({'code': stock_code, 'price': price})
                results.append(result)

            for result in results:
                rr = result['rr_ratio']
                if rr_total > 0 and rr > 0 and result['expected_loss'] > 0:
                    max_loss_allowed = TOTAL_RISK_BUDGET_ALL * (rr / rr_total)
                    if result['expected_loss'] < 100:
                        result['expected_loss'] = 100
                    qty_by_risk = max_loss_allowed / result['expected_loss']
                    budget_limited_qty = MAX_BUY_BUDGET // result['price']
                    result['optimal_qty'] = int(min(qty_by_risk, budget_limited_qty))
                else:
                    result['optimal_qty'] = 0
                print(f"[{result['code']}] 가격:{result['price']} RR:{rr:.2f} Qty:{result['optimal_qty']}")

            for result in results:
                stock_code = result['code']
                price = result['price']
                optimal_qty = result['optimal_qty']
                current_qty = portfolio.get(stock_code, {}).get('qty', 0)

                if optimal_qty > current_qty:
                    add_qty = optimal_qty - current_qty
                    if add_qty > 0:
                        order_result = send_order(stock_code, price, qty=add_qty, order_type="머수")
                        print(f"추가 매수 요청 결과: {order_result}")
                        log_trade(datetime.now(), stock_code, price, result['prob_up'],
                                  result['expected_profit'], result['expected_loss'], rr, add_qty, "머수", order_result)
                        if order_result.get("rt_cd") == "0":
                            if stock_code in portfolio:
                                portfolio[stock_code]['qty'] += add_qty
                            else:
                                portfolio[stock_code] = {'buy_price': price, 'qty': add_qty}

                elif optimal_qty < current_qty:
                    sell_qty = current_qty - optimal_qty
                    if sell_qty > 0:
                        order_result = send_order(stock_code, price, qty=sell_qty, order_type="매도")
                        print(f"부분 매도 요청 결과: {order_result}")
                        log_trade(datetime.now(), stock_code, price, result['prob_up'],
                                  result['expected_profit'], result['expected_loss'], rr, sell_qty, "매도", order_result)
                        if order_result.get("rt_cd") == "0":
                            portfolio[stock_code]['qty'] -= sell_qty
                            if portfolio[stock_code]['qty'] <= 0:
                                del portfolio[stock_code]
                else:
                    print(f"[유지] {stock_code} 현재 수량 유지")

            total_value = 0
            for stock_code, pos in portfolio.items():
                shares = pos['qty']
                if shares > 0:
                    last_price = get_current_price(stock_code)
                    total_value += shares * last_price
            portfolio_values.append(total_value)
            print(f"[Loop {loop_count}] 평가금액: {total_value:,.0f}")

            loop_count += 1
            time.sleep(600)

    except KeyboardInterrupt:
        print("사용자 중단! 누적 수익률 그래프 저장 중...")

    finally:
        if portfolio_values:
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_values, label="누적 포트폴리오 값")
            plt.title("누적 수익률")
            plt.xlabel("룰 회수")
            plt.ylabel("포트폴리오 값")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "누적수익률_그래프.png"), dpi=300)
            print(f"누적 수익률 그래프 저장 완료 ({OUTPUT_DIR}/누적수익률_그래프.png)")
        else:
            print("저장할 데이터가 없습니다.")
