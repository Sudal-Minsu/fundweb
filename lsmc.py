import keyring
import requests
import json
import time
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from config import DB_CONFIG, ACCOUNT_INFO, get_api_keys
from rule_2 import predict_today_candidates
from pathlib import Path
from sqlalchemy import create_engine

# ───────────── 설정 ─────────────
TOTAL_RISK_BUDGET = 1_000_000  # 종목당 최대 리스크 허용 금액
app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"

# ───────────── 토큰 발급 ─────────────
res = requests.post(f"{url_base}/oauth2/tokenP", headers={"content-type": "application/json"},
                    data=json.dumps({"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}))
access_token = res.json().get("access_token", "")
if not access_token:
    print("❌ 액세스 토큰 발급 실패:", res.json())
    sys.exit()
print(f"🔑 액세스 토큰: {access_token}\n")

# ───────────── 공통 API 함수 ─────────────
def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": app_key, "appSecret": app_secret}
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
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
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

# ───────────── 기대수익+손익비+상승확률+최적수량 ─────────────
def lsmc_expected_profit_and_risk_with_prob(stock_code, current_price, total_risk_budget=1_000_000):
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

    max_loss_allowed = total_risk_budget
    optimal_qty = int(max_loss_allowed / expected_loss) if expected_loss > 0 else 0

    return {
        'expected_profit': expected_profit,
        'expected_loss': expected_loss,
        'rr_ratio': rr_ratio,
        'optimal_qty': optimal_qty,
        'prob_up': prob_up
    }

# ───────────── 주문 함수 ─────────────
def send_order(stock_code, price, qty, order_type="매수"):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"
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

# ───────────── 매매 로그 기록 ─────────────
def log_trade(timestamp, stock_code, price, prob_up, exp_profit, exp_loss, rr_ratio, qty, order_type, order_result):
    log_file = Path("trade_log.csv")
    log_entry = {
        "거래시간": timestamp,
        "종목코드": stock_code,
        "현재가": price,
        "상승확률(%)": round(prob_up * 100, 2),
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
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

    try:
        print("📊 GRU 기반 매수 후보 추출 중...")
        top_candidates = predict_today_candidates(engine)
        print(f"✅ 추출된 후보 수: {len(top_candidates)}")

        loop_count = 1
        portfolio = {}

        while True:
            print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            for candidate in top_candidates[:3]:  # 디버깅용 상위 3개만
                stock_code = candidate['code']
                print(f"🔍 종목 코드: {stock_code}")

                price = get_current_price(stock_code)
                if not price:
                    print(f"❌ 현재가 조회 실패: {stock_code}")
                    continue

                result = lsmc_expected_profit_and_risk_with_prob(stock_code, price, total_risk_budget=TOTAL_RISK_BUDGET)

                print(f"💰 현재가: {price} | ProbUp: {result['prob_up']*100:.2f}% | 기대수익: {result['expected_profit']:.2f} | 손익비: {result['rr_ratio']:.2f} | 최적수량: {result['optimal_qty']}")

                if stock_code not in portfolio:
                    if result['optimal_qty'] > 0:
                        order_result = send_order(stock_code, price, qty=result['optimal_qty'], order_type="매수")
                        print(f"✅ 매수 요청 결과: {order_result}")

                        log_trade(
                            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            stock_code=stock_code,
                            price=price,
                            prob_up=result['prob_up'],
                            exp_profit=result['expected_profit'],
                            exp_loss=result['expected_loss'],
                            rr_ratio=result['rr_ratio'],
                            qty=result['optimal_qty'],
                            order_type="매수",
                            order_result=order_result
                        )

                        if order_result.get("rt_cd") == "0":
                            portfolio[stock_code] = {'buy_price': price, 'qty': result['optimal_qty']}
                    else:
                        print(f"🚫 조건 미충족으로 매수 보류: {stock_code}")
                else:
                    buy_price = portfolio[stock_code]['buy_price']
                    if result['expected_profit'] < 300 or price < buy_price * 0.98:
                        order_result = send_order(stock_code, price, qty=portfolio[stock_code]['qty'], order_type="매도")
                        print(f"✅ 매도 요청 결과: {order_result}")

                        log_trade(
                            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            stock_code=stock_code,
                            price=price,
                            prob_up=result['prob_up'],
                            exp_profit=result['expected_profit'],
                            exp_loss=result['expected_loss'],
                            rr_ratio=result['rr_ratio'],
                            qty=portfolio[stock_code]['qty'],
                            order_type="매도",
                            order_result=order_result
                        )

                        if order_result.get("rt_cd") == "0":
                            del portfolio[stock_code]
                    else:
                        print(f"🔒 보유 유지: {stock_code} | 현재가: {price} | 매입가: {buy_price}")

            loop_count += 1
            time.sleep(15)

    except KeyboardInterrupt:
        print("⏹ 자동 매매 종료")
