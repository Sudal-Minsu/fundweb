
import keyring
import requests
import json
import random
import time
import numpy as np
import pandas as pd
import pymysql
import sys
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from config import DB_CONFIG, ACCOUNT_INFO, get_api_keys
from rule_2 import predict_today_candidates  # GRU 기반 후보 추출 함수

# ───────────── 설정 ─────────────
BUY_QUANTITY = 10  # 매수 수량 설정
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

# ───────────── API 함수 ─────────────
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

def simulate_future_prices(current_price, days=10, paths=100, past_returns=None):
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated_prices[i, j] = price
    return simulated_prices

def lsmc_expected_profit(stock_code, current_price):
    prices = get_historical_prices_api(stock_code)
    if prices is None or len(prices) < 20:
        return 0
    returns = np.diff(np.log(prices))
    simulated = simulate_future_prices(current_price, past_returns=returns)
    future_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    X = simulated[:, 0].reshape(-1, 1)
    y = future_profits
    model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    model.fit(X, y)
    return model.predict([[current_price]])[0]

def send_order(stock_code, price, qty, order_type="buy"):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "buy" else "VTTC0801U"
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

# ───────────── 실행 ─────────────
if __name__ == "__main__":
    portfolio = {}
    from sqlalchemy import create_engine
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

    try:
        print("📊 GRU 기반 매수 후보 추출 중...")
        top_candidates = predict_today_candidates(engine)

        loop_count = 1
        while True:
            print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            for candidate in top_candidates:
                stock_code = candidate['code']
                price = get_current_price(stock_code)
                if not price:
                    print(f"❌ 현재가 조회 실패: {stock_code}")
                    time.sleep(5)
                    continue

                expected_profit = lsmc_expected_profit(stock_code, price)
                print(f"[LSMC-MLP] {stock_code} 현재가: {price}, 기대수익: {expected_profit:.2f}")

                if stock_code not in portfolio:
                    if expected_profit > 500:
                        time.sleep(1.2)
                        result = send_order(stock_code, price, qty=BUY_QUANTITY, order_type="buy")
                        print(f"✅ 매수 요청: {stock_code}, 결과: {result}")
                        portfolio[stock_code] = {'buy_price': price, 'qty': BUY_QUANTITY}
                else:
                    buy_price = portfolio[stock_code]['buy_price']
                    if expected_profit < 300 or price < buy_price * 0.98:
                        time.sleep(1.2)
                        result = send_order(stock_code, price, qty=portfolio[stock_code]['qty'], order_type="sell")
                        print(f"✅ 매도 요청: {stock_code}, 결과: {result}")
                        del portfolio[stock_code]

            loop_count += 1
            time.sleep(15)
    except KeyboardInterrupt:
        print("⏹ 자동 매매 종료")
