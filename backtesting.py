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
import matplotlib.pyplot as plt

# ───────────── 설정 ─────────────
BUY_QUANTITY = 10  # 매수 수량 설정
app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"
START_CAPITAL = 10_000_000

# ───────────── 토큰 발급 ─────────────
res = requests.post(f"{url_base}/oauth2/tokenP", headers={"content-type": "application/json"},
                    data=json.dumps({"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}))
access_token = res.json().get("access_token", "")
if not access_token:
    print("❌ 액세스 토큰 발급 실패:", res.json())
    sys.exit()
print(f"🔑 액세스 토큰: {access_token}\n")

# ───────────── API 함수 ─────────────
def fetch_stock_list_from_db():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Code FROM top_stock_price")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in result]

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

def backtest_single_stock(stock_code):
    prices = get_historical_prices_api(stock_code)
    if prices is None or len(prices) < 50:
        print(f"❌ 데이터 부족: {stock_code}")
        return
    capital = START_CAPITAL
    portfolio_value = []
    position = None
    for i in range(30, len(prices) - 10):
        today_price = prices[i]
        expected_profit = lsmc_expected_profit(stock_code, today_price)
        if position is None and expected_profit > 500:
            position = {'entry_price': today_price, 'qty': BUY_QUANTITY}
        elif position and i - 10 >= 0:
            capital += (today_price - position['entry_price']) * position['qty']
            position = None
        portfolio_value.append(capital)

    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_value)
    plt.title(f"{stock_code} 백테스트 자산 변화 (MLP 기반)")
    plt.xlabel("시간")
    plt.ylabel("자산")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ───────────── 실행 ─────────────
if __name__ == "__main__":
    stock_code = input("▶ 백테스트할 종목 코드를 입력하세요 (예: 005930): ").strip()
    backtest_single_stock(stock_code)
