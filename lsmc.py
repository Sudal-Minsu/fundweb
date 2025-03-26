import keyring
import requests
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pymysql
from datetime import datetime
from sklearn.linear_model import LinearRegression
from config import get_api_keys, ACCOUNT_INFO, DB_CONFIG

# 실시간 그래프
plt.ion()
plt.show()  # 그래프 창 띄우기 (로컬 환경 대응)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

# API 키
app_key = keyring.get_password('mock_app_key', '진상원')
app_secret = keyring.get_password('mock_app_secret', '진상원')
url_base = "https://openapivts.koreainvestment.com:29443"

# 토큰 발급
headers = {"content-type": "application/json"}
path = "oauth2/tokenP"
body = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
res = requests.post(f"{url_base}/{path}", headers=headers, data=json.dumps(body))
token_result = res.json()
access_token = token_result.get('access_token', '')

if not access_token:
    print("❌ 액세스 토큰 발급 실패:", token_result)
    exit()

print(f"🔑 발급된 액세스 토큰: {access_token}")
print(f"\n🔑 발급된 액세스 토큰: {access_token}\n")

# 종목 리스트
stock_list = ["005930", "000660", "035420", "068270", "028260"]
trade_history = {}
profit_log = []
profit_rate_log = []

# 해시키 생성
def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    return res.json().get("HASH", "")

# 현재가 조회
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
    if res.json().get('rt_cd') != '0':
        print(f"⚠️ 현재가 API 오류 발생: {res.json()}")
    if 'output' not in res.json():
        print(f"⚠️ API 오류: {res.json()}")
        return None
    return int(res.json()['output']['stck_prpr'])

# DB 저장
def save_trade_to_db(stock_code, order_type, quantity, price, profit=None, profit_rate=None):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    sql = """
    INSERT INTO trades (stock_code, order_type, quantity, price, profit, profit_rate, traded_at)
    VALUES (%s, %s, %s, %s, %s, %s, NOW())
    """
    cursor.execute(sql, (stock_code, order_type, quantity, price, profit, profit_rate))
    conn.commit()
    cursor.close()
    conn.close()

# 시뮬레이션 (GBM)
def simulate_future_prices(current_price, days=10, paths=100, mu=0.0005, sigma=0.02):
    dt = 1
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal())
            simulated_prices[i, j] = price
    return simulated_prices

# LSMC 매도 판단
def lsmc_should_sell(stock_code, quantity):
    current_price = get_current_price(stock_code)
    if current_price is None:
        return False
    simulated = simulate_future_prices(current_price)
    future_profits = np.maximum(simulated.max(axis=1) - current_price, 0)

    X = simulated[:, 0].reshape(-1, 1)
    y = future_profits
    model = LinearRegression().fit(X, y)
    expected_future_profit = model.predict([[current_price]])[0]

    buy_price = trade_history.get(stock_code)
    if not buy_price:
        return False

    immediate_profit = (current_price - buy_price) * quantity
    print(f"🧠 [LSMC] 즉시수익: {immediate_profit:.2f}, 예상수익: {expected_future_profit * quantity:.2f}")

    return immediate_profit > expected_future_profit * quantity

# 랜덤 매수
def random_buy():
    stock_code = random.choice(stock_list)
    quantity = random.randint(1, 10)
    current_price = get_current_price(stock_code)
    if current_price is None:
        return

    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0",
    }

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC0802U",
        "custtype": "P",
        "hashkey": get_hashkey(data)
    }

    res = requests.post(f"{url_base}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(data))
    if res.json().get('rt_cd') != '0':
        print(f"⚠️ 주문 API 오류 발생: {res.json()}")
    if res.json().get('rt_cd') == '0':
        print(f"✅ 매수 성공: {stock_code} ({quantity}주 @ {current_price})")
        trade_history[stock_code] = current_price
        save_trade_to_db(stock_code, "BUY", quantity, current_price)

# LSMC 매도 실행
def trade_with_lsmc():
    stock_code = random.choice(stock_list)
    if stock_code not in trade_history:
        return
    quantity = random.randint(1, 10)

    if lsmc_should_sell(stock_code, quantity):
        current_price = get_current_price(stock_code)
        if current_price is None:
            return

        data = {
            "CANO": ACCOUNT_INFO["CANO"],
            "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
            "PDNO": stock_code,
            "ORD_DVSN": "01",
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
        }

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {access_token}",
            "appKey": app_key,
            "appSecret": app_secret,
            "tr_id": "VTTC0801U",
            "custtype": "P",
            "hashkey": get_hashkey(data)
        }

        res = requests.post(f"{url_base}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(data))
        if res.json().get('rt_cd') == '0':
            sell_price = current_price
            buy_price = trade_history[stock_code]
            profit = (sell_price - buy_price) * quantity
            profit_rate = ((sell_price - buy_price) / buy_price) * 100
            print(f"💰 매도 성공 - 수익: {profit}원, 수익률: {profit_rate:.2f}%")
            profit_log.append(profit)
            profit_rate_log.append(profit_rate)
            save_trade_to_db(stock_code, "SELL", quantity, sell_price, profit, profit_rate)

# 수익 그래프
def plot_profit():
    if not profit_log or not profit_rate_log:
        return

    cumulative_profit = np.cumsum(profit_log)

    ax1.cla()
    ax2.cla()

    ax1.plot(cumulative_profit, marker='o', linestyle='-', label="누적 수익 (원)")
    ax1.set_xlabel("거래 횟수")
    ax1.set_ylabel("누적 수익 (원)")
    ax1.legend()
    ax1.grid()

    ax2.plot(profit_rate_log, marker='o', linestyle='-', color='r', label="수익률 (%)")
    ax2.set_xlabel("거래 횟수")
    ax2.set_ylabel("수익률 (%)")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.pause(0.1)

# 메인 루프
start_time = time.time()
last_buy_time = time.time() - 300  # 처음 시작 시 바로 매수할 수 있도록

while True:
    try:
        current_time = time.time()
        if current_time - last_buy_time >= 300:  # 5분마다 매수
            random_buy()
            last_buy_time = current_time

        trade_with_lsmc()

        if (current_time - start_time) > 60:
            plot_profit()
            start_time = current_time

        time.sleep(2.5)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        time.sleep(5)
