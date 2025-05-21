
import keyring
import requests
import json
import random
import time
import numpy as np
import pymysql
import sys
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
<<<<<<< HEAD
from config import get_api_keys, ACCOUNT_INFO, DB_CONFIG
import traceback
=======
from config import ACCOUNT_INFO, DB_CONFIG
>>>>>>> fe7303e3c8aeb8b5a065400749fda801371ac62d

app_key = keyring.get_password('mock_app_key', '진상원')
app_secret = keyring.get_password('mock_app_secret', '진상원')
url_base = "https://openapivts.koreainvestment.com:29443"

headers = {"content-type": "application/json"}
path = "oauth2/tokenP"
body = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
res = requests.post(f"{url_base}/{path}", headers=headers, data=json.dumps(body))
token_result = res.json()
access_token = token_result.get('access_token', '')

if not access_token:
    print("[ERROR] 액세스 토큰 발급 실패:", token_result)
    sys.exit()

print(f"[INFO] 발급된 액세스 토큰: {access_token}\n")

error_codes = set()

def fetch_stock_list_from_db():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Code FROM top_stock_price")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0].zfill(6) for row in result] 

stock_list = fetch_stock_list_from_db()
trade_history = {}
cooldown = {}
total_profit = 0
profit_log = []

def save_trade_to_db(stock_code, action, price, quantity):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(20),
            action VARCHAR(10),
            price FLOAT,
            quantity INT,
            timestamp DATETIME
        )
    """)
    cursor.execute("""
        INSERT INTO trade_log (stock_code, action, price, quantity, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (stock_code, action, price, quantity, now))
    conn.commit()
    cursor.close()
    conn.close()

def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": app_key, "appSecret": app_secret}
    res = requests.post(url, headers=headers, data=json.dumps(data))
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
    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res_json = res.json()
        if res_json.get('rt_cd') != '0' or 'output' not in res_json:
            error_codes.add(stock_code)
            return None
        return int(res_json['output']['stck_prpr'])
    except Exception as e:
        error_codes.add(stock_code)
        return None

def send_buy_order(stock_code, quantity):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC3001U"
    }
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0"
    }
    headers["hashkey"] = get_hashkey(data)
    print(f"[DEBUG] 매수요청 데이터: {json.dumps(data)}")
    res = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"[ORDER] 매수 주문 응답: {res.json()}")
    return res.json()

def send_sell_order(stock_code, quantity, price):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC3006U"
    }
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0"
    }
    headers["hashkey"] = get_hashkey(data)
    print(f"[DEBUG] 매도요청 데이터: {json.dumps(data)}")
    res = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"[ORDER] 매도 주문 응답: {res.json()}")
    return res.json()

def load_past_returns(stock_code):
    return np.random.normal(0, 0.02, 100)

def simulate_future_prices(current_price, days=10, paths=100, past_returns=None):
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated_prices[i, j] = price
    return simulated_prices

def lsmc_should_buy(stock_code, current_price, threshold=0.005):
    past_returns = load_past_returns(stock_code)
    simulated = simulate_future_prices(current_price, past_returns=past_returns)
    future_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    X = simulated[:, 0].reshape(-1, 1)
    y = future_profits
    model = LinearRegression().fit(X, y)
    expected_profit = model.predict([[current_price]])[0]
    return expected_profit > threshold

def lsmc_should_sell(stock_code, quantity):
    global total_profit
    current_price = get_current_price(stock_code)
    if current_price is None:
        return False
    past_returns = load_past_returns(stock_code)
    simulated = simulate_future_prices(current_price, past_returns=past_returns)
    future_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    X = simulated[:, 0].reshape(-1, 1)
    y = future_profits
    model = LinearRegression().fit(X, y)
    expected = model.predict([[current_price]])[0]
    buy_price = trade_history.get(stock_code)
    if not buy_price:
        return False
    immediate_profit = (current_price - buy_price) * quantity
    if immediate_profit >= 0 and immediate_profit > expected * quantity * 0.3:
        send_sell_order(stock_code, quantity, current_price)
        total_profit += immediate_profit
        save_trade_to_db(stock_code, "SELL", current_price, quantity)
        print(f"[SELL] {stock_code} 매도 (즉시수익: {immediate_profit:.2f}, 기대수익: {expected * quantity:.2f})")
        del trade_history[stock_code]
        return True
    return False

loop_count = 1
while True:
    print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        for stock_code in stock_list:
            try:
                if stock_code in cooldown and datetime.now() < cooldown[stock_code]:
                    continue
                if stock_code in trade_history or stock_code in error_codes:
                    continue
                quantity = 10
                current_price = get_current_price(stock_code)
                if current_price is None:
                    continue
                if not lsmc_should_buy(stock_code, current_price):
                    continue
                res = send_buy_order(stock_code, quantity)
                if res.get("rt_cd") == "0":
                    trade_history[stock_code] = current_price
                    cooldown[stock_code] = datetime.now() + timedelta(minutes=5)
                    save_trade_to_db(stock_code, "BUY", current_price, quantity)
                    print(f"[BUY] {stock_code} 매수 실행 (가격: {current_price})")
                else:
                    print(f"[FAIL] {stock_code} 매수 실패: {res}")
                time.sleep(1.1)
                if lsmc_should_sell(stock_code, quantity):
                    cooldown[stock_code] = datetime.now() + timedelta(minutes=5)
            except Exception as e:
                print(f"[ERROR] {stock_code} 예외 발생: {e}")
                print(traceback.format_exc())
                error_codes.add(stock_code)
        profit_log.append(total_profit)
        print(f"[INFO] 루프 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 누적 수익: {total_profit:.2f}")
        print("[WAIT] 100초 대기 중...\n")
        loop_count += 1
        time.sleep(100)
    except Exception as e:
        print("[FATAL] 루프 바깥 예외 발생:", e)
        print(traceback.format_exc())
        break
