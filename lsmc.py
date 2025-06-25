
import keyring
import requests
import json
import random
import time
import numpy as np
<<<<<<< HEAD
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
=======
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
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
    sys.exit()
print(f"🔑 액세스 토큰: {access_token}\n")

<<<<<<< HEAD
# ───────────── API 함수 ─────────────
=======
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

>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
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
<<<<<<< HEAD
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200 or 'output' not in res.json():
=======
    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res_json = res.json()
        if res_json.get('rt_cd') != '0' or 'output' not in res_json:
            error_codes.add(stock_code)
            return None
        return int(res_json['output']['stck_prpr'])
    except Exception as e:
        error_codes.add(stock_code)
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
        return None

<<<<<<< HEAD
def get_historical_prices_api(stock_code, start_date="20220101", end_date="20240101"):
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
=======
def send_buy_order(stock_code, quantity):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC3001U"
    }
<<<<<<< HEAD
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
=======
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
<<<<<<< HEAD
        "ORD_DVSN": "00",
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(price)
    }
    hashkey = get_hashkey(data)
=======
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
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
<<<<<<< HEAD
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
=======
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
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633

loop_count = 1
while True:
    print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
<<<<<<< HEAD
        while True:
            print("📊 GRU 기반 매수 후보 추출 중...")
            top_candidates = predict_today_candidates(engine)

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

            time.sleep(5)
    except KeyboardInterrupt:
        print("⏹ 자동 매매 종료")
=======
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
>>>>>>> 9f47a3108bff877f8858617419c3bb52e429a633
