import keyring
import requests
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import sys
from datetime import datetime
from sklearn.linear_model import LinearRegression
from config import get_api_keys, ACCOUNT_INFO, DB_CONFIG

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
    sys.exit()

print(f"🔑 발급된 액세스 토큰: {access_token}\n")

# ✅ top_stock_price 테이블에서 종목 리스트 불러오기
def fetch_stock_list_from_db():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Code FROM top_stock_price")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in result]

stock_list = fetch_stock_list_from_db()
trade_history = {}
profit_log = [0]
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
    if res.json().get('rt_cd') != '0' or 'output' not in res.json():
        print(f"⚠️ 현재가 API 오류 발생: {res.json()}")
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


# 과거 수익률 불러오기 함수 (CSV 또는 API 사용 가능)

def load_past_returns_from_api(stock_code, start_date="20220101", end_date="20240101"):
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
    if res.status_code != 200 or 'output' not in res.json():
        print(f"❌ 과거 데이터 조회 실패: {res.json()}")
        return None
    df = pd.DataFrame(res.json()['output'])
    df = df[df['stck_clpr'] != '']
    close_prices = df['stck_clpr'].astype(float).values[::-1]  # 날짜순 정렬
    log_returns = np.diff(np.log(close_prices))
    return log_returns

# 시뮬레이션 (GBM 또는 Historical)
def simulate_future_prices(current_price, days=10, paths=100, mu=0.0005, sigma=0.02, kappa=2.0, theta=0.0005, v0=0.0004, lambda_jump=0.1, mu_jump=0.0, sigma_jump=0.02, past_returns=None):
    dt = 1
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        vol = sigma  # 초기 변동성
        for j in range(days):
            # Historical Simulation: 과거 수익률 샘플링이 주어졌다면 그걸 우선 사용
            if past_returns is not None:
                sampled_return = np.random.choice(past_returns)
                price *= np.exp(sampled_return)
            else:
                # Heston 변동성 시뮬레이션
                vol += kappa * (theta - vol) * dt + 0.1 * np.sqrt(vol) * np.random.normal()
                vol = max(vol, 1e-6)

                # Jump-Diffusion
                jump = 0
                if np.random.rand() < lambda_jump:
                    jump = np.random.normal(mu_jump, sigma_jump)

                drift = (mu - 0.5 * vol**2) * dt
                diffusion = vol * np.random.normal()
                price *= np.exp(drift + diffusion + jump)

            simulated_prices[i, j] = price
        

    return simulated_prices

# LSMC 매도 판단
def lsmc_should_sell(stock_code, quantity):
    current_price = get_current_price(stock_code)
    if current_price is None:
        return False
    past_returns = load_past_returns_from_api(stock_code)
    if past_returns is not None:
        print(f"📊 과거 수익률 불러오기 완료: {len(past_returns)}개 데이터")
        print(f"예시 수익률 (상위 5개): {past_returns[:5]}")  # CSV 파일 경로에 맞게 수정
    simulated = simulate_future_prices(current_price, past_returns=past_returns)
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

    return immediate_profit >= 0 and immediate_profit > expected_future_profit * quantity * 0.5

# 랜덤 매수
def random_buy():
    stock_code = random.choice(stock_list)
    position = trade_history[stock_code]
    quantity = random.randint(1, position["quantity"])
    if quantity == 0:
        return
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
        trade_history[stock_code] = {
            "buy_price": current_price,
            "quantity": quantity
        }
        save_trade_to_db(stock_code, "BUY", quantity, current_price)

# LSMC 매도 실행
def trade_with_lsmc():
    if not trade_history:
        return

    stock_code = random.choice(list(trade_history.keys()))
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
            position = trade_history[stock_code]
            buy_price = position["buy_price"]
            profit = (sell_price - buy_price) * quantity
            profit_rate = ((sell_price - buy_price) / buy_price) * 100
            print(f"💰 매도 성공 - 수익: {profit}원, 수익률: {profit_rate:.2f}%")
            profit_log.append(profit)
            profit_rate_log.append(profit_rate)
            save_trade_to_db(stock_code, "SELL", quantity, sell_price, profit, profit_rate)





# 실시간 자동 매매 실행
if __name__ == "__main__":
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    plt.show(block=False)
    start_time = time.time()
    last_buy_time = time.time() - 300

    try:
        while True:
            current_time = time.time()

            # 5분마다 매수
            if current_time - last_buy_time >= 300:
                random_buy()
                last_buy_time = current_time

            # LSMC 기반 매도
            trade_with_lsmc()

            # 수익 그래프 실시간 업데이트
            if len(profit_log) > 1 or profit_rate_log:
                ax1.cla()
                ax2.cla()

                cumulative_profit = np.cumsum(profit_log)
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
                plt.draw()
                plt.pause(1)

            time.sleep(2.5)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        time.sleep(5)

        

