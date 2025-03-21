import keyring
import requests
import json
import random
import time
import matplotlib.pyplot as plt
import pymysql
from datetime import datetime
from config import get_api_keys, ACCOUNT_INFO, DB_CONFIG

# 📌 실시간 그래프
plt.ion()

# 🔐 API 키
app_key = keyring.get_password('mock_app_key', '진상원')
app_secret = keyring.get_password('mock_app_secret', '진상원')
url_base = "https://openapivts.koreainvestment.com:29443"

# 🔑 토큰 발급
headers = {"content-type": "application/json"}
path = "oauth2/tokenP"
body = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
res = requests.post(f"{url_base}/{path}", headers=headers, data=json.dumps(body))
access_token = res.json().get('access_token', '')
print(f"🔑 발급된 액세스 토큰: {access_token}")

# 🔹 종목 리스트
stock_list = ["005930", "000660", "035420", "068270", "028260"]
trade_history = {}
profit_log = []
profit_rate_log = []

# 🔹 해시키 생성
def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    return res.json().get("HASH", "")

# 🔹 현재가 조회
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
    if 'output' not in res.json():
        print(f"⚠️ API 오류: {res.json()}")
        return None
    return int(res.json()['output']['stck_prpr'])

# 🔹 거래 DB 저장
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

# 🔹 랜덤 매매
def random_trade():
    stock_code = random.choice(stock_list)
    order_type = random.choice(["BUY", "SELL"])
    quantity = random.randint(1, 10)

    print(f"🛒 종목: {stock_code}, 주문: {order_type}, 수량: {quantity}주")

    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
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
        "tr_id": "VTTC0802U" if order_type == "BUY" else "VTTC0801U",
        "custtype": "P",
        "hashkey": get_hashkey(data)
    }

    res = requests.post(url, headers=headers, data=json.dumps(data))

    if res.json().get('rt_cd') == '0':
        print(f"✅ {order_type} 주문 성공: {res.json()}")

        if order_type == "BUY":
            current_price = get_current_price(stock_code)
            trade_history[stock_code] = current_price
            save_trade_to_db(stock_code, "BUY", quantity, current_price)

        elif order_type == "SELL" and stock_code in trade_history:
            buy_price = trade_history[stock_code]
            sell_price = get_current_price(stock_code)

            if buy_price and sell_price:
                profit = (sell_price - buy_price) * quantity
                profit_rate = ((sell_price - buy_price) / buy_price) * 100

                profit_log.append(profit)
                profit_rate_log.append(profit_rate)

                save_trade_to_db(stock_code, "SELL", quantity, sell_price, profit, profit_rate)

                print(f"💰 수익: {profit}원, 수익률: {profit_rate:.2f}%")
    else:
        print(f"⚠️ 주문 실패: {res.json()}")

# 🔹 실시간 그래프
def plot_profit():
    plt.clf()
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(profit_log, marker='o', linestyle='-', label="총 수익 (원)")
    plt.xlabel("거래 횟수")
    plt.ylabel("총 수익 (원)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(profit_rate_log, marker='o', linestyle='-', color='r', label="수익률 (%)")
    plt.xlabel("거래 횟수")
    plt.ylabel("수익률 (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.1)

# 🔹 메인 루프
start_time = time.time()
while True:
    try:
        random_trade()

        if (time.time() - start_time) > 60:
            plot_profit()
            start_time = time.time()

        time.sleep(random.randint(5, 10))

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        time.sleep(5)
