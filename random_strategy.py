import pymysql
import requests
import json
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
import config

# 🔹 거래 내역 MySQL 저장 함수
def save_to_db(db_config, trade_table_name, stock_code, order_type, quantity, price, trade_time, profit=None, profit_rate=None):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 테이블이 없으면 생성
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {trade_table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                stock_code VARCHAR(10),
                order_type VARCHAR(10),
                quantity INT,
                price INT,
                trade_time DATETIME,
                profit INT DEFAULT NULL,
                profit_rate FLOAT DEFAULT NULL
            )
        """)
        conn.commit()
        
        sql = """
        INSERT INTO trade_history (stock_code, order_type, quantity, price, trade_time, profit, profit_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (stock_code, order_type, quantity, price, trade_time, profit, profit_rate))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ 거래 내역 저장 완료: {stock_code}, {order_type}, {quantity}주, {price}원, {trade_time}, 수익: {profit}, 수익률: {profit_rate}%")
    except Exception as e:
        print(f"❌ MySQL 저장 오류: {e}")

# 🔹 해시키 생성 함수
def get_hashkey(url_base, app_key, app_secret, data):
    path = "uapi/hashkey"
    url = f"{url_base}/{path}"
    
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    
    res = requests.post(url, headers=headers, data=json.dumps(data))
    return res.json().get("HASH", "")

# 🔹 현재 주가 조회 함수
def get_current_price(url_base, access_token, app_key, app_secret, stock_code):
    path = "uapi/domestic-stock/v1/quotations/inquire-price"
    url = f"{url_base}/{path}"

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

# 🔹 랜덤 매매 함수 (하나의 거래 실행)
def random_trade(url_base, access_token, app_key, app_secret, db_config, trade_table_name, trade_history, profit_log, profit_rate_log):
    # 랜덤 종목 리스트 (한국 주식)
    stock_list = ["005930", "000660", "035420", "068270", "028260"]  # 삼성전자, SK하이닉스, NAVER, 셀트리온, 삼성물산
    stock_code = random.choice(stock_list)  
    order_type = random.choice(["BUY", "SELL"])  
    quantity = random.randint(1, 10)  
    trade_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"🛒 종목: {stock_code}, 주문: {order_type}, 수량: {quantity}주")

    path = "/uapi/domestic-stock/v1/trading/order-cash"
    url = f"{url_base}/{path}"
    
    data = {
        "CANO": config.ACCOUNT_INFO["CANO"],  
        "ACNT_PRDT_CD": config.ACCOUNT_INFO["ACNT_PRDT_CD"],  
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
        "hashkey": get_hashkey(url_base, app_key, app_secret, data)
    }

    res = requests.post(url, headers=headers, data=json.dumps(data))

    if res.json().get('rt_cd') == '0':  
        print(f"✅ {order_type} 주문 성공: {res.json()}")
        
        price = get_current_price(url_base, access_token, app_key, app_secret, stock_code)

        if order_type == "BUY":
            trade_history[stock_code] = price
            save_to_db(db_config, trade_table_name, stock_code, order_type, quantity, price, trade_time)

        elif order_type == "SELL" and stock_code in trade_history:
            buy_price = trade_history[stock_code]
            sell_price = price

            if buy_price and sell_price:
                profit = (sell_price - buy_price) * quantity
                profit_rate = ((sell_price - buy_price) / buy_price) * 100

                profit_log.append(profit)
                profit_rate_log.append(profit_rate)

                print(f"💰 수익: {profit}원, 수익률: {profit_rate:.2f}%")
                
                save_to_db(db_config, trade_table_name, stock_code, order_type, quantity, sell_price, trade_time, profit, profit_rate)

    else:
        print(f"⚠️ 주문 실패: {res.json()}")

# 🔹 수익률 그래프 함수
def plot_profit(profit_log, profit_rate_log):
    plt.clf()
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(profit_log, marker='o', linestyle='-', label="총 수익 (원)")
    plt.xlabel("거래 횟수")
    plt.ylabel("총 수익 (원)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(profit_rate_log, marker='o', linestyle='-', label="수익률 (%)")
    plt.xlabel("거래 횟수")
    plt.ylabel("수익률 (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.1)

# 🔹 랜덤 자동매매 실행 함수
def run_random_trading(url_base, access_token, app_key, app_secret, db_config, trade_table_name):
    # 거래 내역 및 수익 기록 저장용 변수
    trade_history = {}
    profit_log = []
    profit_rate_log = []
    
    start_time = time.time()
    while True:
        try:
            random_trade(url_base, access_token, app_key, app_secret, db_config, trade_table_name, trade_history, profit_log, profit_rate_log)

            # 60초마다 수익률 그래프 업데이트
            if (time.time() - start_time) > 60:
                plot_profit(profit_log, profit_rate_log)
                start_time = time.time()

            time.sleep(random.randint(5, 10))  
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            time.sleep(5)