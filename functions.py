import pymysql
import requests
import json
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
from config import DB_CONFIG, ACCOUNT_INFO
import keyring
import os

# 🔹 api key 불러오기
def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '고민수')
    app_secret = keyring.get_password('mock_app_secret', '고민수')
    return app_key, app_secret

# 🔹 접근토큰 발급
def get_access_token(app_key, app_secret):
    url_base = "https://openapivts.koreainvestment.com:29443"
    path = "oauth2/tokenP"
    url = f"{url_base}/{path}"
    
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }

    res = requests.post(url, headers=headers, data=json.dumps(body))
    return res.json().get("access_token", "")

# 🔹 토큰 매니저
def get_auth_info():
    TOKEN_FILE = "access_token.json"
    # 캐시 파일이 있으면 읽기
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            cache = json.load(f)
        if time.time() - cache["timestamp"] < 3600:
            print("🟢 기존 토큰 재사용 중 (파일)")
            return cache["app_key"], cache["app_secret"], cache["token"]

    # 새로 발급
    app_key, app_secret = get_api_keys()
    access_token = get_access_token(app_key, app_secret)
    cache = {
        "token": access_token,
        "timestamp": time.time(),
        "app_key": app_key,
        "app_secret": app_secret
    }

    with open(TOKEN_FILE, "w") as f:
        json.dump(cache, f)
    print("🔄 새로운 토큰 발급 완료")
    return app_key, app_secret, access_token



# 🔹 해시키 생성 함수
def get_hashkey(app_key, app_secret, data):
    url_base = "https://openapivts.koreainvestment.com:29443"
    path = "/uapi/hashkey"
    url = f"{url_base}/{path}"
    
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    
    res = requests.post(url, headers=headers, data=json.dumps(data))
    return res.json().get("HASH", "")

# 🔹 현재 주가 조회 함수
def get_current_price(access_token, app_key, app_secret, stock_code):
    url_base = "https://openapivts.koreainvestment.com:29443"
    path = "/uapi/domestic-stock/v1/quotations/inquire-price"
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

# 🔹 매수, 매도 코드(지정가, 시장가)
def execute_order(stock_code, quantity, order_type, order_style, app_key, app_secret, access_token, price=None): #order_type('매수','매도'), order_style('시장가','지정가')
    url_base = "https://openapivts.koreainvestment.com:29443"
    path = "/uapi/domestic-stock/v1/trading/order-cash"
    url = f"{url_base}{path}"

    # ✅ 주문 방식 설정
    if order_style == "시장가":
        ord_dvsn = "01"
        ord_unpr = "0"
    elif order_style == "지정가" and price is not None:
        ord_dvsn = "00"
        ord_unpr = str(price)
    else:
        raise ValueError("지정가 주문에는 price 값을 명시해야 합니다.")

    # ✅ 매수/매도에 따라 tr_id 설정
    if order_type == "매수":
        tr_id = "VTTC0802U"
    elif order_type == "매도":
        tr_id = "VTTC0801U"
    else:
        raise ValueError("order_type은 '매수' 또는 '매도'여야 합니다.")

    data = {
        "CANO": ACCOUNT_INFO['CANO'],
        "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],
        "PDNO": stock_code,
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(quantity),
        "ORD_UNPR": ord_unpr
    }

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",
        "hashkey": get_hashkey(app_key, app_secret, data)
    }

    res = requests.post(url, headers=headers, data=json.dumps(data))

    try:
        res_json = res.json()
    except ValueError:
        print("❌ JSON 응답 아님. 서버 응답 원문:")
        print(res.text)
        return None

    if res_json.get("rt_cd") == "0":
        output = res_json.get("output")
        if isinstance(output, dict) and "ODNO" in output:
            order_no = output["ODNO"]
            print(f"✅ {order_type} 성공: 주문번호 {order_no}")
            return order_no
        else:
            print(f"❌ {order_type} 실패: 주문번호 없음. output: {output}")
            return None
    else:
        print(f"❌ {order_type} 실패: {res_json.get('msg1')}")
        return None




# 🔹 체결 내역 조회 + 저장 (매수/매도 공용)
def fetch_and_save_trade(order_type, order_no, access_token, app_key, app_secret, db_config, table_name):

    url_base = "https://openapivts.koreainvestment.com:29443"
    path = "/uapi/domestic-stock/v1/trading/inquire-ccn" 
    url = f"{url_base}{path}"

    params = {
        "CANO": ACCOUNT_INFO['CANO'],
        "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],
        "ORD_NO": order_no,
        "INQR_DVSN": "00",  # 전체
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC0434R",  # 체결 조회 (모의투자용)
        "custtype": "P",
        "hashkey": get_hashkey(app_key, app_secret, params)
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res_json = res.json()
    except Exception as e:
        print(f"❌ 체결 조회 실패: JSON 응답 아님 또는 서버 오류\n📦 응답 코드: {res.status_code}\n📦 응답 내용: {res.text[:300]}")
        return

    if res_json.get("rt_cd") != "0":
        print(f"❌ 체결 API 실패: {res_json.get('msg1')}")
        return

    output = res_json.get("output", [])
    if not output:
        print("ℹ️ 체결 내역 없음 (아직 체결 전일 수 있음)")
        return

    saved = False  # ✅ 체결 저장 여부 추적용

    for trade in output:
        try:
            stock_code = trade.get("PDNO")
            quantity = int(trade.get("CNQTY", 0))
            price = int(trade.get("CNPR", 0))
            trade_time_raw = trade.get("CNTM", "")  # HHMMSS

            # 시간 형식이 정확할 때만 변환
            if len(trade_time_raw) == 6:
                trade_time = datetime.now().replace(
                    hour=int(trade_time_raw[:2]),
                    minute=int(trade_time_raw[2:4]),
                    second=int(trade_time_raw[4:6]),
                    microsecond=0
                )
            else:
                trade_time = datetime.now()  # fallback

            save_to_db(
                db_config=db_config,
                trade_table_name=table_name,
                stock_code=stock_code,
                order_type=order_type,
                quantity=quantity,
                price=price,
                trade_time=trade_time
            )
            saved = True  # ✅ 저장 성공 표시

        except Exception as e:
            print(f"❌ 체결 데이터 저장 중 오류: {e}")

    return saved  # ✅ 최소 1건이라도 저장됐으면 True, 아니면 False



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

# 🔹 거래 내역 MySQL 저장 함수
def save_to_db(trade_table_name, stock_code, order_type, quantity, price, trade_time, profit=None, profit_rate=None):
    try:
        conn = pymysql.connect(**DB_CONFIG)
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







