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
import pandas as pd

# 🔹 api key 불러오기
def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '고민수')
    app_secret = keyring.get_password('mock_app_secret', '고민수')
    print("✅ app_key:", app_key)
    print("✅ app_secret:", app_secret)
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
    print("🔐 토큰 응답 전체:", res.json())
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




def check_account(access_token, app_key, app_secret):

    output1 = []
    output2 = []
    CTX_AREA_NK100 = ''
    url_base = "https://openapivts.koreainvestment.com:29443"
    while True:

        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{url_base}/{path}"

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {access_token}",
            "appKey": app_key,
            "appSecret": app_secret,
            "tr_id": "VTTC8434R"
        }

        params = {
            "CANO": ACCOUNT_INFO['CANO'],
            "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],
            "AFHR_FLPR_YN": "N",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": '',
            "CTX_AREA_NK100": CTX_AREA_NK100
        }
        res = requests.get(url, headers=headers, params=params)
        print("📡 응답 상태코드:", res.status_code)
        print("📡 응답 헤더:", res.headers)
        print("📡 응답 본문 일부:", res.text[:500])
        output1.append(pd.DataFrame.from_records(res.json()['output1']))
        
        CTX_AREA_NK100 = res.json()['ctx_area_nk100'].strip()

        if CTX_AREA_NK100 == '':
            output2.append(res.json()['output2'][0])
            break

    if not output1[0].empty:
        res1 = pd.concat(output1)[['pdno', 'hldg_qty', 'pchs_avg_pric']].rename(columns={
            'pdno': '종목코드',
            'hldg_qty': '보유수량',
            'pchs_avg_pric': '매입단가'
        }).reset_index(drop=True)
    else:
        res1 = pd.DataFrame(columns=['종목코드', '보유수량', '매입단가'])

    res2 = output2[0]
    
    return [res1, res2]



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
        
        sql = f"""
        INSERT INTO {trade_table_name} (stock_code, order_type, quantity, price, trade_time, profit, profit_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (stock_code, order_type, quantity, price, trade_time, profit, profit_rate))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ 거래 내역 저장 완료: {stock_code}, {order_type}, {quantity}주, {price}원, {trade_time}, 수익: {profit}, 수익률: {profit_rate}%")
    except Exception as e:
        print(f"❌ MySQL 저장 오류: {e}")


def read_trades_mysql(table_name):
    # DB 연결
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # 거래 로그 조회 쿼리
    query = f"SELECT * FROM {table_name} ORDER BY id DESC"
    cursor.execute(query)
    result = cursor.fetchall()

    # pandas DataFrame 변환
    df = pd.DataFrame(result)

    # 종료
    cursor.close()
    conn.close()
    return df






