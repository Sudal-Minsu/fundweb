import pymysql
import requests
import json
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
from config import DB_CONFIG, get_account
import keyring
import os
import pandas as pd

account = get_account("acc4")
ACCOUNT_INFO = account["ACCOUNT_INFO"]

def get_api_keys():
    return account["APP_KEY"], account["APP_SECRET"]

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

def single_trade(stock_code, quantity, order_type="매수", order_style="시장가"):
    app_key, app_secret, access_token = get_auth_info()

    print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 매매 요청 수신")

    # 1. 주문 실행
    order_no = execute_order(
        stock_code=stock_code,
        quantity=quantity,
        order_type=order_type,
        order_style=order_style,
        app_key=app_key,
        app_secret=app_secret,
        access_token=access_token
    )

    time.sleep(3)  # 체결 대기

    # 2. 수익 확인
    _, res2 = check_account(access_token, app_key, app_secret)
    profit = int(res2['asst_icdc_amt'])
    profit_rate = float(res2['asst_icdc_erng_rt']) * 100
    trade_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 3. DB 저장
    save_to_db(
        trade_table_name="trade_history",
        stock_code=stock_code,
        order_type=order_type,
        quantity=quantity,
        price=None,
        trade_time=trade_time,
        profit=profit,
        profit_rate=profit_rate
    )

    return {"status": "success", "order_no": order_no, "profit": profit, "profit_rate": profit_rate}


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

        try:
            data = res.json()
        except Exception:
            print("❌ JSON 파싱 실패:", res.text[:300])
            return None, None

        if data.get("rt_cd") != "0" or "output1" not in data:
            print("❌ API 실패: 토큰이 만료되었거나, 권한 문제가 있습니다.")
            # 캐시된 토큰 삭제
            if os.path.exists("access_token.json"):
                os.remove("access_token.json")
                print("🗑️ 캐시된 토큰 삭제 완료")
            return None, None

        output1.append(pd.DataFrame.from_records(data['output1']))
        CTX_AREA_NK100 = data.get('ctx_area_nk100', '').strip()

        if CTX_AREA_NK100 == '':
            output2.append(data.get('output2', [{}])[0])
            break

    if output1 and not output1[0].empty:
        res1 = pd.concat(output1)[['pdno', 'hldg_qty', 'pchs_avg_pric']].rename(columns={
            'pdno': '종목코드',
            'hldg_qty': '보유수량',
            'pchs_avg_pric': '매입단가'
        }).reset_index(drop=True)
    else:
        res1 = pd.DataFrame(columns=['종목코드', '보유수량', '매입단가'])

    res2 = output2[0] if output2 else {}

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


def log_account(account: dict):
    import os, csv, datetime

    BASE_DIR = os.path.join("data", "results")
    CSV_PATH = os.path.join(BASE_DIR, "account_log.csv")
    os.makedirs(BASE_DIR, exist_ok=True)

    START_CAPITAL = 100_000_000  # 초기 투자금
    today = datetime.date.today().isoformat()
    tot_evlu_amt = int(account['tot_evlu_amt'])              # 총평가금액
    tot_ret = (tot_evlu_amt - START_CAPITAL) / 1_000_000     # 총수익률 (백만 단위 증감)
    pchs_amt = int(account['pchs_amt_smtl_amt'])             # 매수금액
    cur_evlu_amt = int(account['evlu_amt_smtl_amt'])         # 현재 평가금액
    pnl = int(account['evlu_pfls_smtl_amt'])                 # 평가 손익
    cash = int(account['tot_evlu_amt']) - int(account['scts_evlu_amt'])  # 예수금

    row = [today, tot_evlu_amt, tot_ret, pchs_amt, cur_evlu_amt, pnl, cash]

    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["날짜","총평가금액","총수익률","매수금액","현재평가금액","평가손익","예수금"])
        writer.writerow(row)

    print(f"계좌 정보 저장 완료: {row}")




