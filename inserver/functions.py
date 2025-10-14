import pymysql
import requests
import json
import random
import time
from datetime import datetime, date
import matplotlib.pyplot as plt
from config_ko import DB_CONFIG, ACCOUNT_INFO
import keyring
import os, datetime as dt
import pandas as pd
import csv


def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '고민수')
    app_secret = keyring.get_password('mock_app_secret', '고민수')
    print("app_key:", app_key)
    print("app_secret:", app_secret)
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
def get_current_price(app_key, app_secret, access_token, stock_code):
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
def execute_order(app_key, app_secret, access_token, stock_code, quantity, order_type, order_style, price=None): #order_type('매수','매도'), order_style('시장가','지정가')
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
            price = get_current_price(app_key, app_secret, access_token, stock_code)
            order_no = output["ODNO"]
            print(f"✅ {order_type} 성공: 주문번호 {order_no}")
            print(f"코드 : {stock_code} 금액: {price}, 수량: {quantity}, 주문타입: {order_type}, 주문스타일: {order_style}")
            # 로그 데이터 기록
            log_order_to_csv({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "주문번호": order_no,
                "종목코드": stock_code,
                "현재가": price,
                "수량": quantity,
                "주문구분": order_type,
                "주문타입": order_style,
                "상태": "성공"
            })
            update_holdings_on_order(stock_code, quantity, order_type, order_dt=datetime.now())
            return order_no
        else:
            print(f"❌ {order_type} 실패: 주문번호 없음. output: {output}")
            return None
    else:
        print(f"❌ {order_type} 실패: {res_json.get('msg1')}")
        return None


def check_account(app_key, app_secret, access_token):
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

def log_order_to_csv(log_data, log_path="results_3/order_log.csv"):
    # 폴더 없으면 생성
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 파일이 없으면 헤더 추가
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)


BASE_DIR   = "results_3"
CSV_PATH   = os.path.join(BASE_DIR, "account_log.csv")
OUT_EQUITY = os.path.join(BASE_DIR, "총평가금액_추세.png")
OUT_RET    = os.path.join(BASE_DIR, "수익률_추세.png")
os.makedirs(BASE_DIR, exist_ok=True)
START_CAPITAL = 100_000_000  # 초기 투자금
HOLDINGS_CSV = os.path.join(BASE_DIR, "holdings.csv")

def _ensure_holdings_csv():
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.exists(HOLDINGS_CSV):
        pd.DataFrame(columns=["stock_code", "entry_date", "quantity"]).to_csv(
            HOLDINGS_CSV, index=False, encoding="utf-8-sig"
        )

def update_holdings_on_order(stock_code, quantity, order_type, order_dt=None):
    """
    - order_type: '매수' 또는 '매도'
    - FIFO 로 매도 수량 차감
    """
    _ensure_holdings_csv()

    if order_dt is None:
        order_dt = datetime.now().date()
    else:
        order_dt = pd.to_datetime(order_dt).date()

    df = pd.read_csv(HOLDINGS_CSV, encoding="utf-8-sig")

    if order_type == "매수":
        new_row = {
            "stock_code": stock_code,
            "entry_date": order_dt.isoformat(),
            "quantity": int(quantity),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    elif order_type == "매도":
        mask = df["stock_code"] == stock_code
        df_stock = df[mask].copy()
        sell_qty = int(quantity)

        if df_stock.empty:
            print(f"⚠️ 보유 수량 없음: {stock_code}")
        else:
            df_stock["entry_date"] = pd.to_datetime(df_stock["entry_date"])
            df_stock = df_stock.sort_values("entry_date")  # FIFO

            rows_remaining = []
            for _, row in df_stock.iterrows():
                if sell_qty <= 0:
                    rows_remaining.append(row)
                    continue

                hold_qty = int(row["quantity"])
                if hold_qty <= sell_qty:
                    sell_qty -= hold_qty  # 이 lot 전량 매도 -> 삭제
                else:
                    row["quantity"] = hold_qty - sell_qty
                    sell_qty = 0
                    rows_remaining.append(row)

            df_remaining = pd.DataFrame(rows_remaining, columns=df_stock.columns)
            df = pd.concat([df[~mask], df_remaining], ignore_index=True)

            if sell_qty > 0:
                print(f"⚠️ 매도 수량이 보유 수량을 초과했습니다. 초과 {sell_qty}주는 무시되었습니다.")
    else:
        raise ValueError("order_type은 '매수' 또는 '매도'여야 합니다.")

    df.to_csv(HOLDINGS_CSV, index=False, encoding="utf-8-sig")

def log_account(account: dict):
    
    today = date.today().isoformat()
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

def auto_sell_on_horizon(app_key, app_secret, access_token, horizon=6):
    """
    holdings.csv를 확인해서 horizon일 지난 종목은 자동 매도
    """
    _ensure_holdings_csv()
    df = pd.read_csv(HOLDINGS_CSV, encoding="utf-8-sig")
    if df.empty:
        return

    today = datetime.now().date()
    rows_to_sell = []

    for _, row in df.iterrows():
        stock_code = str(row["stock_code"]).zfill(6)
        entry_date = pd.to_datetime(row["entry_date"]).date()
        qty = int(row["quantity"])

        # entry_date + horizon일차가 오늘이거나 지났으면 매도
        if (today - entry_date).days >= horizon:
            rows_to_sell.append((stock_code, qty))

    for stock_code, qty in rows_to_sell:
        price = get_current_price(app_key, app_secret, access_token, stock_code)
        if not price or qty <= 0:
            print(f"⚠️ 매도 스킵: {stock_code}, qty={qty}, price={price}")
            continue

        print(f"⏰ Horizon 만료 매도: {stock_code}, 수량 {qty}, 종가 {price}")
        execute_order(app_key, app_secret, access_token, stock_code, qty, "매도", "시장가")
        time.sleep(1)


def plot_equity_and_return_from_csv():
    # 기간 설정(포함 범위)
    RANGE_START = "2025-09-01"
    RANGE_END   = "2025-10-02"
    
    if not os.path.exists(CSV_PATH):
        print("CSV가 없습니다:", CSV_PATH)
        return

    import matplotlib.dates as mdates

    # 폰트(윈도우 기준)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 1) 데이터 읽기
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    if "날짜" not in df.columns or "총평가금액" not in df.columns:
        raise ValueError("CSV에 '날짜', '총평가금액' 컬럼이 있어야 합니다.")

    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜").drop_duplicates(subset=["날짜"], keep="last")

    # 2) 기간 필터 (포함)
    start = pd.Timestamp(RANGE_START)
    end   = pd.Timestamp(RANGE_END)
    mask = (df["날짜"] >= start) & (df["날짜"] <= end)
    df = df.loc[mask].copy()
    if df.empty:
        print(f"선택된 기간({RANGE_START} ~ {RANGE_END})에 데이터가 없습니다.")
        return

    # 3) 일단위 리샘플(결측일은 직전값으로 채움)
    df = (df.set_index("날짜")
            .resample("D")
            .last()
            .ffill()
            .reset_index())

    # 4) 수익률 파생
    df["수익률_%"]  = (df["총평가금액"] / START_CAPITAL - 1.0) * 100.0
    df["수익률_백만"] = (df["총평가금액"] - START_CAPITAL) / 1_000_000

    # 마지막(오늘) 값
    last_date   = df["날짜"].iloc[-1]
    last_equity = int(df["총평가금액"].iloc[-1])
    last_ret_pct = df["수익률_%"].iloc[-1]
    last_ret_mil = df["수익률_백만"].iloc[-1]

    # X축 포맷러(날짜 간결 표시)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    # ── [A] 총평가금액 ──
    fig = plt.figure(figsize=(11, 6.5))
    fig.suptitle(
        f"총평가금액 추세 ({RANGE_START} ~ {RANGE_END}) | "
        f"마지막 {last_date.date()}  총평가금액 {last_equity:,.0f}원  "
        f"수익률 {last_ret_pct:+.2f}% ({last_ret_mil:+.2f} 백만)",
        fontsize=13, fontweight="bold", y=0.98
    )
    ax = plt.gca()
    ax.plot(df["날짜"], df["총평가금액"], marker="o", linewidth=1.8, label="총평가금액")
    ax.scatter([last_date], [last_equity], s=50)
    ax.text(last_date, last_equity, f" {last_equity:,.0f}원",
            va="bottom", ha="left", fontsize=10)

    ax.grid(True, alpha=0.35)
    ax.set_xlabel("날짜")
    ax.set_ylabel("총평가금액 (원)")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_equity = os.path.join(BASE_DIR, f"총평가금액_추세_{RANGE_START}_{RANGE_END}.png")
    plt.savefig(out_equity, dpi=200)
    plt.close()
    print("저장:", out_equity)

    # ── [B] 수익률(%) ──
    plt.figure(figsize=(11, 6.5))
    plt.title(f"수익률(%) 추세 ({RANGE_START} ~ {RANGE_END})")
    plt.plot(df["날짜"], df["수익률_%"], marker="o", linewidth=1.8, label="수익률(%)")
    plt.scatter([last_date], [last_ret_pct], s=50)
    plt.text(last_date, last_ret_pct, f" {last_ret_pct:+.2f}%",
             va="bottom", ha="left", fontsize=10)

    plt.grid(True, alpha=0.35)
    plt.xlabel("날짜")
    plt.ylabel("수익률 (%)")
    plt.xlim(start, end)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    out_ret = os.path.join(BASE_DIR, f"수익률_추세_{RANGE_START}_{RANGE_END}.png")
    plt.savefig(out_ret, dpi=200)
    plt.close()
    print("저장:", out_ret)



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
    _, res2 = check_account()
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