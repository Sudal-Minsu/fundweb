import keyring
import requests
import json
import random
import time
import matplotlib.pyplot as plt

# 📌 Matplotlib 인터랙티브 모드 활성화 (실시간 업데이트)
plt.ion()

# 🔹 API Key 설정 (모의계좌)
app_key = keyring.get_password('mock_app_key', '진상원')
app_secret = keyring.get_password('mock_app_secret', '진상원')

# 🔹 모의투자 API 기본 URL
url_base = "https://openapivts.koreainvestment.com:29443"

# 🔹 접근토큰 발급
headers = {"content-type": "application/json"}
path = "oauth2/tokenP"
body = {
    "grant_type": "client_credentials",
    "appkey": app_key,
    "appsecret": app_secret
}
url = f"{url_base}/{path}"

res = requests.post(url, headers=headers, data=json.dumps(body))
access_token = res.json().get('access_token', '')
print(f"🔑 발급된 액세스 토큰: {access_token}")

# 🔹 랜덤 종목 리스트 (한국 주식)
stock_list = ["005930", "000660", "035420", "068270", "028260"]  # 삼성전자, SK하이닉스, NAVER, 셀트리온, 삼성물산

# 🔹 거래 내역 저장
trade_history = {}
profit_log = []
profit_rate_log = []

# 🔹 해시키 생성 함수
def get_hashkey(data):
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
def get_current_price(stock_code):
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
    
    # 🔹 API 응답 오류 처리
    if 'output' not in res.json():
        print(f"⚠️ API 오류: {res.json()}")
        return None
    
    return int(res.json()['output']['stck_prpr'])  # 현재 주가 반환

# 🔹 랜덤 매매 함수
def random_trade():
    stock_code = random.choice(stock_list)  # 🔹 랜덤 종목 선택
    order_type = random.choice(["BUY", "SELL"])  # 🔹 랜덤으로 매수/매도 선택
    quantity = random.randint(1, 10)  # 🔹 랜덤 주문 수량 (1~10주)
    
    print(f"🛒 종목: {stock_code}, 주문: {order_type}, 수량: {quantity}주")

    path = "/uapi/domestic-stock/v1/trading/order-cash"
    url = f"{url_base}/{path}"
    
    data = {
        "CANO": "50127965",  # 계좌번호 앞 8자리
        "ACNT_PRDT_CD": "01",  # 계좌번호 뒤 2자리
        "PDNO": stock_code,  # 종목코드
        "ORD_DVSN": "01",  # 주문 방법 (시장가 주문)
        "ORD_QTY": str(quantity),  # 주문 수량
        "ORD_UNPR": "0",  # 시장가 주문 (0)
    }

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC0802U" if order_type == "BUY" else "VTTC0801U",  # 🔹 매수/매도 tr_id 설정
        "custtype": "P",
        "hashkey": get_hashkey(data)
    }

    res = requests.post(url, headers=headers, data=json.dumps(data))

    if res.json().get('rt_cd') == '0':  # 정상 주문 응답 코드
        print(f"✅ {order_type} 주문 성공: {res.json()}")
        
        # 🔹 매수 시, 구매 가격 저장
        if order_type == "BUY":
            trade_history[stock_code] = get_current_price(stock_code)

        # 🔹 매도 시, 수익 계산
        elif order_type == "SELL" and stock_code in trade_history:
            buy_price = trade_history[stock_code]
            sell_price = get_current_price(stock_code)

            if buy_price and sell_price:
                profit = (sell_price - buy_price) * quantity
                profit_rate = ((sell_price - buy_price) / buy_price) * 100

                profit_log.append(profit)
                profit_rate_log.append(profit_rate)

                print(f"💰 수익: {profit}원, 수익률: {profit_rate:.2f}%")
    
    else:
        print(f"⚠️ 주문 실패: {res.json()}")

# 🔹 실시간 수익률 그래프 함수
def plot_profit():
    plt.clf()
    plt.figure(figsize=(10, 5))

    # 🔹 총 수익 (원) 그래프
    plt.subplot(2, 1, 1)
    plt.plot(profit_log, marker='o', linestyle='-', color='b', label="총 수익 (원)")
    plt.xlabel("거래 횟수")
    plt.ylabel("총 수익 (원)")
    plt.legend()
    plt.grid()

    # 🔹 수익률 (%) 그래프
    plt.subplot(2, 1, 2)
    plt.plot(profit_rate_log, marker='o', linestyle='-', color='r', label="수익률 (%)")
    plt.xlabel("거래 횟수")
    plt.ylabel("수익률 (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.1)

# 🔹 랜덤 매매 실행 (실시간 그래프 업데이트)
start_time = time.time()
while True:
    try:
        random_trade()  # 🔹 랜덤으로 매매 실행

        # 🔹 1분마다 그래프 업데이트
        if (time.time() - start_time) > 20:
            plot_profit()
            start_time = time.time()

        time.sleep(random.randint(5, 10))  # 🔹 5~10초 대기 후 다음 거래 실행

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        time.sleep(5)  # 5초 후 재시도
