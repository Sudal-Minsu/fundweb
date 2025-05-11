from functions import get_api_keys, get_access_token, get_hashkey
from config import ACCOUNT_INFO
import requests
import json

url_base = "https://openapivts.koreainvestment.com:29443"

app_key, app_secret =get_api_keys()
access_token = get_access_token(url_base, app_key, app_secret)

# 매수
path = "/uapi/domestic-stock/v1/trading/order-cash"
url = f"{url_base}/{path}"
data = {
    "CANO": ACCOUNT_INFO['CANO'],  # 계좌번호 앞 8지리
    "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],  # 계좌번호 뒤 2자리
    "PDNO": "005930",  # 종목코드
    "ORD_DVSN": "01",  # 주문 방법
    "ORD_QTY": "10",  # 주문 수량
    "ORD_UNPR": "0",  # 주문 단가 (시장가의 경우 0)
}

headers = {
    "Content-Type": "application/json",
    "authorization": f"Bearer {access_token}",
    "appKey": app_key,
    "appSecret": app_secret,
    "tr_id": "VTTC0802U",
    "custtype": "P",
    "hashkey": get_hashkey(url_base, app_key, app_secret, data)
}

res = requests.post(url, headers=headers, data=json.dumps(data))
res.json()