import os
import json
import time
import requests
from config import ACCOUNT_INFO, get_api_keys

# ───────────── 설정 ─────────────
app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"

# ───────────── 토큰 발급 ─────────────
def get_access_token():
    res = requests.post(
        f"{url_base}/oauth2/tokenP",
        headers={"content-type": "application/json"},
        data=json.dumps({
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret
        })
    )
    token = res.json().get("access_token", "")
    if not token:
        print("❌ 액세스 토큰 발급 실패:", res.json())
    return token

# ───────────── 실계좌 잔고 전체 조회 ─────────────
def get_real_balance(access_token):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",  # 모의투자 TR_ID
    }
    params = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200 or 'output1' not in res.json():
        print("❌ 잔고 조회 실패:", res.text)
        return []
    return res.json()['output1']

# ───────────── portfolio.json 재구성 ─────────────
def rebuild_portfolio_json():
    access_token = get_access_token()
    if not access_token:
        print("❌ 액세스 토큰이 없습니다.")
        return
    holdings = get_real_balance(access_token)
    portfolio = {}
    for item in holdings:
        stock_code = item['pdno']
        qty = int(item['hldg_qty'])
        buy_price = int(float(item.get('pchs_avg_pric', 0)))
        if qty > 0:
            portfolio[stock_code] = {
                "buy_price": buy_price,
                "qty": qty
            }
    with open("portfolio.json", "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)
    print("✅ portfolio.json 파일이 실제 계좌 보유현황으로 복구되었습니다!")
    print(json.dumps(portfolio, ensure_ascii=False, indent=2))

# ───────────── 실행 ─────────────
if __name__ == "__main__":
    rebuild_portfolio_json()
