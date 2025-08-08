import requests
from config import ACCOUNT_INFO, get_api_keys

app_key, app_secret = get_api_keys()
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6ImJiZGJhNGRlLWRjNTgtNGE3ZC1iODRhLWIwYTZkZjQ0ZTAwZSIsInByZHRfY2QiOiIiLCJpc3MiOiJ1bm9ndyIsImV4cCI6MTc1NDYxNTA1NCwiaWF0IjoxNzU0NTI4NjU0LCJqdGkiOiJQU0ZTYXBneDZqVmJaZG94TnJXeGUxV2tqOE9QNTdLZXZ6akMifQ.TuxoUa7gQMeJG8EoRRRYnNFcMeojyyD-7McYT-In8Isz42-NlgCCK5NAHcHjagt_A2iXFkAHoRRynKckizi7_w"  # 본인 토큰

url_base = "https://openapivts.koreainvestment.com:29443"
url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"

headers = {
    "authorization": f"Bearer {access_token}",
    "appKey": app_key,
    "appSecret": app_secret,
    "tr_id": "VTTC8434R"
}
params = {
    "CANO": str(ACCOUNT_INFO["CANO"]),
    "ACNT_PRDT_CD": str(ACCOUNT_INFO["ACNT_PRDT_CD"]),
    "AFHR_FLPR_YN": "N",
    "OFL_YN": "",
    "INQR_DVSN": "02",
    "UNPR_DVSN": "01",
    "FUND_STTL_ICLD_YN": "N",
    "FNCG_AMT_AUTO_RDPT_YN": "N",
    "PRCS_DVSN": "01"
}

print("headers:", headers)
print("params:", params)

res = requests.get(url, headers=headers, params=params)
print(res.status_code, res.text)
