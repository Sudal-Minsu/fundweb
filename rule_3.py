from functions import get_api_keys, get_access_token, get_hashkey, fetch_and_save_trade, save_to_db, execute_order, get_auth_info
from config import ACCOUNT_INFO, DB_CONFIG
import requests, json, time
from datetime import datetime



# app_key, app_secret, access_token = get_auth_info()

# order_no = execute_order(
#     stock_code="005930",
#     quantity=10,
#     order_type="매수",       
#     order_style="시장가",    
#     app_key=app_key,
#     app_secret=app_secret,
#     access_token=access_token
# )
app_key, app_secret = get_api_keys()
access_token = get_access_token(app_key, app_secret)

order_no = '0000010634'
if order_no:
    print("⏳ 체결 대기 중...")

    for attempt in range(5):  # 최대 5번 시도 (약 15초)
        success = fetch_and_save_trade('매수', order_no, access_token, app_key, app_secret, DB_CONFIG, '돌파매매')
        if success:
            print("✅ 체결 후 저장 완료!")
            break
        else:
            with open("unfilled_orders.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} - 체결 실패: 주문번호 {order_no}\n")
            print("⏱ 아직 체결되지 않음, 다시 시도 중...")
            
            time.sleep(3)
    else:
        print("❌ 체결되지 않아서 저장 실패")
