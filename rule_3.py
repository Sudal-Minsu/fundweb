from functions import get_api_keys, get_access_token, get_hashkey, save_to_db, execute_order, get_auth_info, check_account
from config import ACCOUNT_INFO, DB_CONFIG
import requests, json, time
from datetime import datetime



import time
from datetime import datetime

def auto_trading_loop(
    stock_code, interval_sec=60, db_path="trading_db", table_name="trade_history"
):
    app_key, app_secret, access_token = get_auth_info()

    while True:
        print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 매매 시도")

        # 1. 주문 실행
        quantity = 10
        order_type = "매수"
        order_style = "시장가"
        price = None  # 시장가

        order_no = execute_order(
            stock_code=stock_code,
            quantity=quantity,
            order_type=order_type,
            order_style=order_style,
            app_key=app_key,
            app_secret=app_secret,
            access_token=access_token
        )

        time.sleep(3)  # 체결 대기 시간 (시장가면 빠르게 가능)

        # 2. 잔고 및 수익 확인
        res1, res2 = check_account(access_token, app_key, app_secret)


        profit = int(res2['asst_icdc_amt'])
        profit_rate = float(res2['asst_icdc_erng_rt']) * 100
        trade_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 3. DB 저장
        save_to_db(
            trade_table_name=table_name,
            stock_code=stock_code,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trade_time=trade_time,
            profit=profit,
            profit_rate=profit_rate
        )

        # 4. 다음 매매까지 대기
        time.sleep(interval_sec)
