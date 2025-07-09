from functions import get_api_keys, get_access_token, get_hashkey, save_to_db, execute_order, get_auth_info, check_account
from config import ACCOUNT_INFO, DB_CONFIG
import requests, json, time
from datetime import datetime
import time

import pandas as pd
import pymysql

# CSV 파일 불러오기
df = pd.read_csv("stock_data.csv", encoding="utf-8-sig", dtype={"종목코드": str})

conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()

# ✅ 테이블이 없으면 생성
create_table_query = """
CREATE TABLE IF NOT EXISTS stock_recommendations (
    종목코드 VARCHAR(10) PRIMARY KEY,
    상승확률 FLOAT,
    기대수익 INT,
    손익비 FLOAT,
    매수제안 VARCHAR(10)
);
"""
cursor.execute(create_table_query)

# INSERT 쿼리 준비
insert_query = """
INSERT INTO stock_recommendations (종목코드, 상승확률, 기대수익, 손익비, 매수제안)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    상승확률 = VALUES(상승확률),
    기대수익 = VALUES(기대수익),
    손익비 = VALUES(손익비),
    매수제안 = VALUES(매수제안);
"""

# 각 행을 튜플로 변환 후 삽입
data_tuples = list(df.itertuples(index=False, name=None))
cursor.executemany(insert_query, data_tuples)
conn.commit()

print("✅ CSV 데이터 삽입 완료!")

# 연결 종료
cursor.close()
conn.close()





# # 데이터 예제
# data = [
#     {"종목코드": "005930", "상승확률": 0.72, "기대수익": 1200, "손익비": 2.1, "매수제안": "매수"},
#     {"종목코드": "035720", "상승확률": 0.69, "기대수익": 850, "손익비": 1.6, "매수제안": "매수"},
#     {"종목코드": "000660", "상승확률": 0.55, "기대수익": 300, "손익비": 1.2, "매수제안": "제외"},
# ]

# df = pd.DataFrame(data)
# df.to_csv("stock_data.csv", index=False, encoding="utf-8-sig")

# print("CSV 파일 생성 완료: stock_data.csv")


def auto_trading_loop(
    stock_code, interval_sec=60, db_path="news_db", table_name="trade_history"
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

#auto_trading_loop("005930", interval_sec=60)