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





