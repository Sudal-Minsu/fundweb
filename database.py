import json
import pymysql
import os
from datetime import datetime
from config import DB_CONFIG


def save_json_to_mysql(json_file):
    """JSON 파일 데이터를 MySQL에 저장 (title, published_date만 사용)"""
    if not os.path.exists(json_file):
        print(f"파일을 찾을 수 없습니다: {json_file}")
        return

    # JSON 파일 열기
    with open(json_file, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    # MySQL 연결
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # SQL INSERT문 (중복 데이터 방지)
    sql = """
    INSERT IGNORE INTO news (title, published_date)
    VALUES (%s, %s)
    """

    # JSON 데이터를 MySQL에 삽입
    for news in news_data:
        cursor.execute(sql, (
            news["title"],
            datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
        ))

    # 변경 사항 저장
    conn.commit()
    cursor.close()
    conn.close()

    print(f"MySQL 저장 완료: {len(news_data)}개의 뉴스")

    # JSON 파일 삭제 (선택 사항)
    os.remove(json_file)
    print(f"{json_file} 파일 삭제 완료")

# 실행 예제
today_json = f"stock_news.json"
save_json_to_mysql(today_json)
