from datetime import datetime
from crawler import update_news  
from sentiment import (
    fetch_news_from_mysql, # 뉴스 불러오기
    analyze_sentiment, # 감성 분석하기
    update_sentiment_in_mysql, # 감성 점수를 news 테이블에 저장
    calculate_daily_avg_sentiment, # 평균 감성 점수 계산
    save_daily_avg_to_mysql # 평균 감성 점수를 avg_sentiment 테이블에 저장         
)
from stock_data import run_stock_data  
import pymysql
from config import DB_CONFIG


""" 데이터베이스 생성 """
def create_database_if_not_exists():
    db_config_no_db = DB_CONFIG.copy()
    db_config_no_db.pop('database', None)

    conn = pymysql.connect(**db_config_no_db)
    cursor = conn.cursor()
    # 기본 문자 인코딩, 정렬 및 문자열 비교 설정
    cursor.execute("CREATE DATABASE IF NOT EXISTS news_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn.commit()
    cursor.close()
    conn.close()
    print("데이터베이스 확인 및 생성 완료")
    
    
""" 테이블 생성 """
def create_tables_if_not_exist():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 1. news 테이블 생성
    create_news_table_sql = """
    CREATE TABLE IF NOT EXISTS news (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title TEXT NOT NULL,
        published_date DATETIME NOT NULL,
        sentiment_score FLOAT DEFAULT NULL
    );
    """

    # 2. avg_sentiment 테이블 생성
    create_avg_sentiment_table_sql = """
    CREATE TABLE IF NOT EXISTS avg_sentiment (
        date DATE PRIMARY KEY,
        avg_sentiment FLOAT
    );
    """

    # 3. stock_data 테이블 생성
    create_stock_data_table_sql = """
    CREATE TABLE IF NOT EXISTS stock_data (
        Date DATE NOT NULL,
        Code VARCHAR(20) NOT NULL,
        Open FLOAT,  
        Close FLOAT,
        Volume BIGINT,
        High FLOAT,
        Low FLOAT,
        PRIMARY KEY (Date, Code)
    );
    """
    
    # 4. market_index 테이블 생성 
    create_market_index_table_sql = """
    CREATE TABLE IF NOT EXISTS market_index (
        Date DATE NOT NULL,
        IndexType VARCHAR(20) NOT NULL,  -- 'KOSPI', 'KOSDAQ', 'USD_KRW', 'US_RATE' 
        Close FLOAT,
        PRIMARY KEY (Date, IndexType)
    );
    """
    
    cursor.execute(create_news_table_sql)
    cursor.execute(create_avg_sentiment_table_sql)
    cursor.execute(create_stock_data_table_sql)
    cursor.execute(create_market_index_table_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print("테이블 확인 및 생성 완료")


""" 파이프라인 """
def run_auto_pipeline():
    print(f"\n데이터베이스 체크 및 생성")
    create_database_if_not_exists()
    
    print(f"\n테이블 체크 및 생성")
    create_tables_if_not_exist()

    print("주식 데이터 수집 시작")  
    run_stock_data()

    print(f"\n뉴스 수집 시작")
    update_news()

    print("감성 분석 시작")
    news_data = fetch_news_from_mysql()
    if news_data:
        analyzed_data = [
            (0.0 if title.strip() == "0" else analyze_sentiment(title), news_id)
            for news_id, title in news_data
        ]
        update_sentiment_in_mysql(analyzed_data)
    else:
        print("감성 분석할 뉴스가 없습니다.")

    print("일별 업종 평균 감성 점수 계산 및 저장")
    daily_averages = calculate_daily_avg_sentiment()
    save_daily_avg_to_mysql(daily_averages)

    print("파이프라인 완료")

if __name__ == "__main__":
    run_auto_pipeline()