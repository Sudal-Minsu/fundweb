from datetime import datetime
from crawler import update_one_year_news  
from sentiment import (
    fetch_news_from_mysql,
    analyze_sentiment,
    update_sentiment_in_mysql,
    calculate_daily_avg_sentiment,  
    save_daily_avg_to_mysql          
)
from top_stock_price import main as run_top_stock_price  
import pymysql
from config import DB_CONFIG

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

    # 3. top_stock_price 테이블 생성
    create_top_stock_table_sql = """
    CREATE TABLE IF NOT EXISTS top_stock_price (
        Date DATE NOT NULL,
        Code VARCHAR(20) NOT NULL,
        Close FLOAT,
        PRIMARY KEY (Date, Code)
    );
    """

    cursor.execute(create_news_table_sql)
    cursor.execute(create_avg_sentiment_table_sql)
    cursor.execute(create_top_stock_table_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print("테이블 확인 및 생성 완료")

def run_auto_pipeline():
    print(f"\n {datetime.now().strftime('%H:%M:%S')} | 테이블 체크 및 뉴스 수집 시작")
    create_tables_if_not_exist()

    update_one_year_news()

    print("감성 분석 시작")
    news_data = fetch_news_from_mysql()
    if news_data:
        analyzed_data = [(analyze_sentiment(title), news_id) for news_id, title in news_data]
        update_sentiment_in_mysql(analyzed_data)
    else:
        print("감성 분석할 뉴스가 없습니다.")

    print("일별 평균 감성 점수 계산 및 저장")
    daily_averages = calculate_daily_avg_sentiment()
    save_daily_avg_to_mysql(daily_averages)

    print("주식 데이터 수집 시작")  
    run_top_stock_price()  

    print("파이프라인 완료")

if __name__ == "__main__":
    run_auto_pipeline()