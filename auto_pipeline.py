from stock_data import run_stock_data  
import pymysql
from config_choi import DB_CONFIG
from rule_2_predict import run_rule_2_predict


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

    # stock_data 테이블 생성
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
    
    # market_index 테이블 생성 
    create_market_index_table_sql = """
    CREATE TABLE IF NOT EXISTS market_index (
        Date DATE NOT NULL,
        IndexType VARCHAR(20) NOT NULL,  -- 'KOSPI', 'KOSDAQ', 'USD_KRW', 'US_RATE' 
        Close FLOAT,
        PRIMARY KEY (Date, IndexType)
    );
    """
    
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

    print("실시간 매수 후보 예측")
    run_rule_2_predict()
    
    print("파이프라인 완료")

if __name__ == "__main__":
    run_auto_pipeline()