import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import pymysql
import FinanceDataReader as fdr
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

# 날짜 설정
today = datetime.date.today()
start_date = today - datetime.timedelta(days=365)

# db 연결 함수
def get_connection():
    return pymysql.connect(**config.DB_CONFIG)

# 1. 테이블 생성
def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    create_sql = """
    CREATE TABLE IF NOT EXISTS top_stock_price (
        Date DATE NOT NULL,
        Code VARCHAR(20) NOT NULL,
        Close FLOAT,
        PRIMARY KEY (Date, Code)
    );
    """
    cursor.execute(create_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print("테이블 'top_stock_price' 확인 또는 생성 완료")

# 2. 상위 200개 종목 추출
def get_top_200_codes():
    def get_stock_data(market_type):
        all_data = []
        for page in tqdm(range(1, 31), desc=f"{'코스피' if market_type == 0 else '코스닥'} 종목 크롤링"):
            url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={market_type}&page={page}'
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.select_one('table.type_2')
            df = pd.read_html(str(table))[0]
            df = df.dropna(how='all')

            codes = []
            for a in soup.select('table.type_2 a.tltle'):
                href = a.get('href')
                code = href.split('=')[-1]
                codes.append(code)

            df = df.iloc[:len(codes)]
            df['종목코드'] = codes
            all_data.append(df)
            time.sleep(0.3)

        final_df = pd.concat(all_data, ignore_index=True)
        final_df['시장'] = '코스피' if market_type == 0 else '코스닥'
        return final_df

    kospi_df = get_stock_data(0)
    kosdaq_df = get_stock_data(1)
    df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

    df = df[['종목코드', '종목명', '시장', 'PER', 'ROE']]
    df = df.replace('-', None).dropna(subset=['PER', 'ROE'])
    df['PER'] = pd.to_numeric(df['PER'], errors='coerce')
    df['ROE'] = pd.to_numeric(df['ROE'], errors='coerce')
    df = df.dropna(subset=['PER', 'ROE'])
    df = df[df['PER'] > 0]

    df['1/PER'] = 1 / df['PER']
    df['ROE_rank'] = df['ROE'].rank(ascending=False)
    df['invPER_rank'] = df['1/PER'].rank(ascending=False)
    df['avg_rank'] = (df['ROE_rank'] + df['invPER_rank']) / 2

    top_200 = df.sort_values('avg_rank').head(200)

    # 중복 제거
    top_200_codes = list(set(top_200['종목코드'].tolist()))[:200]
    print(f"상위 종목코드 추출 완료 (중복 제거 후 {len(top_200_codes)}개)")
    return top_200_codes

# 3. 단일 종목 처리
def process_stock(code):
    try:
        df = fdr.DataReader(code, start_date, today)
        if df.empty:
            return

        df.reset_index(inplace=True)
        df = df[['Date', 'Close']]
        df['Code'] = code

        conn = get_connection()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            close_price = row['Close']
            cursor.execute("DELETE FROM top_stock_price WHERE Code = %s AND Date = %s", (code, date_str))
            cursor.execute("INSERT INTO top_stock_price (Date, Code, Close) VALUES (%s, %s, %s)",
                           (date_str, code, close_price))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"{code} 처리 중 오류: {e}")

# 4. 메인 실행
def main():
    create_table()

    # 오래된 데이터 삭제
    conn = get_connection()
    cursor = conn.cursor()
    cutoff_date = today - datetime.timedelta(days=365)
    cursor.execute("DELETE FROM top_stock_price WHERE Date < %s", (cutoff_date,))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[삭제 완료] {cutoff_date} 이전 데이터")

    # 상위 200 종목 가져오기
    top_200_codes = get_top_200_codes()

    print(f"[처리 시작] 상위 {len(top_200_codes)}개 종목의 데이터 업데이트 중...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_stock, code) for code in top_200_codes]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="종목 저장 진행 중", ncols=100):
            pass

    print("전체 데이터 저장 완료")

# 실행
if __name__ == "__main__":
    main()