# pip install html5lib finance-datareader
import pymysql
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import config

# 오늘 날짜 기준
today = datetime.date.today()
start_date = today - datetime.timedelta(days=365)

def get_connection():
    return pymysql.connect(**config.DB_CONFIG)

# 최대 250개 추출해서 200개 통과 시점까지 처리
def get_top_200_codes():
    def get_stock_data(market_type):
        all_data = []
        for page in tqdm(range(1, 50), desc=f"{'코스피' if market_type == 0 else '코스닥'} 종목 크롤링"):
            url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={market_type}&page={page}'
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.select_one('table.type_2')
            html_str = str(table)
            df = pd.read_html(StringIO(html_str), flavor="bs4")[0]
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

    # 최대 250개 추출 후 조건 통과한 200개만 최종 선정
    top_pool = df.sort_values('avg_rank').drop_duplicates(subset='종목코드').head(250)
    top_codes_pool = top_pool['종목코드'].tolist()

    valid_codes = []
    print("\n[FDR 데이터 확인 중: 거래량 0 없는 종목 선별]")
    for code in tqdm(top_codes_pool, desc="조건 필터링 중"):
        try:
            df_stock = fdr.DataReader(code, start_date, today)
            if len(df_stock) >= 200:
                recent_100 = df_stock.tail(100)
                if not (recent_100['Volume'].isna().any() or (recent_100['Volume'] <= 1).any()):
                    valid_codes.append(code)
            if len(valid_codes) >= 200:
                break  # 200개 확보 시 중단
        except:
            continue

    print(f"\n최종 조건 만족 종목 수: {len(valid_codes)}개")
    return valid_codes

# 상위 200개 종목의 주가 및 거래량 데이터 저장
def process_stock(code):
    try:
        df = fdr.DataReader(code, start_date, today)

        df.reset_index(inplace=True)
        df = df[['Date', 'Close', 'Volume']]
        df['Code'] = code

        conn = get_connection()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            close_price = row['Close']
            volume = row['Volume']
            cursor.execute("""
                INSERT INTO top_stock_price (Date, Code, Close, Volume)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE Close = VALUES(Close), Volume = VALUES(Volume)
            """, (date_str, code, close_price, volume))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"[{code}] 처리 중 오류: {e}")
        
def main():
    conn = get_connection()
    cursor = conn.cursor()

    # 1년 이전 데이터 삭제
    cutoff_date = today - datetime.timedelta(days=365)
    cursor.execute("DELETE FROM top_stock_price WHERE Date < %s", (cutoff_date,))
    conn.commit()
    print(f"[삭제 완료] {cutoff_date} 이전 데이터")

    # 상위 200개 종목 추출
    top_200_codes = get_top_200_codes()

    # 기존 종목 중 top_200_codes 에 포함되지 않는 것 삭제
    format_strings = ','.join(['%s'] * len(top_200_codes))
    cursor.execute(f"""
        DELETE FROM top_stock_price 
        WHERE Code NOT IN ({format_strings})
    """, top_200_codes)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[기존 종목 정리 완료] 상위 200개 외 종목 삭제")

    # 데이터 저장
    print(f"[처리 시작] 상위 {len(top_200_codes)}개 종목의 데이터 업데이트 중...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_stock, code) for code in top_200_codes]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="종목 저장 진행 중", ncols=100):
            pass

    print("전체 데이터 저장 완료")

    # 최종 종목 수 확인
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT Code) FROM top_stock_price")
    count = cursor.fetchone()[0]
    print(f"실제 저장된 종목 수: {count}개")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
