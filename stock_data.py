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
import pandas_datareader.data as web

# 오늘 날짜 및 시작 날짜 설정
today = datetime.date.today()
start_date = today - datetime.timedelta(days=365 * 10)  


""" MySQL DB 연결 함수 """
def get_connection():
    return pymysql.connect(**config.DB_CONFIG)


""" 최대 250개 뽑은 다음 그 중에서 200개 추출 """
def get_top_200_codes():
    
    """ 종목들의 투자지표(ROE, PER, 시가총액)를 크롤링해서 dataframe으로 정리 """
    def get_stock_data(market_type):
        all_data = []
        for page in tqdm(range(1, 50), desc=f"{'코스피' if market_type == 0 else '코스닥'} 종목 크롤링"):
            url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={market_type}&page={page}'
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.select_one('table.type_2')
            html_str = str(table)

            # StringIO를 사용하여 HTML 테이블 문자열을 파일처럼 처리(pandas future warning 방지)
            df = pd.read_html(StringIO(html_str), flavor="bs4")[0]
            # HTML 테이블의 모든 값이 NaN인 행은 제거(레이아웃용 빈 줄 제거, 이후 특정 컬럼이 NaN인 행도 제거함)
            df = df.dropna(how='all')

            # 종목코드 추출
            codes = []
            for a in soup.select('table.type_2 a.tltle'):
                href = a.get('href')
                code = href.split('=')[-1]
                codes.append(code)

            # df의 행 수를 종목코드 개수만큼 자르기(read_html()로 뽑은 테이블의 종목 외의 다른 행 존재 가능성 고려)
            df = df.iloc[:len(codes)]
            df['종목코드'] = codes
            all_data.append(df)
            time.sleep(0.3)

        # 합치고 새로운 연속된 인덱스를 부여
        final_df = pd.concat(all_data, ignore_index=True)
        final_df['시장'] = '코스피' if market_type == 0 else '코스닥'
        return final_df

    kospi_df = get_stock_data(0)
    kosdaq_df = get_stock_data(1)
    df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

    # ROE, PER, 시가총액 컬럼만 선택 
    df = df[['종목코드', '종목명', '시장', 'PER', 'ROE', '시가총액']]
    df = df.replace('-', None).dropna(subset=['PER', 'ROE', '시가총액'])

    # PER, ROE, 시가총액을 숫자로 변환 (실패 시 NaN 처리)
    df['PER'] = pd.to_numeric(df['PER'], errors='coerce')
    df['ROE'] = pd.to_numeric(df['ROE'], errors='coerce')
    df['시가총액'] = df['시가총액'].replace(',', '', regex=True)
    df['시가총액'] = pd.to_numeric(df['시가총액'], errors='coerce')

    df = df.dropna(subset=['PER', 'ROE', '시가총액'])
    df = df[df['PER'] > 0]

    # 1/PER 계산
    df['1/PER'] = 1 / df['PER']

    # 각 항목에 대해 순위 부여 (값이 클수록 좋은 순위니까 내림차순: 최고가 1등)
    df['ROE_rank'] = df['ROE'].rank(ascending=False)
    df['invPER_rank'] = df['1/PER'].rank(ascending=False)
    df['시총_rank'] = df['시가총액'].rank(ascending=False)

    # 가중치 부여: 0.3 * ROE_rank + 0.3 * invPER_rank + 0.4 * 시총_rank 
    # (숫자가 낮을수록 상위. weighted_score가 낮은 순으로 좋음)
    df['weighted_score'] = 0.3 * df['ROE_rank'] + 0.3 * df['invPER_rank'] + 0.4 * df['시총_rank']

    # weighted_score 기준 오름차순 정렬 후, 중복 종목코드를 제거하고 상위 250개 종목 추출
    top_pool = df.sort_values('weighted_score').drop_duplicates(subset='종목코드').head(250)
    top_codes_pool = top_pool['종목코드'].tolist()

    # 최종적으로 조건에 통과한 종목 코드들만 담는 리스트
    valid_codes = []
    print("\n[FDR 데이터 확인 중: 거래량 0 없는 종목 선별]")
    for code in tqdm(top_codes_pool, desc="조건 필터링 중"):
        try:
            df_stock = fdr.DataReader(code, start_date, today)
            # 데이터가 1000일 이상 있어야 함 (10년치면 보통 1250일 이상)
            if len(df_stock) >= 1000:
                # 최근 100일 데이터 추출
                recent_100 = df_stock.tail(100)
                # 최근 100일 중 거래량 결측 또는 1 이하인 날이 50일 초과면 제외
                low_volume_days = recent_100['Volume'].isna().sum() + (recent_100['Volume'] <= 1).sum()
                if low_volume_days <= 50:
                    valid_codes.append(code)
            # 최종 200개 확보 시 중단
            if len(valid_codes) >= 200:
                break  
        except Exception as e:
            continue

    print(f"\n최종 조건 만족 종목 수: {len(valid_codes)}개")
    return valid_codes


""" 종목의 데이터 저장 """
def process_stock(code):
    try:
        df = fdr.DataReader(code, start_date, today)
        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'Close', 'Volume', 'High', 'Low']] 
        df['Code'] = code

        conn = get_connection()
        cursor = conn.cursor()
        for _, row in df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            cursor.execute("""
                INSERT INTO stock_data (Date, Code, Open, Close, Volume, High, Low)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    Open = VALUES(Open),
                    Close = VALUES(Close),
                    Volume = VALUES(Volume),
                    High = VALUES(High),
                    Low = VALUES(Low)
            """, (
                date_str, code,
                row['Open'], row['Close'], row['Volume'],
                row['High'], row['Low']
            ))
        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"[{code}] 처리 중 오류: {e}")


""" 시장 지수 """
def update_market_indices():
    index_map = {
        'KOSPI': '^KS11',
        'KOSDAQ': '^KQ11'
    }

    print("[지수 수집 시작] KOSPI / KOSDAQ")

    for name, ticker in index_map.items():
        try:
            df = fdr.DataReader(ticker, start_date, today)

            df = df.rename_axis("Date").reset_index()
            df = df[['Date', 'Close']]
            df['IndexType'] = name
            
            df = df.dropna(subset=['Date', 'Close'])

            conn = get_connection()
            cursor = conn.cursor()

            for _, row in df.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                close_price = row['Close']
                index_type = row['IndexType']
                cursor.execute("""
                    INSERT INTO market_index (Date, IndexType, Close)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE Close = VALUES(Close)
                """, (date_str, index_type, close_price))

            conn.commit()
            cursor.close()
            conn.close()
            print(f"[{name}] 저장 완료: {len(df)}개")

        except Exception as e:
            print(f"[{name}] 오류 발생: {e}")


""" 거시지표 저장 """
def save_macro_data():
    
    # 1. 환율 (USD/KRW) from FDR
    df_fx = fdr.DataReader("USD/KRW", start_date, today)[['Close']]
    df_fx.rename(columns={'Close': 'USD_KRW'}, inplace=True)

    # 2. 미국 기준금리 (연방기금금리 DFF) from FRED
    df_us_rate = web.DataReader('DFF', 'fred', start_date, today)
    df_us_rate.rename(columns={'DFF': 'US_RATE'}, inplace=True)

    # 3. 날짜 기준 병합
    df = pd.merge(df_fx, df_us_rate, left_index=True, right_index=True, how='outer')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)  

    # 4. 저장
    conn = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        date = row['Date'].date()  # datetime.date로 변환

        # 환율 저장
        if not pd.isna(row['USD_KRW']):
            cursor.execute("""
                INSERT INTO market_index (Date, IndexType, Close)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE Close = VALUES(Close)
            """, (date, 'USD_KRW', row['USD_KRW']))

        # 미국 금리 저장
        if not pd.isna(row['US_RATE']):
            cursor.execute("""
                INSERT INTO market_index (Date, IndexType, Close)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE Close = VALUES(Close)
            """, (date, 'US_RATE', row['US_RATE']))

    conn.commit()
    cursor.close()
    conn.close()
    print("[완료] 환율 및 미국 금리 저장 완료")
    

""" 상위 200개 종목의 데이터 업데이트 """
def run_stock_data():
    conn = get_connection()
    cursor = conn.cursor()

    # 이전 데이터 삭제
    cutoff_date = today - datetime.timedelta(days=365 * 10)
    cursor.execute("DELETE FROM stock_data WHERE Date < %s", (cutoff_date,))
    conn.commit()
    print(f"[삭제 완료] {cutoff_date} 이전 데이터")

    # 상위 200개 종목 추출 
    top_200_codes = get_top_200_codes()

    # 추출한 종목코드들을 %s, %s, %s, ... 형태로 변환
    format_strings = ','.join(['%s'] * len(top_200_codes))
    cursor.execute(f"""
        DELETE FROM stock_data 
        WHERE Code NOT IN ({format_strings})
    """, top_200_codes)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[기존 종목 정리 완료] 상위 200개 외 종목 삭제")

    print(f"[처리 시작] 상위 {len(top_200_codes)}개 종목의 데이터 업데이트 중...")

    # 동시에 최대 8개의 작업(process_stock(code)을 실행
    # submit: 비동기 작업
    with ThreadPoolExecutor(max_workers=8) as executor:
        # future 리스트에 저장
        futures = [executor.submit(process_stock, code) for code in top_200_codes]
        # as_completed(futures): 각 future가 완료되는 순서대로 반환
        for _ in tqdm(as_completed(futures), total=len(futures), desc="종목 저장 진행 중", ncols=100):
            # 진행률만 표시
            pass

    print("전체 데이터 저장 완료")

    print("지수 저장 시작")
    update_market_indices()  
    
    print("거시지표 저장 시작")
    save_macro_data()

    # 최종 종목 수 확인
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT Code) FROM stock_data")
    count = cursor.fetchone()[0]
    print(f"실제 저장된 종목 수: {count}개")
    cursor.close()
    conn.close()


""" 메인 함수 실행 """
if __name__ == "__main__":
    run_stock_data()