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

today = datetime.date.today()
start_date = today - datetime.timedelta(days=365)

def get_connection():
    return pymysql.connect(**config.DB_CONFIG)

""" 최대 250개 뽑은 다음 그 중에서 200개 추출 """

def get_top_200_codes():
    
    """ 종목들의 투자지표를 크롤링해서 dataframe으로 정리 """
    
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
            
            # 종목코드 리스트
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

    df = df[['종목코드', '종목명', '시장', 'PER', 'ROE']]
    df = df.replace('-', None).dropna(subset=['PER', 'ROE'])
    # PER, ROE를 숫자로 변환, 변환 실패 시 NaN으로 처리
    df['PER'] = pd.to_numeric(df['PER'], errors='coerce')
    df['ROE'] = pd.to_numeric(df['ROE'], errors='coerce')
    # 변환하면서 생긴 NaN 제거(이중 확인)
    df = df.dropna(subset=['PER', 'ROE'])
    df = df[df['PER'] > 0]

    df['1/PER'] = 1 / df['PER']
    # 내림차순
    df['ROE_rank'] = df['ROE'].rank(ascending=False)
    df['invPER_rank'] = df['1/PER'].rank(ascending=False)
    df['avg_rank'] = (df['ROE_rank'] + df['invPER_rank']) / 2

    # 종목코드 중복 제거해서 상위 250개 종목 추출
    top_pool = df.sort_values('avg_rank').drop_duplicates(subset='종목코드').head(250)
    top_codes_pool = top_pool['종목코드'].tolist()

    # 최종적으로 조건에 통과한 종목 코드들만 담을 리스트
    valid_codes = []
    print("\n[FDR 데이터 확인 중: 거래량 0 없는 종목 선별]")
    for code in tqdm(top_codes_pool, desc="조건 필터링 중"):
        try:
            df_stock = fdr.DataReader(code, start_date, today)
            # 데이터가 200일 이상 있어야 통과(1년치 데이터는 보통 250일 안팎)
            if len(df_stock) >= 200:
                # 최근 100일간 데이터 따로 추출
                recent_100 = df_stock.tail(100)
                # 최근 100일 중 거래량이 비어있거나 1 이하인 경우 제외(isna(): Nan이면 True, any(): True가 하나라도 있으면 True)
                if not (recent_100['Volume'].isna().any() or (recent_100['Volume'] <= 1).any()):
                    valid_codes.append(code)
            # 200개 확보 시 중단
            if len(valid_codes) >= 200:
                break  
        except:
            continue

    print(f"\n최종 조건 만족 종목 수: {len(valid_codes)}개")
    return valid_codes

""" 종목의 주가 및 거래량 데이터 저장 """

def process_stock(code):
    try:
        df = fdr.DataReader(code, start_date, today)
        # 인덱스 초기화(원래 날짜가 인덱스인데 sql에 넣을 때는 날짜가 일반 컬럼으로 가도록)
        df.reset_index(inplace=True)
        df = df[['Date', 'Close', 'Volume']]
        df['Code'] = code

        conn = get_connection()
        cursor = conn.cursor()
        
        # _: 튜플에서 인덱스 무시(데이터만 사용), itterrows(): dataframe의 각 행을 돌며 튜플 반환
        for _, row in df.iterrows():
            # datetime 형식에서 문자열 형태로 변환(sql에 날짜를 텍스트로 넘김)
            date_str = row['Date'].strftime('%Y-%m-%d')
            close_price = row['Close']
            volume = row['Volume']
            # ON DUPLICATE KEY UPDATE: 중복된 Date, Code가 있을 경우 Close, Volume 업데이트
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

""" 상위 200개 종목의 주가 및 거래량 데이터 업데이트 """
  
def run_top_stock_price():
    conn = get_connection()
    cursor = conn.cursor()

    # 1년 이전 데이터 삭제
    cutoff_date = today - datetime.timedelta(days=365)
    cursor.execute("DELETE FROM top_stock_price WHERE Date < %s", (cutoff_date,))
    conn.commit()
    print(f"[삭제 완료] {cutoff_date} 이전 데이터")

    # 상위 200개 종목 추출
    top_200_codes = get_top_200_codes()

    # 추출한 종목코드들을 %s, %s, %s, ... 형태로 변환
    format_strings = ','.join(['%s'] * len(top_200_codes))
    cursor.execute(f"""
        DELETE FROM top_stock_price 
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

    # 최종 종목 수 확인
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT Code) FROM top_stock_price")
    count = cursor.fetchone()[0]
    print(f"실제 저장된 종목 수: {count}개")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_top_stock_price()
