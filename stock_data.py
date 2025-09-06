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
from config_choi import DB_CONFIG  

# 오늘 날짜 및 시작 날짜 설정
today = datetime.date.today()
start_date = today - datetime.timedelta(days=365 * 14)  

""" MySQL DB 연결 함수 """
def get_connection():
    return pymysql.connect(**DB_CONFIG)

# 네이버 금융에서 사용할 필드 IDs 동적 추출 함수
def fetch_field_ids(market_type):
    # 첫 페이지 GET
    url = f'https://finance.naver.com/sise/sise_market_sum.nhn?sosok={market_type}&page=1'
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    subcnt = soup.select_one('div.subcnt_sise_item_top')
    field_ids = [ipt['value'] for ipt in subcnt.select('input[name=fieldIds]')]
    return field_ids

# 각 페이지에서 POST 요청으로 데이터를 가져오는 함수
def fetch_market_page(market_type, page, field_ids):
    data = {
        'menu': 'market_sum',
        'fieldIds': field_ids,
        'returnUrl': f'https://finance.naver.com/sise/sise_market_sum.nhn?sosok={market_type}&page={page}'
    }
    res = requests.post('https://finance.naver.com/sise/field_submit.nhn', data=data)
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.select_one('table.type_2')
    html_str = str(table)
    df = pd.read_html(StringIO(html_str), flavor='bs4')[0]
    df = df.dropna(how='all')
    # 종목코드 추출
    codes = [a['href'].split('=')[-1] for a in soup.select('table.type_2 a.tltle')]
    df = df.iloc[:len(codes)]
    df['종목코드'] = codes
    return df

# 유니버스 추출 함수
# ROE, PER, 시가총액, 영업이익증가율을 모두 활용하며 각각 가중치 적용
def get_top_200_codes():
    all_markets = []
    for market_type in [1]:  # 코스닥만 대상
        # 동적 필드 추출 및 페이지 수 확인
        field_ids = fetch_field_ids(market_type)
        # 전체 페이지 수
        first_page = requests.get(
            f'https://finance.naver.com/sise/sise_market_sum.nhn?sosok={market_type}&page=1',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        first_soup = BeautifulSoup(first_page.text, 'html.parser')
        pgRR = first_soup.select_one('td.pgRR > a')
        total_pages = int(pgRR['href'].split('=')[-1])

        # 병렬 없이 진행 (필요시 ThreadPoolExecutor 적용)
        market_data = []
        for page in tqdm(range(1, total_pages + 1), desc="코스닥 크롤링"):
            try:
                df_page = fetch_market_page(market_type, page, field_ids)
                market_data.append(df_page)
                time.sleep(0.2)
            except Exception:
                continue
        df_market = pd.concat(market_data, ignore_index=True)
        df_market['시장'] = '코스닥'
        all_markets.append(df_market)

    df = pd.concat(all_markets, ignore_index=True)

    # 관심 필드
    cols = ['시가총액', '영업이익증가율']
    # 대체 결측값 처리
    df.replace({'-': None, ',': ''}, regex=True, inplace=True)
    df = df.dropna(subset=cols + ['종목코드'])

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cols)

    # 랭킹
    df['시총_rank'] = df['시가총액'].rank(ascending=False)

    # 가중합 (현재는 시총 랭크만 사용)
    df['weighted_score'] = (df['시총_rank'])

    # 상위 500개 후보
    top_pool = df.sort_values('weighted_score').drop_duplicates('종목코드').head(500)
    top_codes = top_pool['종목코드'].tolist()

    # FDR 필터링
    valid = []
    for code in tqdm(top_codes, desc="FDR 필터링"):
        try:
            hist = fdr.DataReader(code, start_date, today)
            first_date = hist.index.min().date()
            if len(hist) >= 2800:
                recent = hist.tail(100)
                low_vol = recent['Volume'].isna().sum() + (recent['Volume']<=1).sum()
                if low_vol <= 50:
                    valid.append((code, first_date))
        except:
            continue
    if not valid:
        print("[경고] 조건 만족 종목 없음.")
        return []
    base_date = min(d for _, d in valid)
    final = [c for c,d in valid if d <= base_date]
    return final[:200]


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


""" 환율 저장 """
def save_macro_data():
    df_fx = fdr.DataReader("USD/KRW", start_date, today)[['Close']]
    df_fx.rename(columns={'Close': 'USD_KRW'}, inplace=True)
    df_fx.reset_index(inplace=True)
    df_fx.rename(columns={'index': 'Date'}, inplace=True)
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in df_fx.iterrows():
        date = row['Date'].date()
        # 환율 저장
        if not pd.isna(row['USD_KRW']):
            cursor.execute("""
                INSERT INTO market_index (Date, IndexType, Close)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE Close = VALUES(Close)
            """, (date, 'USD_KRW', row['USD_KRW']))
    conn.commit()
    cursor.close()
    conn.close()
    print("[완료] 환율 저장 완료")
    

""" 상위 200개 종목의 데이터 업데이트 """
def run_stock_data():
    conn = get_connection()
    cursor = conn.cursor()

    # 이전 데이터 삭제
    cutoff_date = today - datetime.timedelta(days=365 * 14)
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
    update_market_indices()  # KOSPI/KOSDAQ 둘 다 저장
    
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
