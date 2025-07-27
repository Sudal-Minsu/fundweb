import pymysql
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

# 오늘 날짜 및 시작 날짜 설정
today = datetime.date.today()
start_date = today - datetime.timedelta(days=365 * 12)

""" MySQL DB 연결 함수 """
def get_connection():
    return pymysql.connect(**config.DB_CONFIG)
           
""" [1] 네이버 테마별 종목코드 최대 n개까지 크롤링 """
def get_codes_from_theme(theme_url):
    res = requests.get(theme_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, 'html.parser')
    codes = []
    for a in soup.select('td.name > div.name_area > a'):
        href = a.get('href', '')
        if '/item/main.naver?code=' in href:
            code = href.split('code=')[-1]
            codes.append(code)
    return codes

def get_top_theme_codes(n=400):
    url = "https://finance.naver.com/sise/theme.naver"
    res = requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, 'html.parser')
    theme_urls = []
    for a in soup.select('td.col_type1 > a'):
        href = a.get('href', '')
        if href.startswith('/sise/sise_group_detail.naver?type=theme'):
            theme_urls.append("https://finance.naver.com" + href)
    result = []
    used = set()
    for idx, theme_url in enumerate(theme_urls):
        try:
            codes = get_codes_from_theme(theme_url)
            for code in codes:
                if code not in used:
                    result.append(code)
                    used.add(code)
                    if len(result) >= n:
                        return result
            time.sleep(0.2)
        except Exception as e:
            pass
    return result

""" [2] 상위 200개 종목 코드 (테마 + FDR 필터 적용) """
def get_top_200_codes():
    all_theme_codes = get_top_theme_codes(n=400)
    valid = []
    for code in tqdm(all_theme_codes, desc="FDR 필터링"):
        try:
            hist = fdr.DataReader(code, start_date, today)
            if hist.empty or len(hist) < 2800 or 'Volume' not in hist.columns:
                continue
            recent = hist.tail(100)
            low_vol = recent['Volume'].isna().sum() + (recent['Volume'] <= 1).sum()
            if low_vol <= 50:
                first_date = hist.index.min().date()
                valid.append((code, first_date))
        except:
            continue

    if not valid:
        print("[경고] 조건 만족 종목 없음.")
        return []

    # 날짜 기준으로 가장 오래된 날짜을 뽑아서
    base_date = min(d for _, d in valid)
    final = [c for c, d in valid if d <= base_date]
    # 가장 먼저 valid에 들어온 순으로 200개
    return final[:200]

""" [3] 종목 데이터 저장 """
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

""" [4] 시장 지수 """
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

""" [5] 환율 저장 """
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

""" [6] 메인 실행: 상위 200개 종목의 데이터 업데이트 """
def run_stock_data():
    conn = get_connection()
    cursor = conn.cursor()
    # 이전 데이터 삭제
    cutoff_date = today - datetime.timedelta(days=365 * 12)
    cursor.execute("DELETE FROM stock_data WHERE Date < %s", (cutoff_date,))
    conn.commit()
    print(f"[삭제 완료] {cutoff_date} 이전 데이터")
    # 상위 200개 종목 추출 (테마 기반)
    top_200_codes = get_top_200_codes()
    if not top_200_codes:
        print("저장할 종목이 없습니다.")
        return
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

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_stock, code) for code in top_200_codes]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="종목 저장 진행 중", ncols=100):
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
