import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pymysql
from config import DB_CONFIG
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

# 크롤링 대상 키워드 목록
KEYWORDS = [
    # 금리 및 통화정책 (국내 + 미국)
    "기준금리", "금리 인상", "금리 인하", "한국은행", "통화정책",
    "연방준비제도", "미국 금리", "연준", "파월", "FOMC",

    # 경기 흐름 및 경기지표
    "경기침체", "경기둔화", "경기전망", "경제성장률", "내수부진",
    "제조업 지수", "PMI", "소매판매", "무역수지", "수출입",

    # 물가 및 인플레이션
    "소비자물가", "생산자물가", "물가상승률", "물가안정",
    "인플레이션", "디플레이션", "스태그플레이션",
    "유가 상승", "유가", "공급망", "식료품 가격",

    # 정책 및 재정
    "정부 정책", "기획재정부", "재정정책", "부양책",
    "경기부양", "확장재정", "SOC 투자", "예산안", "정책 발표",
    "세금 인하", "규제 완화", "세수",

    # 소비/대출/소득 관련
    "민간소비", "가계대출", "가계부채", "실질소득", "체감경기",
    "고용률", "실업률",

    # 지정학적 리스크 및 글로벌 변수
    "전쟁", "지정학", "중국 경기", "미중",
    "중동", "원자재", "공급망", "세계경제"
]

today = datetime.now().date()
BASE_URL = "https://finance.naver.com/news/mainnews.naver"
MAX_PAGE = 30  # 최대 페이지 제한


""" 뉴스 크롤링 """
def get_stock_news(date_str, page):
    url = f"{BASE_URL}?date={date_str}&page={page}"
    response = requests.get(url)
    response.encoding = "euc-kr"
    
    # 200: 성공, 빈 리스트 반환
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select("li.block1")

    news_list = []
    for article in articles:
        title_tag = article.select_one("dl > dd > a")
        date_tag = article.select_one("span.wdate")

        if title_tag and date_tag:
            # strip(): 문자열의 필요 없는 부분 제거
            title = title_tag.text.strip()
            published_date = date_tag.text.strip()
            news_list.append({
                "title": title,
                "published_date": published_date
            })

    return news_list


""" 뉴스 중복 체크 """
def is_news_new(conn, news, similarity_threshold=0.85):
    with conn.cursor() as cursor:
        try:
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False

        date_only = pub_date.date()

        # 해당 날짜의 뉴스 제목들만 조회
        sql = "SELECT title FROM news WHERE DATE(published_date) = %s"
        cursor.execute(sql, (date_only,))
        existing_titles = [row[0] for row in cursor.fetchall()]

        # 유사도 85% 이상인 제목이 이미 있다면 중복으로 간주
        for existing_title in existing_titles:
            similarity = SequenceMatcher(None, existing_title, news["title"]).ratio()
            if similarity >= similarity_threshold:
                return False

        return True


""" MySQL에 뉴스 저장 """
def save_news_to_mysql(conn, news_list):
    # 뉴스가 없는 경우
    if not news_list:
        return

    cursor = conn.cursor()
    # 값이 중복될 경우 넘어감(어차피 값이 일정하므로 업데이트할 필요 없음)
    sql = """
    INSERT IGNORE INTO news (title, published_date)
    VALUES (%s, %s)
    """

    for news in news_list:
        try:
            # 문자열을 datetime 객체로 변환해서 저장
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
            cursor.execute(sql, (news["title"], pub_date))
        except:
            continue

    conn.commit()
    cursor.close()


""" 뉴스 크롤링 및 DB 업데이트 """
def crawl_news_for_date(date_str):
    page = 1 
    collected_news = []
    conn = pymysql.connect(**DB_CONFIG)

    while page <= MAX_PAGE and len(collected_news) < 10:
        
        # 뉴스 크롤링
        news_list = get_stock_news(date_str, page)
        if not news_list:
            break
        
        # 키워드 필터링
        for news in news_list:
            if any(keyword in news["title"] for keyword in KEYWORDS):
                if news not in collected_news:
                    collected_news.append(news)
                    if len(collected_news) >= 10:
                        break
        page += 1

    # db에 있는 뉴스는 걸러내고 새로운 뉴스만 저장
    new_news_list = [n for n in collected_news if is_news_new(conn, n)]
    if new_news_list:
        save_news_to_mysql(conn, new_news_list)
    # 새 뉴스가 하나도 없다면 더미 뉴스를 저장(뉴스가 없었던 날로 학습)
    else:
        dummy_news = {"title": "0", "published_date": f"{date_str} 00:00:00"}
        save_news_to_mysql(conn, [dummy_news])

    conn.close()


""" 뉴스 업데이트 """
def update_news():
    today = datetime.now().date()
    start_date = today - timedelta(days=1824)  

    # 1. 전체 날짜 생성
    full_date_list = [start_date + timedelta(days=i) for i in range(1824)]
    full_date_str_list = [d.strftime("%Y-%m-%d") for d in full_date_list]

    # 2. DB에서 현재 존재하는 뉴스 날짜 조회
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT DATE(published_date) FROM news")
            existing_dates = {row[0].strftime("%Y-%m-%d") for row in cursor.fetchall()}

            # 3. 초과된 뉴스 삭제
            delete_cutoff = start_date 
            cursor.execute("DELETE FROM news WHERE DATE(published_date) < %s", (delete_cutoff,))
            conn.commit()
    finally:
        conn.close()

    # 4. 누락된 날짜 필터링
    missing_dates = [d for d in full_date_str_list if d not in existing_dates]

    if not missing_dates:
        print("뉴스 데이터가 모두 채워져 있습니다.")
        return

    print(f"누락된 {len(missing_dates)}일에 대해 뉴스 크롤링을 시작합니다.")

    # 5. 누락 날짜 뉴스 병렬 크롤링
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(crawl_news_for_date, date_str) for date_str in missing_dates]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="뉴스 보완 중"):
            pass

if __name__ == "__main__":
    update_news()