import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pymysql
from config import DB_CONFIG
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 크롤링 대상 키워드 목록
KEYWORDS = [
    "기준금리", "금리 인상", "금리 인하", "한국은행", "통화정책",
    "경기침체", "경기둔화", "경기전망", "경제성장률", "내수부진",
    "소비자물가", "물가상승률", "물가안정", "인플레이션", "디플레이션", "스태그플레이션",
    "정부 정책", "기획재정부", "재정정책", "부양책", "경기부양", "확장재정", "세수 감소", "가계부채",
    "고용률", "실업률", "민간소비", "가계대출", "실질소득", "체감경기"
]

today = datetime.now().date()
BASE_URL = "https://finance.naver.com/news/mainnews.naver"
MAX_PAGE = 30  # 최대 페이지 제한

def get_stock_news(date_str, page):
    url = f"{BASE_URL}?date={date_str}&page={page}"
    response = requests.get(url)
    response.encoding = "euc-kr"

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select("li.block1")

    news_list = []
    for article in articles:
        title_tag = article.select_one("dl > dd > a")
        date_tag = article.select_one("span.wdate")

        if title_tag and date_tag:
            title = title_tag.text.strip()
            published_date = date_tag.text.strip()
            news_list.append({
                "title": title,
                "published_date": published_date
            })

    return news_list

def is_news_new(conn, news):
    with conn.cursor() as cursor:
        try:
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False

        sql = "SELECT COUNT(*) FROM news WHERE title = %s AND published_date = %s"
        cursor.execute(sql, (news["title"], pub_date))
        result = cursor.fetchone()
        return result[0] == 0

def save_news_to_mysql(conn, news_list):
    if not news_list:
        return

    cursor = conn.cursor()
    sql = """
    INSERT IGNORE INTO news (title, published_date)
    VALUES (%s, %s)
    """

    for news in news_list:
        try:
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
            cursor.execute(sql, (news["title"], pub_date))
        except:
            continue

    conn.commit()
    cursor.close()

def crawl_news_for_date(date_str):
    page = 1
    collected_news = []
    conn = pymysql.connect(**DB_CONFIG)

    while page <= MAX_PAGE and len(collected_news) < 10:
        news_list = get_stock_news(date_str, page)
        if not news_list:
            break

        for news in news_list:
            if any(keyword in news["title"] for keyword in KEYWORDS):
                if news not in collected_news:
                    collected_news.append(news)
                    if len(collected_news) >= 10:
                        break
        page += 1

    new_news_list = [n for n in collected_news if is_news_new(conn, n)]
    if new_news_list:
        save_news_to_mysql(conn, new_news_list)
    else:
        dummy_news = {"title": "0", "published_date": f"{date_str} 00:00:00"}
        save_news_to_mysql(conn, [dummy_news])

    conn.close()

def update_one_year_news():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM news")
            count = cursor.fetchone()[0]
    finally:
        conn.close()

    if count == 0:
        # 초기화: 최근 365일 전체를 병렬로 크롤링
        start_date = today - timedelta(days=364)
        date_list = [start_date + timedelta(days=i) for i in range(365)]
        date_str_list = [d.strftime('%Y-%m-%d') for d in date_list]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(crawl_news_for_date, date_str) for date_str in date_str_list]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="크롤링 진행"):
                pass  # 진행률만 표시
    else:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MAX(DATE(published_date)) FROM news")
                max_date = cursor.fetchone()[0]
                cursor.execute("SELECT MIN(DATE(published_date)) FROM news")
                min_date = cursor.fetchone()[0]

            if max_date < today:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM news WHERE DATE(published_date) = %s", (min_date,))
                conn.commit()

                crawl_news_for_date(today.strftime('%Y-%m-%d'))
            else:
                print("DB가 최신 상태입니다.")
        finally:
            conn.close()

if __name__ == "__main__":
    update_one_year_news()