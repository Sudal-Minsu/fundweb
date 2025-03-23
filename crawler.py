import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pymysql
from config import DB_CONFIG

today = datetime.now().strftime('%Y-%m-%d')
# today = "2025-03-16"

NAVER_FINANCE_NEWS_URL = f"https://finance.naver.com/news/mainnews.naver?date={today}"

def get_stock_news():
    """네이버 금융에서 주식 뉴스를 크롤링"""
    response = requests.get(NAVER_FINANCE_NEWS_URL)
    response.encoding = "euc-kr"  # 한글 깨짐 방지

    if response.status_code != 200:
        print("페이지를 가져오지 못했습니다.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    news_list = []

    articles = soup.select("li.block1")  # 뉴스 리스트 태그 구조 확인 필수

    for article in articles:
        title_tag = article.select_one("dl > dd > a")  # 뉴스 제목 태그
        date_tag = article.select_one("span.wdate")    # 날짜 태그

        if title_tag and date_tag:
            title = title_tag.text.strip()
            published_date = date_tag.text.strip()

            news_list.append({
                "title": title,
                "published_date": published_date
            })

    return news_list


def save_news_to_mysql(news_list):
    """뉴스 리스트를 MySQL에 저장"""
    if not news_list:
        print("저장할 뉴스가 없습니다.")
        return

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    sql = """
    INSERT IGNORE INTO news (title, published_date)
    VALUES (%s, %s)
    """

    for news in news_list:
        try:
            # 날짜 포맷 변환 시도
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 포맷이 다를 경우 예외 처리
            print(f"[오류] 날짜 포맷 문제: {news['published_date']}")
            continue

        cursor.execute(sql, (news["title"], pub_date))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"MySQL 저장 완료: {len(news_list)}개의 뉴스")


# 실행
stock_news = get_stock_news()
save_news_to_mysql(stock_news)