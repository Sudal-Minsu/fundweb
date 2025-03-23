import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pymysql
from config import DB_CONFIG

today = datetime.now().strftime('%Y-%m-%d')
#today = "2025-03-21"

BASE_URL = "https://finance.naver.com/news/mainnews.naver"
MAX_PAGE = 30  # 안전을 위한 최대 페이지 제한

def get_stock_news(date, page):
    """특정 날짜 + 페이지의 뉴스 목록 크롤링"""
    url = f"{BASE_URL}?date={date}&page={page}"
    response = requests.get(url)
    response.encoding = "euc-kr"

    if response.status_code != 200:
        print(f"[오류] {page} 페이지 로딩 실패")
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
    """DB에 이미 존재하는 뉴스인지 확인"""
    with conn.cursor() as cursor:
        try:
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False  # 날짜 포맷 문제는 무시

        sql = "SELECT COUNT(*) FROM news WHERE title = %s AND published_date = %s"
        cursor.execute(sql, (news["title"], pub_date))
        result = cursor.fetchone()
        return result[0] == 0


def save_news_to_mysql(conn, news_list):
    """뉴스 리스트를 MySQL에 저장"""
    if not news_list:
        print("저장할 뉴스가 없습니다.")
        return

    cursor = conn.cursor()
    sql = """
    INSERT IGNORE INTO news (title, published_date)
    VALUES (%s, %s)
    """

    saved_count = 0
    for news in news_list:
        try:
            pub_date = datetime.strptime(news["published_date"], "%Y-%m-%d %H:%M:%S")
            cursor.execute(sql, (news["title"], pub_date))
            saved_count += 1
        except Exception as e:
            print(f"[오류] 저장 실패: {news['title']} - {e}")

    conn.commit()
    cursor.close()
    print(f"MySQL 저장 완료: {saved_count}개")


def crawl_all_pages(date):
    """지정한 날짜의 모든 페이지 크롤링"""
    page = 1
    total_saved = 0
    conn = pymysql.connect(**DB_CONFIG)

    while page <= MAX_PAGE:
        news_list = get_stock_news(date, page)
        if not news_list:
            print(f"{page} 페이지: 뉴스 없음 → 종료")
            break

        # 새 뉴스만 추려냄
        new_news_list = [n for n in news_list if is_news_new(conn, n)]
        if not new_news_list:
            print(f"{page} 페이지: 새 뉴스 없음 → 종료")
            break

        save_news_to_mysql(conn, new_news_list)
        total_saved += len(new_news_list)
        page += 1

    conn.close()
    print(f"[완료] {date} 뉴스 총 {total_saved}개 저장")


# 실행
if __name__ == "__main__":
    crawl_all_pages(today)