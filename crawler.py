import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

# today = datetime.now().strftime('%Y-%m-%d')
today = "2025-03-16"

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
        
        date_tag = article.select_one("span.wdate")  # 날짜 태그

        if title_tag and date_tag:
            title = title_tag.text.strip()
            
            
            published_date = date_tag.text.strip()

            news_list.append({
                "title": title,
                "published_date": published_date
            })

    return news_list


stock_news = get_stock_news()

with open("stock_news.json", "w", encoding="utf-8") as f:
    json.dump(stock_news, f, ensure_ascii=False, indent=4)

print(f"{len(stock_news)}개의 주식 뉴스를 저장했습니다.")
