from datetime import datetime
from crawler import crawl_all_pages
from sentiment import fetch_news_from_mysql, analyze_sentiment, update_sentiment_in_mysql
import time

def run_auto_pipeline():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n🕒 {datetime.now().strftime('%H:%M:%S')} | 📡 뉴스 수집 시작")
    crawl_all_pages(today)

    print("🧠 감성 분석 시작")
    news_data = fetch_news_from_mysql()
    if news_data:
        analyzed_data = [(analyze_sentiment(title), news_id) for news_id, title in news_data]
        update_sentiment_in_mysql(analyzed_data)
    else:
        print("감성 분석할 뉴스 없음")

    print("✅ 파이프라인 완료")

# 실행
if __name__ == "__main__":
    run_auto_pipeline()     # 시작 시 1회 실행