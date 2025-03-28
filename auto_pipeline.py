from datetime import datetime
from crawler import update_one_year_news  
from sentiment import (
    fetch_news_from_mysql,
    analyze_sentiment,
    update_sentiment_in_mysql,
    calculate_daily_avg_sentiment,  
    save_daily_avg_to_mysql          
)

def run_auto_pipeline():
    print(f"\n {datetime.now().strftime('%H:%M:%S')} | 뉴스 수집 시작")
    update_one_year_news()

    print("감성 분석 시작")
    news_data = fetch_news_from_mysql()
    if news_data:
        analyzed_data = [(analyze_sentiment(title), news_id) for news_id, title in news_data]
        update_sentiment_in_mysql(analyzed_data)
    else:
        print("감성 분석할 뉴스가 없습니다.")

    print("일별 평균 감성 점수 계산 및 저장")
    daily_averages = calculate_daily_avg_sentiment()
    save_daily_avg_to_mysql(daily_averages)

    print("파이프라인 완료")

# 실행
if __name__ == "__main__":
    run_auto_pipeline()