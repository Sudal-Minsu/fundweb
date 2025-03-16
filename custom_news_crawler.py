import requests as rq
from bs4 import BeautifulSoup
import pymysql
import datetime
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ KR-FinBERT 모델 로드
MODEL_NAME = "snunlp/KR-FinBERT-SC"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# ✅ 원하는 기업 리스트 (예시)
selected_companies = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("035420", "NAVER"),
    ("207940", "삼성바이오로직스"),
    ("051910", "LG화학"),
    ("068270", "셀트리온"),
    ("005380", "현대차"),
    ("035720", "카카오"),
    ("105560", "KB금융"),
    ("055550", "신한지주"),
]

# ✅ MySQL 연결 정보
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "news_db",
    "charset": "utf8mb4",
}

# ✅ 새 테이블 이름
NEW_TABLE_NAME = "custom_stock_news"

# ✅ 날짜 리스트 생성 (최근 20일)
date_range = [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y.%m.%d.") for i in range(20)]


# ✅ 네이버 뉴스 일별 5개 크롤링
def get_news(company_name):
    base_url = "https://search.naver.com/search.naver"
    headers = {"User-Agent": "Mozilla/5.0"}
    news_data = []

    for search_date in date_range:  # ✅ 최근 20일 동안 반복
        params = {
            "where": "news",
            "query": company_name,
            "sm": "tab_opt",
            "ds": search_date,  # ✅ 시작 날짜
            "de": search_date,  # ✅ 종료 날짜
            "sort": 1  # 최신순 정렬
        }

        response = rq.get(base_url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("ul.list_news > li")

        daily_count = 0  # ✅ 하루에 5개씩 제한

        for article in articles:
            if daily_count >= 5:  # ✅ 하루 5개 제한
                break

            try:
                title_tag = article.select_one("a.news_tit")
                if title_tag is None:
                    continue  # ✅ 뉴스 제목이 없는 경우 건너뜀

                title = title_tag.text
                news_datetime = datetime.datetime.strptime(search_date, "%Y.%m.%d.")

                news_data.append((company_name, title, news_datetime))
                daily_count += 1
            except:
                continue
        
        time.sleep(0.5)  # ✅ 크롤링 속도 조절

    if len(news_data) < 100:
        print(f"⚠️ {company_name} 뉴스 부족 ({len(news_data)}개)")

    return news_data

# ✅ 감성 분석 함수
def analyze_sentiment(text):
    """감성 분석 수행 (softmax 적용, 긍정 - 부정)"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = (probabilities[0][2] - probabilities[0][0]).item()  # ✅ 긍정 확률 - 부정 확률
    return round(sentiment_score, 4)

# ✅ MySQL에 데이터 저장 (새로운 테이블 사용)
def save_to_mysql(news_list, company):
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()

    # ✅ 새로운 테이블 생성 (없으면 생성)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {NEW_TABLE_NAME} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            company VARCHAR(255),
            title TEXT,
            news_datetime DATE,
            sentiment_score FLOAT DEFAULT NULL
        )
    """)

    for _, title, news_datetime in news_list:
        # ✅ 중복 확인 (같은 회사 + 같은 날짜 + 같은 제목이 있는지)
        cursor.execute(f"SELECT COUNT(*) FROM {NEW_TABLE_NAME} WHERE company = %s AND title = %s AND news_datetime = %s", 
                       (company, title, news_datetime))
        if cursor.fetchone()[0] == 0:  # 중복이 없을 경우만 INSERT
            sentiment_score = analyze_sentiment(title)
            cursor.execute(f"""
                INSERT INTO {NEW_TABLE_NAME} (company, title, news_datetime, sentiment_score) 
                VALUES (%s, %s, %s, %s)
            """, (company, title, news_datetime, sentiment_score))
    
    # ✅ 최신 뉴스 100개만 유지 (일별 5개 × 20일)
    cursor.execute(f"""
        DELETE FROM {NEW_TABLE_NAME} 
        WHERE company = %s 
        AND id NOT IN (
            SELECT id FROM (
                SELECT id FROM {NEW_TABLE_NAME} WHERE company = %s ORDER BY news_datetime DESC LIMIT 100
            ) AS subquery
        )
    """, (company, company))

    conn.commit()
    conn.close()

# ✅ 기존 뉴스에 감성 점수 업데이트 (감성 분석이 안된 뉴스만)
def update_sentiment_in_mysql():
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()

    # ✅ 감성 분석이 되지 않은 뉴스만 가져오기
    cursor.execute(f"SELECT id, title FROM {NEW_TABLE_NAME} WHERE sentiment_score IS NULL;")
    news_data = cursor.fetchall()

    if not news_data:
        print("✅ 감성 분석할 뉴스가 없습니다.")
        return

    print(f"🔍 {len(news_data)}개 뉴스 감성 분석 중...")
    
    analyzed_data = [(analyze_sentiment(title), news_id) for news_id, title in news_data]

    # ✅ MySQL 업데이트 실행
    update_sql = f"UPDATE {NEW_TABLE_NAME} SET sentiment_score = %s WHERE id = %s;"
    cursor.executemany(update_sql, analyzed_data)
    conn.commit()

    print(f"✅ MySQL 감성 점수 업데이트 완료: {len(analyzed_data)}개 항목")

    cursor.close()
    conn.close()
    
# ✅ 전체 실행
def main():
    print(f"총 {len(selected_companies)}개 종목 크롤링 시작!")

    for idx, (code, name) in enumerate(selected_companies):
        print(f"크롤링 중: {name} ({idx+1}/{len(selected_companies)})")
        news = get_news(name)
        if news:
            save_to_mysql(news, name)
        time.sleep(1)  # 네이버 크롤링 차단 방지

if __name__ == "__main__":
    main()