import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pymysql
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DB_CONFIG

# 모델 불러오기
MODEL_NAME = "snunlp/KR-FinBERT-SC"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 1. 뉴스 불러오기
def fetch_news_from_mysql():
    """MySQL에서 감성 분석이 필요한 뉴스 데이터 가져오기"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM news WHERE sentiment_score IS NULL;")
    news_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return news_data

# 2. 감성 분석
def analyze_sentiment(text):
    """감성 분석 수행 (softmax 적용)"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    sentiment_score = (probabilities[0][2] - probabilities[0][0]).item()  # 긍정 - 부정
    return round(sentiment_score, 4)

# 3. 감성 점수 news 테이블에 저장
def update_sentiment_in_mysql(analyzed_data):
    """MySQL에 감성 점수 업데이트"""
    if not analyzed_data:
        print("업데이트할 감성 분석 데이터가 없습니다.")
        return
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    update_sql = "UPDATE news SET sentiment_score = %s WHERE id = %s;"
    cursor.executemany(update_sql, analyzed_data)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"MySQL 감성 점수 업데이트 완료: {len(analyzed_data)}개 항목")

# 4. 일별 평균 감성 계산
def calculate_daily_avg_sentiment():
    """날짜별 평균 감성 점수 계산 (NULL은 0으로 간주)"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = """
    SELECT DATE(published_date) AS date, AVG(IFNULL(sentiment_score, 0))
    FROM news
    GROUP BY DATE(published_date);
    """
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results 

# 5. 일별 평균 감성 점수를 avg_sentiment 테이블에 저장
def save_daily_avg_to_mysql(daily_averages):
    """계산된 날짜별 평균 감성 점수를 DB에 저장"""
    if not daily_averages:
        print("저장할 평균 감성 데이터가 없습니다.")
        return
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO avg_sentiment (date, avg_sentiment)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE avg_sentiment = VALUES(avg_sentiment);
    """
    cursor.executemany(insert_sql, daily_averages)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"{len(daily_averages)}개의 일별 감성 점수가 저장되었습니다.")
