# MySQL에 ALTER TABLE news ADD COLUMN sentiment_score FLOAT DEFAULT NULL; 
import pymysql
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DB_CONFIG

MODEL_NAME = "snunlp/KR-FinBERT-SC"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def fetch_all_news_from_mysql():
    """MySQL에서 모든 뉴스 가져오기 (감성 분석된 것도 다시 분석)"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM news;")  # 조건 없이 모든 데이터 가져옴
    news_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return news_data

def fetch_news_from_mysql():
    """MySQL에서 감성 분석이 필요한 뉴스 데이터 가져오기"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM news WHERE sentiment_score IS NULL;")
    news_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return news_data

def analyze_sentiment(text):
    """감성 분석 수행 (softmax 적용)"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)  # 감성 확률 계산
    sentiment_score = (probabilities[0][2] - probabilities[0][0]).item()  # 긍정 - 부정
    return round(sentiment_score, 4)  # 소수점 4자리 반영

def update_sentiment_in_mysql(analyzed_data):
    """MySQL에 감성 점수 업데이트 (기존 값 덮어쓰기)"""
    if not analyzed_data:
        print("업데이트할 감성 분석 데이터가 없습니다.")
        return
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    update_sql = "UPDATE news SET sentiment_score = %s WHERE id = %s;"
    cursor.executemany(update_sql, analyzed_data)  # 기존 값 덮어쓰기
    conn.commit()
    cursor.close()
    conn.close()
    print(f"MySQL 감성 점수 업데이트 완료: {len(analyzed_data)}개 항목")

news_data = fetch_news_from_mysql()  # 뉴스 가져오기
if news_data:
    analyzed_data = [(analyze_sentiment(title), news_id) for news_id, title in news_data]
    update_sentiment_in_mysql(analyzed_data)
else:
    print("감성 분석할 뉴스가 없습니다.")
