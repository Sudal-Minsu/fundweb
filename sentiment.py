import os
# 허깅페이스 모델 내부의 tensorflow와 pythorch간의 충돌 방지
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
import pymysql
import torch
# pytorch의 신경망 관련 함수들을 모아놓은 모듈(여기서는 softmax 사용)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DB_CONFIG

MODEL_NAME = "snunlp/KR-FinBERT-SC"
# 사전 학습된 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 문장 분류용 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

""" MySQL에서 감성 분석이 필요한 뉴스 데이터 가져오기 """ 
    
def fetch_news_from_mysql():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM news WHERE sentiment_score IS NULL;")
    news_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return news_data

""" 감성 분석 수행 (softmax 적용) """

def analyze_sentiment(text):
    # return_tensors="pt": PyTorch 텐서 형식으로 반환
    # padding=True: 길이가 다른 문장들을 동일한 길이로 맞춤(0으로 채움)
    # truncation=True: max_length(128)토큰 이상이면 자르기
    # BERT에서는 입력 길이가 모두 같아야 병렬 처리 가능
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # gradient 계산을 끄기(학습은 안하고 추론만 할 것)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # logits: 모델의 출력값
    logits = outputs.logits
    # dim=1: 각 행마다 softmax를 적용(부정, 중립, 긍정)
    probabilities = F.softmax(logits, dim=1)
    # 긍정 - 부정, item(): 텐서를 파이썬 숫자로 변환
    sentiment_score = (probabilities[0][2] - probabilities[0][0]).item()
    return round(sentiment_score, 4)

""" MySQL에 감성 점수 업데이트 """

def update_sentiment_in_mysql(analyzed_data):
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

""" 날짜별 평균 감성 점수 계산 (NULL은 0으로 간주) """

def calculate_daily_avg_sentiment():
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

""" 계산된 날짜별 평균 감성 점수를 DB에 저장 """

def save_daily_avg_to_mysql(daily_averages):
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
