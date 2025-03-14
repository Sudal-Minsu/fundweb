import pymysql
from config import DB_CONFIG

# 2️⃣ MySQL 연결
conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()

# 3️⃣ SQL 실행 (최신 뉴스 10개 가져오기)
sql = "SELECT id, title, published_date FROM news ORDER BY published_date DESC LIMIT 10;"
cursor.execute(sql)

# 4️⃣ 결과 가져오기
news_data = cursor.fetchall()

# 5️⃣ 결과 출력
print("📢 MySQL에서 가져온 뉴스 데이터:")
for row in news_data:
    news_id, title, published_date = row
    print(f"📰 ID: {news_id}, 제목: {title}, 날짜: {published_date}")

# 6️⃣ MySQL 연결 종료
cursor.close()
conn.close()

print(news_data[id])