from flask import Flask, render_template, abort
import os
import pymysql
import config
from datetime import datetime
from auto_pipeline import run_auto_pipeline
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# 투자 규칙 데이터
rules = [
    {'id': 1, 'name': '랜덤자동매매', 'profit': 0, 'yield': 0},
    {'id': 2, 'name': '규칙 2', 'profit': -150, 'yield': -4.2},
    {'id': 3, 'name': '규칙 3', 'profit': 100, 'yield': 3.0}
]

# 자동매매 집계 정보 조회
def get_auto_trading_summary():
    try:
        conn = pymysql.connect(**config.DB_CONFIG)
        cursor = conn.cursor()
        query = """
            SELECT SUM(profit) AS total_profit,
                   AVG(profit_rate) AS average_yield
            FROM trade_history
            WHERE profit IS NOT NULL
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0] or 0, result[1] or 0
    except Exception as e:
        print("MySQL 조회 오류:", e)
        return 0, 0

# APScheduler를 사용한 백그라운드 작업 스케줄링
scheduler = BackgroundScheduler()
scheduler.add_job(run_auto_pipeline, 'interval', minutes=5)
scheduler.start()

# 서버 실행 시 1회 실행
run_auto_pipeline()

@app.route('/')
def index():
    total_profit, average_yield = get_auto_trading_summary()
    rules[0]['profit'] = total_profit
    rules[0]['yield'] = average_yield
    return render_template('index.html', rules=rules)

@app.route('/rule/<int:rule_id>')
def rule_detail(rule_id):
    rule = next((r for r in rules if r['id'] == rule_id), None)
    if rule is None:
        return abort(404)

    dates, avg_scores = [], []
    recent_news = []

    if rule_id == 2:
        try:
            conn = pymysql.connect(**config.DB_CONFIG)
            cursor = conn.cursor()

            # 1년치 감성 평균
            cursor.execute("""
                SELECT date, avg_sentiment
                FROM avg_sentiment
                ORDER BY date ASC
            """)
            daily_data = cursor.fetchall()
            dates = [row[0].strftime('%Y-%m-%d') for row in daily_data]
            avg_scores = [round(row[1], 4) for row in daily_data]

            # 최근 10일 뉴스 제목 + 감성 점수
            cursor.execute("""
                SELECT DATE(published_date) AS date, title, sentiment_score
                FROM news
                WHERE title != '0' AND sentiment_score IS NOT NULL
                ORDER BY published_date DESC
            """)
            rows = cursor.fetchall()

            news_by_day = {}
            for date, title, score in rows:
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in news_by_day:
                    news_by_day[date_str] = []
                news_by_day[date_str].append((title, round(score, 4)))

            # 상위 10일치만 추림
            recent_news = list(news_by_day.items())[:10]

            cursor.close()
            conn.close()
        except Exception as e:
            print("감성 그래프/뉴스 데이터 조회 오류:", e)

    return render_template(
        'rule_detail.html',
        rule=rule,
        rule_id=rule_id,
        dates=dates,
        avg_scores=avg_scores,
        recent_news=recent_news
    )

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
