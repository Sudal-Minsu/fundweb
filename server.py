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

    labels, scores, titles = [], [], []
    if rule_id == 2:
        try:
            conn = pymysql.connect(**config.DB_CONFIG)
            cursor = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT title, published_date, sentiment_score
                FROM news
                WHERE DATE(published_date) = %s
                ORDER BY published_date DESC
            """, (today,))
            data = cursor.fetchall()
            cursor.close()
            conn.close()

            labels = [row[1].strftime('%H:%M') for row in data]
            scores = [row[2] for row in data]
            titles = [row[0] for row in data]
        except Exception as e:
            print("뉴스 데이터 조회 오류:", e)

        zipped_data = zip(labels, titles, scores)

        return render_template(
            'rule_detail.html',
            rule=rule,
            rule_id=rule_id,
            labels=labels,
            scores=scores,
            titles=titles,
            zipped_data=zipped_data
        )


# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
