from flask import Flask, render_template
import os
import pymysql
from config import DB_CONFIG

app = Flask(__name__)

# 🔹 투자 규칙 목록
rules = [
    {'id': 1, 'name': '랜덤자동매매', 'profit': 0, 'yield': 0},
    {'id': 2, 'name': '규칙 2', 'profit': -150, 'yield': -4.2},
    {'id': 3, 'name': '규칙 3', 'profit': 100, 'yield': 3.0}
]

# 🔹 자동매매 요약 정보 (수익금, 수익률 평균)
def get_auto_trading_summary():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            SELECT 
                SUM(profit) AS total_profit,
                AVG(profit_rate) AS average_yield
            FROM trades
            WHERE profit IS NOT NULL
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        total_profit = result[0] if result[0] is not None else 0
        average_yield = result[1] if result[1] is not None else 0

        return total_profit, average_yield
    except Exception as e:
        print("MySQL 조회 오류:", e)
        return 0, 0

# 🔹 홈 화면
@app.route('/')
def index():
    total_profit, average_yield = get_auto_trading_summary()
    rules[0]['profit'] = total_profit
    rules[0]['yield'] = average_yield
    return render_template('index.html', rules=rules)

# 🔹 규칙 상세 보기
@app.route('/rule/<int:rule_id>')
def rule_detail(rule_id):
    rule = next((r for r in rules if r['id'] == rule_id), None)

    if rule_id == 1:
        try:
            total_profit, average_yield = get_auto_trading_summary()
            rule['profit'] = total_profit
            rule['yield'] = average_yield

            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()

            query = """
                SELECT profit, profit_rate, traded_at
                FROM trades
                WHERE profit IS NOT NULL
                ORDER BY traded_at ASC
                LIMIT 50
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()

            profit_list = [float(r[0]) for r in results]
            profit_rate_list = [float(r[1]) for r in results]
            labels = [r[2].strftime("%Y-%m-%d %H:%M") for r in results]

            return render_template(
                'rule_detail.html',
                rule=rule,
                profit_data=profit_list,
                yield_data=profit_rate_list,
                labels=labels
            )
        except Exception as e:
            print("📛 거래 데이터 조회 오류:", e)
            return render_template('rule_detail.html', rule=rule)

    return render_template('rule_detail.html', rule=rule)

# 🔹 앱 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
