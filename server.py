from flask import Flask, render_template, make_response
import os
import pymysql
import config  # config.py에서 DB_CONFIG를 가져옴

app = Flask(__name__)

# 🔹 투자 규칙 목록 (수익 데이터는 rule_id=1에만 연동)
rules = [
    {'id': 1, 'name': '랜덤자동매매', 'profit': 0, 'yield': 0},
    {'id': 2, 'name': '규칙 2', 'profit': -150, 'yield': -4.2},
    {'id': 3, 'name': '규칙 3', 'profit': 100, 'yield': 3.0}
]

# 🔹 자동매매 수익 요약 조회 함수
def get_auto_trading_summary():
    try:
        conn = pymysql.connect(**config.DB_CONFIG)
        cursor = conn.cursor()
        query = """
            SELECT 
                SUM(profit) AS total_profit,
                AVG(profit_rate) AS average_yield
            FROM trade_history
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

# 🔹 메인 화면
@app.route('/')
def index():
    total_profit, average_yield = get_auto_trading_summary()
    rules[0]['profit'] = total_profit
    rules[0]['yield'] = average_yield
    return render_template('index.html', rules=rules)

# 🔹 상세 규칙 페이지 (rule_id 기준)
@app.route('/rule/<int:rule_id>')
def rule_detail(rule_id):
    rule = next((r for r in rules if r['id'] == rule_id), None)

    if rule_id == 1:
        try:
            # 요약 수익 갱신
            total_profit, average_yield = get_auto_trading_summary()
            rule['profit'] = total_profit
            rule['yield'] = average_yield

            conn = pymysql.connect(**config.DB_CONFIG)
            cursor = conn.cursor()
            query = """
                SELECT profit, profit_rate, trade_time
                FROM trade_history
                WHERE profit IS NOT NULL
                ORDER BY trade_time ASC
                LIMIT 50
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()

            profit_list = [float(r[0]) for r in results]
            yield_list = [float(r[1]) for r in results]
            labels = [r[2].strftime('%Y-%m-%d %H:%M') for r in results]

            response = make_response(render_template(
                'rule_detail.html',
                rule=rule,
                profit_data=profit_list,
                yield_data=yield_list,
                labels=labels
            ))
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return response

        except Exception as e:
            print("📛 거래 데이터 조회 오류:", e)
            return render_template('rule_detail.html', rule=rule)

    # rule_id != 1일 경우
    return render_template('rule_detail.html', rule=rule)

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
