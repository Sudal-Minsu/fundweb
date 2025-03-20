from flask import Flask  # Flask 웹 프레임워크 임포트
from flask import render_template  # HTML 템플릿 렌더링을 위한 함수 임포트
import os  
import pymysql
import json
import config

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
    
app = Flask(__name__)  # Flask 애플리케이션 인스턴스 생성

# 투자 규칙 데이터를 리스트로 정의 (각 규칙은 ID, 이름, 수익금, 수익률을 가짐)
rules = [
    {'id': 1, 'name': '랜덤자동매매', 'profit': 0, 'yield': 0},
    {'id': 2, 'name': '규칙 2', 'profit': -150, 'yield': -4.2},
    {'id': 3, 'name': '규칙 3', 'profit': 100, 'yield': 3.0}
]

@app.route('/')  # 루트 URL ('/')에 접속하면 아래 함수 실행
def index():
    total_profit, average_yield = get_auto_trading_summary()
    # 규칙 1에 자동매매 집계 결과 반영
    rules[0]['profit'] = total_profit
    rules[0]['yield'] = average_yield
    return render_template('index.html', rules=rules)  # index.html을 렌더링하고 rules 데이터를 전달

@app.route('/rule/<int:rule_id>')  # /rule/숫자 형식의 URL이 호출될 때 실행
def rule_detail(rule_id):
    # rules 리스트에서 해당 ID를 가진 규칙을 찾음 (없으면 None 반환)
    rule = next((r for r in rules if r['id'] == rule_id), None)
    return render_template('rule_detail.html', rule=rule)  # rule_detail.html을 렌더링하고 해당 규칙을 전달

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # 환경변수 PORT가 있으면 사용하고, 없으면 기본값 10000 사용
    app.run(host="0.0.0.0", port=port)  # 모든 네트워크 인터페이스에서 접근 가능하도록 Flask 서버 실행


