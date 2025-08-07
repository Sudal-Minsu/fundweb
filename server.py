from flask import Flask, render_template, abort, jsonify, request, send_from_directory, redirect, url_for, Response
import os
import pymysql
import pandas as pd
from datetime import datetime
from auto_pipeline import run_auto_pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from functions import read_trades_mysql, single_trade
from flask_sqlalchemy import SQLAlchemy
import matplotlib.pyplot as plt
import io
from config import DB_CONFIG

app = Flask(__name__)
modeling_status = {"state": "idle", "metrics": {}, "plot_path": ""}

# DB 설정 (SQLite로 간단하게 시작)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portfolio.db'
db = SQLAlchemy(app)

# DB 모델 정의
class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20))
    weight = db.Column(db.Float)

with app.app_context():
    db.create_all()

trading_thread = None

@app.route('/ping')
def show_table():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stock_recommendations")
        rows = cursor.fetchall()

    except Exception as e:
        print("❌ DB 오류 발생:", e)
        return f"<h2>DB 에러: {e}</h2>"

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

    return render_template('ping.html', data=rows)

@app.route('/buy', methods=['POST'])
def buy_stock():
    data = request.get_json()
    stock_code = data.get("stock_code")
    quantity = int(data.get("quantity", 1))

    try:
        result = single_trade(stock_code=stock_code, quantity=quantity)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

#홈 페이지 라우트
@app.route("/home")
def home():
    return render_template("home.html")

#리포트 페이지 라우트
@app.route("/report")
def report():
    return render_template("report.html")

#예시 데이터 추후에 코드와 연결 필요요
@app.route("/confusion-data")
def confusion_data():
    try:
        result_dir = os.path.join(app.root_path, "rule_2_결과")
        file_path = os.path.join(result_dir, "confusion_matrix.csv")

        if not os.path.exists(file_path):
            print(f"[ERROR] confusion_matrix.csv 파일이 존재하지 않음: {file_path}")
            return jsonify({"matrix": [], "total": 0})

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            df.columns[0]: "실제 라벨",
            df.columns[1]: "예측 라벨",
            df.columns[2]: "합계"
        })

        pivot = {
            (int(row["실제 라벨"]), int(row["예측 라벨"])): int(row["합계"])
            for _, row in df.iterrows()
        }

        matrix = [
            [pivot.get((0, 0), 0), pivot.get((0, 1), 0)],
            [pivot.get((1, 0), 0), pivot.get((1, 1), 0)],
        ]
        total = sum(sum(row) for row in matrix)

        return jsonify({
            "matrix": matrix,
            "total": total
        })

    except Exception as e:
        print(f"[ERROR] confusion_matrix.csv 로드 실패: {e}")
        return jsonify({"matrix": [], "total": 0})

#매매 로그 로드
@app.route("/trade-log")
def trade_log():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "trade_log.csv")
        if not os.path.exists(file_path):
            print(f"[ERROR] 파일이 존재하지 않음: {file_path}")
            return jsonify([])
        df = pd.read_csv(file_path, encoding="utf-8-sig").fillna("")
        df = df.tail(20).iloc[::-1]
        selected_columns = [
            "거래시간",
            "기대수익",
            "상증확률(%)",
            "손익비",
            "예상손실",
            "종목코드",
            "주문수량",
            "주문종류",
            "현재가"
        ]
        df = df[selected_columns]
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        print(f"[ERROR] trade_log.csv 로드 실패: {e}")
        return jsonify([])

#포트폴리오 페이지 라우트    
@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

#백테스트 페이지 라우트
@app.route("/backtest")
def backtest():
    # 그래프를 바로 생성하고 반환하는 API 호출
    return render_template("backtest.html")

# 그래프를 동적으로 생성하고 HTTP 응답으로 반환
@app.route("/generate-graph")
def generate_backtest_graph():
    # 예시 데이터로 그래프 생성 (적절한 백테스트 데이터를 사용해 그려야 합니다)
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 4, 9, 16]

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Growth over Time')
    ax.set_title('Backtest Results')
    ax.set_xlabel('Time')
    ax.set_ylabel('Growth')

    # 이미지를 메모리 버퍼에 저장
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)  # 메모리 해제

    # 이미지를 HTTP 응답으로 전송
    return Response(img_buf, mimetype='image/png')

@app.route("/")
def root():
    return redirect("/home")

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)