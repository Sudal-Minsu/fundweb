from flask import Flask, render_template, abort, jsonify, request, send_from_directory, redirect
import os
import pymysql
from config import DB_CONFIG
import pandas as pd
from datetime import datetime
from auto_pipeline import run_auto_pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from functions import read_trades_mysql
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

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




@app.route('/backtest')
def backtest():
    timestamp = int(datetime.now().timestamp())

    # 경로 기준을 fundweb 상위 디렉토리의 rule_2_결과로 설정
    base_dir = os.path.abspath(os.path.join(app.root_path, '..', 'rule_2_결과'))

    # 성능 지표
    score_table = None
    score_path = os.path.join(base_dir, "성능 지표.csv")
    if os.path.exists(score_path):
        df_score = pd.read_csv(score_path)
        df_score = df_score.sort_values(by="f1_score", ascending=False).head(10)
        score_table = df_score.to_dict(orient="records")

    # 매매 로그
    trade_log_table = None
    trade_log_path = os.path.join(base_dir, "매매 로그.csv")
    if os.path.exists(trade_log_path):
        df_log = pd.read_csv(trade_log_path)
        df_log = df_log.sort_values("buy_date", ascending=False).head(10)
        trade_log_table = df_log.to_dict(orient="records")

    # 거래 성공률
    success_rate_table = None
    success_rate_path = os.path.join(base_dir, "거래 성공률.csv")
    if os.path.exists(success_rate_path):
        df_success = pd.read_csv(success_rate_path)
        df_success = df_success.sort_values(by="success_rate", ascending=False).head(10)
        success_rate_table = df_success.to_dict(orient="records")

    return render_template(
        'backtest.html',
        timestamp=timestamp,
        score_table=score_table,
        trade_log_table=trade_log_table,
        success_rate_table=success_rate_table
    )

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/save-portfolio", methods=["POST"])
def save_portfolio():
    Portfolio.query.delete()
    for i in range(5):
        ticker = request.form.get(f"ticker{i}")
        weight = request.form.get(f"weight{i}")
        if ticker and weight:
            db.session.add(Portfolio(ticker=ticker, weight=float(weight)))
    db.session.commit()
    return redirect("/portfolio")  # 또는 "/home" 등 리디렉션 대상 경로

@app.route("/portfolio-data")
def portfolio_data():
    base_dir = os.path.abspath(os.path.join(app.root_path, '..', 'rule_2_결과'))

    # 포트폴리오 비중 데이터
    pie_path = os.path.join(base_dir, "포트폴리오 비중.csv")
    if os.path.exists(pie_path):
        df_pie = pd.read_csv(pie_path)
    else:
        df_pie = pd.DataFrame({"ticker": ["삼성전자", "현대차"], "weight": [50, 50]})

    # 누적 수익률
    perf_path = os.path.join(base_dir, "누적 수익률.csv")
    if os.path.exists(perf_path):
        df_perf = pd.read_csv(perf_path)
        df_perf = df_perf.sort_values(by=df_perf.columns[0])
        perf_labels = df_perf.iloc[:, 0].tolist()
        perf_values = df_perf.iloc[:, 1].tolist()
    else:
        perf_labels = ["1월", "2월", "3월", "4월", "5월"]
        perf_values = [0, 5, 10, 12, 15]

    # 월별 수익률
    heatmap_path = os.path.join(base_dir, "월별 수익률.csv")
    if os.path.exists(heatmap_path):
        df_heatmap = pd.read_csv(heatmap_path)
        heatmap_labels = df_heatmap.iloc[:, 0].tolist()
        heatmap_values = df_heatmap.iloc[:, 1].tolist()
    else:
        heatmap_labels = ["1월", "2월", "3월", "4월", "5월"]
        heatmap_values = [2, -1, 3, 0, 4]

    return jsonify({
        "pie": {
            "labels": df_pie["ticker"].tolist(),
            "values": df_pie["weight"].tolist()
        },
        "performance": {
            "labels": perf_labels,
            "values": perf_values
        },
        "heatmap": {
            "labels": heatmap_labels,
            "values": heatmap_values
        }
    })

@app.route('/run-backtest', methods=['POST'])
def run_backtest():
    try:
        run_auto_pipeline()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/external/<path:filename>')
def external_static(filename):
    external_dir = os.path.abspath(os.path.join(app.root_path, '..', 'rule_2_결과'))
    return send_from_directory(external_dir, filename)

@app.route("/")
def root():
    return redirect("/home")

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)