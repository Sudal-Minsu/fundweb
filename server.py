from flask import Flask, render_template, abort, jsonify, request, send_from_directory, redirect
import os
import pymysql
from config import DB_CONFIG
import pandas as pd
from datetime import datetime
from auto_pipeline import run_auto_pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from functions import read_trades_mysql, single_trade
from flask_sqlalchemy import SQLAlchemy
from matplotlib.animation import FuncAnimation

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

#결과 폴더 가져오는 용도
@app.route('/external/<path:filename>')
def external_static(filename):
    external_dir = os.path.abspath(os.path.join(app.root_path, '..', 'rule_2_결과'))
    return send_from_directory(external_dir, filename)

#리포트 페이지 라우트
@app.route("/report")
def report():
    return render_template("report.html")

#예시 데이터 추후에 코드와 연결 필요요
@app.route("/confusion-data")
def confusion_data():
    matrix = [
        [596, 214],
        [411, 251]  
    ]
    total = sum(sum(row) for row in matrix)

    return jsonify({
        "matrix": matrix,
        "total": total
    })

#매매 로그 로드
@app.route("/trade-log")
def trade_log():
    try:
        external_dir = os.path.abspath(os.path.join(app.root_path, '..', 'rule_2_결과'))
        file_path = os.path.join(external_dir, 'trade_log.csv')

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        print(f"[ERROR] trade_log.csv 로드 실패: {e}")
        return jsonify([])

#포트폴리오 페이지 라우트    
@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

#백테스트 페이지 라우트트
@app.route("/backtest")
def backtest():
    return render_template("backtest.html")

#그래프 출력용, 예시 데이터 사용, 데이터는 추후 변경 필요
@app.route("/run-modeling", methods=["POST"])
def run_modeling():
    global modeling_status
    try:
        data = request.get_json()
        symbol = data.get("symbol", "005930.KS")
        epochs = int(data.get("epochs", 30))
        window_size = int(data.get("window_size", 20))
        batch_size = int(data.get("batch_size", 32))

        modeling_status["state"] = "running"

        x_data = np.arange(100)
        y_data = np.sin(x_data / 5) + np.random.normal(scale=0.1, size=100)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, 100)
        ax.set_ylim(min(y_data) - 0.5, max(y_data) + 0.5)
        ax.set_title(f"Forecast Result - {symbol}", fontsize=12)
        ax.set_facecolor("#2f2f2f")
        fig.patch.set_facecolor("#3a3a3a")
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(x_data[:frame], y_data[:frame])
            return line,

        ani = FuncAnimation(fig, update, frames=len(x_data), init_func=init, blit=True, repeat=False)
        save_path = os.path.join("static", "forecast_example.gif")
        ani.save(save_path, writer='pillow')
        plt.close()

        modeling_status["metrics"] = {
            "mape": 7.3,
            "rmse": 0.118,
            "accuracy": 81.2,
            "plot_path": "/static/forecast_example.gif"
        }
        modeling_status["state"] = "done"

        return jsonify({"status": "ok", "metrics": modeling_status["metrics"]})
    except Exception as e:
        modeling_status["state"] = "error"
        print("[ERROR]", e)
        return jsonify({"status": "error", "message": str(e)})

@app.route("/model-status")
def model_status():
    return jsonify({"state": modeling_status["state"], "metrics": modeling_status["metrics"]})


@app.route("/")
def root():
    return redirect("/home")

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)