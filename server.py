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
import yaml
import requests
from config import DB_CONFIG

app = Flask(__name__)
modeling_status = {"state": "idle", "metrics": {}, "plot_path": ""}
UPLOAD_BASE = os.path.join(app.root_path, "rule_2_결과")

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


def job_fetch_data():
    run_auto_pipeline()

scheduler = BackgroundScheduler()
scheduler.add_job(job_fetch_data, "cron", hour=9, minute=0)
scheduler.start()

@app.route('/ping')
def show_table():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 포트폴리오 요약 가져오기
        cursor.execute("SELECT * FROM portfolio_summary")
        summary_rows = cursor.fetchall()

        # 보유 종목(holdings) 가져오기
        cursor.execute("SELECT symbol, quantity, avg_price, current_price FROM holdings")
        holding_rows = cursor.fetchall()

    except Exception as e:
        print("❌ DB 오류 발생:", e)
        return f"<h2>DB 에러: {e}</h2>"

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

    return render_template('ping.html', data=summary_rows, holdings=holding_rows)

# 자동 업데이트
with open("config/users.yaml", encoding="utf-8") as f:
    USERS = yaml.safe_load(f)

def generate_user_csv(user_id, config):
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    url = "https://openapivts.koreainvestment.com:29443"
    r = requests.get(url, headers=headers)
    data = r.json()

    df = pd.DataFrame(data)
    today = datetime.date.today().strftime("%Y-%m-%d")

    user_dir = os.path.join("rule_2_결과", user_id)
    os.makedirs(user_dir, exist_ok=True)

    file_path = os.path.join(user_dir, config["output"])
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"[{user_id}] {file_path} 저장 완료")

def daily_job():
    print("auto_pipeline 실행 시작")
    run_auto_pipeline()
    print("auto_pipeline 완료")

    print("사용자별 CSV 생성 시작")
    for user_id, cfg in USERS.items():
        try:
            generate_user_csv(user_id, cfg)
        except Exception as e:
            print(f"[ERROR] {user_id} 업데이트 실패:", e)
    print("사용자별 CSV 생성 완료")

scheduler = BackgroundScheduler()
scheduler.add_job(daily_job, "cron", hour=9, minute=0)
scheduler.start()

@app.route("/upload/<user_id>", methods=["POST"])
def upload_csv(user_id):
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400

    file = request.files["file"]
    user_dir = os.path.join(UPLOAD_BASE, user_id)
    os.makedirs(user_dir, exist_ok=True)

    save_path = os.path.join(user_dir, file.filename)
    file.save(save_path)

    return jsonify({"message": f"{file.filename} 업로드 성공", "path": save_path})

#홈 페이지 라우트
@app.route("/home")
def home():
    return render_template("home.html")

#포트폴리오 페이지 라우트    
@app.route("/portfolio")
def portfolio():
    try:
        result_dir = os.path.join(app.root_path, "rule_2_결과")
        holdings_path = os.path.join(result_dir, "holdings.csv")

        cand_names = ["portfolio_summary.csv", "portfolio.csv"]
        summary_path = None
        for name in cand_names:
            p = os.path.join(result_dir, name)
            if os.path.exists(p):
                summary_path = p
                break

        if not os.path.exists(holdings_path):
            return "<h2>holdings.csv 파일이 존재하지 않습니다.</h2>"
        if summary_path is None:
            return "<h2>포트폴리오 요약 CSV 파일이 존재하지 않습니다.</h2>"

        try:
            df_h = pd.read_csv(holdings_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_h = pd.read_csv(holdings_path, encoding="cp949")
        df_h = df_h.fillna("")

        for col in ["보유수량", "매입단가", "현재가"]:
            if col in df_h.columns:
                s = pd.to_numeric(df_h[col], errors="coerce")
                df_h[col] = s.map(lambda x: f"{int(x):,}" if pd.notna(x) else "")

        try:
            df_s = pd.read_csv(summary_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_s = pd.read_csv(summary_path, encoding="cp949")
        df_s = df_s.fillna("")

        for c in df_s.columns:
            s = pd.to_numeric(df_s[c], errors="coerce")
            if s.notna().any():
                df_s[c] = s.map(lambda x: f"{int(x):,}" if pd.notna(x) and float(x).is_integer() else (f"{x:,.2f}" if pd.notna(x) else ""))

        holdings_cols = df_h.columns.tolist()
        holdings_rows = df_h.to_dict(orient="records")
        summary_cols = df_s.columns.tolist()
        summary_rows = df_s.to_dict(orient="records")

        return render_template(
            "portfolio.html",
            holdings_cols=holdings_cols,
            holdings_rows=holdings_rows,
            summary_cols=summary_cols,
            summary_rows=summary_rows
        )
    except Exception as e:
        print(f"[ERROR] /portfolio 실패: {e}")
        return f"<h2>/portfolio 로드 실패: {e}</h2>"

# 누적 수익률 그래프 로드
@app.route("/cumulative-returns")
def cumulative_returns():
    try:
        result_dir = os.path.join(app.root_path, "rule_2_결과")
        file_path = os.path.join(result_dir, "총평가금액.csv")
        if not os.path.exists(file_path):
            return jsonify({"labels": [], "values": []})
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="cp949")
        df.columns = df.columns.str.strip()
        if "loop" not in df.columns or "총평가금액" not in df.columns:
            return jsonify({"labels": [], "values": []})
        df["총평가금액"] = pd.to_numeric(df["총평가금액"], errors="coerce")
        df = df.dropna(subset=["총평가금액"])
        labels = df["loop"].astype(str).tolist()
        values = df["총평가금액"].astype(float).tolist()
        return jsonify({"labels": labels, "values": values})
    except Exception as e:
        print("[ERROR] cumulative-returns:", e)
        return jsonify({"labels": labels, "returns": values})

#forecast 페이지 라우트
@app.route("/forecast")
def forecast():
    return render_template("forecast.html")

#리포트 페이지 라우트
@app.route("/strategy")
def strategy():
    return render_template("strategy.html")

# confusion matrix 로드
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
        df["거래시간"] = pd.to_datetime(
            df["거래시간"].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce"
        )
        df["거래시간"] = df["거래시간"].dt.strftime("%Y-%m-%d %H:%M")
        df = df[df["주문결과"] == "모의투자 매수주문이 완료 되었습니다."]
        df = df.tail(10).iloc[::-1]
        selected_columns = [
            "거래시간",
            "종목코드",
            "현재가",
            "주문수량",
            "주문종류",
        ]
        df = df[selected_columns]
        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        print(f"[ERROR] trade_log.csv 로드 실패: {e}")
        return jsonify([])

@app.route("/")
def root():
    return redirect("/home")

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)