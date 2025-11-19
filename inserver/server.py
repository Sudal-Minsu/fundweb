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
import requests
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


def job_fetch_data():
    run_auto_pipeline()

scheduler = BackgroundScheduler()
scheduler.add_job(job_fetch_data, "cron", hour=9, minute=0)
scheduler.start()

# 홈 페이지 라우트
@app.route("/home")
def home():
    return render_template("home.html")

# 홈 페이지 기본으로 설정
@app.route("/")
def root():
    return redirect("/home")

# daily_trading 페이지 라우트
@app.route("/daily_trading")
def daily_trading():
    return render_template("daily_trading.html")

# fullkelly 페이지 라우트
@app.route("/fullkelly")
def fullkelly():
    return render_template("fullkelly.html")

# triplebarrier_bot 페이지 라우트
@app.route("/triplebarrier_bot")
def triplebarrier_bot():
    return render_template("triplebarrier_bot.html")

# ranking 페이지 라우트
@app.route("/ranking")
def ranking():
    return render_template("ranking.html")

# 데이터셋 접근
@app.route("/data/<folder>/<path:filename>")
def serve_csv(folder, filename):
    allowed_dirs = ["results"]
    if folder not in allowed_dirs:
        abort(403) 
    base_dir = os.path.join(app.root_path, "data", folder)
    return send_from_directory(base_dir, filename)

# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)