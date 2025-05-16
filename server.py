from flask import Flask, render_template, abort
import os
import pymysql
from config import DB_CONFIG
from datetime import datetime
from auto_pipeline import run_auto_pipeline
from apscheduler.schedulers.background import BackgroundScheduler
from rule_3 import auto_trading_loop
import threading
from functions import read_trades_mysql

app = Flask(__name__)

trading_thread = None

def run_trading_loop():
    auto_trading_loop("005930", interval_sec=60)

@app.route('/ping')
def ping():
    global trading_thread
    if trading_thread is None or not trading_thread.is_alive():
        trading_thread = threading.Thread(target=run_trading_loop)
        trading_thread.daemon = True
        trading_thread.start()

    # ✅ 거래 로그 읽어오기
    df = read_trades_mysql("trade_history")
    trades = df.to_dict(orient='records')

    return render_template("ping.html", trades=trades)



# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    
