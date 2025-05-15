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


@app.route('/ping')
def ping():
    return '서버는 살아있습니다.'

@app.route('/backtest')
def backtest():
    timestamp = int(datetime.now().timestamp())

    # 성능 지표
    score_table = None
    score_path = os.path.join("static", "rule_2_결과", "성능 지표.csv")
    if os.path.exists(score_path):
        df_score = pd.read_csv(score_path)
        score_table = df_score.to_dict(orient="records")

    # 매매 로그
    trade_log_table = None
    trade_log_path = os.path.join("static", "rule_2_결과", "매매 로그.csv")
    if os.path.exists(trade_log_path):
        df_log = pd.read_csv(trade_log_path)
        df_log = df_log.sort_values("buy_date", ascending=False).head(10)
        trade_log_table = df_log.to_dict(orient="records")

    # 거래 성공률
    success_rate_table = None
    success_rate_path = os.path.join("static", "rule_2_결과", "거래 성공률.csv")
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


# 🔹 Flask 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    
