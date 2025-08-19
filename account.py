from functions import check_account, get_auth_info, get_current_price
import time
from config import DB_CONFIG
import pymysql
from datetime import datetime
import pandas as pd
from pathlib import Path

# 인증 정보 가져오기
app_key, app_secret, access_token = get_auth_info()

# 계좌 정보 조회
res1, res2 = check_account(access_token, app_key, app_secret)

# 현재가 추가
for idx, row in res1.iterrows():
    symbol = row['종목코드']
    current_price = get_current_price(access_token, app_key, app_secret, symbol)
    res1.at[idx, '현재가'] = current_price
    time.sleep(3)  

# 포트폴리오 핵심 항목 필터링
portfolio = {
    '보유주식 평가금액': res2.get('scts_evlu_amt'),
    '총 매수금액': res2.get('pchs_amt_smtl_amt'),
    '평가손익': res2.get('evlu_pfls_smtl_amt'),
    '예수금': res2.get('dnca_tot_amt'),
    '총 평가자산': res2.get('tot_evlu_amt')
}

# 📌 보유 종목 출력
print("\n보유 종목 (현재가 포함):")
print(res1[['종목코드', '보유수량', '매입단가', '현재가']].to_string(index=False))

# 📊 포트폴리오 요약 출력
print("\n포트폴리오 요약:")
for key, value in portfolio.items():
    if value is not None:
        print(f"{key}: {int(value):,} 원")
    else:
        print(f"{key}: 데이터 없음")


# ✅ 테이블 자동 생성 함수
def init_tables():
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cursor:
        # holdings 테이블 생성 (updated_at 제거됨)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20),
            quantity INT,
            avg_price INT,
            current_price INT
        )
        """)
        # portfolio_summary 테이블 생성 (updated_at 제거됨)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_summary (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_eval_amount BIGINT,
            total_purchase BIGINT,
            eval_profit_loss BIGINT,
            cash_balance BIGINT,
            total_eval_amount BIGINT
        )
        """)
    conn.commit()
    conn.close()
    print("테이블 생성 완료")

# ✅ 데이터 저장 함수 (초기화 후 저장)
def save_to_db(res1, portfolio_data):
    conn = pymysql.connect(**DB_CONFIG)

    with conn.cursor() as cursor:
        # holdings 초기화
        cursor.execute("DELETE FROM holdings")
        # portfolio_summary 초기화
        cursor.execute("DELETE FROM portfolio_summary")

        # holdings 저장
        for _, row in res1.iterrows():
            cursor.execute("""
                INSERT INTO holdings (symbol, quantity, avg_price, current_price)
                VALUES (%s, %s, %s, %s)
            """, (
                row['종목코드'],
                int(row['보유수량']),
                float(row['매입단가']),
                float(row['현재가'])
            ))

        # portfolio_summary 저장 (updated_at 없이)
        cursor.execute("""
            INSERT INTO portfolio_summary (
                stock_eval_amount, total_purchase, eval_profit_loss,
                cash_balance, total_eval_amount
            )
            VALUES (%s, %s, %s, %s, %s)
        """, (
            int(portfolio_data.get('보유주식 평가금액', 0)),
            int(portfolio_data.get('총 매수금액', 0)),
            int(portfolio_data.get('평가손익', 0)),
            int(portfolio_data.get('예수금', 0)),
            int(portfolio_data.get('총 평가자산', 0)),
        ))

    conn.commit()
    conn.close()
    print("DB에 저장 완료 (최신 데이터만 유지)")

def save_to_csv(res1, portfolio_data, out_dir="rule_2_결과"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) 보유 종목 CSV
    holdings_cols = ['종목코드', '보유수량', '매입단가', '현재가']
    df_holdings = res1.loc[:, holdings_cols].copy()

    # 숫자형 안전 캐스팅
    df_holdings['보유수량'] = pd.to_numeric(df_holdings['보유수량'], errors='coerce').fillna(0).astype(int)
    df_holdings['매입단가'] = pd.to_numeric(df_holdings['매입단가'], errors='coerce').fillna(0).astype(int)
    df_holdings['현재가']   = pd.to_numeric(df_holdings['현재가'], errors='coerce').fillna(0).astype(int)

    # atomic replace 로 덮어쓰기
    tmp_holdings = out_path / "holdings.tmp.csv"
    final_holdings = out_path / "holdings.csv"
    df_holdings.to_csv(tmp_holdings, index=False, encoding="utf-8-sig")
    tmp_holdings.replace(final_holdings)

    # 2) 포트폴리오 요약 CSV (한 행)
    df_portfolio = pd.DataFrame([{
        '보유주식 평가금액': int(portfolio_data.get('보유주식 평가금액') or 0),
        '총 매수금액':       int(portfolio_data.get('총 매수금액') or 0),
        '평가손익':         int(portfolio_data.get('평가손익') or 0),
        '예수금':           int(portfolio_data.get('예수금') or 0),
        '총 평가자산':       int(portfolio_data.get('총 평가자산') or 0),
    }])

    tmp_port = out_path / "portfolio_summary.tmp.csv"
    final_port = out_path / "portfolio_summary.csv"
    df_portfolio.to_csv(tmp_port, index=False, encoding="utf-8-sig")
    tmp_port.replace(final_port)

    print(f"CSV 저장 완료 → {final_holdings}, {final_port}")

init_tables()  # 처음 한 번 또는 매 실행 시 호출 OK
save_to_db(res1, portfolio)
save_to_csv(res1, portfolio, out_dir="rule_2_결과")

