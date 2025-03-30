import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sqlalchemy import create_engine
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from sklearn.metrics import precision_score, recall_score, f1_score
from config import DB_CONFIG

# 설정
SEQ_LEN = 20
PRED_DAYS = 60
THRESHOLD = 0.6
INITIAL_CASH = 1_000_000

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# DB 연결 함수
def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def load_stock_data(code, engine):
    query = f"""
    SELECT Date, Close FROM top_stock_price
    WHERE Code = '{code}'
    ORDER BY Date
    """
    df = pd.read_sql(query, engine, parse_dates=['Date'])
    df['Code'] = code
    return df

def load_sentiment_data(engine):
    query = "SELECT date AS Date, avg_sentiment FROM avg_sentiment"
    return pd.read_sql(query, engine, parse_dates=['Date'])

def get_distinct_codes(engine, limit=200):
    query = f"""
    SELECT DISTINCT Code 
    FROM top_stock_price
    ORDER BY Code
    LIMIT {limit}
    """
    df = pd.read_sql(query, engine)
    return df['Code'].tolist()

# LSTM 모델 정의
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# 다변량 시퀀스 생성 함수 (종가 + 감성 점수)
def prepare_sequences(data_2d):
    sequences, targets = [], []
    for i in range(len(data_2d) - SEQ_LEN - 1):
        window = data_2d[i:i+SEQ_LEN]
        label_close_today = data_2d[i+SEQ_LEN-1][0]
        label_close_tomorrow = data_2d[i+SEQ_LEN][0]
        movement = [1, 0] if label_close_tomorrow < label_close_today else [0, 1]
        sequences.append(window)
        targets.append(movement)
    return np.array(sequences), np.array(targets)

# 트레이딩 시뮬레이션
def simulate_trading(code, predictions, prices, dates):
    cash = INITIAL_CASH
    shares = 0
    portfolio_values = []
    buy_price = None

    for i in range(len(predictions)):
        prob_down, prob_up = predictions[i]
        price = prices[i]
        date = pd.to_datetime(dates[i]).strftime('%Y-%m-%d')

        if price <= 0 or np.isnan(price):
            portfolio_values.append(cash)
            continue

        if shares > 0 and buy_price and price < buy_price * 0.9:
            print(f"[{date}] [{code}] 손절매도: {shares}주 @ {price:.2f}")
            cash += shares * price
            shares = 0
            buy_price = None

        elif shares > 0 and buy_price and price > buy_price * 1.05:
            print(f"[{date}] [{code}] 익절매도: {shares}주 @ {price:.2f}")
            cash += shares * price
            shares = 0
            buy_price = None

        elif prob_up > 0.6:
            qty = 10
            total_cost = price * qty
            if cash >= total_cost:
                print(f"[{date}] [{code}] 매수: {qty}주 @ {price:.2f}")
                cash -= total_cost
                shares += qty
                if buy_price is None:
                    buy_price = price

        elif prob_down > 0.3 and shares > 0:
            sell_qty = shares // 2
            if sell_qty > 0:
                print(f"[{date}] [{code}] 부분 매도: {sell_qty}주 @ {price:.2f}")
                cash += sell_qty * price
                shares -= sell_qty

        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    return portfolio_values

# 단일 종목 백테스트
def backtest_single_stock(code, engine, sentiment_df, skipped_codes_detail):
    stock_df = load_stock_data(code, engine)
    if len(stock_df) < 200:
        skipped_codes_detail.append({'Code': code, 'Reason': 'Not enough data (<200 rows)'})
        return None, None, None, None, None

    df = stock_df.merge(sentiment_df, on='Date', how='left').fillna(0)
    df['Close'] = df['Close'].astype(float)
    df.sort_values('Date', inplace=True)

    original_close = df['Close'].values.copy()
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[['Close', 'avg_sentiment']])

    X_data, y_data = prepare_sequences(features_scaled)

    if len(X_data) < PRED_DAYS:
        skipped_codes_detail.append({'Code': code, 'Reason': 'Not enough sequence data'})
        return None, None, None, None, None

    X_train, y_train = X_data[:-PRED_DAYS], y_data[:-PRED_DAYS]
    X_test = X_data[-PRED_DAYS:]
    y_test = y_data[-PRED_DAYS:]
    original_prices_for_test = original_close[-PRED_DAYS:]

    model = create_lstm_model((SEQ_LEN, 2))
    model.fit(X_train, y_train, epochs=10, verbose=0)

    test_dates = df['Date'].values[-PRED_DAYS:]
    preds = model.predict(X_test, verbose=0)

    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    values = simulate_trading(code, preds, original_prices_for_test, test_dates)

    if len(set(true_labels)) < 2 or len(set(pred_labels)) < 2:
        return values, None, None, None, code

    precision = round(precision_score(true_labels, pred_labels, zero_division=0), 2)
    recall = round(recall_score(true_labels, pred_labels, zero_division=0), 2)
    f1 = round(f1_score(true_labels, pred_labels, zero_division=0), 2)

    return values, precision, recall, f1, None

# 메인 함수
def main():
    engine = get_engine()
    sentiment_df = load_sentiment_data(engine)
    top_codes = get_distinct_codes(engine, limit=200)

    print(f"\n[백테스트 시작] 대상 종목 수: {len(top_codes)}개")

    results = {}
    score_records = []
    skipped_codes = []
    skipped_codes_detail = []

    for code in tqdm(top_codes, desc="종목별 백테스트"):
        result, precision, recall, f1, skipped = backtest_single_stock(code, engine, sentiment_df, skipped_codes_detail)
        if result is not None:
            results[code] = result
        if f1 is not None:
            score_records.append({'Code': code, 'Precision': precision, 'Recall': recall, 'F1-score': f1})
        if skipped:
            skipped_codes.append(skipped)

    print(f"\n[결과 요약] 실제 매매가 발생한 종목 수: {len(results)}개")
    print(f"[결과 요약] 총 초기 투자 금액: {len(results) * INITIAL_CASH:,} 원")

    if score_records:
        df_scores = pd.DataFrame(score_records)
        df_scores = df_scores.sort_values(by='F1-score', ascending=False)

        if skipped_codes_detail:
            skipped_df_detail = pd.DataFrame(skipped_codes_detail)
            skipped_df_detail.to_csv("skipped_stocks_detail.csv", index=False, encoding="utf-8-sig")

        if skipped_codes:
            skipped_df = pd.DataFrame({'Code': skipped_codes, 'Precision': '', 'Recall': '', 'F1-score': ''})
            df_scores = pd.concat([df_scores, skipped_df], ignore_index=True)

        df_scores.to_csv("f1_scores_by_stock.csv", index=False, encoding="utf-8-sig")

        df_scores['F1-score'] = df_scores['F1-score'].astype(str).replace('', np.nan)
        df_scores['F1-score'] = pd.to_numeric(df_scores['F1-score'], errors='coerce')
        avg_f1 = df_scores['F1-score'].dropna().mean()
        print(f"[전체 평균 F1-score] {avg_f1:.2%}")

    num_days = PRED_DAYS
    total_portfolio = np.zeros(num_days)

    for values in results.values():
        if len(values) == num_days:
            total_portfolio += np.array(values)

    dates = pd.date_range(end=datetime.date.today(), periods=num_days)

    plt.figure(figsize=(14, 6))
    plt.plot(dates, total_portfolio, label="전체 포트폴리오", color='blue')
    plt.title("상위 종목 LSTM 전략 평가손익 (최근 2개월)")
    plt.xlabel("날짜")
    plt.ylabel("총 평가손익 (원)")
    formatter = ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()