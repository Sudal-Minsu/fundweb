import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 에러 미만 메시지 무시
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
from sklearn.utils import class_weight
from config import DB_CONFIG
import logging
tf.get_logger().setLevel(logging.ERROR) # 에러 미만 메시지 무시

# 설정
SEQ_LEN = 20
PRED_DAYS = 30
INITIAL_CASH = 1_000_000

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def load_stock_data(code, engine):
    query = f"SELECT Date, Close, Volume FROM top_stock_price WHERE Code = '{code}' ORDER BY Date"
    df = pd.read_sql(query, engine, parse_dates=['Date'])
    df['Code'] = code
    return df

def load_sentiment_data(engine):
    query = "SELECT date AS Date, avg_sentiment FROM avg_sentiment"
    return pd.read_sql(query, engine, parse_dates=['Date'])

def get_distinct_codes(engine, limit=200):
    query = "SELECT DISTINCT Code FROM top_stock_price ORDER BY Code LIMIT {}".format(limit)
    df = pd.read_sql(query, engine)
    return df['Code'].tolist()

def compute_rsi(series: pd.Series, period: int = 5) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def prepare_sequences(data_2d, balance_classes=False):
    sequences, targets = [], []
    for i in range(len(data_2d) - SEQ_LEN - 1):
        window = data_2d[i:i+SEQ_LEN]
        today_close = data_2d[i+SEQ_LEN-1][0]
        tomorrow_close = data_2d[i+SEQ_LEN][0]
        movement = [1, 0] if tomorrow_close < today_close else [0, 1]
        sequences.append(window)
        targets.append(movement)

    sequences = np.array(sequences)
    targets = np.array(targets)

    if not balance_classes:
        return sequences, targets

    # 클래스 균형 맞추기
    labels = np.argmax(targets, axis=1)
    class_0_idx = np.where(labels == 0)[0]
    class_1_idx = np.where(labels == 1)[0]

    min_class_size = min(len(class_0_idx), len(class_1_idx))

    if min_class_size == 0:
        return sequences, targets  # 편향된 경우 그대로 반환

    np.random.seed(42)
    selected_0 = np.random.choice(class_0_idx, min_class_size, replace=False)
    selected_1 = np.random.choice(class_1_idx, min_class_size, replace=False)
    selected_idx = np.concatenate([selected_0, selected_1])
    np.random.shuffle(selected_idx)

    return sequences[selected_idx], targets[selected_idx]

def simulate_trading(code, predictions, prices, dates, sentiments):
    cash = INITIAL_CASH
    shares = 0
    portfolio_values = []
    buy_price = None

    for i in range(len(predictions)):
        prob_down, prob_up = predictions[i]
        price = prices[i]
        sentiment = sentiments[i]
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

        elif prob_up > 0.6 or (prob_up > 0.55 and sentiment > 0.3):
            qty = 20
            total_cost = price * qty
            if cash >= total_cost:
                print(f"[{date}] [{code}] 매수: {qty}주 @ {price:.2f} (prob_up={prob_up:.2f}, sentiment={sentiment:.2f})")
                cash -= total_cost
                shares += qty
                if buy_price is None:
                    buy_price = price

        elif prob_down > 0.55 or (prob_down > 0.5 and sentiment < -0.3) and shares > 0:
            sell_qty = shares // 2
            if sell_qty > 0:
                print(f"[{date}] [{code}] 부분 매도: {sell_qty}주 @ {price:.2f} (prob_down={prob_down:.2f}, sentiment={sentiment:.2f})")
                cash += sell_qty * price
                shares -= sell_qty

        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    return portfolio_values

def backtest_single_stock(code, engine, sentiment_df, skipped_codes_detail):
    # 1. 데이터 로드
    stock_df = load_stock_data(code, engine)
    if len(stock_df) < 200:
        skipped_codes_detail.append({'Code': code, 'Reason': 'Not enough data (<200 rows)'})
        return None, None, None, None, None, None

    # 2. 감성 점수는 매매 전략 조건에만 사용 (모델 입력 X)
    df = stock_df.merge(sentiment_df, on='Date', how='left')
    df['avg_sentiment'] = df['avg_sentiment'].fillna(0)

    # 3. 모델 입력용 피처 생성
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].fillna(0)
    df['RSI_5'] = compute_rsi(df['Close'], period=5)

    # 4. 결측값 제거 및 정렬
    df = df.dropna(subset=['RSI_5'])  # RSI_5 계산에 필요한 초기 구간 제거
    df.sort_values('Date', inplace=True)

    # 5. 입력 피처 스케일링
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[['Close', 'Volume', 'RSI_5']])

    # 6. 테스트용 정보 추출
    original_close = df['Close'].values.copy()
    sentiment_series = df['avg_sentiment'].values[-PRED_DAYS:]
    test_dates = df['Date'].values[-PRED_DAYS:]

    # 7. 시퀀스 생성 및 클래스 균형 적용
    X_data, y_data = prepare_sequences(features_scaled, balance_classes=True)
    if len(X_data) < PRED_DAYS:
        skipped_codes_detail.append({'Code': code, 'Reason': 'Not enough sequence data'})
        return None, None, None, None, None, None

    # 8. 훈련/테스트 분할
    X_train, y_train = X_data[:-PRED_DAYS], y_data[:-PRED_DAYS]
    X_test = X_data[-PRED_DAYS:]
    y_test = y_data[-PRED_DAYS:]
    original_prices_for_test = original_close[-PRED_DAYS:]

    # 9. 클래스 가중치 계산
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))

    # 10. 모델 생성 및 학습
    model = create_lstm_model((SEQ_LEN, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=15, verbose=0, class_weight=class_weights_dict)

    # 11. 예측
    preds = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # 12. 매매 시뮬레이션 (감성 점수는 매매 조건에서만 사용)
    values = simulate_trading(code, preds, original_prices_for_test, test_dates, sentiment_series)

    # 13. F1-score 계산
    if len(set(true_labels)) < 2 or len(set(pred_labels)) < 2:
        return values, None, None, None, code, pred_labels

    precision = round(precision_score(true_labels, pred_labels, zero_division=0), 2)
    recall = round(recall_score(true_labels, pred_labels, zero_division=0), 2)
    f1 = round(f1_score(true_labels, pred_labels, zero_division=0), 2)

    return values, precision, recall, f1, None, pred_labels

def save_score_results(score_records, skipped_codes, skipped_codes_detail):
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

def main():
    engine = get_engine()
    sentiment_df = load_sentiment_data(engine)
    top_codes = get_distinct_codes(engine, limit=200)

    print(f"\n[백테스트 시작] 대상 종목 수: {len(top_codes)}개")

    results = {}
    score_records = []
    skipped_codes = []
    skipped_codes_detail = []
    label_distributions = []

    for code in tqdm(top_codes, desc="종목별 백테스트"):
        result, precision, recall, f1, skipped, pred_labels = backtest_single_stock(
            code, engine, sentiment_df, skipped_codes_detail
        )

        if result is not None:
            results[code] = result

        if f1 is not None:
            score_records.append({
                'Code': code,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1
            })

        if pred_labels is not None:
            counts = np.bincount(pred_labels, minlength=2)
            label_distributions.append({
                'Code': code,
                'F1_Calculated': f1 is not None,
                'Class_0_Count': int(counts[0]),
                'Class_1_Count': int(counts[1])
            })

        if skipped:
            skipped_codes.append(skipped)

    print(f"\n[결과 요약] 실제 매매가 발생한 종목 수: {len(results)}개")
    print(f"[결과 요약] 총 초기 투자 금액: {len(results) * INITIAL_CASH:,} 원")

    save_score_results(score_records, skipped_codes, skipped_codes_detail)

    if label_distributions:
        df_pred_dist = pd.DataFrame(label_distributions)
        df_pred_dist = df_pred_dist.sort_values(by='F1_Calculated', ascending=False)
        df_pred_dist.to_csv("pred_label_distribution.csv", index=False, encoding="utf-8-sig")
        print("[pred_labels 분포 저장 완료] → pred_label_distribution.csv")

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