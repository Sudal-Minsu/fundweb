# 데이터 관련 설정
SEQ_LEN = 20             # 입력 시퀀스 길이 (일 단위)
TRAIN_YEARS = 3          # 학습에 사용할 데이터 기간 (년 단위)
TEST_PERIOD_DAYS = 60    # 백테스트 기간 (일 단위)
RSI_n = 5                # RSI 계산에 사용할 기간 (n일)
STOCK_NUMBER = 100        # 종목 수 (최대 200개)

# 백테스트 시작 날짜 (사용자가 원하는 날짜로 수정 가능)
import pandas as pd
BACKTEST_START_DATE = pd.to_datetime("2024-09-05")          

# 예측 파라미터 (n과 a)
FUTURE_N = 2             # 몇 일 뒤의 종가를 기준으로 할지 (n일 뒤)
TARGET_PERCENT = 0.02    # 종가 변화 기준 

# 모델 학습 관련
LEARNING_RATE = 0.001    # 학습률
EPOCHS = 200             # 학습 반복 횟수
BATCH_SIZE = 32          # 배치 사이즈     

# 전역 feature 설정: 연속형과 이진 시그널 분리 
CONTINUOUS_COLS = [
    #'avg_sentiment',
    #'Close_shifted',
    #'LogVolume',
    #'TradingValue',
    #'Volatility_5',
    'Momentum_2',
    'Price_vs_MA20',
    #'RSI_n',
    #'MACD_Histogram',
    #'KOSPI',
    #'KOSDAQ', 
    #'USD_KRW',
    #'US_RATE'
]

BINARY_COLS = [
    #'RSI_oversold', 
    #'RSI_overbought',
    #'MACD_cross_up',
    #'MACD_cross_down',
    #'ShortTerm_Drop',
    #'Volume_jump',
    #'Volatility_spike',
    'RSI_normal_zone',
    #'MACD_bullish_cross',
    #'Price_near_low_20d',
    #'MA20_above_MA60',
]
FEATURE_COLUMNS = CONTINUOUS_COLS + BINARY_COLS

# 백테스트 전략 관련
INITIAL_CAPITAL = 100_000_000     # 초기 자본 (1억 원)
     
# 매매 조건 (예측 확률 기준)
BUY_PROB_THRESHOLD = 0.55          # 매수 신호 확률 기준 (상승 예측: class 0)
SELL_PROB_THRESHOLD = 0.55         # 매도 신호 확률 기준 (하락 예측: class 1)

# 매매 지표
RSI_b = 2  # RSI 계산에 사용할 기간 
              
# 순위 기반 전략 관련 
TOP_N_FOR_BUY = 10                # 매수 후보 종목 수 (상위 n위 안에 있는 종목)
BOTTOM_N_FOR_SELL_RANK = 190      # 매도 후보 종목 수 (상위 n위 밖에 있는 종목)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sqlalchemy import create_engine
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit  
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import logging
tf.get_logger().setLevel(logging.ERROR)
import shap
import seaborn as sns
import warnings

# 결과 저장 디렉토리 설정 
OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 백테스트 기간 (TEST_PERIOD_DAYS 일)
global_test_dates = pd.date_range(start=BACKTEST_START_DATE, periods=TEST_PERIOD_DAYS, freq='B')
TEST_END_DATE = global_test_dates[-1]

# 폰트 설정 (Windows의 말굽체 사용 예시)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False


""" DB 연결 엔진 생성 """
def get_engine():
    from config import DB_CONFIG
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


""" stock_data 테이블 가져오기 """
def load_stock_data(code, engine):
    query = f"SELECT Date, Open, High, Low, Close, Volume FROM stock_data WHERE Code = '{code}' ORDER BY Date"
    df = pd.read_sql(query, engine, parse_dates=['Date'])
    df['Code'] = code
    return df


""" market_index 테이블 가져오기 """
def load_market_index(engine):
    query = "SELECT Date, IndexType, Close FROM market_index"
    df_market = pd.read_sql(query, engine, parse_dates=['Date'])
    df_market_pivot = df_market.pivot(index='Date', columns='IndexType', values='Close').reset_index()
    return df_market_pivot


""" avg_sentiment 테이블 가져오기 """
def load_sentiment_data(engine):
    query = "SELECT date AS Date, avg_sentiment FROM avg_sentiment"
    return pd.read_sql(query, engine, parse_dates=['Date'])


""" stock_data 테이블에서 필요한 수의 종목 코드 가져오기 """
def get_distinct_codes(engine, limit=200):
    query = f"SELECT DISTINCT Code FROM stock_data ORDER BY Code LIMIT {limit}"
    df = pd.read_sql(query, engine)
    return df['Code'].tolist()


""" RSI 계산 함수 """
def compute_rsi(series: pd.Series, period) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


""" 시퀀스 생성 함수 """
def prepare_sequences(features_2d, close_prices, future_n, target_percent):
    sequences, targets = [], []
    for i in range(0, len(features_2d) - SEQ_LEN - future_n + 1):
        window = features_2d[i : i + SEQ_LEN]
        base_price = close_prices[i + SEQ_LEN - 2] 
        future_price = close_prices[i + SEQ_LEN - 1 + future_n]
        if future_price > base_price * (1 + target_percent):
            movement = [1, 0]  # 상승 → 매수 (class 0)
        elif future_price < base_price * (1 - target_percent):
            movement = [0, 1]  # 하락 → 매도 (class 1)
        else:
            continue
        sequences.append(window)
        targets.append(movement)
    sequences = np.array(sequences)
    targets = np.array(targets)
    return sequences, targets


""" 데이터 전처리, 모델 생성 및 학습 """
def prepare_stock_models(engine, top_codes):
    stock_models = {}
    loss_dict = {}
    for code in top_codes:
        df_stock = load_stock_data(code, engine)
        df_sentiment = load_sentiment_data(engine)
        df_stock = df_stock.merge(df_sentiment, on='Date', how='left')
        df_market = load_market_index(engine)
        df_stock = df_stock.merge(df_market, on='Date', how='left')
        
        # 감성 관련
        df_stock['avg_sentiment'] = df_stock['avg_sentiment'].fillna(0)
        df_stock['Sentiment_spike_up'] = (df_stock['avg_sentiment'].diff(1) > 0.05).astype(int)
        
        # 주가 관련
        df_stock['Open'] = df_stock['Open'].astype(float)
        df_stock['High'] = df_stock['High'].astype(float)
        df_stock['Low'] = df_stock['Low'].astype(float)
        df_stock['Close'] = df_stock['Close'].astype(float)
        df_stock['Close_shifted'] = df_stock['Close'].astype(float)
        
        # 거래량 관련
        df_stock['Volume'] = pd.to_numeric(df_stock['Volume'], errors='coerce').fillna(0)
        df_stock['LogVolume'] = np.log1p(df_stock['Volume'])
        df_stock['TradingValue'] = df_stock['Close'] * df_stock['Volume']
        df_stock['Volatility_5'] = df_stock['Close'].rolling(5).std()
        df_stock['Volatility_spike'] = (df_stock['Volatility_5'] > df_stock['Volatility_5'].rolling(30).mean()).astype(int)
        df_stock['Volume_jump'] = (df_stock['Volume'] > df_stock['Volume'].rolling(20).mean() * 1.5).astype(int)
        
        # 종가 관련
        df_stock['Momentum_2'] = df_stock['Close'] / df_stock['Close'].shift(2) - 1
        df_stock['ShortTerm_Drop'] = (df_stock['Close'].pct_change(1) < -0.02).astype(int)
        df_stock['Price_near_low_20d'] = (df_stock['Close'] < df_stock['Close'].rolling(20).min() * 1.05).astype(int)
        
        # 이동평균선 관련
        df_stock['MA_20'] = df_stock['Close'].rolling(20).mean()
        df_stock['MA_60'] = df_stock['Close'].rolling(60).mean()
        df_stock['Price_vs_MA20'] = df_stock['Close'] / df_stock['MA_20'] - 1
        df_stock['MA20_above_MA60'] = (df_stock['MA_20'] > df_stock['MA_60']).astype(int)
        
        # RSI 관련
        df_stock['RSI_n'] = compute_rsi(df_stock['Close'], period=RSI_n)
        df_stock['RSI_b'] = compute_rsi(df_stock['Close'], period=RSI_b)
        df_stock['RSI_normal_zone'] = ((df_stock['RSI_n'] >= 30) & (df_stock['RSI_n'] <= 70)).astype(int)
        df_stock['RSI_oversold'] = (df_stock['RSI_n'] < 30).astype(int)
        df_stock['RSI_overbought'] = (df_stock['RSI_n'] > 70).astype(int)
        
        # MACD 관련
        df_stock['EMA12'] = df_stock['Close'].ewm(span=12, adjust=False).mean()
        df_stock['EMA26'] = df_stock['Close'].ewm(span=26, adjust=False).mean()
        df_stock['MACD_Line'] = df_stock['EMA12'] - df_stock['EMA26']
        df_stock['Signal_Line'] = df_stock['MACD_Line'].ewm(span=9, adjust=False).mean()
        df_stock['MACD_Histogram'] = df_stock['MACD_Line'] - df_stock['Signal_Line']
        df_stock['MACD_cross_up'] = ((df_stock['MACD_Histogram'].shift(1) < 0) & (df_stock['MACD_Histogram'] > 0)).astype(int)
        df_stock['MACD_cross_down'] = ((df_stock['MACD_Histogram'].shift(1) > 0) & (df_stock['MACD_Histogram'] < 0)).astype(int)
        df_stock['MACD_bullish_cross'] = ((df_stock['MACD_Line'].shift(1) < df_stock['Signal_Line'].shift(1)) & (df_stock['MACD_Line'] > df_stock['Signal_Line'])).astype(int)
        
        # market_index 병합 
        df_stock[['KOSPI', 'KOSDAQ', 'USD_KRW', 'US_RATE']] = df_stock[['KOSPI', 'KOSDAQ', 'USD_KRW', 'US_RATE']].ffill().fillna(0)

        df_stock[FEATURE_COLUMNS] = df_stock[FEATURE_COLUMNS].shift(1)
        
        # 결측치 제거 및 정렬
        df_stock = df_stock.dropna(subset=FEATURE_COLUMNS)
        df_stock.sort_values('Date', inplace=True)

        # 학습 데이터 분리: BACKTEST_START_DATE 이전의 TRAIN_YEARS 기간
        train_cutoff = BACKTEST_START_DATE - pd.DateOffset(years=TRAIN_YEARS)
        df_train = df_stock[(df_stock['Date'] >= train_cutoff) & (df_stock['Date'] < BACKTEST_START_DATE)]
        if len(df_train) < SEQ_LEN + FUTURE_N:
            continue

        # 피처 분리 및 정규화
        scaler = MinMaxScaler()
        scaled_continuous = scaler.fit_transform(df_train[CONTINUOUS_COLS])
        binary_data = df_train[BINARY_COLS].values
        features_train = np.concatenate([scaled_continuous, binary_data], axis=1)
        
        # 타겟 생성 (종가 이용)
        close_target = df_train['Close'].values
        
        # 시퀀스 생성
        X_train, y_train = prepare_sequences(features_train, close_target, future_n=FUTURE_N, target_percent=TARGET_PERCENT)
        
        print(f"[{code}] 시퀀스 수: {len(X_train)}")
        if len(X_train) < 100:
            print(f"[{code}] 데이터 샘플 부족으로 제외됨 (샘플 수: {len(X_train)})")
            continue
        
        # 라벨 분포 확인
        label_counts = np.bincount(np.argmax(y_train, axis=1), minlength=2)
        print(f"[{code}] 클래스 분포 - 상승: {label_counts[0]}, 하락: {label_counts[1]}")
        
        # 모델 생성 
        def create_model(input_shape):
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.GRU(32, return_sequences=False)(inputs)
            x = tf.keras.layers.Dense(32, activation='gelu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        model = create_model((SEQ_LEN, len(FEATURE_COLUMNS)))
        
        # 시계열 데이터 분할: 마지막 fold를 검증셋으로 사용
        tscv = TimeSeriesSplit(n_splits=8)
        splits = list(tscv.split(X_train))
        train_index, val_index = splits[-1]
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # 콜백 설정
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, verbose=1, min_lr=1e-5),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
        ]
        
        # 모델 학습
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks
        )
        
        print(f"[{code}] 마지막 학습 loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            print(f"[{code}] 마지막 검증 loss (val_loss): {history.history['val_loss'][-1]:.4f}")
        
        # 손실 곡선 저장
        loss_dict[code] = (history.history['loss'], history.history.get('val_loss', []))
        
        # 모델 및 관련 데이터 저장
        stock_models[code] = {
            'df': df_stock,     # 모든 피처가 포함된 데이터
            'scaler': scaler,   # 정규화 객체 (MinMaxScaler)
            'model': model,     # 학습된 모델
            'shares': 0,        # 현재 보유 주식 수 저장 
            'buy_price': None,  # 현재 보유 포지션의 평균 매수 단가 저장
            "rolling_true": [], # 예측 성능을 평가하기 위한 실제 라벨 기록
            "rolling_pred": [], # 최근 예측 라벨 (10일 누적)
            "daily_scores": []  # 일자별 precision/recall/f1 기록
        }
    return stock_models, loss_dict


""" 상위/하위 손실 곡선 시각화 """
def plot_loss_curves(loss_curve_dict, top_n=10):
    loss_summary = []
    for code, (_, val_loss) in loss_curve_dict.items():
        if val_loss:
            final_loss = val_loss[-1]
            loss_summary.append((code, final_loss))
    sorted_by_loss = sorted(loss_summary, key=lambda x: x[1])
    bottom_codes = [code for code, _ in sorted_by_loss[:top_n]]
    top_codes = [code for code, _ in sorted_by_loss[-top_n:]]

    def plot_group(codes, title, filename):
        plt.figure(figsize=(10, 5))
        for code in codes:
            if code in loss_curve_dict:
                train_loss, val_loss = loss_curve_dict[code]
                plt.plot(train_loss, label=f"{code} Train")
                if val_loss:
                    plt.plot(val_loss, linestyle='--', label=f"{code} Val")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()
        print(f"[그래프 저장 완료] {filename}")

    plot_group(top_codes, f"Top {top_n} Val Loss 종목 (높은 Val Loss)", "상위 10개 손실 곡선.png")
    plot_group(bottom_codes, f"Bottom {top_n} Val Loss 종목 (낮은 Val Loss)", "하위 10개 손실 곡선.png")
    
    
""" 백테스트 함수 """
def run_backtest(stock_models, global_test_dates):
    all_portfolio_values = []
    code_list = list(stock_models.keys())
    
    # 초기 자본 설정
    total_cash = INITIAL_CAPITAL
    
    # 매매 수수료(매수/매도) 및 세금
    FEE_RATE = 0.000140527
    TAX_RATE = 0.0015
    
    # 백테스트 루프
    for test_date in tqdm(global_test_dates, desc="날짜별 전략 백테스트"):
        
        # 투자금 설정
        day_max_invest = total_cash * 0.1
        per_stock_max_invest = day_max_invest * 0.1
        
        # 매수 후보 종목 
        buy_candidates = {}
        
        # 매수 조건 평가 (매수 관련 조건 및 예측, f1 score 등)
        for code in code_list:
            data = stock_models[code]
            df = data['df']
            
            # test_date 이전 데이터만 사용 (과거 SEQ_LEN일치 데이터가 없으면 스킵)
            df_subset = df[df['Date'] < test_date]
            if len(df_subset) < SEQ_LEN:
                continue
            
            # 마지막 데이터 이후 FUTURE_N일의 미래 데이터가 존재하는지 확인
            pos = df.index.get_loc(df_subset.index[-1])
            if pos + 1 + FUTURE_N >= len(df):
                continue
            
            # 종가 변화량 계산
            close_yesterday = df_subset['Close'].iloc[-1]
            close_day_before = df_subset['Close'].iloc[-2]
            pct_change = (close_yesterday / close_day_before) - 1
            
            # RSI, MA 계산
            if len(df_subset) >= 60:
                ma20 = df_subset['Close'].rolling(20).mean().iloc[-1]
                ma60 = df_subset['Close'].rolling(60).mean().iloc[-1]
                ma20_prev = df_subset['Close'].rolling(20).mean().iloc[-2]
                ma60_prev = df_subset['Close'].rolling(60).mean().iloc[-2]
            else:
                ma20 = None
                ma60 = None
            try:
                rsi_yesterday = df_subset['RSI_b'].dropna().iloc[-1]
            except IndexError:
                rsi_yesterday = 50
                
            # 모델 예측: 최근 SEQ_LEN일 데이터를 입력으로 사용
            window_df = df_subset.iloc[-SEQ_LEN:]
            scaled_cont = data['scaler'].transform(window_df[CONTINUOUS_COLS])
            binary_data = window_df[BINARY_COLS].values
            features_input = np.concatenate([scaled_cont, binary_data], axis=1)
            features_input = np.expand_dims(features_input, axis=0)
            pred_prob = data['model'].predict(features_input, verbose=0)[0]
            
            # f1 score 계산을 위해, 해당 예측에 대한 정답 산출 (예측 시점의 마지막 데이터 기준 FUTURE_N일 뒤)
            pos = df.index.get_loc(df_subset.index[-1])
            if pos + 1 + FUTURE_N < len(df):
                base_price = df['Close'].iloc[pos]
                future_price = df['Close'].iloc[pos + 1 + FUTURE_N]
                if future_price > base_price * (1 + TARGET_PERCENT):
                    true_label = 0
                elif future_price < base_price * (1 - TARGET_PERCENT):
                    true_label = 1
                else:
                    continue
            else:
                continue
            pred_class = np.argmax(pred_prob)
            
            # 누적 저장 (최근 10일치 평가)
            data.setdefault("rolling_true", []).append(true_label)
            data.setdefault("rolling_pred", []).append(pred_class)
            if len(data["rolling_true"]) > 10:
                data["rolling_true"] = data["rolling_true"][-10:]
                data["rolling_pred"] = data["rolling_pred"][-10:]
                
            # 누적(true/pred) 저장 (전체 백테스트 구간)
            data.setdefault("cumulative_true", []).append(true_label)
            data.setdefault("cumulative_pred", []).append(pred_class)
            
            if len(data["rolling_true"]) == 10:
                window_true = data["rolling_true"]
                window_pred = data["rolling_pred"]
                precision = precision_score(window_true, window_pred, average='macro', zero_division=0)
                recall = recall_score(window_true, window_pred, average='macro', zero_division=0)
                f1_val = f1_score(window_true, window_pred, average='macro', zero_division=0)
                
                # 매매 전략용 f1-score (최근 2일 평균)
                recent_f1_list = [s["f1_score"] for s in data["daily_scores"][-2:]]
                if len(recent_f1_list) == 2:
                    data["recent_avg_f1"] = sum(recent_f1_list) / 2
                
                # 기록용 성능 저장
                data.setdefault("daily_scores", []).append({
                    "date": test_date,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_val
                })
            else:
                data["recent_avg_f1"] = None
                
            # 매수 조건 평가 (매수 후보 종목 선정)
            if ma20 and ma60 and (ma20 > ma20_prev) and (ma60 > ma60_prev) and (pred_prob[0] > BUY_PROB_THRESHOLD) and (pct_change < 0):
                buy_candidates[code] = {"price": close_yesterday}
                    
        # 매도 조건 평가 
        for code in code_list:
            data = stock_models[code]
            # 포지션 보유 여부 확인
            if data.get("shares", 0) > 0 and data.get("buy_price") is not None:
                df_subset_sell = data["df"][data["df"]['Date'] < test_date]
                if len(df_subset_sell) < 1:
                    continue
                close_yesterday = df_subset_sell['Close'].iloc[-1]
                buy_price = data["buy_price"]
                try:
                    rsi_yesterday = df_subset_sell['RSI_b'].dropna().iloc[-1]
                except IndexError:
                    rsi_yesterday = 50
                if close_yesterday > buy_price * 1.05:
                    data["sell_candidate"] = 100
                    data["sell_reason"] = "익절"
                elif close_yesterday <= buy_price * 0.94:
                    data["sell_candidate"] = 100
                    data["sell_reason"] = "손절"
                elif close_yesterday > buy_price and rsi_yesterday > 55:
                    data["sell_candidate"] = 100
                    data["sell_reason"] = "RSI > 55"
                else:
                    # 추가 매도 조건: 예측 기반 매도 및 RSI 조건 (충분한 데이터가 있을 경우만 예측 수행)
                    if len(df_subset_sell) >= SEQ_LEN:
                        window_df_sell = df_subset_sell.iloc[-SEQ_LEN:]
                        scaled_cont_sell = data['scaler'].transform(window_df_sell[CONTINUOUS_COLS])
                        binary_data_sell = window_df_sell[BINARY_COLS].values
                        features_input_sell = np.concatenate([scaled_cont_sell, binary_data_sell], axis=1)
                        features_input_sell = np.expand_dims(features_input_sell, axis=0)
                        pred_prob_sell = data['model'].predict(features_input_sell, verbose=0)[0]
                    else:
                        pred_prob_sell = None
                    if (pred_prob_sell is not None and close_yesterday > buy_price 
                        and pred_prob_sell[1] > SELL_PROB_THRESHOLD):
                        data["sell_candidate"] = 100
                        data["sell_reason"] = "예측 기반 매도"
                    else:
                        data["sell_candidate"] = 0
                        data["sell_reason"] = None
                        
        # 최근 2일간 f1-score 평균 순위가 상위 BOTTOM_N_FOR_SELL_RANK 밖이면 100% 매도 
        ranking_list = []
        for code in code_list:
            recent_avg_f1 = stock_models[code].get("recent_avg_f1")
            if recent_avg_f1 is not None:
                ranking_list.append((code, recent_avg_f1))
        # 내림차순 정렬
        ranking_list.sort(key=lambda x: x[1], reverse=True)
        rank_mapping = {}
        # 정렬된 종목 리스트에 순위 매기기
        for rank, (code, _) in enumerate(ranking_list, start=1):
            rank_mapping[code] = rank
        for code in code_list:
            data = stock_models[code]
            shares = data.get("shares", 0)
            # 기존 매도 조건과 중복 방지
            if shares > 0 and data.get("sell_candidate", 0) == 0:
                if code in rank_mapping and rank_mapping[code] > BOTTOM_N_FOR_SELL_RANK:
                    data["sell_candidate"] = 100
                    data["sell_reason"] = "f1-score 순위 하위"
                    
        # 매도 주문 실행
        for code in code_list:
            data = stock_models[code]
            shares = data.get("shares", 0)
            if shares > 0:
                df_subset = data["df"][data["df"]['Date'] < test_date]
                if len(df_subset) < 1:
                    continue
                # 실제 마지막 거래일의 날짜와 종가 사용
                sell_date_actual = test_date
                close_yesterday = df_subset['Close'].iloc[-1]
                sell_percent = data.get("sell_candidate", 0)
                if sell_percent > 0:
                    qty_to_sell = int(shares * sell_percent / 100)
                    if qty_to_sell > 0:
                        sale_proceeds = qty_to_sell * close_yesterday * (1 - FEE_RATE - TAX_RATE)
                        total_cash += sale_proceeds
                        data["shares"] -= qty_to_sell
                        
                        # 매도 기록 저장 (포지션 청산 시)
                        if data["shares"] == 0 and data.get("buy_price") is not None:
                            # 수수료·세금 반영한 실질 매수가/매도가로 수익률 계산 
                            profit_pct = (
                                (close_yesterday * (1 - FEE_RATE - TAX_RATE))
                                / (data["buy_price"] * (1 + FEE_RATE))
                            ) - 1
                            trade_success = True if profit_pct >= TARGET_PERCENT else False
                            
                            trade_list = data.setdefault("trade_history", [])
                            for trade in reversed(trade_list):
                                if trade.get("sell_date") is None:
                                    trade["sell_date"] = sell_date_actual
                                    trade["sell_price"] = close_yesterday
                                    trade["profit_pct"] = profit_pct
                                    trade["success"] = trade_success
                                    trade["sell_reason"] = data.get("sell_reason")
                                    break
                            
                            # 청산 후 초기화
                            data["buy_price"] = None
                            
        # 매수 주문 실행 
        if buy_candidates:
            sorted_candidates = sorted(
                buy_candidates.items(), 
                key=lambda x: stock_models[x[0]].get("recent_avg_f1", 0) if stock_models[x[0]].get("recent_avg_f1") is not None else 0,
                reverse=True  
            )
            top_buy_candidates = dict(sorted_candidates[:TOP_N_FOR_BUY])
        else:
            top_buy_candidates = {}
        for code in top_buy_candidates.keys():
            data = stock_models[code]
            price = top_buy_candidates[code]["price"]
            max_qty = int(per_stock_max_invest // (price * (1 + FEE_RATE)))
            if max_qty > 0:
                total_cost = max_qty * price * (1 + FEE_RATE)
                if total_cost > total_cash:
                    max_qty = int(total_cash // (price * (1 + FEE_RATE)))
                    total_cost = max_qty * price * (1 + FEE_RATE)
                if max_qty > 0:
                    total_cash -= total_cost
                    if data.get("shares", 0) > 0 and data.get("buy_price") is not None:
                        current_shares = data["shares"]
                        current_buy_price = data["buy_price"]
                        new_avg = ((current_buy_price * current_shares) + (price * max_qty)) / (current_shares + max_qty)
                        data["buy_price"] = new_avg
                    else:
                        data["buy_price"] = price
                    data["shares"] = data.get("shares", 0) + max_qty
                    # 매수 거래 기록 추가 (매수 시점 기록)
                    data.setdefault("trade_history", []).append({
                        "buy_date": test_date,
                        "buy_price": price,
                        "qty": max_qty,
                        "sell_date": None,
                        "sell_price": None,
                        "profit_pct": None,
                        "success": None,
                        "sell_reason": None
                    })
                    
        # 당일 포트폴리오 가치 업데이트
        total_portfolio_value = total_cash
        for code in code_list:
            data = stock_models[code]
            if data.get("shares", 0) > 0:
                df_subset = data["df"][data["df"]['Date'] <= test_date]
                if len(df_subset) > 0:
                    current_price = df_subset['Close'].iloc[-1]
                    total_portfolio_value += data["shares"] * current_price
        all_portfolio_values.append(total_portfolio_value)
        
    # 백테스트 루프 종료 후: 강제 청산 처리
    final_test_date = global_test_dates[-1]
    for code, data in stock_models.items():
        if data.get("shares", 0) > 0 and data.get("buy_price") is not None:
            df_subset = data["df"][data["df"]["Date"] <= final_test_date]
            if len(df_subset) == 0:
                continue
            final_trading_date = df_subset["Date"].iloc[-1]
            final_price = df_subset["Close"].iloc[-1]
            # 수수료·세금 반영한 실질 매수가/매도가로 수익률 계산 
            profit_pct = (
                (final_price * (1 - FEE_RATE - TAX_RATE))
                / (data["buy_price"] * (1 + FEE_RATE))
            ) - 1
            trade_success = True if profit_pct >= TARGET_PERCENT else False

            trade_list = data.setdefault("trade_history", [])
            for trade in reversed(trade_list):
                if trade.get("sell_date") is None:
                    trade["sell_date"] = final_trading_date
                    trade["sell_price"] = final_price
                    trade["profit_pct"] = profit_pct
                    trade["success"] = trade_success
                    trade["sell_reason"] = "Forced liquidation"
                    break
                    
    return all_portfolio_values


""" 백테스트 결과 시각화 """
def plot_backtest(stock_models, global_test_dates):
    total_portfolio = run_backtest(stock_models, global_test_dates)
    plt.figure(figsize=(14, 6))
    plt.plot(global_test_dates, total_portfolio, label="총 자산", color='blue')
    plt.title("날짜 기준 총 자산")
    plt.xlabel("날짜")
    plt.ylabel("총 자산 (원)")
    formatter = ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "총 자산.png"), dpi=300)
    plt.close()
    return total_portfolio


""" 평균 성능 CSV 저장 """
def score_csv(stock_models, loss_curve_dict, filename="성능 지표.csv"):
    records = []
    for code, data in stock_models.items():
        cumulative_true = data.get("cumulative_true", [])
        cumulative_pred = data.get("cumulative_pred", [])
        if not cumulative_true or not cumulative_pred:
            continue
        precision_cum = precision_score(cumulative_true, cumulative_pred, average='macro', zero_division=0)
        recall_cum = recall_score(cumulative_true, cumulative_pred, average='macro', zero_division=0)
        f1_cum = f1_score(cumulative_true, cumulative_pred, average='macro', zero_division=0)
        
        # train_loss와 val_loss 추출 
        train_loss_curve, val_loss_curve = loss_curve_dict.get(code, ([None], []))
        last_train_loss = round(train_loss_curve[-1], 4) if train_loss_curve else None
        last_val_loss = round(val_loss_curve[-1], 4) if val_loss_curve else None
        
        records.append({
            "code": code,
            "cumulative_precision": round(precision_cum, 4),
            "cumulative_recall": round(recall_cum, 4),
            "cumulative_f1_score": round(f1_cum, 4),
            "train_loss": last_train_loss,
            "val_loss": last_val_loss
        })
    if not records:
        print("[경고] 저장할 평균 성능 데이터가 없습니다.")
        return
    df = pd.DataFrame(records)
    df = df.sort_values(by="cumulative_f1_score", ascending=False)
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, encoding="utf-8-sig")
    print(f"[CSV 저장 완료] 종목별 누적 성능 → {filename}")


""" 거래 성공률 CSV 저장 """
def success_rate_csv(stock_models, filename="거래 성공률.csv"):
    records = []
    for code, data in stock_models.items():
        trades = data.get("trade_history", [])
        if not trades:
            continue
        # 매도 거래가 기록된 경우만 고려 (sell_date가 존재하는 경우)
        completed_trades = [t for t in trades if t.get("sell_date") is not None]
        if not completed_trades:
            continue
        success_count = sum(1 for t in completed_trades if t["success"])
        total_trades = len(completed_trades)
        avg_profit = sum(t["profit_pct"] for t in completed_trades) / total_trades
        records.append({
            "code": code,
            "total_trades": total_trades,
            "successful_trades": success_count,
            "success_rate": round(success_count / total_trades, 4),
            "avg_profit_pct": round(avg_profit, 4)
        })
    if not records:
        print("[경고] 저장할 거래 성공률 데이터가 없습니다.")
        return
    df = pd.DataFrame(records)
    df = df.sort_values(by="success_rate", ascending=False)
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, encoding="utf-8-sig")
    print(f"[CSV 저장 완료] 거래 성공률 데이터 → {filename}")


""" 매매 로그 CSV 저장 """
def trade_log_csv(stock_models, filename="매매 로그.csv"):
    all_records = []
    for code, data in stock_models.items():
        trade_history = data.get("trade_history", [])
        for trade in trade_history:
            if trade.get("buy_date"):
                all_records.append({
                    "code": code,
                    "buy_date": trade["buy_date"],
                    "buy_price": trade["buy_price"],
                    "qty": trade["qty"],
                    "sell_date": trade["sell_date"],
                    "sell_price": trade["sell_price"],
                    "profit_pct": round(trade["profit_pct"] * 100, 2) if trade["profit_pct"] is not None else None,
                    "success": trade["success"],
                    "sell_reason": trade.get("sell_reason")
                })
    if not all_records:
        print("[경고] 저장할 매매 로그가 없습니다.")
        return
    df = pd.DataFrame(all_records)
    df.sort_values(by="buy_date", inplace=True)
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, encoding="utf-8-sig")
    print(f"[CSV 저장 완료] 매매 로그 기록 → {filename}")


""" SHAP 분석 함수 """
def shap_analysis(stock_models):
    
    # 매매 기록 추출
    trade_records = []
    for code, data in stock_models.items():
        trade_history = data.get("trade_history", [])
        for trade in trade_history:
            if trade.get("buy_date") is not None and trade.get("profit_pct") is not None:
                trade_records.append({
                    "code": code,
                    "buy_date": trade["buy_date"],
                    "profit_pct": trade["profit_pct"]
                })

    if not trade_records:
        print("[SHAP] 매매 기록이 없습니다.")
        return

    df_trades = pd.DataFrame(trade_records)

    top3 = df_trades.sort_values(by="profit_pct", ascending=False).head(3)
    bottom3 = df_trades.sort_values(by="profit_pct", ascending=True).head(3)

    if top3.empty and bottom3.empty:
        print("[SHAP] 선택된 매매 기록이 없습니다.")
        return

    # SHAP heatmap 그리기용 subplot 준비 (3행 2열: 왼쪽 상위, 오른쪽 하위)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    # top3는 왼쪽 열, bottom3는 오른쪽 열
    trade_groups = [(top3, 0), (bottom3, 1)]

    for group, col in trade_groups:
        for row, (_, trade) in enumerate(group.iterrows()):
            code = trade["code"]
            buy_date = trade["buy_date"]
            profit_pct = trade["profit_pct"]
            data = stock_models.get(code)
            if data is None:
                continue

            df_stock = data["df"]
            scaler = data["scaler"]
            model = data["model"]

            # SHAP 값을 계산할 때 사용할 입력 시퀀스 (매수일 직전 SEQ_LEN일)
            df_seq = df_stock[df_stock["Date"] < buy_date].tail(SEQ_LEN)
            if len(df_seq) < SEQ_LEN:
                print(f"[SHAP] {code}의 {buy_date} 이전 시퀀스 부족")
                continue
            scaled_cont = scaler.transform(df_seq[CONTINUOUS_COLS])
            binary_data = df_seq[BINARY_COLS].values
            X_input = np.concatenate([scaled_cont, binary_data], axis=1)
            X_input = np.expand_dims(X_input, axis=0)

            # 배경 데이터: 학습 구간에서 무작위 시퀀스 10개 샘플링
            train_cutoff = BACKTEST_START_DATE - pd.DateOffset(years=TRAIN_YEARS)
            df_train = df_stock[(df_stock["Date"] >= train_cutoff) & (df_stock["Date"] < buy_date)]
            if len(df_train) < SEQ_LEN + 10:
                print(f"[SHAP] {code}의 배경 학습 시퀀스 부족")
                continue
            
            scaled_cont_train = scaler.transform(df_train[CONTINUOUS_COLS])
            binary_data_train = df_train[BINARY_COLS].values
            features_matrix_train = np.concatenate([scaled_cont_train, binary_data_train], axis=1)
            
            # 생성 가능한 시퀀스의 개수
            num_seq = len(features_matrix_train) - SEQ_LEN + 1
            if num_seq < 10:
                print(f"[SHAP] {code}의 배경 시퀀스 부족")
                continue
            
            # 생성 가능한 시퀀스 인덱스 범위 내에서 10개 무작위 선택
            rand_indices = np.random.choice(num_seq, size=10, replace=False)
            background_data = np.array([
                features_matrix_train[i:i + SEQ_LEN] for i in rand_indices
            ])

            # SHAP 계산
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                explainer = shap.GradientExplainer(model, background_data)
                shap_values = explainer.shap_values(X_input)

            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            if shap_vals.ndim == 4 and shap_vals.shape[-1] == 2:
                shap_vals = shap_vals[..., 0]
            shap_matrix = shap_vals[0]

            # heatmap 출력
            ax = axes[row, col]
            sns.heatmap(
                shap_matrix.T, cmap="coolwarm", center=0,
                xticklabels=[f"T{i}" for i in range(SEQ_LEN)],
                yticklabels=FEATURE_COLUMNS,
                ax=ax
            )
            ax.set_title(f"{code} | {buy_date.date()} | 수익률: {profit_pct:.2%}")

    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "shap_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[SHAP] shap_heatmap.png 저장 완료")


""" 메인 함수 """
def main():
    engine = get_engine()
    top_codes = get_distinct_codes(engine, limit=STOCK_NUMBER)
    print(f"\n[백테스트 시작] 대상 종목 수: {len(top_codes)}개")
    stock_models, loss_curve_dict = prepare_stock_models(engine, top_codes)
    print(f"[학습 완료] 사용 가능한 종목 수: {len(stock_models)}")
    
    plot_loss_curves(loss_curve_dict, top_n=10)
    plot_backtest(stock_models, global_test_dates)
    score_csv(stock_models, loss_curve_dict)
    success_rate_csv(stock_models)
    trade_log_csv(stock_models)   
    shap_analysis(stock_models)


""" 메인 함수 실행 """
if __name__ == "__main__":
    main()