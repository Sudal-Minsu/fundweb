import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression

# ✅ 한글 폰트 설정 (윈도우용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 종목 코드 → 기업명 매핑
stock_code_name_map = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "051910": "LG화학",
    "068270": "셀트리온",
    # 필요한 만큼 추가 가능
}

# 가정된 보유일수와 기대수익 임계치
HOLD_DAYS = 10
EXPECTED_PROFIT_THRESHOLD = 500  
BUY_QUANTITY = 50

# 과거 데이터 불러오기 함수
def get_historical_prices_api(stock_code, start_date="20210101", end_date="20240101"):
    np.random.seed(hash(stock_code) % 2**32)
    prices = np.cumprod(1 + np.random.normal(0, 0.02, 500)) * 10000
    return prices

# 시뮬레이션 함수
def simulate_future_prices(current_price, days=HOLD_DAYS, paths=100, past_returns=None):
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated_prices[i, j] = price
    return simulated_prices

# LSMC 기대수익 계산
def lsmc_expected_profit(prices, current_index):
    if current_index < 30 or current_index >= len(prices) - HOLD_DAYS:
        return 0
    current_price = prices[current_index]
    past_returns = np.diff(np.log(prices[current_index-30:current_index]))
    simulated = simulate_future_prices(current_price, past_returns=past_returns)
    future_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    X = simulated[:, 0].reshape(-1, 1)
    y = future_profits
    model = LinearRegression().fit(X, y)
    return model.predict([[current_price]])[0]

# 백테스트 수행
def backtest(stock_code):
    prices = get_historical_prices_api(stock_code)
    capital = 10000000  # 시작 자산 1천만 원
    position = None
    trades = []
    daily_value = []

    stock_name = stock_code_name_map.get(stock_code, "알 수 없음")

    for i in range(30, len(prices) - HOLD_DAYS):
        today_price = prices[i]
        expected_profit = lsmc_expected_profit(prices, i)

        # 매수 조건
        if position is None and expected_profit > EXPECTED_PROFIT_THRESHOLD:
            buy_price = today_price
            position = {
                "entry_day": i,
                "entry_price": buy_price
            }
            print(f"🟢 {i}일차 매수 @ {buy_price:.2f}")

        # 매도 조건
        if position and i - position["entry_day"] >= HOLD_DAYS:
            sell_price = today_price
            profit = sell_price - position["entry_price"]
            capital += profit
            trades.append(profit)
            print(f"🔴 {i}일차 매도 @ {sell_price:.2f}, 수익: {profit:.2f}")
            position = None

        daily_value.append(capital)

    print(f"\n📈 [{stock_name}] 총 수익: {sum(trades):,.2f} 원")
    print(f"✅ 최종 자산: {capital:,.2f} 원")

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(daily_value, linewidth=2)
    plt.title(f"백테스트 결과 - {stock_name} ({stock_code})")
    plt.xlabel("일")
    plt.ylabel("자산(원)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 실행 예시
backtest("005930")
