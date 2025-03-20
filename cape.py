"""cape 지수 계산"""

import yfinance as yf
import pandas as pd
import numpy as np

# 🔹 S&P 500 지수의 현재 가격 가져오기
def get_sp500_price():
    print("📢 Fetching S&P 500 price from Yahoo Finance...")
    try:
        sp500 = yf.Ticker("^GSPC")
        price_data = sp500.history(period="1d")
        if price_data.empty:
            raise ValueError("❌ S&P 500 가격 데이터를 가져올 수 없습니다.")
        latest_price = price_data["Close"].iloc[-1]
        print(f"📈 S&P 500 현재 가격: {latest_price:.2f}")
        return latest_price
    except Exception as e:
        print(f"🚨 오류 발생: {e}")
        return None

# 🔹 S&P 500 EPS 데이터 수동 입력
def get_sp500_eps():
    # 예시: S&P 500 EPS 데이터를 수동으로 입력합니다.
    # 실제 데이터는 위의 대체 데이터 소스에서 최신 값을 확인하여 입력하세요.
    eps_data = {
        '2015': 100.00,
        '2016': 105.00,
        '2017': 110.00,
        '2018': 115.00,
        '2019': 120.00,
        '2020': 125.00,
        '2021': 130.00,
        '2022': 135.00,
        '2023': 140.00,
        '2024': 145.00
    }
    eps_series = pd.Series(eps_data)
    avg_eps = eps_series.mean()
    print(f"📊 최근 10년 평균 EPS: {avg_eps:.2f}")
    return avg_eps

# 🔹 CAPE 지수 계산
def calculate_cape():
    try:
        sp500_price = get_sp500_price()
        avg_eps = get_sp500_eps()
        if sp500_price is None or avg_eps is None:
            raise ValueError("❌ CAPE 계산을 위한 데이터가 부족합니다.")
        cape_ratio = sp500_price / avg_eps
        print(f"✅ S&P 500의 CAPE Ratio: {cape_ratio:.2f}")
        return cape_ratio
    except Exception as e:
        print(f"🚨 오류 발생: {e}")
        return None

# ✅ 실행
cape_value = calculate_cape()
if cape_value is not None:
    print(f"📈 S&P 500의 CAPE Ratio: {cape_value:.2f}")
else:
    print("❌ CAPE 계산에 실패했습니다. 로그를 확인하세요.")
