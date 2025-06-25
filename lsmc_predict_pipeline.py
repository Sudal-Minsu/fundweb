# combined_pipeline.py
from predict_pipeline import predict_today_candidates
from trading_utils import (
    get_current_price, lsmc_expected_profit, send_order,
    get_engine, BUY_QUANTITY
)
import time

def run_combined_pipeline():
    portfolio = {}  # 현재 보유 종목
    engine = get_engine()

    while True:
        try:
            print("\n📈 GRU 기반 매수 후보 추출 중...")
            top_candidates = predict_today_candidates(engine)

            for candidate in top_candidates:
                code = candidate['code']
                price = get_current_price(code)
                if price is None:
                    print(f"❌ 현재가 실패: {code}")
                    continue

                expected_profit = lsmc_expected_profit(code, price)
                print(f"[{code}] 확률: {candidate['prob_up']:.2f}, 기대수익: {expected_profit:.2f}, 현재가: {price}")

                # 매수 조건
                if code not in portfolio and expected_profit > 500:
                    res = send_order(code, price, BUY_QUANTITY, order_type="buy")
                    print(f"✅ 매수: {code} | 결과: {res}")
                    portfolio[code] = {'buy_price': price, 'qty': BUY_QUANTITY}

                # 매도 조건
                elif code in portfolio:
                    buy_price = portfolio[code]['buy_price']
                    if expected_profit < 300 or price < buy_price * 0.98:
                        res = send_order(code, price, portfolio[code]['qty'], order_type="sell")
                        print(f"✅ 매도: {code} | 결과: {res}")
                        del portfolio[code]

            time.sleep(15)  # 15초마다 재평가

        except KeyboardInterrupt:
            print("⏹ 자동매매 종료")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_combined_pipeline()
