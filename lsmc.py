import os
import sys
import time
import json
import keyring
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from config import DB_CONFIG, ACCOUNT_INFO, get_api_keys


# ───────────── 설정 ─────────────
OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_RISK_BUDGET_ALL = 5_000_000_00
MAX_BUY_BUDGET = 10_000_000
app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"

def adjust_price_to_tick(price):
    if price < 1000:
        tick = 1
    elif price < 5000:
        tick = 5
    elif price < 10000:
        tick = 10
    elif price < 50000:
        tick = 50
    elif price < 100000:
        tick = 100
    elif price < 500000:
        tick = 500
    else:
        tick = 1000
    return price - (price % tick)

# ───────────── 토큰 발급 ─────────────
res = requests.post(f"{url_base}/oauth2/tokenP",
                    headers={"content-type": "application/json"},
                    data=json.dumps({
                        "grant_type": "client_credentials",
                        "appkey": app_key,
                        "appsecret": app_secret
                    }))
access_token = res.json().get("access_token", "")
if not access_token:
    print("❌ 액세스 토큰 발급 실패:", res.json(), flush=True)
    sys.exit()
print(f"액세스 토큰: {access_token}\n", flush=True)

def get_historical_prices_api(stock_code, start_date="20220101", end_date="20240101"):
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010400"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code,
        "fid_org_adj_prc": "0",
        "fid_period_div_code": "D",
        "fid_begin_date": start_date,
        "fid_end_date": end_date
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200 or 'output' not in res.json():
        return None
    df = pd.DataFrame(res.json()['output'])
    df = df[df['stck_clpr'] != '']
    return df['stck_clpr'].astype(float).values[::-1]

def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    return res.json().get("HASH", "")

def get_current_price(stock_code):
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200 or 'output' not in res.json():
        return None
    return adjust_price_to_tick(int(res.json()['output']['stck_prpr']))

def get_real_balance_qty(stock_code):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",
    }
    params = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200 or 'output1' not in res.json():
        return 0
    for item in res.json()['output1']:
        if item['pdno'] == stock_code:
            return int(item['hldg_qty'])
    return 0

def send_order(stock_code, price, qty, order_type="매수"):
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"
    adjusted_price = adjust_price_to_tick(price)
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(adjusted_price)
    }
    hashkey = get_hashkey(data)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "hashkey": hashkey
    }
    time.sleep(1.2)
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    return res.json()

def load_portfolio():
    path = Path("portfolio.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_portfolio(data):
    with open("portfolio.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def wait_until_all_non_candidate_sold(portfolio, current_buy_codes):
    has_non_candidates = True
    while has_non_candidates:
        has_non_candidates = False
        for stock_code in list(portfolio.keys()):
            if stock_code not in current_buy_codes:
                real_shares = get_real_balance_qty(stock_code)
                if real_shares > 0:
                    last_price = get_current_price(stock_code)
                    order_result = send_order(stock_code, last_price, qty=real_shares, order_type="매도")
                    print(f"🔁 [비후보 종목 매도] {stock_code}: {real_shares}주 → {order_result}", flush=True)
                    log_trade(datetime.now(), stock_code, last_price, 0, 0, 0, 0, real_shares, "매도", order_result)
                    if order_result.get("rt_cd") == "0" or order_result.get("msg_cd") == "40240000":
                        del portfolio[stock_code]
                        has_non_candidates = True
                else:
                    del portfolio[stock_code]
                    has_non_candidates = True
        if has_non_candidates:
            print("비후보 종목 매도 체결 대기중... 10초 대기", flush=True)
            time.sleep(10)

def simulate_future_prices(current_price, days=10, paths=100, past_returns=None):
    simulated_prices = np.zeros((paths, days))
    for i in range(paths):
        price = current_price
        for j in range(days):
            sampled_return = np.random.choice(past_returns)
            price *= np.exp(sampled_return)
            simulated_prices[i, j] = price
    return simulated_prices

def lsmc_expected_profit_and_risk_with_prob(stock_code, current_price):
    prices = get_historical_prices_api(stock_code)
    if prices is None or len(prices) < 20:
        return {'expected_profit': 0, 'expected_loss': 0, 'rr_ratio': 0, 'optimal_qty': 0, 'prob_up': 0}
    returns = np.diff(np.log(prices))
    simulated = simulate_future_prices(current_price, past_returns=returns)
    max_profits = np.maximum(simulated.max(axis=1) - current_price, 0)
    max_losses = np.maximum(current_price - simulated.min(axis=1), 0)
    expected_profit = np.mean(max_profits)
    expected_loss = np.mean(max_losses)
    rr_ratio = expected_profit / expected_loss if expected_loss > 0 else 0
    prob_up = np.mean(max_profits > 0)
    return {
        'expected_profit': expected_profit,
        'expected_loss': expected_loss,
        'rr_ratio': rr_ratio,
        'prob_up': prob_up
    }

def log_trade(timestamp, stock_code, price, prob_up, exp_profit, exp_loss, rr_ratio, qty, order_type, order_result):
    log_file = Path("trade_log.csv")
    log_entry = {
        "거래시간": timestamp,
        "종목코드": stock_code,
        "현재가": price,
        "상증확률(%)": round(prob_up * 100, 2),
        "기대수익": round(exp_profit, 2),
        "예상손실": round(exp_loss, 2),
        "손익비": round(rr_ratio, 2),
        "주문수량": qty,
        "주문종류": order_type,
        "주문결과": order_result.get("msg1", "NO_RESPONSE")
    }
    if log_file.exists():
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(log_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    print("📊 buy_list.csv에서 매수 후보 불러오는 중...", flush=True)
    buy_list_path = os.path.join(OUTPUT_DIR, "buy_list.csv")
    if not os.path.exists(buy_list_path):
        print("❌ buy_list.csv 파일이 존재하지 않습니다.", flush=True)
        sys.exit()

    top_candidates = pd.read_csv(buy_list_path, dtype={'종목코드': str})
    top_candidates = [
        {**row, '종목코드': row['종목코드'].zfill(6)} for _, row in top_candidates.iterrows()
    ]
    current_buy_codes = set([c['종목코드'] for c in top_candidates])
    print(f"✅ [get_today_candidates] 불러온 후보 수: {len(top_candidates)}", flush=True)

    loop_count = 1
    portfolio = load_portfolio() if Path("portfolio.json").exists() else {}
    portfolio_values = []

    try:
        while True:
            print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            
            # 비후보 종목 전략 매도
            wait_until_all_non_candidate_sold(portfolio, current_buy_codes)
            save_portfolio(portfolio)

            results = []
            rr_total = 0

            # 1. 전체 후보에 대해 기대수익 등 시뮬레이션
            for candidate in top_candidates:
                stock_code = candidate['종목코드']
                print(f"종목 코드: {stock_code}", flush=True)
                price = get_current_price(stock_code)
                if not price:
                    print(f"❌ 현재가 조회 실패: {stock_code}", flush=True)
                    continue
                result = lsmc_expected_profit_and_risk_with_prob(stock_code, price)
                rr_total += result['rr_ratio']
                result.update({'code': stock_code, 'price': price})
                results.append(result)

            # 2. 기대수익 상위 10개만 추출
            results_sorted = sorted(results, key=lambda x: x['expected_profit'], reverse=True)[:10]

            # 3. 기대수익 상위 10개에 대해서만 매매 로직 (optimal_qty 계산 포함)
            for result in results_sorted:
                rr = result['rr_ratio']
                if rr_total > 0 and rr > 0 and result['expected_loss'] > 0:
                    max_loss_allowed = TOTAL_RISK_BUDGET_ALL * (rr / rr_total)
                    if result['expected_loss'] < 100:
                        result['expected_loss'] = 100
                    qty_by_risk = max_loss_allowed / result['expected_loss']
                    budget_limited_qty = MAX_BUY_BUDGET // result['price']
                    result['optimal_qty'] = int(min(qty_by_risk, budget_limited_qty))
                else:
                    result['optimal_qty'] = 0
                print(f"[{result['code']}] 가격:{result['price']} RR:{rr:.2f} 기대수익:{result['expected_profit']:.2f} Qty:{result['optimal_qty']}", flush=True)

            # ✅ 3.5 상위 10개 종목 통계 CSV 저장 (요청 기능)
            try:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(OUTPUT_DIR, f"candidates_stats_{timestamp_str}.csv")
                latest_csv_path = os.path.join(OUTPUT_DIR, "latest_candidates_stats.csv")
                export_rows = []
                for r in results_sorted:
                    export_rows.append({
                        '종목코드': r['code'],
                        '현재가': int(r['price']),
                        '기대수익': round(float(r['expected_profit']), 2),
                        '예상손실': round(float(r['expected_loss']), 2),
                        '손익비': round(float(r['rr_ratio']), 2),
                        '상승확률(%)': round(float(r['prob_up']) * 100, 2),
                        '권장수량': int(r.get('optimal_qty', 0)),
                        '루프': loop_count,
                        '타임스탬프': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                if export_rows:
                    df_stats = pd.DataFrame(export_rows)
                    df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    df_stats.to_csv(latest_csv_path, index=False, encoding='utf-8-sig')
                    print(f"📄 후보 통계 CSV 저장 완료: {csv_path}", flush=True)
                else:
                    print("⚠️ 저장할 후보 통계가 없습니다.", flush=True)
            except Exception as e:
                print(f"CSV 저장 중 오류: {e}", flush=True)

            # 4. 주문 실행
            for result in results_sorted:
                stock_code = result['code']
                price = result['price']
                optimal_qty = result['optimal_qty']
                current_qty = portfolio.get(stock_code, {}).get('qty', 0)

                if optimal_qty > current_qty:
                    add_qty = optimal_qty - current_qty
                    if add_qty > 0:
                        order_result = send_order(stock_code, price, qty=add_qty, order_type="매수")
                        print(f"✅ 추가 매수 요청 결과: {order_result}", flush=True)
                        log_trade(datetime.now(), stock_code, price, result['prob_up'],
                                  result['expected_profit'], result['expected_loss'], result['rr_ratio'], add_qty, "매수", order_result)
                        if order_result.get("rt_cd") == "0":
                            if stock_code in portfolio:
                                portfolio[stock_code]['qty'] += add_qty
                            else:
                                portfolio[stock_code] = {'buy_price': price, 'qty': add_qty}

                elif optimal_qty < current_qty:
                    sell_qty = current_qty - optimal_qty
                    if sell_qty > 0:
                        order_result = send_order(stock_code, price, qty=sell_qty, order_type="매도")
                        print(f"부분 매도 요청 결과: {order_result}", flush=True)
                        log_trade(datetime.now(), stock_code, price, result['prob_up'],
                                  result['expected_profit'], result['expected_loss'], result['rr_ratio'], sell_qty, "매도", order_result)
                        if order_result.get("rt_cd") == "0":
                            portfolio[stock_code]['qty'] -= sell_qty
                            if portfolio[stock_code]['qty'] <= 0:
                                del portfolio[stock_code]
                else:
                    print(f"[유지] {stock_code} 현재 수량 유지", flush=True)

            save_portfolio(portfolio)

            total_value = 0
            for stock_code, pos in portfolio.items():
                shares = pos['qty']
                if shares > 0:
                    last_price = get_current_price(stock_code)
                    total_value += shares * last_price
            portfolio_values.append(total_value)
            print(f"[Loop {loop_count}] 평가금액: {total_value:,.0f}", flush=True)

            loop_count += 1
            time.sleep(600)

    except KeyboardInterrupt:
        print("사용자 중단! 누적 수익률 그래프 저장 중...", flush=True)

    finally:
        if portfolio_values:
            plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정 (윈도우)
            plt.rcParams['axes.unicode_minus'] = False     # 마이너스 부호 깨짐 방지

            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_values, label="누적 포트폴리오 값")
            plt.title("누적 수익률")
            plt.xlabel("룰 회수")
            plt.ylabel("포트폴리오 값")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "누적수익률_그래프.png"), dpi=300)
            print(f"누적 수익률 그래프 저장 완료 ({OUTPUT_DIR}/누적수익률_그래프.png)", flush=True)
        else:
            print("저장할 데이터가 없습니다.", flush=True)
