# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

# ─────────────────────────────────────────────
# config.py에서 개인정보/설정 가져오기
#  - 필요한 항목: DB_CONFIG, get_api_keys(), ACCOUNT_INFO
# ─────────────────────────────────────────────
try:
    from config import DB_CONFIG, get_account
except Exception as e:
    print("❌ config.py 불러오기 실패! 같은 폴더에 있고, 아래를 제공하는지 확인하세요.")
    print("   - DB_CONFIG (dict)")
    print("   - get_api_keys() -> (app_key, app_secret)")
    print("   - ACCOUNT_INFO (dict: CANO, ACNT_PRDT_CD)")
    print(f"오류: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# [전략/실행 설정]
# ─────────────────────────────────────────────

# 분석/그래프/로그 저장 폴더: results
OUTPUT_DIR = os.path.join("data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 매매 로그 파일
LOG_DIR = OUTPUT_DIR
LOG_FILE = Path(LOG_DIR) / "trade_log.csv"

# 과거 로그가 있으면 이동(선택)
OLD_LOG_FILE = Path("full_kelly_결과") / "trade_log.csv"
if OLD_LOG_FILE.exists() and not LOG_FILE.exists():
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(OLD_LOG_FILE), str(LOG_FILE))
        print(f"기존 로그를 {OLD_LOG_FILE} → {LOG_FILE} 로 이동했습니다.")
    except Exception as e:
        print(f"기존 로그 이동 실패: {e}")

# 총 매수 예산(이번 루프 전체)
TOTAL_BUY_BUDGET_ALL = 100_000_000   # 1억
# 종목당 최대 매수 예산 상한
MAX_BUY_BUDGET = 10_000_000

# 손절/익절 규칙(손익비 R = TAKE/STOP)
STOP_LOSS_PCT   = 0.025  # -2.5%
TAKE_PROFIT_PCT = 0.05   # +5%

# 총 노출 한도 적용 여부
ENFORCE_TOTAL_BUDGET_CAP = True

# 포지션 유지/정리 정책
SELL_NON_CANDIDATES      = True   # 후보에 없는 종목 전량 정리
CHECK_TP_SL_EVERY_LOOP   = True   # 매 루프마다 TP/SL 점검

# API 키 불러오기
account = get_account("acc2")
app_key = account["APP_KEY"]
app_secret = account["APP_SECRET"]
ACCOUNT_INFO = account["ACCOUNT_INFO"]
if not app_key or not app_secret:
    print("❌ app_key/app_secret이 비어 있습니다. config.py의 get_api_keys()를 확인하세요.")
    sys.exit(1)

# 모의투자 베이스 URL
url_base = "https://openapivts.koreainvestment.com:29443"

# ───────────── 유틸 ─────────────
def adjust_price_to_tick(price: int) -> int:
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
    return int(price - (price % tick))

# ───────────── 토큰 발급 ─────────────
def issue_access_token():
    res = requests.post(
        f"{url_base}/oauth2/tokenP",
        headers={"content-type": "application/json"},
        data=json.dumps({
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret
        })
    )
    try:
        j = res.json()
    except Exception:
        print("❌ 토큰 응답 JSON 파싱 실패:", res.text)
        sys.exit(1)
    token = j.get("access_token", "")
    if not token:
        print("❌ 액세스 토큰 발급 실패:", j)
        sys.exit(1)
    print("액세스 토큰 발급 성공.\n", flush=True)
    return token

access_token = issue_access_token()

# ───────────── 시세/잔고/주문 API ─────────────
def get_current_price(stock_code: str) -> int | None:
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100",
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code,
    }
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200:
        return None
    try:
        j = res.json()
        pr = int(j['output']['stck_prpr'])
        return adjust_price_to_tick(pr)
    except Exception:
        return None

def get_real_balance_qty(stock_code: str) -> int:
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
    if res.status_code != 200:
        return 0
    try:
        for item in res.json().get('output1', []):
            if item.get('pdno') == stock_code:
                return int(item.get('hldg_qty', 0))
    except Exception:
        pass
    return 0

def get_hashkey(data: dict) -> str:
    url = f"{url_base}/uapi/hashkey"
    headers = {
        "Content-Type": "application/json",
        "appKey": app_key,
        "appSecret": app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json().get("HASH", "")
    except Exception:
        return ""

def send_order(stock_code: str, price: int, qty: int, order_type: str = "매수") -> dict:
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"
    adjusted_price = adjust_price_to_tick(price)
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(adjusted_price)),
    }
    hashkey = get_hashkey(data)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "hashkey": hashkey,
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json()
    except Exception:
        return {"rt_cd": "-1", "msg1": "INVALID_JSON"}

# ───────────── 포트폴리오 & 로깅 ─────────────
def load_portfolio() -> dict:
    path = Path("portfolio.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_portfolio(data: dict):
    with open("portfolio.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def log_trade(timestamp, stock_code, price, p, R, fstar, qty, order_type, order_result):
    ts = timestamp if isinstance(timestamp, str) else timestamp.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "거래시간": ts,
        "종목코드": stock_code,
        "현재가": price,
        "상승확률(p)": round(float(p) * 100, 2),
        "손익비(R)": round(float(R), 3),
        "켈리비율(f*)": round(float(fstar), 4),
        "주문수량": int(qty),
        "주문종류": order_type,
        "주문결과": order_result.get("msg1", "NO_RESPONSE")
    }
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

# ───────────── 보조: 비후보 정리 & TP/SL ─────────────
def wait_until_all_non_candidate_sold(portfolio: dict, current_buy_codes: set[str]):
    if not SELL_NON_CANDIDATES:
        return
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
                    log_trade(datetime.now(), stock_code, last_price, portfolio[stock_code].get("p", 0),
                              portfolio[stock_code].get("R", 0), portfolio[stock_code].get("fstar", 0),
                              real_shares, "매도", order_result)
                    if order_result.get("rt_cd") == "0" or order_result.get("msg_cd") == "40240000":
                        del portfolio[stock_code]
                        has_non_candidates = True
                else:
                    del portfolio[stock_code]
                    has_non_candidates = True
        if has_non_candidates:
            print("비후보 종목 매도 체결 대기중... 10초 대기", flush=True)
            time.sleep(10)

def check_takeprofit_stoploss(portfolio: dict):
    if not CHECK_TP_SL_EVERY_LOOP:
        return
    for stock_code, pos in list(portfolio.items()):
        shares = int(pos.get('qty', 0))
        if shares <= 0:
            continue
        last_price = get_current_price(stock_code)
        if last_price is None:
            continue
        tp = pos.get('tp_price')
        sl = pos.get('sl_price')
        if tp and last_price >= tp:
            order_result = send_order(stock_code, last_price, qty=shares, order_type="매도")
            print(f"🎯 [익절] {stock_code} {shares}주 @ {last_price} → {order_result}", flush=True)
            log_trade(datetime.now(), stock_code, last_price, pos.get("p", 0), pos.get("R", 0), pos.get("fstar", 0), shares, "매도", order_result)
            if order_result.get("rt_cd") == "0":
                del portfolio[stock_code]
                continue
        if sl and last_price <= sl:
            order_result = send_order(stock_code, last_price, qty=shares, order_type="매도")
            print(f"🛑 [손절] {stock_code} {shares}주 @ {last_price} → {order_result}", flush=True)
            log_trade(datetime.now(), stock_code, last_price, pos.get("p", 0), pos.get("R", 0), pos.get("fstar", 0), shares, "매도", order_result)
            if order_result.get("rt_cd") == "0":
                del portfolio[stock_code]

# ───────────── 켈리 계산 ─────────────
def extract_prob_from_row(row: dict) -> float:
    """buy_list.csv 한 행에서 상승확률 p를 추출(다양한 컬럼명 지원). 0~1로 반환."""
    keys = ["prob_up", "p", "prob", "상승확률", "확률"]
    for k in row.keys():
        lk = str(k).lower()
        for cand in keys:
            if lk == cand.lower():
                val = row[k]
                try:
                    p = float(val)
                    if p > 1.0:  # % 스케일
                        p = p / 100.0
                    return max(0.0, min(1.0, p))
                except Exception:
                    pass
    return 0.55  # 기본값

def compute_kelly_fraction(p: float, R: float) -> float:
    q = 1.0 - p
    return p - (q / R)

# ───────────── 메인 루프 ─────────────
if __name__ == "__main__":
    print("📊 buy_list.csv에서 매수 후보 불러오는 중...", flush=True)
    buy_list_path = os.path.join(OUTPUT_DIR, "buy_list.csv")
    if not os.path.exists(buy_list_path):
        print("❌ buy_list.csv 파일이 존재하지 않습니다.", flush=True)
        print("   위치:", os.path.abspath(buy_list_path))
        sys.exit(1)

    # buy_list.csv 로드 (종목코드는 6자리 0패딩)
    df_cand = pd.read_csv(buy_list_path, dtype={'종목코드': str, 'code': str})
    rows = []
    for _, row in df_cand.iterrows():
        d = row.to_dict()
        code = (d.get('종목코드') or d.get('code') or '').zfill(6)
        if not code:
            continue
        d['종목코드'] = code
        rows.append(d)

    if not rows:
        print("❌ 유효한 후보가 없습니다.", flush=True)
        sys.exit(1)

    current_buy_codes = set([r['종목코드'] for r in rows])
    print(f"✅ [get_today_candidates] 불러온 후보 수: {len(rows)}", flush=True)

    loop_count = 1
    portfolio = load_portfolio() if Path("portfolio.json").exists() else {}

    # 평가금액(Equity Curve) 기록
    equity_curve = []  # {time, total_value}
    # 고정 손익비 (R = TAKE/STOP)
    R = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    INITIAL_ASSET = 100_000_000

    try:
        while True:
            print(f"\n[LOOP {loop_count}] 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

            # 0) 비후보 정리 및 TP/SL 점검
            if SELL_NON_CANDIDATES:
                wait_until_all_non_candidate_sold(portfolio, current_buy_codes)
                save_portfolio(portfolio)
            if CHECK_TP_SL_EVERY_LOOP:
                check_takeprofit_stoploss(portfolio)
                save_portfolio(portfolio)

            # 1) 각 후보 p, 현재가, f* 계산
            kelly_list = []
            for d in rows:
                code = d['종목코드']
                price = get_current_price(code)
                if not price or price <= 0:
                    print(f"❌ 현재가 조회 실패: {code}", flush=True)
                    continue
                p = extract_prob_from_row(d)
                fstar = compute_kelly_fraction(p, R)
                if fstar <= 0:
                    # 엣지 없음
                    continue
                kelly_list.append({
                    'code': code,
                    'price': price,
                    'p': p,
                    'R': R,
                    'fstar': fstar,
                })

            if not kelly_list:
                print("⚠️ 켈리 양수 종목이 없습니다. 다음 루프로 넘어갑니다.", flush=True)
                loop_count += 1
                time.sleep(600)
                continue

            # 2) f* 합으로 비례 배분 (총예산 및 개별 캡)
            sum_f = sum(x['fstar'] for x in kelly_list)
            allocated_total = 0
            for x in sorted(kelly_list, key=lambda z: z['fstar'], reverse=True):
                target_value = TOTAL_BUY_BUDGET_ALL * (x['fstar'] / sum_f)
                target_value = min(target_value, MAX_BUY_BUDGET)
                if ENFORCE_TOTAL_BUDGET_CAP:
                    remain = TOTAL_BUY_BUDGET_ALL - allocated_total
                    if remain <= 0:
                        target_value = 0
                    else:
                        target_value = min(target_value, remain)
                qty_target = int(target_value // x['price'])
                x['target_value'] = int(target_value)
                x['target_qty'] = max(0, qty_target)
                allocated_total += x['target_value']

            # 3) 리밸런싱 (증감 주문)
            for x in kelly_list:
                code = x['code']
                price = x['price']
                p = x['p']
                fstar = x['fstar']
                target_qty = x['target_qty']
                cur_qty = int(portfolio.get(code, {}).get('qty', 0))

                if target_qty > cur_qty:
                    add = target_qty - cur_qty
                    if add > 0:
                        order_result = send_order(code, price, qty=add, order_type="매수")
                        print(f"✅ 매수 {code}: +{add}주 @{price} → {order_result}", flush=True)
                        log_trade(datetime.now(), code, price, p, R, fstar, add, "매수", order_result)
                        if order_result.get("rt_cd") == "0":
                            buy_price = price
                            tp_price = adjust_price_to_tick(int(buy_price * (1 + TAKE_PROFIT_PCT)))
                            sl_price = adjust_price_to_tick(int(buy_price * (1 - STOP_LOSS_PCT)))
                            if code in portfolio:
                                portfolio[code]['qty'] += add
                                portfolio[code]['tp_price'] = portfolio[code].get('tp_price', tp_price)
                                portfolio[code]['sl_price'] = portfolio[code].get('sl_price', sl_price)
                                portfolio[code]['p'] = p
                                portfolio[code]['R'] = R
                                portfolio[code]['fstar'] = fstar
                            else:
                                portfolio[code] = {
                                    'buy_price': buy_price,
                                    'qty': add,
                                    'tp_price': tp_price,
                                    'sl_price': sl_price,
                                    'p': p,
                                    'R': R,
                                    'fstar': fstar,
                                }

                elif target_qty < cur_qty:
                    sell = cur_qty - target_qty
                    if sell > 0:
                        order_result = send_order(code, price, qty=sell, order_type="매도")
                        print(f"↘️ 부분 매도 {code}: -{sell}주 @{price} → {order_result}", flush=True)
                        log_trade(datetime.now(), code, price, p, R, fstar, sell, "매도", order_result)
                        if order_result.get("rt_cd") == "0":
                            portfolio[code]['qty'] -= sell
                            if portfolio[code]['qty'] <= 0:
                                del portfolio[code]
                else:
                    print(f"[유지] {code} 수량 {cur_qty}주 유지", flush=True)

            save_portfolio(portfolio)

            # 4) 평가금액 기록 & CSV 저장
            total_value = 0
            for code, pos in portfolio.items():
                shares = int(pos.get('qty', 0))
                if shares > 0:
                    last_price = get_current_price(code)
                    if last_price:
                        total_value += shares * last_price

            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cash = INITIAL_ASSET - sum(pos['qty'] * pos['buy_price'] for pos in portfolio.values())
            total_value = cash
            for code, pos in portfolio.items():
                shares = int(pos.get('qty', 0))
                if shares > 0:
                    last_price = get_current_price(code)
                    if last_price:
                        total_value += shares * last_price

            cum_return = (total_value / INITIAL_ASSET) - 1.0

            equity_curve.append({
                "time": now_str,
                "total_value": int(total_value),
                "cum_return": round(cum_return, 6)
            })

            csv_path = Path(OUTPUT_DIR) / "equity_curve.csv"
            pd.DataFrame(equity_curve).to_csv(
                csv_path,
                index=False,
                encoding='utf-8-sig'
            )
            print(f"💾 equity_curve 저장 완료: {csv_path}", flush=True)


            print(f"[Loop {loop_count}] 평가금액: {total_value:,.0f}, 누적수익률: {cum_return*100:.2f}%", flush=True)


            loop_count += 1
            time.sleep(600)  # 10분 간격

    except KeyboardInterrupt:
        print("사용자 중단! 누적 수익률 그래프/CSV 저장 중...", flush=True)

    finally:
        # 최종 Equity Curve CSV 재저장
        try:
            if 'equity_curve' in locals() and len(equity_curve) > 0:
                df_eq = pd.DataFrame(equity_curve)
                csv_path = Path(OUTPUT_DIR) / "equity_curve.csv"
                df_eq.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"누적 평가금액 CSV 저장 완료 ({csv_path})", flush=True)
        except Exception as e:
            print(f"CSV 저장 중 오류: {e}", flush=True)

        # 누적수익률 그래프 저장
        try:
            if 'equity_curve' in locals() and len(equity_curve) > 0:
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False

                plt.figure(figsize=(10, 6))
                plt.plot([x['total_value'] for x in equity_curve], label="누적 포트폴리오 값")
                plt.title("누적 수익률")
                plt.xlabel("룰 회수")
                plt.ylabel("포트폴리오 값")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(OUTPUT_DIR, "누적수익률_그래프.png")
                plt.savefig(out_path, dpi=300)
                print(f"누적 수익률 그래프 저장 완료 ({out_path})", flush=True)
            else:
                print("저장할 데이터가 없습니다.", flush=True)
        except Exception as e:
            print(f"그래프 저장 중 오류: {e}", flush=True)
