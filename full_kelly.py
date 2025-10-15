# -*- coding: utf-8 -*-

import os
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
# inserver 기준 경로 고정
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))     # .../fundweb/inserver
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BUYLIST_PATH   = os.path.join(OUTPUT_DIR, "buy_list.csv")
LOG_FILE       = Path(OUTPUT_DIR) / "trade_log.csv"
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portfolio.json")

# ─────────────────────────────────────────────
# 설정 파일: inserver/config_local.py
# ─────────────────────────────────────────────
try:
    from config_local import DB_CONFIG, get_api_keys, ACCOUNT_INFO
except Exception as e:
    print("❌ config_local.py 불러오기 실패! inserver 폴더에 존재해야 합니다.")
    print("   - config_local.py (secrets.json을 읽어 app_key/app_secret, CANO/ACNT_PRDT_CD 제공)")
    print("   - secrets.json (상시모의투자 키/계좌)")
    print(f"오류: {e}")
    raise SystemExit(1)

# ─────────────────────────────────────────────
# 전략/실행 설정
# ─────────────────────────────────────────────
USE_FULL_EQUITY        = True
EQUITY_UTILIZATION     = 1.0   # 100%
TOTAL_BUY_BUDGET_ALL   = 100_000_000
MAX_BUY_BUDGET         = 10_000_000

STOP_LOSS_PCT          = 0.025   # -2.5%
TAKE_PROFIT_PCT        = 0.05    # +5%
ENFORCE_TOTAL_BUDGET_CAP = True

SELL_NON_CANDIDATES    = True
CHECK_TP_SL_EVERY_LOOP = True

# 실주문 방지 스위치 (기본 ON)
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"

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

# ───────────── 토큰 ─────────────
def issue_access_token():
    app_key, app_secret = get_api_keys()
    if not app_key or not app_secret:
        print("❌ app_key/app_secret 비어 있음.")
        raise SystemExit(1)
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
        print("❌ 토큰 JSON 파싱 실패:", res.text)
        raise SystemExit(1)
    token = j.get("access_token", "")
    if not token:
        print("❌ 액세스 토큰 발급 실패:", j)
        raise SystemExit(1)
    print("🔐 액세스 토큰 발급 성공", flush=True)
    return token, app_key, app_secret

access_token, app_key, app_secret = issue_access_token()

# ───────────── API ─────────────
def get_current_price(stock_code: str) -> int | None:
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
    if res.status_code != 200:
        return None
    try:
        pr = int(res.json()['output']['stck_prpr'])
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
        "tr_id": "VTTC8434R"
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
        "CTX_AREA_NK100": ""
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
    if DRY_RUN:
        return {"rt_cd": "0", "msg1": "DRY_RUN: no order sent"}
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"
    adjusted_price = adjust_price_to_tick(price)
    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": "00",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(adjusted_price))
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "hashkey": get_hashkey(data)
    }
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    try:
        return res.json()
    except Exception:
        return {"rt_cd": "-1", "msg1": "INVALID_JSON"}

def get_account_totals(portfolio_snapshot: dict | None = None) -> tuple[int, int, int]:
    """
    반환: (total_equity, cash, stock_eval)
    """
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R"
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
        "CTX_AREA_NK100": ""
    }
    cash = 0
    stock_eval = 0
    total_equity = 0
    try:
        res = requests.get(url, headers=headers, params=params)
        time.sleep(1.2)
        j = res.json()

        out2 = (j.get("output2") or [{}])[0]
        if isinstance(out2, dict):
            for k in ["dnca_tot_amt", "dnca_avlb_amt", "nxdy_excc_amt", "prvs_rcdl_excc_amt"]:
                if k in out2 and str(out2[k]).strip():
                    cash = int(float(out2[k]))
                    break
            for k in ["tot_evlu_amt", "scts_evlu_amt", "evlu_amt_smtl"]:
                if k in out2 and str(out2[k]).strip():
                    stock_eval = int(float(out2[k]))
                    break

        tmp_eval = 0
        for it in j.get("output1", []):
            try:
                qty = int(it.get("hldg_qty", 0))
                prpr = int(it.get("prpr", 0))
                if prpr <= 0:
                    cd = it.get("pdno")
                    last = get_current_price(cd) if cd else 0
                    prpr = last or 0
                tmp_eval += qty * prpr
            except Exception:
                pass
        if tmp_eval > 0:
            stock_eval = tmp_eval

        if stock_eval and cash:
            total_equity = stock_eval + cash
        elif stock_eval:
            total_equity = stock_eval
        elif cash:
            total_equity = cash
        else:
            est = 0
            if portfolio_snapshot:
                for code, pos in portfolio_snapshot.items():
                    try:
                        est += int(pos.get("qty", 0)) * (get_current_price(code) or 0)
                    except Exception:
                        pass
            total_equity = est
    except Exception:
        est = 0
        if portfolio_snapshot:
            for code, pos in portfolio_snapshot.items():
                try:
                    est += int(pos.get("qty", 0)) * (get_current_price(code) or 0)
                except Exception:
                    pass
        total_equity = est
        cash = 0
        stock_eval = est
    return int(total_equity), int(cash), int(stock_eval)

# ───────────── 포트폴리오 & 로깅 ─────────────
def load_portfolio() -> dict:
    p = Path(PORTFOLIO_PATH)
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}

def save_portfolio(data: dict):
    json.dump(data, open(PORTFOLIO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def log_trade(timestamp, stock_code, price, p, R, fstar, qty, order_type, order_result):
    ts = timestamp if isinstance(timestamp, str) else timestamp.strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": ts,
        "code": stock_code,
        "price": int(price),
        "p_pct": round(float(p) * 100, 2),
        "R": round(float(R), 3),
        "kelly_f": round(float(fstar), 4),
        "qty": int(qty),
        "side": order_type,
        "result_msg": order_result.get("msg1", "NO_RESPONSE")
    }
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

# ───────────── 보조: 비후보 정리 & TP/SL ─────────────
def wait_until_all_non_candidate_sold(portfolio: dict, current_buy_codes: set[str]):
    if not SELL_NON_CANDIDATES:
        return
    has_non = True
    while has_non:
        has_non = False
        for code in list(portfolio.keys()):
            if code not in current_buy_codes:
                real_shares = get_real_balance_qty(code)
                if real_shares > 0:
                    last_price = get_current_price(code) or 0
                    res = send_order(code, last_price, qty=real_shares, order_type="매도")
                    print(f"🔁 [비후보 매도] {code}: {real_shares}주 → {res}", flush=True)
                    log_trade(datetime.now(), code, last_price,
                              portfolio[code].get("p", 0),
                              portfolio[code].get("R", 0),
                              portfolio[code].get("fstar", 0),
                              real_shares, "매도", res)
                    if res.get("rt_cd") == "0" or res.get("msg_cd") == "40240000":
                        del portfolio[code]
                        has_non = True
                else:
                    del portfolio[code]
                    has_non = True
        if has_non:
            print("비후보 정리 체결 대기… 10초", flush=True)
            time.sleep(10)

def check_takeprofit_stoploss(portfolio: dict):
    if not CHECK_TP_SL_EVERY_LOOP:
        return
    for code, pos in list(portfolio.items()):
        shares = int(pos.get('qty', 0))
        if shares <= 0:
            continue
        last = get_current_price(code)
        if last is None:
            continue
        tp = pos.get('tp_price')
        sl = pos.get('sl_price')
        if tp and last >= tp:
            res = send_order(code, last, qty=shares, order_type="매도")
            print(f"🎯 [익절] {code} {shares}주 @ {last} → {res}", flush=True)
            log_trade(datetime.now(), code, last, pos.get("p", 0), pos.get("R", 0), pos.get("fstar", 0), shares, "매도", res)
            if res.get("rt_cd") == "0":
                del portfolio[code]
                continue
        if sl and last <= sl:
            res = send_order(code, last, qty=shares, order_type="매도")
            print(f"🛑 [손절] {code} {shares}주 @ {last} → {res}", flush=True)
            log_trade(datetime.now(), code, last, pos.get("p", 0), pos.get("R", 0), pos.get("fstar", 0), shares, "매도", res)
            if res.get("rt_cd") == "0":
                del portfolio[code]

# ───────────── 켈리 ─────────────
def extract_prob_from_row(row: dict) -> float:
    keys = ["prob_up", "p", "prob", "상승확률", "확률"]
    for k in row.keys():
        if str(k).lower() in [x.lower() for x in keys]:
            try:
                p = float(row[k])
                if p > 1.0:
                    p = p / 100.0
                return max(0.0, min(1.0, p))
            except Exception:
                pass
    return 0.55

def compute_kelly_fraction(p: float, R: float) -> float:
    q = 1.0 - p
    return p - (q / R)

# ───────────── 메인 ─────────────
if __name__ == "__main__":
    print("📊 buy_list.csv 로드 중…", flush=True)
    if not os.path.exists(BUYLIST_PATH):
        print("❌ buy_list.csv 없음:", os.path.abspath(BUYLIST_PATH))
        raise SystemExit(1)

    df_cand = pd.read_csv(BUYLIST_PATH, dtype={'종목코드': str, 'code': str})
    rows = []
    for _, row in df_cand.iterrows():
        d = row.to_dict()
        code = (d.get('종목코드') or d.get('code') or '').zfill(6)
        if not code:
            continue
        d['종목코드'] = code
        rows.append(d)
    if not rows:
        print("❌ 유효 후보 없음")
        raise SystemExit(1)

    current_buy_codes = set([r['종목코드'] for r in rows])
    print(f"✅ 후보 수: {len(rows)}", flush=True)

    loop_count = 1
    portfolio = load_portfolio()
    equity_curve = []
    R = TAKE_PROFIT_PCT / STOP_LOSS_PCT

    try:
        while True:
            print(f"\n[LOOP {loop_count}] {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

            if SELL_NON_CANDIDATES:
                wait_until_all_non_candidate_sold(portfolio, current_buy_codes)
                save_portfolio(portfolio)
            if CHECK_TP_SL_EVERY_LOOP:
                check_takeprofit_stoploss(portfolio)
                save_portfolio(portfolio)

            if USE_FULL_EQUITY:
                total_equity, cash, stock_eval = get_account_totals(portfolio)
                effective_total_budget = int(total_equity * EQUITY_UTILIZATION)
                if effective_total_budget <= 0:
                    print("⚠️ 총자산 0으로 인식 → skip", flush=True)
                    loop_count += 1
                    time.sleep(600)
                    continue
                effective_max_per_stock = float("inf")
                print(f"💰 Total equity={total_equity:,} / utilization={EQUITY_UTILIZATION*100:.0f}% → budget={effective_total_budget:,}", flush=True)
            else:
                effective_total_budget = int(TOTAL_BUY_BUDGET_ALL)
                effective_max_per_stock = MAX_BUY_BUDGET

            # 후보별 p, f* 계산
            kelly_list = []
            for d in rows:
                code = d['종목코드']
                price = get_current_price(code)
                if not price or price <= 0:
                    print(f"❌ 현재가 실패: {code}", flush=True)
                    continue
                p = extract_prob_from_row(d)
                fstar = compute_kelly_fraction(p, R)
                if fstar <= 0:
                    continue
                kelly_list.append({
                    'code': code,
                    'price': price,
                    'p': p,
                    'R': R,
                    'fstar': fstar
                })

            if not kelly_list:
                print("⚠️ 양수 켈리 없음 → sleep", flush=True)
                loop_count += 1
                time.sleep(600)
                continue

            # f* 비례 배분
            sum_f = sum(x['fstar'] for x in kelly_list)
            allocated_total = 0
            for x in sorted(kelly_list, key=lambda z: z['fstar'], reverse=True):
                target_value = effective_total_budget * (x['fstar'] / sum_f)
                if np.isfinite(MAX_BUY_BUDGET):
                    target_value = min(target_value, float('inf'))  # no cap when using full equity
                if ENFORCE_TOTAL_BUDGET_CAP:
                    remain = effective_total_budget - allocated_total
                    target_value = 0 if remain <= 0 else min(target_value, remain)
                qty_target = int(target_value // x['price'])
                x['target_value'] = int(target_value)
                x['target_qty'] = max(0, qty_target)
                allocated_total += x['target_value']

            # 리밸런싱
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
                        res = send_order(code, price, qty=add, order_type="매수")
                        print(f"✅ BUY {code}: +{add} @{price} → {res}", flush=True)
                        log_trade(datetime.now(), code, price, p, R, fstar, add, "매수", res)
                        if res.get("rt_cd") == "0":
                            buy_price = price
                            tp = adjust_price_to_tick(int(buy_price * (1 + TAKE_PROFIT_PCT)))
                            sl = adjust_price_to_tick(int(buy_price * (1 - STOP_LOSS_PCT)))
                            if code in portfolio:
                                portfolio[code]['qty'] += add
                                portfolio[code]['tp_price'] = portfolio[code].get('tp_price', tp)
                                portfolio[code]['sl_price'] = portfolio[code].get('sl_price', sl)
                                portfolio[code]['p'] = p
                                portfolio[code]['R'] = R
                                portfolio[code]['fstar'] = fstar
                            else:
                                portfolio[code] = {
                                    'buy_price': buy_price,
                                    'qty': add,
                                    'tp_price': tp,
                                    'sl_price': sl,
                                    'p': p,
                                    'R': R,
                                    'fstar': fstar
                                }

                elif target_qty < cur_qty:
                    sell = cur_qty - target_qty
                    if sell > 0:
                        res = send_order(code, price, qty=sell, order_type="매도")
                        print(f"↘️ SELL {code}: -{sell} @{price} → {res}", flush=True)
                        log_trade(datetime.now(), code, price, p, R, fstar, sell, "매도", res)
                        if res.get("rt_cd") == "0":
                            portfolio[code]['qty'] -= sell
                            if portfolio[code]['qty'] <= 0:
                                del portfolio[code]
                else:
                    print(f"[HOLD] {code} qty {cur_qty}", flush=True)

            save_portfolio(portfolio)

            # 평가금액 기록 & CSV 저장
            total_value = 0
            for code, pos in portfolio.items():
                shares = int(pos.get('qty', 0))
                if shares > 0:
                    last = get_current_price(code)
                    if last:
                        total_value += shares * last

            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            equity_curve.append({"time": now_str, "total_value": int(total_value)})
            pd.DataFrame(equity_curve).to_csv(Path(OUTPUT_DIR) / "equity_curve.csv",
                                              index=False, encoding='utf-8-sig')
            print(f"[Loop {loop_count}] total value: {total_value:,.0f}", flush=True)

            loop_count += 1
            time.sleep(600)

    except KeyboardInterrupt:
        print("사용자 중단! 누적 수익률 그래프/CSV 저장 중...", flush=True)

    finally:
        try:
            if len(equity_curve) > 0:
                df_eq = pd.DataFrame(equity_curve)
                csv_path = Path(OUTPUT_DIR) / "equity_curve.csv"
                df_eq.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ Equity CSV saved ({csv_path})", flush=True)
        except Exception as e:
            print(f"CSV 저장 중 오류: {e}", flush=True)

        try:
            if len(equity_curve) > 0:
                plt.rcParams['axes.unicode_minus'] = False
                plt.figure(figsize=(10, 6))
                plt.plot([x['total_value'] for x in equity_curve], label="Cumulative Portfolio Value")
                plt.title("Cumulative Return")
                plt.xlabel("Loop")
                plt.ylabel("Portfolio Value")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
                plt.savefig(out_path, dpi=300)
                print(f"✅ Equity curve saved ({out_path})", flush=True)
            else:
                print("저장할 데이터가 없습니다.", flush=True)
        except Exception as e:
            print(f"그래프 저장 중 오류: {e}", flush=True)
