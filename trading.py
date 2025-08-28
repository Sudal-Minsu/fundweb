import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, time as dtime
from pathlib import Path
import keyring
import pymysql

# ───────────── 설정 ─────────────
APP_USER = "최진혁"
APP_KEY_SERVICE = "mock_app_key"
APP_SECRET_SERVICE = "mock_app_secret"

DEFAULT_APP_KEY = "PSbWOQW9CsjVIq8MwF3oeHG9gY9JjLHJVu8t"
DEFAULT_APP_SECRET = (
    "uzxSVMytr/jWcbCYMBGcRMloeCM9A1fiTOur3Y3j30RY6gtvf3G0Bn1y/"
    "z6J2pa0CKKZRFf6OXpk/umYfxZaWQr4eVmoCJG6BX7wfQ/GOYlEDotyouzkMwevv7hjI06tzruSpPuN6EMS1nirtIeTnh8kxxN4LBS70XggdFevyM3KR87RG7k="
)

def _ensure_keyring():
    if keyring.get_password(APP_KEY_SERVICE, APP_USER) is None:
        keyring.set_password(APP_KEY_SERVICE, APP_USER, DEFAULT_APP_KEY)
    if keyring.get_password(APP_SECRET_SERVICE, APP_USER) is None:
        keyring.set_password(APP_SECRET_SERVICE, APP_USER, DEFAULT_APP_SECRET)

def get_api_keys():
    _ensure_keyring()
    return (
        keyring.get_password(APP_KEY_SERVICE, APP_USER),
        keyring.get_password(APP_SECRET_SERVICE, APP_USER),
    )

ACCOUNT_INFO = {"CANO": "50139282", "ACNT_PRDT_CD": "01"}

OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 종목당 투자금: 전일 거래대금의 0.25% (최대 5천만)
MAX_PER_STOCK_BUDGET = 50_000_000     # 종목당 상한
INVEST_RATE_FROM_PREV_TV = 0.0025      # 0.25%

LOOP_SLEEP_SEC = 180

# ───────────── 시간 상수 ─────────────
CANCEL_BUY_START_TIME = dtime(14, 55) # 매수 미체결 취소 시작 시각
FORCE_SELL_TIME       = dtime(15, 0)  # 15:00 강제 매도
MARKET_CLOSE_TIME     = dtime(15, 30) # 15:30

# 상태 파일
BOUGHT_TODAY_PATH       = os.path.join(OUTPUT_DIR, "bought_today.json")
NOT_TRADABLE_TODAY_PATH = os.path.join(OUTPUT_DIR, "not_tradable_today.json")
DAILY_PNL_CSV           = os.path.join(OUTPUT_DIR, "평가자료.csv")

# ───────────── DB 설정 ─────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "port": 3306,
    "database": "news_db",
    "charset": "utf8mb4"
}

# ───────────── 세션/인증 상태 (import 시 실행되지 않도록 분리) ─────────────
SESSION = {
    "url_base": "https://openapivts.koreainvestment.com:29443",  # 모의투자 VTS
    "app_key": None,
    "app_secret": None,
    "access_token": None,
    "inited": False,
}

def init_session(app_key=None, app_secret=None, url_base=None):
    """
    인증/세션 초기화. import 시 자동으로 실행되지 않으며,
    main() 또는 외부에서 명시적으로 호출하거나 ensure_session()이 필요 시 호출합니다.
    """
    if url_base:
        SESSION["url_base"] = url_base

    if not app_key or not app_secret:
        app_key, app_secret = get_api_keys()

    # 토큰 발급
    res = requests.post(
        f"{SESSION['url_base']}/oauth2/tokenP",
        headers={"content-type": "application/json"},
        data=json.dumps({
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret
        })
    )
    access_token = res.json().get("access_token", "")
    if not access_token:
        raise RuntimeError(f"❌ 액세스 토큰 발급 실패: {res.text}")

    SESSION["app_key"] = app_key
    SESSION["app_secret"] = app_secret
    SESSION["access_token"] = access_token
    SESSION["inited"] = True
    print("🔑 액세스 토큰 OK", flush=True)

def is_session_ready():
    return SESSION.get("inited") and bool(SESSION.get("access_token"))

def ensure_session():
    """
    세션이 없으면 즉시 초기화. 외부에서 함수 단독 호출할 때도 안전하게 사용 가능.
    """
    if not is_session_ready():
        init_session()

# ───────────── 유틸 ─────────────
def _num(x):
    if x is None: return None
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "null": return None
    try: return float(s)
    except: return None

def _num0(x):
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() == "null": return 0.0
        return float(s)
    except:
        return 0.0

def is_market_closed_msg(msg: str) -> bool:
    if not msg: return False
    m = msg.strip().lower()
    return ("장종료" in m) or ("장 종료" in m) or ("closed" in m) or ("market closed" in m)

def _z6(code):  # 종목코드 zero-fill
    return str(code).zfill(6)

# ───────────── 스로틀: 조회/취소/주문 ─────────────
_last_get_ts = 0.0
def throttle_reads(min_interval=0.6):
    """조회 계열(API GET) 최소 간격 보장 + 지터로 정각 충돌 분산"""
    import random
    global _last_get_ts
    dt = time.monotonic() - _last_get_ts
    wait = max(0.0, min_interval - dt) + random.uniform(0, 0.15)
    if wait > 0:
        time.sleep(wait)
    _last_get_ts = time.monotonic()

_last_order_ts = 0.0
def send_order_throttled(stock_code, price, qty, order_type="매수", ord_dvsn="00", min_interval=0.8):
    """주문 API 최소 간격 보장"""
    global _last_order_ts
    dt = time.monotonic() - _last_order_ts
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_order_ts = time.monotonic()
    return send_order(stock_code, price, qty, order_type=order_type, ord_dvsn=ord_dvsn)

_last_cancel_ts = 0.0
def send_cancel_order_throttled(*, ord_orgno, orgn_odno, ord_dvsn, qty_all=True, qty=0, min_interval=0.6):
    """취소 API 최소 간격 보장"""
    global _last_cancel_ts
    dt = time.monotonic() - _last_cancel_ts
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_cancel_ts = time.monotonic()
    return send_cancel_order(ord_orgno=ord_orgno, orgn_odno=orgn_odno, ord_dvsn=ord_dvsn, qty_all=qty_all, qty=qty)

# ───────────── 슬립 헬퍼: 15:00 보장 + 크리티컬 타임 보장 ─────────────
def sleep_until_3pm_or(default_sec: int):
    """
    기본 슬립(default_sec) 하되, 오늘 15:00을 절대 넘기지 않도록 절단.
    """
    now = datetime.now()
    target = datetime.combine(now.date(), dtime(15, 0))
    remain = max(0.0, (target - now).total_seconds())
    time.sleep(max(0.2, min(default_sec, remain - 0.3)))  # 0.3초 여유

CRITICALS = [CANCEL_BUY_START_TIME, dtime(14, 57), FORCE_SELL_TIME, MARKET_CLOSE_TIME]

def smart_sleep(default_sec: int):
    """
    다음 크리티컬 시각(14:55/14:57/15:00/15:30) 이전까지만 수면.
    내부적으로도 15:00 절단을 한 번 더 보장.
    """
    now = datetime.now()
    remains = []
    for t in CRITICALS:
        tgt = datetime.combine(now.date(), t)
        if tgt <= now:
            continue
        remains.append((tgt - now).total_seconds())
    limit = min(remains) if remains else default_sec
    planned = max(0.2, min(default_sec, limit - 0.3))
    # 15:00 가드까지 겹쳐서 보장
    now2 = datetime.now()
    target_3pm = datetime.combine(now2.date(), dtime(15, 0))
    remain_3pm = max(0.0, (target_3pm - now2).total_seconds())
    final_sleep = max(0.2, min(planned, remain_3pm - 0.3))
    time.sleep(final_sleep)

# ───────────── ATR14% 계산용 DB 접근 ─────────────
def fetch_ohlc_for_codes_from_db(codes, rows_per_code=20):
    if not codes:
        return {}
    codes6 = [_z6(c) for c in codes]

    conn = pymysql.connect(**DB_CONFIG)
    out = {}
    try:
        with conn.cursor() as cur:
            for c in codes6:
                sql = """
                    SELECT Date, Code, High, Low, Close
                    FROM stock_data
                    WHERE (Code = %s OR Code = LPAD(%s, 6, '0') OR Code = CAST(%s AS UNSIGNED))
                    ORDER BY Date DESC
                    LIMIT %s
                """
                c_int = int(c)
                cur.execute(sql, (c, c, c_int, rows_per_code))
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=["Date","Code","High","Low","Close"])
                if df.empty:
                    out[c] = pd.DataFrame(columns=["Date","Code","High","Low","Close"])
                    continue

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                for col in ["High","Low","Close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna(subset=["Date","High","Low","Close"]).copy()
                df["Code"] = df["Code"].astype(str).str.zfill(6)
                df = df.sort_values(["Date"]).reset_index(drop=True)

                out[c] = df[["Date","Code","High","Low","Close"]].reset_index(drop=True)
    finally:
        conn.close()

    return out

def compute_atr14_from_df(df):
    if df is None or df.empty:
        return None

    for col in ["High","Low","Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["High","Low","Close"]).copy()

    # rolling(14) 계산을 위해 최소 15행 필요
    if len(df) < 15:
        return None

    prev_close = df["Close"].shift(1)
    tr1 = (df["High"] - df["Low"]).abs()
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"]  - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr14 = tr.rolling(14, min_periods=14).mean()

    last_atr = atr14.iloc[-1]
    last_close = df["Close"].iloc[-1]
    if pd.notna(last_atr) and pd.notna(last_close) and last_close > 0:
        return float(last_atr / last_close * 100.0)
    return None

def rank_by_atr14_from_db(codes, rows_per_code=20):
    dfs = fetch_ohlc_for_codes_from_db(codes, rows_per_code=rows_per_code)
    scored = []
    for c in codes:
        c6 = _z6(c)
        atr_pct = compute_atr14_from_df(dfs.get(c6))
        if atr_pct is not None and atr_pct > 0:
            scored.append((c6, atr_pct))

    if not scored:
        return [_z6(c) for c in codes], {}

    scored.sort(key=lambda x: x[1], reverse=True)
    sorted_codes = [c for c, _ in scored]
    atr_map = {c: a for c, a in scored}
    return sorted_codes, atr_map

# ───────────── 전일 거래대금 조회 ─────────────
def get_prev_trading_value_from_db(code):
    """전일 거래대금(전일 Close × 전일 Volume)"""
    code6 = _z6(code)
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT Date, Code, Close, Volume
                FROM stock_data
                WHERE (Code = %s OR Code = LPAD(%s, 6, '0') OR Code = CAST(%s AS UNSIGNED))
                ORDER BY Date DESC
                LIMIT 2
            """
            c_int = int(code6)
            cur.execute(sql, (code6, code6, c_int))
            rows = cur.fetchall()
            if len(rows) < 2:
                return None
            _, _, close_prev, vol_prev = rows[1]
            close_prev = _num(close_prev)
            vol_prev = _num(vol_prev)
            if close_prev is None or vol_prev is None:
                return None
            return float(close_prev) * float(vol_prev)
    finally:
        conn.close()

def build_prev_trading_value_map(codes):
    out = {}
    for c in codes:
        c6 = _z6(c)
        out[c6] = get_prev_trading_value_from_db(c6)
    return out

# ───────────── 시세/주문 ─────────────
def get_quote(stock_code):
    ensure_session()
    throttle_reads()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": "FHKST01010100"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    res = requests.get(url, headers=headers, params=params); time.sleep(1.2)
    try:
        j = res.json()
    except:
        print(f"❌ 시세 조회 실패: {stock_code} / {res.text}", flush=True)
        return None, None, None
    if res.status_code != 200 or 'output' not in j:
        print(f"❌ 시세 조회 실패: {stock_code} / {res.text}", flush=True)
        return None, None, None
    out = j['output']
    to_int = lambda x: int(str(x).replace(",", "").strip()) if x not in (None, "") else None
    cur  = to_int(out.get('stck_prpr'))
    ask1 = to_int(out.get('askp1') or out.get('askp'))
    bid1 = to_int(out.get('bidp1') or out.get('bidp'))
    return cur, ask1, bid1

def get_current_price(stock_code):
    cur, _, _ = get_quote(stock_code); return cur

def get_hashkey(data):
    ensure_session()
    url = f"{SESSION['url_base']}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"]}
    res = requests.post(url, headers=headers, data=json.dumps(data)); time.sleep(1.2)
    try:
        return res.json().get("HASH", "")
    except:
        return ""

def send_order(stock_code, price, qty, order_type="매수", ord_dvsn="00"):
    """
    - 매수: 04(최우선지정가) → ORD_UNPR=0
    - 매도: 01(시장가)      → ORD_UNPR=0
    """
    ensure_session()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"

    price_free_types = {"01","03","04","11","12","13","14","15","16"}
    unpr = "0" if ord_dvsn in price_free_types else str(int(price))

    data = {
        "CANO": ACCOUNT_INFO["CANO"], "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code, "ORD_DVSN": ord_dvsn, "ORD_QTY": str(qty), "ORD_UNPR": unpr
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": tr_id, "hashkey": get_hashkey(data)
    }
    res = requests.post(url, headers=headers, data=json.dumps(data)); time.sleep(1.2)
    try: return res.json()
    except: return {"rt_cd": "-1", "msg1": res.text}

# ───────────── 정정/취소 ─────────────
def send_cancel_order(ord_orgno, orgn_odno, ord_dvsn, qty_all=True, qty=0):
    ensure_session()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/order-rvsecncl"
    tr_id = "VTTC0803U"  # 모의: VTTC0803U
    params = {
        "CANO": ACCOUNT_INFO["CANO"], "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "KRX_FWDG_ORD_ORGNO": str(ord_orgno), "ORGN_ODNO": str(orgn_odno),
        "ORD_DVSN": str(ord_dvsn),
        "RVSE_CNCL_DVSN_CD": "02",           # 취소
        "ORD_QTY": "0" if qty_all else str(int(qty)),
        "ORD_UNPR": "0", "QTY_ALL_ORD_YN": "Y" if qty_all else "N"
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": tr_id, "hashkey": get_hashkey(params),
    }
    res = requests.post(url, headers=headers, data=json.dumps(params)); time.sleep(1.2)
    try: return res.json()
    except: return {"rt_cd": "-1", "msg1": res.text}

def list_cancelable_buy_orders():
    ensure_session()
    throttle_reads()
    """
    1차: 정정/취소 가능 주문 조회 (환경에 따라 비어 나올 수 있음)
    """
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
    tr_id = "VTTC8036R"  # 모의: VTTC8036R
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": tr_id,
    }

    def _call_once(inqr1, inqr2):
        params = {
            "CANO": ACCOUNT_INFO["CANO"], "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
            "INQR_DVSN_1": str(inqr1), "INQR_DVSN_2": str(inqr2),
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        }
        out = []
        while True:
            throttle_reads()
            res = requests.get(url, headers=headers, params=params); time.sleep(1.2)
            j = res.json()
            rows = j.get("output2") or j.get("output") or []
            for r in rows:
                orgno = r.get("krx_fwdg_ord_orgno") or r.get("KRX_FWDG_ORD_ORGNO") or r.get("ord_gno_brno")
                odno  = r.get("odno") or r.get("ODNO")
                dvsn  = r.get("ord_dvsn") or r.get("ORD_DVSN") or "00"
                pdno  = r.get("pdno") or r.get("PDNO")

                rmn1 = _num0(r.get("rmn_qty") or r.get("RMN_QTY"))
                rmn2 = _num0(r.get("unerc_qty") or r.get("UNERC_QTY"))
                ord_qty = _num0(r.get("ord_qty") or r.get("ORD_QTY"))
                tot_ccld_qty = _num0(r.get("tot_ccld_qty") or r.get("TOT_CCLD_QTY"))
                ccld_qty = _num0(r.get("ccld_qty") or r.get("CCLD_QTY"))
                rmn3 = max(0, int(round(ord_qty - max(tot_ccld_qty, ccld_qty))))
                rmn_qty = max(int(rmn1), int(rmn2), int(rmn3))

                side_txt = (r.get("sll_buy_dvsn_cd") or r.get("SLL_BUY_DVSN_CD")
                            or r.get("sll_buy_dvsn_name") or r.get("SLL_BUY_DVSN_NAME") or "")
                side_str = str(side_txt)
                is_buy = ("02" in side_str) or ("매수" in side_str)

                out.append({
                    "krx_fwdg_ord_orgno": str(orgno) if orgno is not None else "",
                    "odno": str(odno) if odno is not None else "",
                    "ord_dvsn": str(dvsn) if dvsn is not None else "00",
                    "rmn_qty": int(rmn_qty),
                    "pdno": str(pdno).zfill(6) if pdno else "",
                    "is_buy": bool(is_buy),
                })
            if j.get("tr_cont", "F") != "M":
                break
            params["CTX_AREA_FK100"] = j.get("ctx_area_fk100", "")
            params["CTX_AREA_NK100"] = j.get("ctx_area_nk100", "")
        return out

    out = _call_once(0, 0)
    if not out:
        out = _call_once(1, 2)
    return out

# ───────────── 조회 ─────────────
def get_all_holdings():
    ensure_session()
    throttle_reads()
    """
    주식잔고조회 (연속조회 포함)
    - API가 1페이지당 최대 20건 반환하므로, tr_cont/ctx 키로 모든 페이지 수집
    - 모의투자 tr_id: VTTC8434R (실계좌는 TTTC8434R)
    """
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-balance"

    headers_base = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"],
        "appSecret": SESSION["app_secret"],
        "tr_id": "VTTC8434R",
    }

    fk, nk = "", ""
    merged = {}

    while True:
        headers = dict(headers_base)
        # 첫 페이지는 tr_cont 생략, 이후 페이지부터 tr_cont='N'
        if fk or nk:
            headers["tr_cont"] = "N"

        params = {
            "CANO": ACCOUNT_INFO["CANO"],
            "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "00",  # 정렬/조회구분 (20건 페이징 허용)
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",  # 처리구분(연속조회에 필요)
            "CTX_AREA_FK100": fk,
            "CTX_AREA_NK100": nk,
        }

        throttle_reads()
        res = requests.get(url, headers=headers, params=params); time.sleep(1.2)
        j = res.json()

        for item in (j.get("output1") or []):
            code = str(item.get("pdno", "")).zfill(6)
            qty  = _num(item.get("hldg_qty"))
            if not qty or qty <= 0:
                continue
            avg = _num(item.get("pchs_avg_pric"))
            cur = _num(item.get("prpr"))
            merged[code] = {
                "qty": int(qty),
                "avg_price": (avg if avg and avg > 0 else None),
                "cur_price": (cur if cur and cur > 0 else None),
            }

        # 다음 페이지 키
        fk = (j.get("ctx_area_fk100") or "").rstrip()
        nk = (j.get("ctx_area_nk100") or "").rstrip()
        # 응답 헤더 우선, 없으면 바디의 tr_cont 참조
        next_flag = (res.headers.get("tr_cont") or j.get("tr_cont") or "").strip()

        # 'M' 또는 'F'면 계속, 그 외면 종료. 또한 키가 비면 종료
        if next_flag not in ("M", "F"):
            break
        if not (fk or nk):
            break

    return merged

def get_today_orders():
    ensure_session()
    throttle_reads()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": "VTTC0081R",
    }
    today = datetime.now().strftime("%Y%m%d")
    params = {
        "CANO": ACCOUNT_INFO["CANO"], "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "INQR_STRT_DT": today, "INQR_END_DT": today,
        "SLL_BUY_DVSN_CD": "00", "INQR_DVSN": "00", "PDNO": "",
        "CCLD_DVSN": "00", "ORD_GNO_BRNO": "", "ODNO": "",
        "INQR_DVSN_3": "00", "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
    }
    items = []
    while True:
        throttle_reads()
        res = requests.get(url, headers=headers, params=params); time.sleep(1.2)
        j = res.json()
        items.extend(j.get("output1", []) or [])
        if j.get("tr_cont", "F") != "M": break
        params["CTX_AREA_FK100"] = j.get("ctx_area_fk100", "")
        params["CTX_AREA_NK100"] = j.get("ctx_area_nk100", "")
    return items

def build_bought_today_set(today_orders):
    bought = set()
    for o in today_orders:
        code = str(o.get("pdno", "")).zfill(6)
        side_txt = (o.get("sll_buy_dvsn_cd") or o.get("sll_buy_dvsn_name")
                    or o.get("trad_dvsn_name") or "").strip()
        is_buy   = ("매수" in side_txt) or (str(side_txt) in ("02","2"))
        if code and is_buy: bought.add(code)
    return bought

def has_open_orders(today_orders):
    def _text(o, *keys):
        for k in keys:
            v = o.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
        return ""
    terminal = ["거부","불가","매매불가","주문거절","취소","정정거부","오류",
                "rejected","reject","cancel","canceled","cancelled","error","invalid"]
    for o in today_orders:
        st = _text(o,"ordr_sttus_name","ccld_dvsn_name","ord_sttus"); stl = st.lower()
        if any(k in st for k in terminal) or any(k in stl for k in terminal): continue
        if "미체결" in st: return True
        rmn = _num0(o.get("rmn_qty")) or _num0(o.get("unerc_qty"))
        if rmn > 0: return True
        ord_qty=_num0(o.get("ord_qty")); c1=_num0(o.get("tot_ccld_qty")); c2=_num0(o.get("ccld_qty"))
        if ord_qty>0 and max(c1,c2)<ord_qty: return True
    return False

def get_open_sell_qty_for_code(today_orders, code: str) -> int:
    code = str(code).zfill(6)
    def _text(o,*keys):
        for k in keys:
            v=o.get(k)
            if isinstance(v,str) and v.strip(): return v.strip()
        return ""
    terminal = ["거부","불가","매매불가","주문거절","취소","정정거부","오류",
                "rejected","reject","cancel","canceled","cancelled","error","invalid"]
    open_qty = 0
    for o in today_orders or []:
        if str(o.get("pdno","")).zfill(6) != code: continue
        side_txt = (o.get("sll_buy_dvsn_cd") or o.get("sll_buy_dvsn_name")
                    or o.get("trad_dvsn_name") or "").strip()
        is_sell = ("매도" in side_txt) or (str(side_txt) in ("01","1"))
        if not is_sell: continue
        st=_text(o,"ordr_sttus_name","ccld_dvsn_name","ord_sttus"); stl=st.lower()
        if any(k in st for k in terminal) or any(k in stl for k in terminal): continue
        rmn = _num0(o.get("rmn_qty") or o.get("unerc_qty"))
        if rmn>0: open_qty += int(rmn)
        ord_qty=_num0(o.get("ord_qty")); c1=_num0(o.get("tot_ccld_qty")); c2=_num0(o.get("ccld_qty"))
        gap=max(0, int(round(ord_qty - max(c1,c2))))
        if gap>open_qty: open_qty=gap
    return int(open_qty)

# ───────────── 폴백: 당일 주문조회 기반 취소 후보 ─────────────
def build_cancelables_from_today_orders(today_orders):
    """
    inquire-daily-ccld 결과에서 '매수 & 잔량>0'을 취소 후보로 추출.
    반환: [{krx_fwdg_ord_orgno, odno, ord_dvsn, rmn_qty, pdno, is_buy}]
    """
    out = []
    for o in today_orders or []:
        side_txt = str(o.get("sll_buy_dvsn_cd") or o.get("sll_buy_dvsn_name") or o.get("trad_dvsn_name") or "")
        is_buy = ("매수" in side_txt) or (str(side_txt) in ("02","2"))
        rmn = int(_num0(o.get("rmn_qty") or o.get("unerc_qty")))
        if not (is_buy and rmn > 0):
            continue
        odno = str(o.get("odno") or o.get("ODNO") or "").strip()
        orgno = (o.get("krx_fwdg_ord_orgno") or o.get("KRX_FWDG_ORD_ORGNO")
                 or o.get("ord_gno_brno") or o.get("ORD_GNO_BRNO") or "")
        ord_dvsn = str(o.get("ord_dvsn") or o.get("ORD_DVSN") or "00").strip() or "00"
        pdno = str(o.get("pdno") or o.get("PDNO") or "").zfill(6)
        if not (odno and orgno):
            continue
        out.append({
            "krx_fwdg_ord_orgno": str(orgno),
            "odno": str(odno),
            "ord_dvsn": ord_dvsn,
            "rmn_qty": int(rmn),
            "pdno": pdno,
            "is_buy": True,
        })
    return out

# ───────────── 보조 ─────────────
def calc_pnl_pct(avg, cur):
    if not avg or not cur: return None
    try: return (float(cur) - float(avg)) / float(avg) * 100.0
    except: return None

# ───────────── 주문가능 조회 (최소 필드만 반환) ─────────────
INQUIRE_PSBL_TR_ID = "VTTC8908R"  # 모의투자용. 실계좌는 "TTTC8908R"로 교체

def inquire_psbl_order(stock_code, price, ord_dvsn="04", include_cma="Y", include_ovrs="N"):
    ensure_session()
    throttle_reads()
    """
    /uapi/domestic-stock/v1/trading/inquire-psbl-order
    반환: { rt_cd, msg, nrcvb_buy_qty }
    """
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"],
        "appSecret": SESSION["app_secret"],
        "tr_id": INQUIRE_PSBL_TR_ID,
    }
    params = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": str(stock_code).zfill(6),
        "ORD_UNPR": str(int(price)),
        "ORD_DVSN": str(ord_dvsn),
        "CMA_EVLU_AMT_ICLD_YN": include_cma,
        "OVRS_ICLD_YN": include_ovrs,
    }
    res = requests.get(url, headers=headers, params=params); time.sleep(1.2)
    j = res.json()
    out = (j or {}).get("output", {}) or {}

    def _i(x):
        try: return int(str(x).replace(",", "").strip())
        except: return 0

    return {
        "rt_cd": str((j or {}).get("rt_cd", "")),
        "msg": (j or {}).get("msg1", ""),
        "nrcvb_buy_qty": _i(out.get("nrcvb_buy_qty")),
    }

# ───────────── 장중 손절(-1%) ─────────────
def intraday_stoploss_sell(holdings, today_orders):
    for code, pos in holdings.items():
        qty = pos.get("qty", 0); avg = pos.get("avg_price", None)
        if qty <= 0 or not avg: continue
        cur = pos.get("cur_price", None) or get_current_price(code)
        if not cur: continue
        pnl = calc_pnl_pct(avg, cur)
        if pnl is None: continue
        if pnl < -1.0:
            open_sell_qty = get_open_sell_qty_for_code(today_orders, code)
            sellable_qty = max(0, int(qty) - int(open_sell_qty))
            if sellable_qty > 0:
                print(f"⚠️ 손절 발동(-1%↓): {code} pnl={pnl:.2f}% → 시장가 전량({sellable_qty}) 매도", flush=True)
                # 스로틀 적용
                result = send_order_throttled(code, 0, sellable_qty, order_type="매도", ord_dvsn="01")
                log_trade(datetime.now(), code, cur, sellable_qty, "매도", result)

# ───────────── 15:00 강제 매도 ─────────────
def force_sell_at_close(now, holdings):
    try:
        today_orders = get_today_orders()
    except:
        today_orders = []

    for code, pos in holdings.items():
        code = str(code).zfill(6)
        qty = int(pos.get("qty", 0) or 0)
        if qty <= 0:
            continue

        avg = pos.get("avg_price", None)
        cur = pos.get("cur_price", None) or get_current_price(code)
        pnl = calc_pnl_pct(avg, cur) if (avg and cur) else None

        open_sell_qty = get_open_sell_qty_for_code(today_orders, code)
        sellable_qty = max(0, qty - open_sell_qty)
        if sellable_qty <= 0:
            continue

        reason = f"pnl={pnl:.2f}%" if pnl is not None else "force_close"
        print(f"⛳ 15:00 전량 매도[{reason}]: {code} sellable={sellable_qty}", flush=True)
        # 스로틀 적용
        result = send_order_throttled(code, 0, sellable_qty, order_type="매도", ord_dvsn="01")
        log_trade(now, code, cur or 0, sellable_qty, "매도", result)

    print("↩️ 15:00 강제 매도 주문 발행 완료(이월 없음) — 루프 계속 진행", flush=True)

# ───────────── 계좌 요약 & 집계 ─────────────
INITIAL_CAPITAL = 100_000_000  # 초기자본 1억
INITIAL_TOT_EVAL = None        # 시작 기준 총평가금액(참고용)

def get_account_summary():
    ensure_session()
    throttle_reads()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": "VTTC8434R",
        "Cache-Control": "no-cache", "Pragma": "no-cache",
    }
    params = {"CANO": ACCOUNT_INFO["CANO"], "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
              "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
              "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
              "PRCS_DVSN": "01", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
    res = requests.get(url, headers=headers, params=params); time.sleep(1.2)
    j = res.json(); out2 = (j.get("output2") or [{}])
    return out2[0] if out2 else {}

def append_daily_pnl(now_dt, total_eval_amount):
    try:
        cumulative_return_pct = ((float(total_eval_amount) / float(INITIAL_CAPITAL)) - 1.0) * 100.0
    except Exception:
        cumulative_return_pct = None

    row = {
        "date": now_dt.strftime("%Y-%m-%d"),
        "time": now_dt.strftime("%H:%M:%S"),
        "총평가금액": float(total_eval_amount),
        "누적수익률(%)": round(cumulative_return_pct, 2) if cumulative_return_pct is not None else None,
    }
    if os.path.exists(DAILY_PNL_CSV):
        df = pd.read_csv(DAILY_PNL_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(DAILY_PNL_CSV, index=False, encoding="utf-8-sig")

    pct_str = f"{cumulative_return_pct:.2f}%" if cumulative_return_pct is not None else "NA"
    print(f"🧾 집계 저장 → {DAILY_PNL_CSV} (총평가금액={total_eval_amount:,.0f}, 누적수익률={pct_str})", flush=True)

# ───────────── 로그 ─────────────
def log_trade(timestamp, stock_code, price, qty, order_type, order_result, extra=None):
    log_file = Path("trade_log_2.csv")
    code_str = str(stock_code).zfill(6)
    log_entry = {
        "거래시간": timestamp, "종목코드": code_str, "현재가": price,
        "주문수량": qty, "주문종류": order_type,
        "주문결과": (order_result or {}).get("msg1", "NO_RESPONSE"),
    }
    if log_file.exists():
        df = pd.read_csv(log_file, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    df.to_csv(log_file, index=False, encoding="utf-8-sig")

# === [취소 로그 모드] 간결하게만 출력 ===
CANCEL_LOG_CONCISE = True

def _get_rmn_from_row(o):
    """inquire-daily-ccld row에서 잔량 추출"""
    return int(_num0(o.get("rmn_qty") or o.get("unerc_qty")))

def get_remaining_qty_by_odno(orders, odno: str) -> int:
    """ODNO로 현재 잔량 조회 (없으면 0으로 간주)"""
    for o in orders or []:
        if str(o.get("odno") or o.get("ODNO") or "").strip() == str(odno).strip():
            return _get_rmn_from_row(o)
    return 0

def cancel_and_report(item):
    """
    item: {"pdno","odno","krx_fwdg_ord_orgno","ord_dvsn","rmn_qty"}
    - 취소 요청 후 당일주문 재조회로 실제 취소된 수량 계산
    - 로그: `❎ 취소 CODE(ODNO) → N주 취소, 잔량 M주 [성공/실패: msg]`
    """
    code = item.get("pdno")
    odno = item.get("odno")
    before_rmn = int(item.get("rmn_qty", 0))

    result = send_cancel_order_throttled(
        ord_orgno=item.get("krx_fwdg_ord_orgno"),
        orgn_odno=odno,
        ord_dvsn=(item.get("ord_dvsn") or "00"),
        qty_all=True
    )

    # 반영 딜레이 대비
    time.sleep(1.2)
    after_orders = get_today_orders()
    after_rmn = get_remaining_qty_by_odno(after_orders, odno)
    canceled = max(0, before_rmn - after_rmn)

    status = "성공" if str(result.get("rt_cd")) == "0" else "실패"
    msg = (result.get("msg1") or "").strip()

    if CANCEL_LOG_CONCISE:
        print(f"❎ 취소 {code}({odno}) → {canceled}주 취소, 잔량 {after_rmn}주 [{status}: {msg}]", flush=True)
    else:
        print(f"취소요청: code={code} odno={odno} before_rmn={before_rmn} → rt_cd={result.get('rt_cd')} {msg} / after_rmn={after_rmn}", flush=True)

    return canceled, after_rmn, result

# ───────────── 매수 (현금 가능 수량 0 → 당일 매수 중단, 종목 제한은 해당 종목만 스킵) ─────────────
def refresh_avg_after_buy(code, tries=3, delay=1.5):
    for _ in range(tries):
        time.sleep(delay)
        acct = get_all_holdings()
        if code in acct:
            qty  = acct[code].get("qty", 0)
            avg  = acct[code].get("avg_price", None)
            if qty > 0 and avg and avg > 0:
                print(f"    ↪ 매수 반영됨: {code} qty={qty} avg={avg}", flush=True)
                return True
    print(f"    ↪ 매수 직후 평균가 미확인: {code} (다음 루프에서 동기화)", flush=True)
    return False

def buy_candidates(holdings, buy_codes, bought_today, not_tradable_today, prev_tv_map):
    """
    buy_codes: ATR14% 내림차순 후보
    prev_tv_map: {code: 전일 거래대금(float)}  # None이면 스킵
    반환: (market_closed, stop_buy_today)
    """
    print(f"▶ 매수 단계 (전일 거래대금의 {INVEST_RATE_FROM_PREV_TV*100:.2f}% 사용, 최대 5천만/종목)", flush=True)
    today_str = datetime.now().strftime("%Y%m%d")
    market_closed = False
    stop_buy_today = False

    ban_keywords = ["매매불가", "거래불가", "거래정지", "주문거절", "매매 금지", "거래 금지"]

    for code in buy_codes:
        if code in not_tradable_today:
            print(f"  ↪ 오늘 매매불가 스킵: {code}", flush=True); continue
        if code in bought_today:
            print(f"  ↪ 오늘 이미 매수/보유 이력 있어 스킵: {code}", flush=True); continue

        prev_tv = prev_tv_map.get(code)
        if prev_tv is None or prev_tv <= 0:
            print(f"  ❌ 전일 거래대금 없음/0: {code} → 스킵", flush=True)
            continue

        invest_amt = min(prev_tv * INVEST_RATE_FROM_PREV_TV, MAX_PER_STOCK_BUDGET)
        cur = get_current_price(code)
        if not cur or cur <= 0:
            print(f"  ❌ 현재가 조회 실패: {code}", flush=True); continue

        # (1) 전략상 목표 수량
        strategy_qty = int(invest_amt // cur)
        if strategy_qty <= 0:
            print(f"  ❌ 계산된 수량=0 (invest={invest_amt:,.0f}, cur={cur}): {code}", flush=True)
            continue

        # (2) 주문가능(현금 기준) 조회
        psbl = inquire_psbl_order(code, price=cur, ord_dvsn="04", include_cma="Y", include_ovrs="N")
        msg = psbl.get("msg", "")
        if is_market_closed_msg(msg):
            print("⛔ 시장 종료 감지(주문가능 조회 응답) → 루프 종료 예정", flush=True)
            market_closed = True
            break

        if any(k in msg for k in ban_keywords):
            not_tradable_today.add(code); save_not_tradable(today_str, not_tradable_today)
            print(f"  ⛔ 종목 거래제한 감지 → 오늘 스킵 등록: {code} / {msg}", flush=True)
            continue

        cash_qty = psbl.get("nrcvb_buy_qty", 0)  # 현금 100% 기준
        if cash_qty <= 0:
            print(f"  💸 현금 기준 주문가능수량=0 → 오늘 매수 중단 트리거 ({code})", flush=True)
            stop_buy_today = True
            break  # 즉시 매수 루프 종료

        qty = min(strategy_qty, cash_qty)
        if qty < strategy_qty:
            print(f"  ↪ 잔액 제한으로 수량 {strategy_qty}→{qty} 축소 ({code})", flush=True)

        # (3) 최우선지정가(04)로 매수  ⬅️ 주문 스로틀 적용
        result = send_order_throttled(code, 0, qty, order_type="매수", ord_dvsn="04")
        msg2 = (result.get("msg1") or "").strip()
        need_approx = cur * qty  # 참고: 주문 필요자금 근사
        print(f"  🟩 매수 04 요청: {code} x{qty} (참고필요자금≈{need_approx:,.0f}) → {result.get('rt_cd')} {msg2}", flush=True)
        log_trade(datetime.now(), code, cur, qty, "매수", result)

        if is_market_closed_msg(msg2):
            print("⛔ 시장 종료 감지(매수 응답) → 루프 종료 예정", flush=True)
            market_closed = True
            break

        if str(result.get("rt_cd")) == "0":
            bought_today.add(code); save_bought_today(today_str, bought_today)
            refresh_avg_after_buy(code, tries=3, delay=1.5)
        else:
            if any(k in msg2 for k in ban_keywords):
                not_tradable_today.add(code); save_not_tradable(today_str, not_tradable_today)
                print(f"  ⛔ 매수 응답에서 매매불가 감지 → 오늘 스킵 등록: {code}", flush=True)

    return market_closed, stop_buy_today

# ───────────── 상태 I/O ─────────────
def load_bought_today(today_str):
    try:
        with open(BOUGHT_TODAY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == today_str:
            return set(data.get("codes", []))
    except: pass
    return set()

def save_bought_today(today_str, codes_set):
    try:
        with open(BOUGHT_TODAY_PATH, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "codes": sorted(list(codes_set))}, f, ensure_ascii=False, indent=2)
    except: pass

def load_not_tradable(today_str):
    try:
        with open(NOT_TRADABLE_TODAY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == today_str:
            return set(data.get("codes", []))
    except: pass
    return set()

def save_not_tradable(today_str, codes_set):
    try:
        with open(NOT_TRADABLE_TODAY_PATH, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "codes": sorted(list(codes_set))}, f, ensure_ascii=False, indent=2)
    except: pass

# ───────────── 메인 ─────────────
def main():
    """
    원래 __main__ 블록의 전체 실행 루프를 함수로 캡슐화.
    외부에서 import 후 main()을 호출해 동작시킬 수 있음.
    """
    # 세션 초기화
    try:
        init_session()
    except Exception as e:
        print(str(e), flush=True)
        sys.exit(1)

    buy_list_path = os.path.join(OUTPUT_DIR, "buy_list.csv")
    if not os.path.exists(buy_list_path):
        print("❌ buy_list.csv 없음", flush=True); sys.exit()

    top_candidates_df = pd.read_csv(buy_list_path, dtype={'종목코드': str})
    today_candidates_all = [str(row['종목코드']).zfill(6) for _, row in top_candidates_df.iterrows()]

    # 🔹 ATR14% 기준 내림차순 정렬 (rows_per_code=20)
    today_candidates, atr_map = rank_by_atr14_from_db(
        today_candidates_all, rows_per_code=20
    )
    print(f"📋 오늘 후보 {len(today_candidates_all)}개 → ATR14% 내림차순 정렬 완료", flush=True)
    for c in today_candidates:
        atr_val = atr_map.get(c)
        if atr_val is not None:
            print(f"   - {c} ATR14%={atr_val:.2f}%", flush=True)
        else:
            print(f"   - {c} ATR14%=NA", flush=True)

    # 전일 거래대금 map 준비
    prev_tv_map = build_prev_trading_value_map(today_candidates)

    # 시작 기준 총평가금액 (참고용)
    global INITIAL_TOT_EVAL
    try:
        summary0 = get_account_summary()
        INITIAL_TOT_EVAL = _num0(summary0.get("tot_evlu_amt"))
        print(f"🧭 시작 기준 총평가금액 = {INITIAL_TOT_EVAL:,.0f}", flush=True)
    except:
        INITIAL_TOT_EVAL = None

    last_date = None
    bought_today = set()
    not_tradable_today = set()
    cancel_1450_done = False
    first_loop = True
    force_sell_done_today = False   # 15:00 강제 매도 1회
    buy_halted_today = False        # 현금-가능수량 0 트리거 시 당일 매수 중단
    buy_phase_done_today = False    # 오늘 매수 단계 1회만 실행 (이후 로그 억제)

    # 재실행 안전 훅: 시작 직후 CANCEL_BUY_START_TIME ~ 15:30 사이면 즉시 '매수 미체결' 취소 한 번 수행
    now0 = datetime.now().time()
    if CANCEL_BUY_START_TIME <= now0 < MARKET_CLOSE_TIME:
        print("🚀 재실행 감지: 시작 즉시 미체결 '매수' 취소 점검 실행", flush=True)
        try:
            # 1) 정규 경로
            cancelables = list_cancelable_buy_orders()
            to_cancel = [it for it in cancelables if it.get("is_buy") and int(it.get("rmn_qty", 0)) > 0]

            # 2) 폴백 경로 (정규 경로가 빈 경우)
            if not to_cancel:
                today_orders_raw = get_today_orders()
                fb = build_cancelables_from_today_orders(today_orders_raw)
                to_cancel = fb

            num = 0
            total_canceled = 0
            for it in to_cancel:
                canceled, after_rmn, _ = cancel_and_report(it)
                total_canceled += canceled
                num += 1

            print(f"✅ 재실행 취소 요약: 취소요청 {num}건 / 총 {total_canceled}주 취소", flush=True)
            cancel_1450_done = True
        except Exception as e:
            print(f"⚠️ 재실행 취소 처리 실패: {e}", flush=True)

    try:
        while True:
            now = datetime.now()
            today_str = now.strftime("%Y%m%d")

            if last_date != today_str:
                bought_today = load_bought_today(today_str)
                not_tradable_today = load_not_tradable(today_str)
                last_date = today_str
                cancel_1450_done = False if now.time() < CANCEL_BUY_START_TIME else cancel_1450_done
                force_sell_done_today = False
                buy_halted_today = False
                buy_phase_done_today = False   # 날짜 바뀌면 다시 1회 허용

            print(f"\n────────── [잔고 체크 중] {now.strftime('%Y-%m-%d %H:%M:%S')} ──────────", flush=True)

            # 최신 잔고/주문
            holdings = get_all_holdings()
            print(f"📦 실계좌 보유 {len(holdings)}종목 동기화", flush=True)
            try:
                today_orders = get_today_orders()
                bought_today |= build_bought_today_set(today_orders)
            except Exception as e:
                print(f"⚠️ 당일 주문조회 실패: {e} (기존 bought_today 유지)", flush=True)
                today_orders = []

            # 보유 중은 당일 재매수 금지
            held_today = {c for c, v in holdings.items() if v.get("qty", 0) > 0}
            bought_today |= held_today

            # 상태 저장
            save_bought_today(today_str, bought_today)
            save_not_tradable(today_str, not_tradable_today)

            # CANCEL_BUY_START_TIME ~ 15:00: 미체결 '매수' 전량 취소 (한 번만, 간결 로그)
            if not cancel_1450_done and CANCEL_BUY_START_TIME <= now.time() < FORCE_SELL_TIME:
                print("🕝 매수 미체결 전량 취소 실행", flush=True)
                try:
                    cancelables = list_cancelable_buy_orders()
                    to_cancel = [it for it in cancelables if it.get("is_buy") and int(it.get("rmn_qty", 0)) > 0]

                    if not to_cancel:
                        fb = build_cancelables_from_today_orders(today_orders)
                        to_cancel = fb

                    num = 0
                    total_canceled = 0
                    for it in to_cancel:
                        canceled, after_rmn, _ = cancel_and_report(it)
                        total_canceled += canceled
                        num += 1

                    print(f"✅ 전량 취소 요청 완료: 취소요청 {num}건 / 총 {total_canceled}주 취소", flush=True)
                    cancel_1450_done = True
                except Exception as e:
                    print(f"⚠️ 취소 처리 실패: {e}", flush=True)

            # 15:00 강제 매도
            if not force_sell_done_today and FORCE_SELL_TIME <= now.time() < MARKET_CLOSE_TIME:
                force_sell_at_close(now, holdings)
                force_sell_done_today = True

            # 손절(-1%)은 14:57 이전까지만 수행
            if now.time() < dtime(14, 57):
                intraday_stoploss_sell(holdings, today_orders)
            else:
                print("🛑 14:57 이후 손절 체크 중단", flush=True)

            # 매수는 하루 1회만 시도하고, 이후에는 호출/로그 모두 억제
            if (not buy_phase_done_today) and (not buy_halted_today):
                market_closed, stop_buy_today = buy_candidates(
                    holdings, today_candidates, bought_today, not_tradable_today, prev_tv_map
                )
                buy_phase_done_today = True  # 오늘 매수 단계 완료

                if market_closed:
                    print("✅ 시장 종료 감지 → 루프 즉시 종료", flush=True)
                    break
                if stop_buy_today:
                    buy_halted_today = True
                    print("⛔ 현금 기준 주문가능수량=0 발생 → 오늘 남은 시간 동안 추가 매수 중단", flush=True)
            # 이후 루프에서는 매수 관련 출력/호출 억제

            # 평가금액 출력
            try:
                summary = get_account_summary()
                total_eval_amount = _num0(summary.get("tot_evlu_amt"))
                eval_amount = _num0(summary.get("scts_evlu_amt"))
            except Exception as e:
                print(f"⚠️ 잔고 요약 조회 실패: {e}", flush=True)
                total_eval_amount = 0.0
                eval_amount = 0.0
            print(f"💰 (API) 총평가금액: {total_eval_amount:,.0f} / 평가금액(주식): {eval_amount:,.0f}", flush=True)

            # ───────────── 종료 조건 ─────────────
            if not first_loop:
                try:
                    no_open_orders = not has_open_orders(today_orders)
                except Exception as e:
                    no_open_orders = False
                    print(f"⚠️ 미체결 체크 실패: {e}", flush=True)

                if no_open_orders and eval_amount == 0:
                    try:
                        summary_now = get_account_summary()
                        total_eval_now = _num0(summary_now.get("tot_evlu_amt"))
                    except Exception:
                        total_eval_now = total_eval_amount
                    append_daily_pnl(now, total_eval_now)
                    print("🛑 장중 조기 종료: 미체결 없음 + 평가금액(주식)=0 → 집계 저장 후 종료", flush=True)
                    break
            else:
                print("🛡️ 시작 보호 모드: 첫 루프는 종료 조건을 건너뜁니다.", flush=True)
                first_loop = False

            # MARKET_CLOSE_TIME 도달 시 저장 후 종료
            if now.time() >= MARKET_CLOSE_TIME:
                try:
                    summary = get_account_summary()
                    total_eval_amount = _num0(summary.get("tot_evlu_amt"))
                except Exception:
                    pass
                append_daily_pnl(now, total_eval_amount)
                print(f"⏰ {MARKET_CLOSE_TIME.strftime('%H:%M')} 도달 — 집계 저장 후 종료", flush=True)
                break

            # 스마트 슬립: 크리티컬 시각 전까지만 대기
            smart_sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        print("\n⏹ 사용자 중단", flush=True)

# 모듈로 import 시 자동 실행되지 않도록 보호
if __name__ == "__main__":
    main()
