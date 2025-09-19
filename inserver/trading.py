import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, time as dtime
from pathlib import Path
import pymysql
from config import DB_CONFIG, get_account

# ───────────── 설정 ─────────────
OUTPUT_DIR = os.path.join("data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 종목당 투자금: 전일 거래대금의 0.25%
INVEST_RATE_FROM_PREV_TV = 0.0025

account = get_account("acc1")
APP_KEY     = account["APP_KEY"]
APP_SECRET  = account["APP_SECRET"]
ACCOUNT_INFO = account["ACCOUNT_INFO"]

# ───────────── 시간 상수 ─────────────
CANCEL_BUY_TIME   = dtime(14, 55)
MARKET_CLOSE_TIME = dtime(15, 30)

# 매수 트리거 시간
OPEN_BUY_TIME     = dtime(8, 59, 0)

# 스냅샷 시각
SNAP_0930_TIME = dtime(9, 30, 0)
SNAP_1500_TIME = dtime(15, 0, 0)

# 15:00 이후 조기 종료 체크 주기(초)
POST_SELL_CHECK_INTERVAL_SEC = 5 * 60

# 상태 파일
BOUGHT_TODAY_PATH       = os.path.join(OUTPUT_DIR, "bought_today.json")
NOT_TRADABLE_TODAY_PATH = os.path.join(OUTPUT_DIR, "not_tradable_today.json")
DAILY_PNL_CSV           = os.path.join(OUTPUT_DIR, "평가자료.csv")

# 스냅샷 파일
PORTFOLIO_CSV = os.path.join(OUTPUT_DIR, "포트폴리오.csv")
HOLDINGS_CSV  = os.path.join(OUTPUT_DIR, "보유종목.csv")

# ───────────── 세션/인증 상태 ─────────────
SESSION = {
    "url_base": "https://openapivts.koreainvestment.com:29443",  # 모의투자 VTS
    "app_key": None,
    "app_secret": None,
    "access_token": None,
    "inited": False,
}

def _resolve_api_keys(app_key=None, app_secret=None):
    return app_key or APP_KEY, app_secret or APP_SECRET

def init_session(app_key=None, app_secret=None, url_base=None):
    if url_base:
        SESSION["url_base"] = url_base

    app_key, app_secret = _resolve_api_keys(app_key, app_secret)
    if not app_key or not app_secret:
        raise RuntimeError("❌ API 키를 불러오지 못했습니다. config.get_api_keys() 설정을 확인하세요.")

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
    print(f"🔑 액세스 토큰 OK (account={ACCOUNT_INFO.get('CANO')}, mock/VTS)", flush=True)

def is_session_ready():
    return SESSION.get("inited") and bool(SESSION.get("access_token"))

def ensure_session():
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

# ───────────── GET/ORDER 스로틀 ─────────────
_last_get_ts = 0.0
def throttle_reads(min_interval=0.6):
    import random
    global _last_get_ts
    dt = time.monotonic() - _last_get_ts
    wait = max(0.0, min_interval - dt) + random.uniform(0, 0.15)
    if wait > 0:
        time.sleep(wait)
    _last_get_ts = time.monotonic()

_last_order_ts = 0.0
def send_order_throttled(stock_code, price, qty, order_type="매수", ord_dvsn="00", min_interval=0.8):
    global _last_order_ts
    dt = time.monotonic() - _last_order_ts
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_order_ts = time.monotonic()
    return send_order(stock_code, price, qty, order_type=order_type, ord_dvsn=ord_dvsn)

_last_cancel_ts = 0.0
def send_cancel_order_throttled(*, ord_orgno, orgn_odno, ord_dvsn, qty_all=True, qty=0, min_interval=0.6):
    global _last_cancel_ts
    dt = time.monotonic() - _last_cancel_ts
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_cancel_ts = time.monotonic()
    return send_cancel_order(ord_orgno=ord_orgno, orgn_odno=orgn_odno, ord_dvsn=ord_dvsn, qty_all=qty_all, qty=qty)

# ───────────── 정시 기상 헬퍼 ─────────────
def wait_until(target_dt: datetime):
    while True:
        now = datetime.now()
        remaining = (target_dt - now).total_seconds()
        if remaining <= 0:
            return
        if remaining > 60:
            time.sleep(remaining - 59.5)
        elif remaining > 1:
            time.sleep(remaining - 0.5)
        elif remaining > 0.05:
            time.sleep(remaining - 0.01)
        else:
            return

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
    code6 = _z6(code)
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT Date, Code, Close, Volume
                FROM stock_data
                WHERE (Code = %s OR Code = LPAD(%s, 6, '0') OR Code = CAST(%s AS UNSIGNED))
                ORDER BY Date DESC
                LIMIT 1
            """
            c_int = int(code6)
            cur.execute(sql, (code6, code6, c_int))
            rows = cur.fetchall()
            if not rows:
                return None

            _, _, close_val, vol_val = rows[0]  # 최신 = 전일
            close_val = _num(close_val)
            vol_val   = _num(vol_val)
            if close_val is None or vol_val is None:
                return None
            return float(close_val) * float(vol_val)
    finally:
        conn.close()

def build_prev_trading_value_map(codes):
    out = {}
    for c in codes:
        c6 = _z6(c)
        out[c6] = get_prev_trading_value_from_db(c6)
    return out

# ───────────── 시세/주문 ─────────────
# 현재가는 수량 계산용 근사로 사용
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
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": str(stock_code).zfill(6)}
    res = requests.get(url, headers=headers, params=params); time.sleep(1.0)
    if res.status_code != 200:
        return None, None, None
    try:
        j = res.json()
        out = (j.get("output") or j.get("output1") or j.get("output2") or {})
        if isinstance(out, list) and out: out = out[0]
        cur = out.get("stck_prpr") or out.get("prpr")
        cur = int(str(cur).replace(",","")) if cur not in (None, "", "null") else None
        return cur, None, None
    except:
        return None, None, None

def get_current_price(stock_code):
    cur, _, _ = get_quote(stock_code); return cur

def get_hashkey(data):
    ensure_session()
    url = f"{SESSION['url_base']}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"]}
    res = requests.post(url, headers=headers, data=json.dumps(data)); time.sleep(1.0)
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
        "PDNO": str(stock_code).zfill(6), "ORD_DVSN": ord_dvsn, "ORD_QTY": str(qty), "ORD_UNPR": unpr
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": tr_id, "hashkey": get_hashkey(data)
    }
    res = requests.post(url, headers=headers, data=json.dumps(data)); time.sleep(1.0)
    try: return res.json()
    except: return {"rt_cd": "-1", "msg1": res.text}

# ───────────── 정정/취소/조회 ─────────────
def send_cancel_order(ord_orgno, orgn_odno, ord_dvsn, qty_all=True, qty=0):
    ensure_session()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/order-rvsecncl"
    tr_id = "VTTC0803U"  # 모의: VTTC0803U
    params = {
        "CANO": ACCOUNT_INFO["CANO"], "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "KRX_FWDG_ORD_ORGNO": str(ord_orgno), "ORGN_ODNO": str(orgn_odno),
        "ORD_DVSN": str(ord_dvsn),
        "RVSE_CNCL_DVSN_CD": "02",
        "ORD_QTY": "0" if qty_all else str(int(qty)),
        "ORD_UNPR": "0", "QTY_ALL_ORD_YN": "Y" if qty_all else "N"
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"], "appSecret": SESSION["app_secret"],
        "tr_id": tr_id, "hashkey": get_hashkey(params),
    }
    res = requests.post(url, headers=headers, data=json.dumps(params)); time.sleep(1.0)
    try: return res.json()
    except: return {"rt_cd": "-1", "msg1": res.text}

def list_cancelable_buy_orders():
    ensure_session()
    throttle_reads()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
    tr_id = "VTTC8036R"
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
            res = requests.get(url, headers=headers, params=params); time.sleep(1.0)
            j = res.json()
            rows = j.get("output2") or j.get("output") or []
            for r in rows:
                orgno = r.get("krx_fwdg_ord_orgno") or r.get("KRX_FWDG_ORD_ORGNO") or r.get("ord_gno_brno")
                odno  = r.get("odno") or r.get("ODNO")
                dvsn  = r.get("ord_dvsn") or r.get("ORD_DVSN") or "00"
                pdno  = r.get("pdno") or r.get("PDNO")
                def _i(v):
                    try: return int(float(str(v).replace(",","").strip()))
                    except: return 0
                rmn1 = _i(r.get("rmn_qty") or r.get("RMN_QTY"))
                rmn2 = _i(r.get("unerc_qty") or r.get("UNERC_QTY"))
                ord_qty = _i(r.get("ord_qty") or r.get("ORD_QTY"))
                tot_ccld_qty = _i(r.get("tot_ccld_qty") or r.get("TOT_CCLD_QTY"))
                ccld_qty = _i(r.get("ccld_qty") or r.get("CCLD_QTY"))
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

def get_all_holdings():
    ensure_session()
    throttle_reads()
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
        if fk or nk:
            headers["tr_cont"] = "N"
        params = {
            "CANO": ACCOUNT_INFO["CANO"],
            "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "00",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": fk,
            "CTX_AREA_NK100": nk,
        }
        throttle_reads()
        res = requests.get(url, headers=headers, params=params); time.sleep(1.0)
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
        fk = (j.get("ctx_area_fk100") or "").rstrip()
        nk = (j.get("ctx_area_nk100") or "").rstrip()
        next_flag = (res.headers.get("tr_cont") or j.get("tr_cont") or "").strip()
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
        res = requests.get(url, headers=headers, params=params); time.sleep(1.0)
        j = res.json()
        items.extend(j.get("output1", []) or [])
        if j.get("tr_cont", "F") != "M": break
        params["CTX_AREA_FK100"] = j.get("ctx_area_fk100", "")
        params["CTX_AREA_NK100"] = j.get("ctx_area_nk100", "")
    return items

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

# === [취소 로그 모드] 간결하게만 출력 ===
CANCEL_LOG_CONCISE = True

def _get_rmn_from_row(o):
    return int(_num0(o.get("rmn_qty") or o.get("unerc_qty")))

def get_remaining_qty_by_odno(orders, odno: str) -> int:
    for o in orders or []:
        if str(o.get("odno") or o.get("ODNO") or "").strip() == str(odno).strip():
            return _get_rmn_from_row(o)
    return 0

def cancel_and_report(item):
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
    time.sleep(1.0)
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

# ───────────── 보조 ─────────────
def calc_pnl_pct(avg, cur):
    if not avg or not cur: return None
    try: return (float(cur) - float(avg)) / float(avg) * 100.0
    except: return None

INQUIRE_PSBL_TR_ID = "VTTC8908R"  # 모의투자용. 실계좌는 "TTTC8908R"

def inquire_psbl_order(stock_code, price, ord_dvsn="04", include_cma="Y", include_ovrs="N"):
    """
    주문가능 조회. 최우선/시장가/시간외 등은 ORD_UNPR=0, 그 외는 지정가 가격 사용.
    """
    ensure_session()
    throttle_reads()
    url = f"{SESSION['url_base']}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {SESSION['access_token']}",
        "appKey": SESSION["app_key"],
        "appSecret": SESSION["app_secret"],
        "tr_id": INQUIRE_PSBL_TR_ID,
    }
    price_free_types = {"01","03","04","11","12","13","14","15","16"}
    ord_unpr = "0" if str(ord_dvsn) in price_free_types else str(int(max(1, int(price or 0))))

    params = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": str(stock_code).zfill(6),
        "ORD_UNPR": ord_unpr,
        "ORD_DVSN": str(ord_dvsn),
        "CMA_EVLU_AMT_ICLD_YN": include_cma,
        "OVRS_ICLD_YN": include_ovrs,
    }
    res = requests.get(url, headers=headers, params=params); time.sleep(1.0)
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

# ───────────── 스냅샷/로그 ─────────────
INITIAL_CAPITAL = 100_000_000

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
    res = requests.get(url, headers=headers, params=params); time.sleep(1.0)
    j = res.json(); out2 = (j.get("output2") or [{}])
    return out2[0] if out2 else {}

def append_daily_pnl(now_dt, total_eval_amount):
    try:
        cumulative_return_pct = ((float(total_eval_amount) / float(INITIAL_CAPITAL)) - 1.0) * 100.0
    except Exception:
        cumulative_return_pct = None
    row_df = pd.DataFrame([{
        "date": now_dt.strftime("%Y-%m-%d"),
        "time": now_dt.strftime("%H:%M:%S"),
        "총평가금액": float(total_eval_amount),
        "누적수익률(%)": round(cumulative_return_pct, 2) if cumulative_return_pct is not None else None,
    }])
    if os.path.exists(DAILY_PNL_CSV):
        row_df.to_csv(DAILY_PNL_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row_df.to_csv(DAILY_PNL_CSV, index=False, encoding="utf-8-sig")
    pct_str = f"{cumulative_return_pct:.2f}%" if cumulative_return_pct is not None else "NA"
    print(f"🧾 집계 저장 → {DAILY_PNL_CSV} (총평가금액={total_eval_amount:,.0f}, 누적수익률={pct_str})", flush=True)

def save_portfolio_snapshot(now_dt, holdings, summary=None):
    if summary is None:
        try:
            summary = get_account_summary()
        except Exception:
            summary = {}
    def _f(key):
        return _num0((summary or {}).get(key))
    row = {
        "date": now_dt.strftime("%Y-%m-%d"),
        "time": now_dt.strftime("%H:%M:%S"),
        "평가금액": _f("scts_evlu_amt"),
        "매입금액": _f("pchs_amt_smtl_amt"),
        "평가손익금액": _f("evlu_pfls_smtl_amt"),
        "예수금": _f("dnca_tot_amt"),
        "총평가금액": _f("tot_evlu_amt"),
    }
    df = pd.DataFrame([row])
    df.to_csv(PORTFOLIO_CSV, mode="w", index=False, encoding="utf-8-sig")
    print(f"💾 포트폴리오 스냅샷 → {PORTFOLIO_CSV}", flush=True)

def save_holdings_snapshot(now_dt, holdings):
    rows = []
    for code, pos in (holdings or {}).items():
        rows.append({
            "date": now_dt.strftime("%Y-%m-%d"),
            "time": now_dt.strftime("%H:%M:%S"),
            "종목코드": str(code).zfill(6),
            "보유수량": int(pos.get("qty", 0) or 0),
            "매입평균가격": _num0(pos.get("avg_price")),
            "현재가": _num0(pos.get("cur_price")),
        })
    if rows:
        df = pd.DataFrame(rows)
    else:
        print("ℹ️ 보유 내역 없음: 빈 보유종목 파일로 저장", flush=True)
        df = pd.DataFrame(columns=["date","time","종목코드","보유수량","매입평균가격","현재가"])
    df.to_csv(HOLDINGS_CSV, mode="w", index=False, encoding="utf-8-sig")
    print(f"💾 보유종목 스냅샷 → {HOLDINGS_CSV}", flush=True)

# ───────────── 로그 ─────────────
def log_trade(timestamp, stock_code, price, qty, order_type, order_result, extra=None):
    log_file = Path(OUTPUT_DIR) / "trade_log_2.csv"
    code_str = str(stock_code).zfill(6)
    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(timestamp, "strftime") else str(timestamp)
    row_df = pd.DataFrame([{
        "거래시간": ts,
        "종목코드": code_str,
        "현재가": price,
        "주문수량": qty,
        "주문종류": order_type,
        "주문결과": (order_result or {}).get("msg1", "NO_RESPONSE"),
    }])
    if log_file.exists():
        row_df.to_csv(log_file, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row_df.to_csv(log_file, index=False, encoding="utf-8-sig")

# ───────────── 조기 종료 관련 유틸 ─────────────
def has_open_orders(today_orders=None):
    """
    미체결 주문 존재 여부 판단 (견고화 버전)
    1) 우선 '정정/취소 가능 주문 조회'로 미체결 잔량 존재 여부 확인
    2) 보조적으로 today_orders 스캔 (대/소문자 키 모두, 상태 문자열 확장)
    """
    # 1) 정정/취소 가능 주문 조회: 미체결 잔량이 있는 주문만 내려오는 것이 보통이라 여기 결과가 있으면 True
    try:
        cancelables = list_cancelable_buy_orders()  # 이름이 buy지만 실제로는 매수/매도 모두 들어옵니다.
        for it in cancelables or []:
            rmn = int(_num0(it.get("rmn_qty")))
            if rmn > 0:
                return True
    except Exception:
        # 조회 실패 시 2)로 폴백
        pass

    # 2) today_orders 스캔 (대/소문자 키, 더 넓은 '종료' 상태)
    if not today_orders:
        try:
            today_orders = get_today_orders()
        except Exception:
            return False

    def _text(o, *keys):
        for k in keys:
            v = o.get(k) or o.get(k.upper())
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _num_any(o, *keys):
        for k in keys:
            v = o.get(k) or o.get(k.upper())
            if v is not None:
                val = _num0(v)
                return val
        return 0.0

    # ‘종료(터미널) 상태’ 키워드 확장
    terminal_keywords = [
        "거부","불가","매매불가","주문거절","취소","취소완료","전량취소","정정거부","오류",
        "체결완료","전량체결","종료","완료",
        "rejected","reject","cancel","canceled","cancelled","error","invalid","filled","completed","done"
    ]

    for o in today_orders or []:
        st = _text(o, "ordr_sttus_name", "ccld_dvsn_name", "ord_sttus", "ord_sttus_name", "ord_sts", "stat")
        st_lower = st.lower()

        # 종료 상태로 명확히 보이면 스킵
        if any(k in st for k in terminal_keywords) or any(k in st_lower for k in terminal_keywords):
            continue

        # 1) API가 주는 잔량 필드 우선
        rmn = max(int(_num_any(o, "rmn_qty", "unerc_qty")), 0)
        if rmn > 0:
            return True

        # 2) 보정: 주문수량 - 체결수량 (취소/완료 행을 종료로 못 알아본 케이스 대비)
        ord_qty = _num_any(o, "ord_qty")
        c1 = _num_any(o, "tot_ccld_qty")
        c2 = _num_any(o, "ccld_qty")
        gap = max(0, int(round(ord_qty - max(c1, c2))))
        if gap > 0:
            # 다만 '취소접수/취소완료'류인데 상태문자열이 비어 온 케이스를 더 걸러줌
            # 취소 수량/사유 같은 필드가 있으면 종료로 간주 (있을 수도, 없을 수도)
            cancel_hint = _text(o, "rvse_cncl_dvsn_cd", "rvsecncl_yn", "cncl_yn", "cncl_stat", "cncl_rsn")
            if not cancel_hint:
                return True

    return False

def save_all_before_exit(tag="early_exit"):
    """
    조기/마감 종료 직전 저장. (요청에 따라 포트폴리오/보유종목 저장 생략)
    - 15:00에 이미 스냅샷을 저장하므로 여기서는 '평가자료.csv'만 append.
    """
    now = datetime.now()
    try:
        summary = get_account_summary()
    except Exception:
        summary = {}
    total_eval_amount = _num0((summary or {}).get("tot_evlu_amt"))
    try:
        append_daily_pnl(now, total_eval_amount)
    except Exception as e:
        print(f"⚠️ 평가자료 저장 오류: {e}", flush=True)

    print(f"🛑 [{tag}] 조기/마감 종료 직전 저장 완료 → 종료합니다.", flush=True)
    sys.exit(0)

def monitor_after_3pm_for_idle_exit():
    print("🕒 15:00 이후 5분 간격 모니터링 시작 (조건: 미체결 없음 AND 주식 평가금액=0) …", flush=True)
    end_dt = datetime.combine(datetime.now().date(), MARKET_CLOSE_TIME)
    while True:
        now = datetime.now()
        if now >= end_dt:
            print("⏰ 마감 시각 도달 — 조기 종료 조건 미충족, 다음 단계 진행.", flush=True)
            return
        try:
            today_orders = get_today_orders()
        except Exception:
            today_orders = []
        try:
            summary = get_account_summary()
            eval_amount = _num0(summary.get("scts_evlu_amt"))
        except Exception:
            eval_amount = 0.0
        no_open = not has_open_orders(today_orders)
        print(f"   · 체크 @ {now.strftime('%H:%M:%S')} → 미체결없음={no_open}, 주식평가금액={eval_amount:,.0f}", flush=True)
        if no_open and eval_amount == 0:
            save_all_before_exit(tag="post_3pm_idle")
        remaining = (end_dt - now).total_seconds()
        sleep_s = max(1, min(POST_SELL_CHECK_INTERVAL_SEC, remaining))
        time.sleep(sleep_s)

# ───────────── 매수 로직 ─────────────
def open_moo_buy_once(buy_codes, bought_today, not_tradable_today, prev_tv_map):
    """
    매수 1회 실행: 최우선지정가(04)로 매수
    - 수량 계산은 현재가로 근사(주문 자체는 ORD_UNPR=0)
    """
    print(f"▶ 매수 시작 (전일 거래대금의 {INVEST_RATE_FROM_PREV_TV*100:.2f}%)", flush=True)
    today_str = datetime.now().strftime("%Y%m%d")
    ban_keywords = ["매매불가", "거래불가", "거래정지", "주문거절", "매매 금지", "거래 금지"]

    for code in buy_codes:
        code = _z6(code)
        if code in not_tradable_today:
            print(f"  ↪ 오늘 매매불가 스킵: {code}", flush=True); continue
        if code in bought_today:
            print(f"  ↪ 오늘 이미 매수/보유 이력 있어 스킵: {code}", flush=True); continue

        prev_tv = prev_tv_map.get(code)
        if not prev_tv or prev_tv <= 0:
            print(f"  ❌ 전일 거래대금 없음/0: {code} → 스킵", flush=True); continue

        # 수량 산출용 현재가(근사) 확보
        cur = get_current_price(code)
        if not cur or cur <= 0:
            print(f"  ❌ 현재가 조회 실패: {code} → 스킵", flush=True); continue

        invest_amt = prev_tv * INVEST_RATE_FROM_PREV_TV
        strategy_qty = int(invest_amt // cur)
        if strategy_qty <= 0:
            print(f"  ❌ 계산된 수량=0 (invest={invest_amt:,.0f}, cur={cur}): {code}", flush=True); continue

        # 주문가능 조회: 최우선지정가 모드(04, ORD_UNPR=0)
        psbl = inquire_psbl_order(code, price=0, ord_dvsn="04", include_cma="Y", include_ovrs="N")
        msg = psbl.get("msg", "")
        if is_market_closed_msg(msg):
            print("⛔ 시장 종료 메시지 감지(주문가능 응답) — 스킵하고 계속", flush=True)
        if any(k in msg for k in ban_keywords):
            not_tradable_today.add(code); save_not_tradable(today_str, not_tradable_today)
            print(f"  ⛔ 종목 거래제한 감지 → 오늘 스킵 등록: {code} / {msg}", flush=True); continue

        cash_qty = psbl.get("nrcvb_buy_qty", 0)
        if cash_qty <= 0:
            print(f"  💸 현금 기준 주문가능수량=0 → 스킵: {code}", flush=True); continue

        qty = min(strategy_qty, cash_qty)
        if qty < strategy_qty:
            print(f"  ↪ 잔액 제한으로 수량 {strategy_qty}→{qty} 축소 ({code})", flush=True)

        # 최우선지정가(04): 가격은 의미 없으므로 0 전달
        result = send_order_throttled(code, 0, qty, order_type="매수", ord_dvsn="04")
        msg2 = (result.get("msg1") or "").strip()
        print(f"  🟩 매수 요청: {code} x{qty} (참고필요자금≈{cur*qty:,.0f}) → {result.get('rt_cd')} {msg2}", flush=True)
        log_trade(datetime.now(), code, cur, qty, "매수", result)

        if is_market_closed_msg(msg2):
            print("⛔ 시장 종료 메시지 감지(매수 응답) — 계속 진행", flush=True)

        if str(result.get("rt_cd")) == "0":
            bought_today.add(code); save_bought_today(today_str, bought_today)
        else:
            if any(k in msg2 for k in ban_keywords):
                not_tradable_today.add(code); save_not_tradable(today_str, not_tradable_today)
                print(f"  ⛔ 매수 응답에서 매매불가 감지 → 오늘 스킵 등록: {code}", flush=True)

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

# ───────────── 이벤트 핸들러/스케줄 ─────────────
def do_open_moo_buy(today_candidates, bought_today, not_tradable_today, prev_tv_map):
    print("▶ 매수 시작", flush=True)
    open_moo_buy_once(today_candidates, bought_today, not_tradable_today, prev_tv_map)

def do_snapshot(tag=""):
    now = datetime.now()
    holdings = get_all_holdings()
    try:
        summary = get_account_summary()
    except Exception:
        summary = None
    print(f"📸 스냅샷({tag})", flush=True)
    save_portfolio_snapshot(now, holdings, summary=summary)
    save_holdings_snapshot(now, holdings)

def do_cancel_buys():
    print("🕝 [정시] 14:55 매수 미체결 전량 취소", flush=True)
    try:
        cancelables = list_cancelable_buy_orders()
        to_cancel = [it for it in cancelables if it.get("is_buy") and int(it.get("rmn_qty", 0)) > 0]
        if not to_cancel:
            today_orders_raw = get_today_orders()
            fb = []
            for o in today_orders_raw or []:
                side_txt = str(o.get("sll_buy_dvsn_cd") or o.get("sll_buy_dvsn_name") or o.get("trad_dvsn_name") or "")
                is_buy = ("매수" in side_txt) or (str(side_txt) in ("02","2"))
                rmn = int(_num0(o.get("rmn_qty") or o.get("unerc_qty")))
                if not (is_buy and rmn > 0): continue
                odno = str(o.get("odno") or o.get("ODNO") or "").strip()
                orgno = (o.get("krx_fwdg_ord_orgno") or o.get("KRX_FWDG_ORD_ORGNO")
                         or o.get("ord_gno_brno") or o.get("ORD_GNO_BRNO") or "")
                ord_dvsn = str(o.get("ord_dvsn") or o.get("ORD_DVSN") or "00").strip() or "00"
                pdno = str(o.get("pdno") or o.get("PDNO") or "").zfill(6)
                if not (odno and orgno): continue
                fb.append({"krx_fwdg_ord_orgno": str(orgno),"odno": str(odno),"ord_dvsn": ord_dvsn,
                           "rmn_qty": int(rmn),"pdno": pdno,"is_buy": True})
            to_cancel = fb
        num = 0
        total_canceled = 0
        for it in to_cancel:
            canceled, after_rmn, _ = cancel_and_report(it)
            total_canceled += canceled
            num += 1
        print(f"✅ 전량 취소 요청 완료: 취소요청 {num}건 / 총 {total_canceled}주 취소", flush=True)
    except Exception as e:
        print(f"⚠️ 취소 처리 실패: {e}", flush=True)

def do_force_sell_and_snapshot():
    do_snapshot(tag="15:00")
    now = datetime.now()
    holdings = get_all_holdings()
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
        result = send_order_throttled(code, 0, sellable_qty, order_type="매도", ord_dvsn="01")
        log_trade(now, code, cur or 0, sellable_qty, "매도", result)
    print("↩️ 15:00 강제 매도 주문 발행 완료 — 5분 간격 모니터링으로 전환", flush=True)
    monitor_after_3pm_for_idle_exit()

# ───────────── 마감(15:30) 전용: 평가자료만 저장 후 종료 ─────────────
def save_pnl_only_and_exit(tag="market_close"):
    now = datetime.now()
    try:
        summary = get_account_summary()
    except Exception:
        summary = {}
    total_eval_amount = _num0((summary or {}).get("tot_evlu_amt"))
    try:
        append_daily_pnl(now, total_eval_amount)
    except Exception as e:
        print(f"⚠️ 평가자료 저장 오류: {e}", flush=True)

    print(f"🛑 [{tag}] 마감 저장 완료 → 종료합니다.", flush=True)
    sys.exit(0)

def do_market_close_and_exit():
    # 마감 시에는 평가자료.csv만 저장하고 종료
    save_pnl_only_and_exit(tag="market_close")

def build_today_events(today_candidates, bought_today, not_tradable_today, prev_tv_map):
    today = datetime.now().date()
    today_events = [
        ("open_moo_buy",  datetime.combine(today, OPEN_BUY_TIME),     lambda: do_open_moo_buy(today_candidates, bought_today, not_tradable_today, prev_tv_map)),
        ("snap_0930",     datetime.combine(today, SNAP_0930_TIME),    lambda: do_snapshot(tag="09:30")),
        ("cancel_buys",   datetime.combine(today, CANCEL_BUY_TIME),   do_cancel_buys),
        ("snap_1500_sell",datetime.combine(today, SNAP_1500_TIME),    do_force_sell_and_snapshot),
        ("close_and_exit",datetime.combine(today, MARKET_CLOSE_TIME), do_market_close_and_exit),
    ]
    now = datetime.now()
    today_events = [ev for ev in today_events if ev[1] > now]
    today_events.sort(key=lambda x: x[1])
    return today_events

def run_today_schedule(today_candidates, bought_today, not_tradable_today, prev_tv_map):
    events = build_today_events(today_candidates, bought_today, not_tradable_today, prev_tv_map)
    if not events:
        print("📭 오늘 남은 일정 없음 — 종료", flush=True)
        return
    for name, ts, fn in events:
        print(f"⏳ 이벤트 대기: {name} @ {ts.strftime('%H:%M:%S')}", flush=True)
        wait_until(ts)
        print(f"⏰ 실행: {name} @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}", flush=True)
        try:
            fn()
        except SystemExit:
            raise
        except Exception as e:
            print(f"⚠️ 이벤트 실행 오류({name}): {e}", flush=True)

# ───────────── 메인 ─────────────
def main():
    # 세션 초기화
    try:
        init_session()
    except Exception as e:
        print(str(e), flush=True)
        sys.exit(1)

    # 후보 리스트 로드
    buy_list_path = os.path.join(OUTPUT_DIR, "buy_list.csv")
    if not os.path.exists(buy_list_path):
        print("❌ buy_list.csv 없음", flush=True); sys.exit()

    top_candidates_df = pd.read_csv(buy_list_path, dtype={'종목코드': str})
    today_candidates_all = [str(row['종목코드']).zfill(6) for _, row in top_candidates_df.iterrows()]

    # ATR14% 정렬
    today_candidates, atr_map = rank_by_atr14_from_db(today_candidates_all, rows_per_code=20)
    print(f"📋 오늘 후보 {len(today_candidates_all)}개 → ATR14% 내림차순 정렬 완료", flush=True)
    for c in today_candidates:
        atr_val = atr_map.get(c)
        if atr_val is not None:
            print(f"   - {c} ATR14%={atr_val:.2f}%", flush=True)
        else:
            print(f"   - {c} ATR14%=NA", flush=True)

    # 전일 거래대금 map
    prev_tv_map = build_prev_trading_value_map(today_candidates)

    # 상태 로드
    today_str = datetime.now().strftime("%Y%m%d")
    bought_today = load_bought_today(today_str)
    not_tradable_today = load_not_tradable(today_str)

    # 하루 일정 실행
    run_today_schedule(today_candidates, bought_today, not_tradable_today, prev_tv_map)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n⏹ 사용자 중단", flush=True)
