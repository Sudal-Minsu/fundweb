# rule3_triplebarrier_bot.py
# ---------------------------------------------------------
# today_recos.csv(rule_3.py 결과) 기반
# ① 자동 매수(시장가, 고정 슬롯 균등비중: 초기현금/K_MAX) — 보유종목 스킵
# ② 장중 트리플배리어(TP/SL/시간) 매도 루프(추천 여부 무관, 배리어로만 매도)
# ③ 로그/리포팅
#    - trades_log.csv : 매수/매도 + 1시간 HEARTBEAT(SNAP)
#    - equity_log.csv : 현금/보유평가/에쿼티 (1시간 주기 + 체결 즉시)
#    - equity_curve.png : 스냅샷마다 자동 업데이트
# ---------------------------------------------------------
import os, json, time, requests, random
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import time as dtime
from typing import Optional, Tuple, Dict
from config_ko import get_api_keys, ACCOUNT_INFO
import logging, os, json
from logging.handlers import RotatingFileHandler

LOG_DIR = "/home/ubuntu/fundweb/logs"
os.makedirs(LOG_DIR, exist_ok=True)

_api_err_logger = logging.getLogger("api_errors")
_api_err_logger.setLevel(logging.INFO)
if not _api_err_logger.handlers:
    h = RotatingFileHandler(os.path.join(LOG_DIR, "api_errors.log"),
                            maxBytes=5_000_000, backupCount=5)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _api_err_logger.addHandler(h)

def _log_api_fail(name, r, body):
    try:
        j = body if isinstance(body, dict) else json.loads(body)
    except Exception:
        j = {"_raw": str(body)[:400]}
    rt_cd = (j.get("rt_cd") if isinstance(j, dict) else None)
    code  = (j.get("code") or j.get("error_code") or j.get("status")) if isinstance(j, dict) else None
    msg   = (j.get("msg1") or j.get("msg") or j.get("message") or j.get("error") or j.get("detail") or j.get("_raw") or "")
    _api_err_logger.info(f"{name} http={getattr(r,'status_code','NA')} rt_cd={rt_cd} code={code} msg={str(msg)[:200]}")
    print(f"❌ API 실패: {name} http={getattr(r,'status_code','NA')} rt_cd={rt_cd} code={code} msg={msg}")

# ===== 주문/루프 슬립 설정 =====
SLEEP_BETWEEN_BUYS  = 1.5   # 매수 주문 간 최소 대기(초)
SLEEP_BETWEEN_SELLS = 1.5   # 매도 주문 간 최소 대기(초)
IDLE_LOOP_SLEEP     = 20    # 매도 루프에서 변화 없을 때 대기(초)
ADD_JITTER          = True  # 슬립에 소량 지터 추가(레이트리밋 완화)

# ========== 경로/고정값 ==========
RESULT_DIR     = "results_3"
TODAY_RECOS    = os.path.join(RESULT_DIR, "today_recos.csv")
HOLDINGS_CSV   = os.path.join(RESULT_DIR, "holdings.csv")

# ----- 로그/리포팅 -----
TRADES_CSV   = os.path.join(RESULT_DIR, "trades_log.csv")
EQUITY_CSV   = os.path.join(RESULT_DIR, "equity_log.csv")
EQUITY_SNAPSHOT_SEC = 3600  # 1시간마다 스냅샷(초). 체결 시엔 즉시 1회 추가 스냅샷

# 초기자산/포트폴리오 CSV 경로
INITIAL_EQUITY_FILE = os.path.join(RESULT_DIR, "initial_equity.json")
PORT_SUMMARY_CSV    = os.path.join(RESULT_DIR, "portfolio_summary.csv")
PORT_HOLDINGS_CSV   = os.path.join(RESULT_DIR, "portfolio_holdings.csv")

MIN_PRICE_KRW  = 1000      # 1000원 미만 매수 금지
CLOSE_HH       = 15        # 시간 배리어(장마감) 기준 시각
CLOSE_MM       = 30        # ← 15:30으로 고정
FEE_CUSHION    = 1.003     # 체결/수수료 여유 (기본 0.3%)

# 스테일 보유행을 브로커 잔고 기준으로 청소할 때의 유예시간(초)
CLEANUP_GRACE_SEC = 180
# 방금 추가된 보유행이 즉시 매도되는 걸 막기 위한 최소 보유시간(초)
MIN_HOLD_SEC = 60

# ========== 트레이딩 파라미터 (고정 기본값) ==========
TP_PCT      = 0.03   # 익절 +3%
SL_PCT      = 0.03   # 손절 -3%
K_MAX       = 4      # 동시 보유 종목 수(슬롯 수)
HORIZON     = 6      # 보유 기간(영업일)

# ========== 인증/계좌 (VTS) ==========
APP_KEY, APP_SECRET = get_api_keys()
CANO          = ACCOUNT_INFO['CANO']
ACNT_PRDT_CD  = ACCOUNT_INFO["ACNT_PRDT_CD"]
BASE_URL      = "https://openapivts.koreainvestment.com:29443"  # VTS 서버

# ========== 공통 유틸 ==========
def _safe_append_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    tmp = pd.DataFrame([row]).dropna(axis=1, how='all')
    for c in tmp.columns:
        if c not in df.columns:
            df[c] = pd.Series(dtype=tmp[c].dtype if tmp[c].dtype != 'object' else object)
    for c in df.columns:
        if c not in tmp.columns:
            tmp[c] = pd.NA
    tmp = tmp[df.columns]
    return pd.concat([df, tmp], ignore_index=True)

def _load_initial_equity() -> Optional[float]:
    if os.path.exists(INITIAL_EQUITY_FILE):
        try:
            j = json.load(open(INITIAL_EQUITY_FILE, "r", encoding="utf-8"))
            v = float(j.get("initial_equity", 0.0))
            return v if v > 0 else None
        except Exception:
            return None
    return None

def _save_initial_equity(val: float):
    try:
        with open(INITIAL_EQUITY_FILE, "w", encoding="utf-8") as f:
            json.dump({"initial_equity": float(val), "ts": str(now_kst())}, f, ensure_ascii=False)
    except Exception as e:
        print("[EQUITY] 초기자산 저장 실패:", e)

def _sleep_with_jitter(base_sec: float):
    if base_sec <= 0: return
    time.sleep(base_sec + (random.uniform(0, 0.4) if ADD_JITTER else 0.0))

def now_kst():
    return pd.Timestamp.now(tz="Asia/Seoul")

def ensure_log_files():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        cols = ["ts","side","code","qty","price","reason","odno","tp_px","sl_px","cash_after",
                "거래시간","주문종류","종목코드","주문수량","현재가"]
        pd.DataFrame(columns=cols).to_csv(TRADES_CSV, index=False, encoding="utf-8")
    if not os.path.exists(EQUITY_CSV):
        cols = ["ts","cash","positions_value","equity","cum_return","total_pnl"]
        pd.DataFrame(columns=cols).to_csv(EQUITY_CSV, index=False, encoding="utf-8")

def ensure_files():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(HOLDINGS_CSV):
        cols = ["code","qty","entry_date","entry_px","tp_px","sl_px","horizon_end","order_id_buy","last_update"]
        pd.DataFrame(columns=cols).to_csv(HOLDINGS_CSV, index=False)
    ensure_log_files()

def load_holdings():
    ensure_files()
    try:
        df = pd.read_csv(HOLDINGS_CSV, parse_dates=["entry_date","horizon_end","last_update"])
    except Exception:
        df = pd.read_csv(HOLDINGS_CSV)
    if "code" in df.columns:
        df["code"] = (df["code"].astype(str)
                                .str.replace(".0","",regex=False)
                                .str.strip()
                                .str.zfill(6))
    return df

def save_holdings(df):
    if not df.empty and "code" in df.columns:
        df["code"] = (df["code"].astype(str)
                                .str.replace(".0","",regex=False)
                                .str.strip()
                                .str.zfill(6))
    df.to_csv(HOLDINGS_CSV, index=False)

def to_float(x) -> float:
    try:
        if x is None: return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).replace(",", "").strip()
        return float(s) if s else 0.0
    except Exception:
        return 0.0

# ==== 레이트 리미터 ====
_LAST_CALL = {}
def throttle(key: str, min_interval: float = 1.2):
    now = time.time()
    last = _LAST_CALL.get(key, 0.0)
    wait = last + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[key] = time.time()

# ===== 로그 함수 =====
def log_trade(side: str, code: str, qty: int, price: float, reason: str, odno: Optional[str],
              tp_px: Optional[float]=None, sl_px: Optional[float]=None, cash_after: Optional[float]=None):
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception:
        df = pd.DataFrame(columns=["ts","side","code","qty","price","reason","odno","tp_px","sl_px","cash_after",
                                   "거래시간","주문종류","종목코드","주문수량","현재가"])
    # 과거 잘못 저장된 숫자형 코드 정규화
    for col in ["code","종목코드"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                              .str.replace(".0","",regex=False)
                              .str.strip()
                              .str.zfill(6)
                              .where(df[col].notna(), ""))

    code6 = (str(code).replace(".0","").zfill(6) if code else "")
    ts_now = now_kst().tz_localize(None)
    row = {
        "ts": ts_now,
        "side": side,
        "code": code6,
        "qty": int(qty) if qty is not None else 0,
        "price": float(price) if price is not None else 0.0,
        "reason": reason or "",
        "odno": odno or "",
        "tp_px": float(tp_px) if tp_px is not None else None,
        "sl_px": float(sl_px) if sl_px is not None else None,
        "cash_after": float(cash_after) if cash_after is not None else None,
        # 국문 표제
        "거래시간": ts_now,
        "주문종류": side,
        "종목코드": code6,
        "주문수량": int(qty) if qty is not None else 0,
        "현재가": float(price) if price is not None else 0.0,
    }
    for c in row.keys():
        if c not in df.columns:
            df[c] = pd.NA
    df = _safe_append_row(df, row)
    df.to_csv(TRADES_CSV, index=False, encoding="utf-8")

# ========== 보유행 추가/병합 ==========
def add_position(code, qty, entry_date, entry_px, tp_pct, sl_pct, horizon_days, order_id_buy=None):
    df = load_holdings()
    code = str(code).replace(".0","").zfill(6)
    qty = int(qty)

    try:
        entry_px = float(entry_px)
    except Exception:
        raise ValueError(f"[add_position] entry_px 파싱 실패: code={code}, entry_px={entry_px}")
    if not pd.notna(entry_px) or entry_px <= 0:
        raise ValueError(f"[add_position] entry_px 비정상: code={code}, entry_px={entry_px}")

    # 기존 보유 병합(가중평균가)
    if not df.empty and (df["code"] == code).any():
        old = df[df["code"] == code].copy()
        old_qty = int(old["qty"].sum())
        old_px  = float((old["entry_px"] * old["qty"]).sum() / max(1, old_qty)) if old_qty > 0 else entry_px
        new_qty = old_qty + qty
        new_px  = (old_px * old_qty + entry_px * qty) / max(1, new_qty)
        tp_px = new_px * (1.0 + tp_pct)
        sl_px = new_px * (1.0 - sl_pct)

        _tz = "Asia/Seoul"
        base_dt = pd.to_datetime(entry_date)
        if base_dt.tzinfo is None:
            base_dt = base_dt.tz_localize(_tz, nonexistent="NaT", ambiguous="NaT")
        horizon_end_ts = (base_dt + BDay(horizon_days)).replace(hour=CLOSE_HH, minute=CLOSE_MM, second=0, microsecond=0)
        horizon_end = horizon_end_ts.tz_localize(None)

        df = df[df["code"] != code]
        row = {
            "code": code, "qty": int(new_qty),
            "entry_date": pd.to_datetime(entry_date), "entry_px": float(new_px),
            "tp_px": float(tp_px), "sl_px": float(sl_px),
            "horizon_end": horizon_end, "order_id_buy": order_id_buy,
            "last_update": now_kst().tz_localize(None)
        }
        df = _safe_append_row(df, row)
        save_holdings(df)
        return

    # 신규
    tp_px = float(entry_px) * (1.0 + tp_pct)
    sl_px = float(entry_px) * (1.0 - sl_pct)
    _tz = "Asia/Seoul"
    base_dt = pd.to_datetime(entry_date)
    if base_dt.tzinfo is None:
        base_dt = base_dt.tz_localize(_tz, nonexistent="NaT", ambiguous="NaT")
    horizon_end_ts = (base_dt + BDay(horizon_days)).replace(hour=CLOSE_HH, minute=CLOSE_MM, second=0, microsecond=0)
    horizon_end = horizon_end_ts.tz_localize(None)

    row = {
        "code": code, "qty": int(qty),
        "entry_date": pd.to_datetime(entry_date), "entry_px": float(entry_px),
        "tp_px": float(tp_px), "sl_px": float(sl_px),
        "horizon_end": horizon_end, "order_id_buy": order_id_buy,
        "last_update": now_kst().tz_localize(None)
    }
    df = _safe_append_row(df, row)
    save_holdings(df)

def remove_position(code):
    df = load_holdings()
    code = str(code).replace(".0","").zfill(6)
    df = df[df["code"] != code]
    save_holdings(df)

# ========== VTS API ==========
def get_access_token(app_key, app_secret) -> Optional[str]:
    url = f"{BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json; charset=utf-8"}
    body = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
    try:
        res = requests.post(url, headers=headers, json=body, timeout=10)
        try:
            j = res.json()
        except Exception:
            j = {"raw": res.text[:300]}
    except requests.RequestException as e:
        print(f"🔐 토큰 요청 네트워크 예외: {e}")
        return None
    if res.status_code != 200:
        print(f"🔐 토큰 실패 [{res.status_code}] {j}")
        print(f"   → APP_KEY 확인: {app_key[:4]}...{app_key[-4:]}")
        return None
    token = j.get("access_token")
    if not token:
        print(f"🔐 토큰 응답에 access_token 없음: {j}")
        return None
    print("🟢 토큰 OK")
    return token

def get_hashkey(app_key, app_secret, data):
    url = f"{BASE_URL}/uapi/hashkey"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "appkey": app_key, "appsecret": app_secret
    }
    throttle("hashkey", 1.2)
    res = requests.post(url, headers=headers, json=data, timeout=10)
    if res.status_code != 200:
        print(f"⚠️ 해시키 실패 [{res.status_code}] {res.text[:300]}")
        res.raise_for_status()
    j = res.json()
    h = j.get("HASH", "")
    if not h:
        raise RuntimeError(f"해시키 없음 응답: {j}")
    return h

def get_current_price(access_token: str, app_key: str, app_secret: str, base_url: str, code: str) -> Optional[float]:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": str(code).zfill(6)}
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": app_key, "appsecret": app_secret,
        "tr_id": "FHKST01010100"
    }
    GLOBAL_GAP = 0.50
    SYMBOL_GAP = 0.35
    retry_delays = [0.6, 1.2, 2.4]
    for attempt in range(1 + len(retry_delays)):
        throttle("price:__global__", GLOBAL_GAP)
        throttle(f"price:{code}",      SYMBOL_GAP)
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            j = r.json()
        except Exception as e:
            print(f"[PRICE] 예외 {code}: {e}")
            j, r = None, None
        if r is not None and r.status_code == 200 and isinstance(j, dict) and "output" in j:
            pr = (j["output"] or {}).get("stck_prpr", "0")
            try:
                val = float(str(pr).replace(",", ""))
            except Exception:
                val = 0.0
            print(f"[PRICE] {code} = {val}")
            return val if val > 0 else None
        msg = (j.get("msg1") or "") if isinstance(j, dict) else ""
        if (r and r.status_code in (429, 500)) and ("초당 거래건수" in msg or "EGW00201" in str(j)):
            if attempt < len(retry_delays) + 1:
                delay = retry_delays[attempt-1] + random.uniform(0, 0.2)
                print(f"[PRICE] 레이트리밋 감지 {code} → {delay:.1f}s 대기 후 재시도")
                time.sleep(delay); continue
        print(f"[PRICE] 오류 {code} [{getattr(r,'status_code','NA')}]: {j}")
        return None

def execute_order(stock_code, quantity, order_type, order_style, app_key, app_secret, access_token, base_url, price=None):
    url  = f"{base_url}/uapi/domestic-stock/v1/trading/order-cash"
    if order_style == "시장가":
        ord_dvsn = "01"; ord_unpr = "0"
    elif order_style == "지정가" and price is not None:
        ord_dvsn = "00"; ord_unpr = str(int(price))
    else:
        raise ValueError("지정가 주문에는 price 값을 명시해야 합니다.")
    if order_type == "매수":
        tr_id = "VTTC0802U"
    elif order_type == "매도":
        tr_id = "VTTC0801U"
    else:
        raise ValueError("order_type은 '매수' 또는 '매도'여야 합니다.")

    data = {
        "CANO": ACCOUNT_INFO['CANO'],
        "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],
        "PDNO": str(stock_code).zfill(6),
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(int(quantity)),
        "ORD_UNPR": ord_unpr
    }
    for attempt in range(3):
        try:
            hashkey = get_hashkey(app_key, app_secret, data)
            headers = {
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {access_token}",
                "appkey": app_key, "appsecret": app_secret,
                "tr_id": tr_id, "custtype": "P", "hashkey": hashkey
            }
            throttle("order", 1.2)
            r = requests.post(url, headers=headers, json=data, timeout=10)
            try:
                j = r.json()
            except ValueError:
                print("❌ 주문 응답 JSON 아님:", r.text[:300])
                j = {"rt_cd": "-1", "msg1": "JSON parse error"}
        except Exception as e:
            j = {"rt_cd": "-1", "msg1": f"req error: {e}"}
        if j.get("rt_cd") == "0":
            odno = (j.get("output") or {}).get("ODNO")
            if odno:
                print(f"✅ {order_type} 성공: 주문번호 {odno}")
                return odno
            print(f"❌ {order_type} 실패: 주문번호 없음. output={j.get('output')}")
            return None
        msg = (j.get("msg1") or "").strip()
        print(f"❌ {order_type} 실패[{attempt+1}/3]: {msg}")
        throttle("order_backoff", 1.2 * (2 ** attempt))
        if "초당 거래건수" not in msg:
            break
    return None

# 🔹 토큰 매니저 (캐시 파일 사용)
def get_auth_info():
    TOKEN_FILE = "access_token.json"
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < 55 * 60:
                print("🟢 기존 토큰 재사용 중 (파일)")
                return cache["app_key"], cache["app_secret"], cache["token"]
        except Exception as e:
            print("⚠️ 캐시 파일 읽기 실패:", e)
    app_key, app_secret = APP_KEY, APP_SECRET
    access_token = get_access_token(app_key, app_secret)
    cache = {"token": access_token, "timestamp": time.time(),
             "app_key": app_key, "app_secret": app_secret}
    with open(TOKEN_FILE, "w") as f:
        json.dump(cache, f)
    print("🔄 새로운 토큰 발급 완료")
    return app_key, app_secret, access_token

# ========== 잔고/예수금/평가 & 스냅샷 ==========
def check_account(access_token, app_key, app_secret):
    output1, output2 = [], []
    CTX_AREA_NK100 = ""
    url_base = BASE_URL
    while True:
        url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {
            "content-type": "application/json; charset=utf-8",   # ← charset 추가
            "authorization": f"Bearer {access_token}",
            "appkey": app_key, "appsecret": app_secret,
            "tr_id": "VTTC8434R",
            "custtype": "P"                                     # ← 필요시 명시
        }
        params = {
            "CANO": ACCOUNT_INFO['CANO'],
            "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],
            "AFHR_FLPR_YN": "N", "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
            "OFL_YN": "", "INQR_DVSN": "01", "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": CTX_AREA_NK100
        }
        res = requests.get(url, headers=headers, params=params, timeout=10)
        print("📡 응답 상태코드:", res.status_code)

        # JSON 파싱
        try:
            data = res.json()
        except Exception:
            _log_api_fail("check_account:json_parse", res, res.text)
            return None, None

        # 성공 판정 & 실패시 본문 이유 남기기
        if (res.status_code != 200) or (data.get("rt_cd") != "0"):
            _log_api_fail("check_account", res, data)
            return None, None

        # 방어: output1가 없거나 빈 경우
        if "output1" not in data or data.get("output1") in (None, [], [{}]):
            # 그래도 output2만으로 예수금은 받을 수 있음
            output2.append((data.get("output2") or [{}])[0])
            break

        # 정상 분기
        try:
            output1.append(pd.DataFrame.from_records(data["output1"]))
        except Exception as e:
            _log_api_fail("check_account:output1_parse", res, data)
            return None, None

        # 페이지 토큰(대소문자 모두 시도)
        next_key = (data.get("ctx_area_nk100") or data.get("CTX_AREA_NK100") or "").strip()
        CTX_AREA_NK100 = next_key

        # 마지막 페이지면 output2(요약) 저장
        if not CTX_AREA_NK100:
            output2.append((data.get("output2") or [{}])[0])
            break

    # holdings(DataFrame) 구성
    if output1 and not output1[0].empty:
        df_all = pd.concat(output1, ignore_index=True)
        cols = ['pdno','hldg_qty','pchs_avg_pric']
        for c in cols:
            if c not in df_all.columns:
                df_all[c] = pd.NA
        res1 = df_all[cols].rename(columns={
            'pdno':'종목코드', 'hldg_qty':'보유수량', 'pchs_avg_pric':'매입단가'
        }).reset_index(drop=True)
        res1['종목코드'] = res1['종목코드'].astype(str).str.zfill(6)
        res1['보유수량'] = pd.to_numeric(res1['보유수량'], errors='coerce').fillna(0).astype(int)
        res1['매입단가'] = pd.to_numeric(res1['매입단가'], errors='coerce').fillna(0.0).astype(float)
    else:
        res1 = pd.DataFrame(columns=['종목코드','보유수량','매입단가'])

    res2 = output2[0] if output2 else {}
    return res1, res2


# 포트폴리오 CSV 내보내기
def export_portfolio_csvs(access_token, app_key, app_secret):
    holdings_df, out2 = check_account(access_token, app_key, app_secret)
    cash = to_float((out2 or {}).get("dnca_tot_amt", 0.0))
    if holdings_df is None or holdings_df.empty:
        pd.DataFrame(columns=["ts","positions_value","total_cost","pnl","cash","equity"]).to_csv(PORT_SUMMARY_CSV, index=False, encoding="utf-8")
        pd.DataFrame(columns=["ts","종목코드","보유수량","매입단가","현재가","평가금액","평가손익"]).to_csv(PORT_HOLDINGS_CSV, index=False, encoding="utf-8")
        return
    rows = []; positions_value = 0.0; total_cost = 0.0
    for _, r in holdings_df.iterrows():
        code = str(r["종목코드"]).zfill(6)
        qty  = int(r["보유수량"]); avg = float(r["매입단가"])
        cur = get_current_price(access_token, app_key, app_secret, BASE_URL, code) or 0.0
        mv  = float(qty) * float(cur); cost = float(qty) * float(avg); pnl = mv - cost
        rows.append({"ts": now_kst().tz_localize(None),"종목코드": code,"보유수량": qty,"매입단가": avg,"현재가": cur,"평가금액": mv,"평가손익": pnl})
        positions_value += mv; total_cost += cost
        _sleep_with_jitter(0.15)
    equity = cash + positions_value
    summary_row = {"ts": now_kst().tz_localize(None),"positions_value": positions_value,"total_cost": total_cost,"pnl": positions_value-total_cost,"cash": cash,"equity": equity}
    try: sdf = pd.read_csv(PORT_SUMMARY_CSV)
    except Exception: sdf = pd.DataFrame(columns=["ts","positions_value","total_cost","pnl","cash","equity"])
    sdf = _safe_append_row(sdf, summary_row); sdf.to_csv(PORT_SUMMARY_CSV, index=False, encoding="utf-8")
    try: hdf = pd.read_csv(PORT_HOLDINGS_CSV)
    except Exception: hdf = pd.DataFrame(columns=["ts","종목코드","보유수량","매입단가","현재가","평가금액","평가손익"])
    for r in rows: hdf = _safe_append_row(hdf, r)
    hdf.to_csv(PORT_HOLDINGS_CSV, index=False, encoding="utf-8")

def get_cash_balance(access_token, app_key, app_secret) -> float:
    _, res2 = check_account(access_token, app_key, app_secret)
    if res2 is None: return 0.0
    return to_float(res2.get("dnca_tot_amt", 0))

def get_orderable_cash(access_token, app_key, app_secret) -> float:
    _, out2 = check_account(access_token, app_key, app_secret)
    if not out2: return 0.0
    keys = ["ord_psbl_cash","ord_psbl_amt","ord_psbl_cash_amt","dnca_tot_amt"]
    for k in keys:
        if k in out2 and to_float(out2[k]) > 0:
            return to_float(out2[k])
    return to_float(out2.get("dnca_tot_amt", 0))

def _portfolio_valuation(access_token, app_key, app_secret) -> Tuple[float, float, float]:
    cash = get_cash_balance(access_token, app_key, app_secret)
    pos_df, _ = check_account(access_token, app_key, app_secret)
    positions_value = 0.0
    if pos_df is not None and not pos_df.empty:
        for _, r in pos_df.iterrows():
            code = str(r["종목코드"]).zfill(6); qty = int(r["보유수량"])
            if qty <= 0: continue
            cur = get_current_price(access_token, app_key, app_secret, BASE_URL, code)
            if cur and cur > 0: positions_value += qty * cur
            _sleep_with_jitter(0.15)
    equity = cash + positions_value
    return float(cash), float(positions_value), float(equity)

def export_equity_png(out_path: str = os.path.join(RESULT_DIR, "equity_curve.png")):
    try:
        df = pd.read_csv(EQUITY_CSV, parse_dates=["ts"])
    except Exception:
        print("[EQUITY] 로그 없음 → PNG 생략"); return
    if df.empty: print("[EQUITY] 데이터 비어있음 → PNG 생략"); return
    df = df.sort_values("ts")
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[EQUITY] matplotlib 불가 → PNG 생략: {e}"); return
    plt.figure(); plt.plot(df["ts"], df["equity"])
    plt.title("Equity Curve"); plt.xlabel("Time"); plt.ylabel("Equity (KRW)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[EQUITY] PNG 업데이트: {out_path}")

_last_equity_snap = 0.0
def maybe_snapshot_equity(access_token, app_key, app_secret, force: bool=False):
    global _last_equity_snap
    now_sec = time.time()
    if not force and (now_sec - _last_equity_snap) < EQUITY_SNAPSHOT_SEC:
        return
    cash, pv, eq = _portfolio_valuation(access_token, app_key, app_secret)
    init_eq = _load_initial_equity()
    if init_eq is None or init_eq <= 0:
        _save_initial_equity(eq); init_eq = eq
    cum_ret = (eq / init_eq - 1.0) if init_eq > 0 else 0.0
    total_pnl = eq - init_eq
    try:
        df = pd.read_csv(EQUITY_CSV)
    except Exception:
        df = pd.DataFrame(columns=["ts","cash","positions_value","equity","cum_return","total_pnl"])
    row = {"ts": now_kst().tz_localize(None),"cash": cash,"positions_value": pv,"equity": eq,"cum_return": cum_ret,"total_pnl": total_pnl}
    for c in row.keys():
        if c not in df.columns: df[c] = pd.NA
    df = _safe_append_row(df, row)
    df.to_csv(EQUITY_CSV, index=False, encoding="utf-8")
    _last_equity_snap = now_sec
    export_equity_png(os.path.join(RESULT_DIR, "equity_curve.png"))
    export_portfolio_csvs(access_token, app_key, app_secret)
    log_trade(side="SNAP", code="", qty=0, price=0.0, reason="HEARTBEAT", odno="", tp_px=None, sl_px=None, cash_after=cash)

# ===== 브로커 보유 동기화 =====
def _broker_positions_map(access_token, app_key, app_secret) -> Dict[str, Tuple[int, float]]:
    res1, _ = check_account(access_token, app_key, app_secret)
    mp: Dict[str, Tuple[int, float]] = {}
    if res1 is not None and not res1.empty:
        for _, r in res1.iterrows():
            mp[str(r["종목코드"]).zfill(6)] = (int(r["보유수량"]), float(r["매입단가"]))
    return mp

def _drop_stale_holdings_against_broker(df_hold: pd.DataFrame, pos_map: Dict[str, Tuple[int,float]], now_ts: pd.Timestamp) -> pd.DataFrame:
    keep_rows = []
    for _, row in df_hold.iterrows():
        code = str(row["code"]).zfill(6)
        qty  = int(row.get("qty", 0) or 0)
        if qty <= 0: continue
        br_qty = pos_map.get(code, (0, 0.0))[0]
        if br_qty <= 0:
            last_upd = pd.to_datetime(row.get("last_update"))
            if pd.isna(last_upd): continue
            age_sec = (now_ts.tz_localize(None) - last_upd).total_seconds()
            if age_sec <= CLEANUP_GRACE_SEC:
                keep_rows.append(row); continue
            print(f"[CLEANUP] 브로커 보유 0 & 유예초과 → holdings 제거: {code}")
            continue
        keep_rows.append(row)
    return pd.DataFrame(keep_rows, columns=df_hold.columns) if keep_rows else pd.DataFrame(columns=df_hold.columns)

def sync_holdings_with_broker(access_token, app_key, app_secret):
    hold = load_holdings()
    have = set(hold["code"].astype(str).str.zfill(6)) if not hold.empty else set()
    br_df, _ = check_account(access_token, app_key, app_secret)
    if br_df is None or br_df.empty: return
    for _, r in br_df.iterrows():
        code = str(r["종목코드"]).zfill(6)
        qty  = int(r["보유수량"]); avg = float(r["매입단가"])
        if qty <= 0 or code in have: continue
        tp_px = avg * (1.0 + TP_PCT); sl_px = avg * (1.0 - SL_PCT)
        entry_date = now_kst().normalize().tz_localize(None)
        horizon_end = (entry_date + BDay(HORIZON)).replace(hour=CLOSE_HH, minute=CLOSE_MM, second=0, microsecond=0)
        row = {
            "code": code, "qty": qty, "entry_date": entry_date, "entry_px": avg,
            "tp_px": tp_px, "sl_px": sl_px, "horizon_end": horizon_end,
            "order_id_buy": "", "last_update": now_kst().tz_localize(None)
        }
        if hold.empty:
            hold = pd.DataFrame(columns=list(row.keys()))
        hold = _safe_append_row(hold, row)
    save_holdings(hold)

# ========== 매수(시장가, 고정 슬롯 균등비중: 초기현금/K_MAX) ==========
def run_buy(tp_pct: float, sl_pct: float, k_max: int, horizon_d: int):
    """
    - today_recos.csv 상위 slots_to_use 종목만 집행
    - 보유 종목은 스킵
    - '초기 주문가능금액 / k_max' 를 각 주문의 목표 예산으로 사용 (항상 1/K_MAX 비중)
    - 실패 시 '1주씩' 줄이며 재시도
    """
    ensure_files()
    _SLEEP_BETWEEN_BUYS = 1.5
    JITTER_SEC = 0.4
    def _sleep(sec: float):
        if sec > 0:
            time.sleep(sec + (JITTER_SEC * 0.5))

    # 1) 토큰
    app_key, app_secret, access_token = get_auth_info()
    base_url = BASE_URL
    if not access_token:
        print("[BUY] 토큰 발급 실패 → 매수 중단"); return

    # 2) 추천 로드
    if not os.path.exists(TODAY_RECOS):
        print(f"[BUY] 추천 파일 없음: {TODAY_RECOS}"); return
    rec = pd.read_csv(TODAY_RECOS, parse_dates=["entry_date"])
    if rec.empty:
        print("[BUY] 추천 비어있음"); return
    rec["code"] = rec["code"].astype(str).str.zfill(6)

    # 2.5) 현재 보유 종목(브로커 기준) 제거
    pos_map_now = _broker_positions_map(access_token, app_key, app_secret)
    held_codes = {c for c,(q,_) in pos_map_now.items() if q > 0}
    rec = rec[~rec["code"].isin(held_codes)].reset_index(drop=True)
    if rec.empty:
        print("[BUY] 추천이 모두 보유 중인 종목 → 매수 없음"); return

    # 3) 초기 주문가능금액과 고정 슬롯 예산(= 1/K_MAX)
    ord_cash_initial = get_orderable_cash(access_token, app_key, app_secret)
    if ord_cash_initial <= 0:
        print("[BUY] 주문가능금액 0원 → 매수 없음"); return
    per_slot_budget = ord_cash_initial / max(1, k_max)   # 상한: 항상 1/K_MAX
    print(f"[BUY] 초기 주문가능금액={ord_cash_initial:,.0f}원, 고정 슬롯 예산(cap)={per_slot_budget:,.0f}원")

    taken = 0
    for _, r in rec.iterrows():
        code = r["code"]
        cur = get_current_price(access_token, app_key, app_secret, base_url, code)
        if (cur is None) or (cur <= 0) or (cur < MIN_PRICE_KRW):
            _sleep(_SLEEP_BETWEEN_BUYS); continue

        # ✅ 남은 슬롯 수 기준 균등분을 계산하고, 1/4 상한으로 캡
        remaining_slots = max(1, k_max - taken)
        ord_cash_now = get_orderable_cash(access_token, app_key, app_secret)
        if ord_cash_now <= 0:
            print("[BUY] 주문가능금액 소진"); break

        equal_share_now = ord_cash_now / remaining_slots
        budget_for_this = min(per_slot_budget, equal_share_now)

        LOCAL_CUSHION = 1.004
        qty = int(budget_for_this // (cur * LOCAL_CUSHION))
        if qty < 1:
            print(f"[SKIP] {code} 예산 {budget_for_this:,.0f}원 → 1주 미만")
            _sleep(_SLEEP_BETWEEN_BUYS); continue

    # (이하 주문 시도/1주씩 감소 백오프/로그는 기존 그대로)


        print(f"[BUY] {code} 목표예산 {budget_for_this:,.0f}원 / 현재가 {cur:.1f} → 초기수량 {qty}주")

        # ---- 주문(1주씩 감소하며 재시도) ----
        odno = None
        attempts = 0
        MAX_ATTEMPTS = max(5, qty + 2)
        while attempts < MAX_ATTEMPTS and qty >= 1:
            attempts += 1

            # 주문 직전 주문가능금액 재확인 → 예산 초과 시 수량 1주씩 감소
            ord_cash_now = get_orderable_cash(access_token, app_key, app_secret)
            while qty >= 1 and (cur * qty * LOCAL_CUSHION) > (ord_cash_now - 1000):
                qty -= 1
            if qty < 1:
                print(f"[SKIP] {code} 주문가능금액 부족으로 수량 0"); break

            odno = execute_order(
                stock_code=code, quantity=qty,
                order_type="매수", order_style="시장가",
                app_key=app_key, app_secret=app_secret,
                access_token=access_token, base_url=base_url
            )
            if odno: break
            qty -= 1
            if qty >= 1: _sleep(1.2)

        if not odno:
            print(f"[FAIL] 시장가 주문 접수 실패: {code}"); _sleep(_SLEEP_BETWEEN_BUYS); continue

        # 잔여 주문가능금액
        remaining_cash = get_orderable_cash(access_token, app_key, app_secret)
        used_budget += (per_slot_budget if (per_slot_budget <= ord_cash_now) else (cur*qty))

        # 포지션 기록
        entry_px_val = cur
        entry_dt = pd.to_datetime(r.get("entry_date", now_kst().date()))
        add_position(code, qty, entry_dt, entry_px_val, tp_pct, sl_pct, horizon_d, order_id_buy=odno)

        # 거래 로그 & 에쿼티 스냅샷(즉시)
        log_trade("BUY", code, qty, entry_px_val, reason="INIT", odno=odno,
                  tp_px=entry_px_val*(1+tp_pct), sl_px=entry_px_val*(1-sl_pct),
                  cash_after=remaining_cash)
        maybe_snapshot_equity(access_token, app_key, app_secret, force=True)

        taken += 1
        _sleep(_SLEEP_BETWEEN_BUYS)

        # 선택: 이미 K_MAX 만큼 집행했으면 종료(추천이 더 있어도)
        if taken >= K_MAX:
            break

    print(f"[BUY] 완료: 집행 {taken}건, 사용예산(개념상) ≈ {used_budget:,.0f}원, 잔여주문가능 {get_orderable_cash(access_token, app_key, app_secret):,.0f}원")

    # 매수 후 동기화(혹시 누락된 종목 담기)
    sync_holdings_with_broker(access_token, app_key, app_secret)

    # 스냅샷
    maybe_snapshot_equity(access_token, app_key, app_secret, force=True)

# ========== 매도(트리플 배리어) 루프 ==========
def run_sell_loop():
    ensure_files()
    app_key, app_secret, access_token = get_auth_info()
    if not access_token:
        print("[SELL] 토큰 발급 실패 → 매도 루프 중단"); return
    base_url = BASE_URL

    # 루프 진입 전 보유 동기화
    sync_holdings_with_broker(access_token, app_key, app_secret)

    pos_map = {}; last_pos_fetch = 0.0; POS_TTL = 15.0
    while True:
        now = now_kst()
        if now.time() >= dtime(CLOSE_HH, CLOSE_MM): break

        if (time.time() - last_pos_fetch) > POS_TTL:
            pos_map = _broker_positions_map(access_token, app_key, app_secret)
            last_pos_fetch = time.time()

        df = load_holdings()
        if not df.empty:
            df = _drop_stale_holdings_against_broker(df, pos_map, now); save_holdings(df)

        if df.empty:
            maybe_snapshot_equity(access_token, app_key, app_secret, force=False)
            _sleep_with_jitter(IDLE_LOOP_SLEEP); continue

        changed = False
        for _, row in df.iterrows():
            code = str(row["code"]).zfill(6)
            br_qty, br_avg = pos_map.get(code, (0, 0.0))
            if br_qty <= 0: continue

            last_update = pd.to_datetime(row.get("last_update"))
            if pd.notna(last_update):
                age_sec = (now.tz_localize(None) - last_update).total_seconds()
                if age_sec < MIN_HOLD_SEC: continue

            _sleep_with_jitter(0.20)
            cur = get_current_price(access_token, app_key, app_secret, base_url, code)
            if (cur is None) or (cur <= 0):
                print(f"[WARN] 가격 조회 실패/0원: {code}"); continue

            try:
                tp = float(row["tp_px"]); sl = float(row["sl_px"])
            except Exception:
                print(f"[WARN] 배리어 파싱 실패: code={code}"); continue
            if not (pd.notna(tp) and pd.notna(sl) and tp > 0 and sl > 0):
                print(f"[WARN] 잘못된 배리어 값: code={code} tp={tp} sl={sl} → 스킵"); continue

            horizon_end = pd.to_datetime(row.get("horizon_end", pd.NaT))
            if pd.notna(horizon_end) and horizon_end.tzinfo is None:
                horizon_end = horizon_end.tz_localize("Asia/Seoul", nonexistent="NaT", ambiguous="NaT")

            reason = None
            if cur >= tp: reason = "TP"
            elif cur <= sl: reason = "SL"
            elif (pd.notna(horizon_end)) and (now >= horizon_end): reason = "TIME"

            if reason:
                print(f"[SELL] {code} x {br_qty} 시장가  reason={reason}  cur={cur:.1f} tp={tp:.1f} sl={sl:.1f}")
                odno = execute_order(stock_code=code, quantity=br_qty, order_type="매도", order_style="시장가",
                                     app_key=app_key, app_secret=app_secret, access_token=access_token, base_url=base_url)
                if odno:
                    cash_after = get_cash_balance(access_token, app_key, app_secret)
                    log_trade("SELL", code, br_qty, cur, reason=reason, odno=odno,
                              tp_px=tp if reason=="TP" else None,
                              sl_px=sl if reason=="SL" else None,
                              cash_after=cash_after)
                    maybe_snapshot_equity(access_token, app_key, app_secret, force=True)
                    remove_position(code)
                    changed = True
                    _sleep_with_jitter(SLEEP_BETWEEN_SELLS)
                else:
                    print(f"[SELL] 실패 → holdings 유지: {code}")

        maybe_snapshot_equity(access_token, app_key, app_secret, force=False)
        if not changed: _sleep_with_jitter(IDLE_LOOP_SLEEP)

# ========== CLI ==========
def main():
    import argparse
    ap = argparse.ArgumentParser(description="rule_3 추천 기반 트리플배리어 매매 봇 (VTS)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("buy", help="시장가 매수 실행 (고정 슬롯 균등비중; 보유 스킵)")
    sub.add_parser("sell-loop", help="장중 트리플배리어 매도 루프")
    sub.add_parser("auto", help="buy 후 sell-loop 연속 실행")
    sub.add_parser("export-equity", help="equity_log.csv로 에쿼티 그래프 PNG 저장")
    args = ap.parse_args()

    cmd = args.cmd or "auto"
    if cmd == "buy":
        run_buy(TP_PCT, SL_PCT, K_MAX, HORIZON)
    elif cmd == "sell-loop":
        run_sell_loop()
    elif cmd == "auto":
        run_buy(TP_PCT, SL_PCT, K_MAX, HORIZON)
        run_sell_loop()
    elif cmd == "export-equity":
        export_equity_png(os.path.join(RESULT_DIR, "equity_curve.png"))

if __name__ == "__main__":
    main()
