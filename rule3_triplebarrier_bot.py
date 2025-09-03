# rule3_triplebarrier_bot.py
# ---------------------------------------------------------
# today_recos.csv(rule_3.py 결과) 기반
# ① 자동 매수(시장가, 동적 슬롯 균등비중) — 보유종목 스킵, 남은 슬롯만 집행
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
APP_KEY       = "PSXtsebcvZLq1ZKGsppEYYxCd0RoOd48INlF"
APP_SECRET    = "pnPjHI+nULtuBz3jTzPhvBQY+9VKfMCql6lN3btyp19EGhi1hALeHrPjhsFj016eaGqACCcDWdZ3ivhNOIVhBZRATrHdiTk8L8uCxVNQn3qpWSk+54SQ/XMCyJvVpUSaPiRBf+n0iSu7blyUjBxQgt9zBMUvBB23ylyMg8yrWCDJZpgQXM4="
CANO          = "50150860"
ACNT_PRDT_CD  = "01"
BASE_URL      = "https://openapivts.koreainvestment.com:29443"  # VTS 서버

ACCOUNT_INFO = {"CANO": CANO, "ACNT_PRDT_CD": ACNT_PRDT_CD}

# ========== 공통 유틸 ==========
def _sleep_with_jitter(base_sec: float):
    if base_sec <= 0:
        return
    if ADD_JITTER:
        time.sleep(base_sec + random.uniform(0, 0.4))
    else:
        time.sleep(base_sec)

def now_kst():
    return pd.Timestamp.now(tz="Asia/Seoul")

def ensure_log_files():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        cols = ["ts","side","code","qty","price","reason","odno","tp_px","sl_px","cash_after"]
        pd.DataFrame(columns=cols).to_csv(TRADES_CSV, index=False, encoding="utf-8")
    if not os.path.exists(EQUITY_CSV):
        cols = ["ts","cash","positions_value","equity"]
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
        df["code"] = df["code"].astype(str).str.zfill(6)
    return df

def save_holdings(df):
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
    """같은 key로 min_interval 초 이내 중복 호출을 지연"""
    now = time.time()
    last = _LAST_CALL.get(key, 0.0)
    wait = last + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[key] = time.time()

# ===== 로그 함수 =====
def log_trade(side: str, code: str, qty: int, price: float, reason: str, odno: Optional[str],
              tp_px: Optional[float]=None, sl_px: Optional[float]=None, cash_after: Optional[float]=None):
    """매수/매도/스냅샷(HEARTBEAT) 로그"""
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception:
        df = pd.DataFrame(columns=["ts","side","code","qty","price","reason","odno","tp_px","sl_px","cash_after"])
    row = {
        "ts": now_kst().tz_localize(None),
        "side": side,
        "code": str(code).zfill(6) if code else "",
        "qty": int(qty) if qty is not None else 0,
        "price": float(price) if price is not None else 0.0,
        "reason": reason or "",
        "odno": odno or "",
        "tp_px": float(tp_px) if tp_px is not None else None,
        "sl_px": float(sl_px) if sl_px is not None else None,
        "cash_after": float(cash_after) if cash_after is not None else None
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADES_CSV, index=False, encoding="utf-8")

# ========== 보유행 추가/병합 ==========
def add_position(code, qty, entry_date, entry_px, tp_pct, sl_pct, horizon_days, order_id_buy=None):
    df = load_holdings()
    code = str(code).zfill(6)
    qty = int(qty)

    # entry_px 검증 강하게
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
        horizon_end_ts = (base_dt + BDay(horizon_days)).replace(
            hour=CLOSE_HH, minute=CLOSE_MM, second=0, microsecond=0
        )
        horizon_end = horizon_end_ts.tz_localize(None)

        df = df[df["code"] != code]
        row = {
            "code": code,
            "qty": int(new_qty),
            "entry_date": pd.to_datetime(entry_date),
            "entry_px": float(new_px),
            "tp_px": float(tp_px),
            "sl_px": float(sl_px),
            "horizon_end": horizon_end,
            "order_id_buy": order_id_buy,
            "last_update": now_kst().tz_localize(None)
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_holdings(df)
        return

    # 신규
    tp_px = float(entry_px) * (1.0 + tp_pct)
    sl_px = float(entry_px) * (1.0 - sl_pct)
    _tz = "Asia/Seoul"
    base_dt = pd.to_datetime(entry_date)
    if base_dt.tzinfo is None:
        base_dt = base_dt.tz_localize(_tz, nonexistent="NaT", ambiguous="NaT")
    horizon_end_ts = (base_dt + BDay(horizon_days)).replace(
        hour=CLOSE_HH, minute=CLOSE_MM, second=0, microsecond=0
    )
    horizon_end = horizon_end_ts.tz_localize(None)

    row = {
        "code": code,
        "qty": int(qty),
        "entry_date": pd.to_datetime(entry_date),
        "entry_px": float(entry_px),
        "tp_px": float(tp_px),
        "sl_px": float(sl_px),
        "horizon_end": horizon_end,
        "order_id_buy": order_id_buy,
        "last_update": now_kst().tz_localize(None)
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_holdings(df)

def remove_position(code):
    df = load_holdings()
    code = str(code).zfill(6)
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
        "appkey": app_key,
        "appsecret": app_secret
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

# >>> 시세 조회(글로벌+심볼 스로틀 + 백오프) <<<
def get_current_price(access_token: str, app_key: str, app_secret: str, base_url: str, code: str) -> Optional[float]:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": str(code).zfill(6)}
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": app_key,
        "appsecret": app_secret,
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

        msg = ""
        if isinstance(j, dict):
            msg = (j.get("msg1") or "")
        if (r and r.status_code in (429, 500)) and ("초당 거래건수" in msg or "EGW00201" in str(j)):
            if attempt < len(retry_delays) + 1:
                delay = retry_delays[attempt-1] + random.uniform(0, 0.2)
                print(f"[PRICE] 레이트리밋 감지 {code} → {delay:.1f}s 대기 후 재시도")
                time.sleep(delay)
                continue

        print(f"[PRICE] 오류 {code} [{getattr(r,'status_code','NA')}]: {j}")
        return None

def execute_order(stock_code, quantity, order_type, order_style, app_key, app_secret, access_token, base_url, price=None):
    """
    order_type: '매수' | '매도'
    order_style: '시장가' | '지정가'
    """
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
                "appkey": app_key,
                "appsecret": app_secret,
                "tr_id": tr_id,
                "custtype": "P",
                "hashkey": hashkey
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
    cache = {
        "token": access_token,
        "timestamp": time.time(),
        "app_key": app_key,
        "app_secret": app_secret
    }
    with open(TOKEN_FILE, "w") as f:
        json.dump(cache, f)
    print("🔄 새로운 토큰 발급 완료")
    return app_key, app_secret, access_token

# ========== 잔고/예수금/평가 & 스냅샷 ==========
def check_account(access_token, app_key, app_secret):
    output1, output2 = [], []
    CTX_AREA_NK100 = ''
    url_base = BASE_URL

    while True:
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{url_base}{path}"

        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {access_token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": "VTTC8434R"
        }

        params = {
            "CANO": ACCOUNT_INFO['CANO'],
            "ACNT_PRDT_CD": ACCOUNT_INFO['ACNT_PRDT_CD'],
            "AFHR_FLPR_YN": "N",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": '',
            "CTX_AREA_NK100": CTX_AREA_NK100
        }

        res = requests.get(url, headers=headers, params=params, timeout=10)
        print("📡 응답 상태코드:", res.status_code)

        try:
            data = res.json()
        except Exception:
            print("❌ JSON 파싱 실패:", res.text[:300])
            return None, None

        if data.get("rt_cd") != "0" or "output1" not in data:
            print("❌ API 실패: 토큰/권한/요청 파라미터 확인 필요.")
            return None, None

        output1.append(pd.DataFrame.from_records(data['output1']))
        CTX_AREA_NK100 = data.get('ctx_area_nk100', '').strip()

        if CTX_AREA_NK100 == '':
            output2.append(data.get('output2', [{}])[0])
            break

    if output1 and not output1[0].empty:
        res1 = pd.concat(output1)[['pdno', 'hldg_qty', 'pchs_avg_pric']].rename(columns={
            'pdno': '종목코드',
            'hldg_qty': '보유수량',
            'pchs_avg_pric': '매입단가'
        }).reset_index(drop=True)
        res1['종목코드'] = res1['종목코드'].astype(str).str.zfill(6)
        res1['보유수량'] = pd.to_numeric(res1['보유수량'], errors='coerce').fillna(0).astype(int)
        res1['매입단가'] = pd.to_numeric(res1['매입단가'], errors='coerce').fillna(0.0).astype(float)
    else:
        res1 = pd.DataFrame(columns=['종목코드', '보유수량', '매입단가'])

    res2 = output2[0] if output2 else {}
    return res1, res2

def get_cash_balance(access_token, app_key, app_secret) -> float:
    res1, res2 = check_account(access_token, app_key, app_secret)
    if res2 is None:
        return 0.0
    return to_float(res2.get("dnca_tot_amt", 0))

def get_orderable_cash(access_token, app_key, app_secret) -> float:
    """실제 주문 가능 금액(증거금/체결예약 포함)을 우선 사용"""
    _, out2 = check_account(access_token, app_key, app_secret)
    if not out2:
        return 0.0
    keys = ["ord_psbl_cash", "ord_psbl_amt", "ord_psbl_cash_amt", "dnca_tot_amt"]
    for k in keys:
        if k in out2 and to_float(out2[k]) > 0:
            return to_float(out2[k])
    return to_float(out2.get("dnca_tot_amt", 0))

def _portfolio_valuation(access_token, app_key, app_secret) -> Tuple[float, float, float]:
    """(cash, positions_value, equity) 반환"""
    cash = get_cash_balance(access_token, app_key, app_secret)
    pos_df, _ = check_account(access_token, app_key, app_secret)
    positions_value = 0.0
    if pos_df is not None and not pos_df.empty:
        for _, r in pos_df.iterrows():
            code = str(r["종목코드"]).zfill(6)
            qty  = int(r["보유수량"])
            if qty <= 0:
                continue
            cur = get_current_price(access_token, app_key, app_secret, BASE_URL, code)
            if cur and cur > 0:
                positions_value += qty * cur
            _sleep_with_jitter(0.15)  # 글로벌 한도 완화
    equity = cash + positions_value
    return float(cash), float(positions_value), float(equity)

def export_equity_png(out_path: str = os.path.join(RESULT_DIR, "equity_curve.png")):
    """equity_log.csv 기반 에쿼티 커브 PNG 생성/덮어쓰기"""
    try:
        df = pd.read_csv(EQUITY_CSV, parse_dates=["ts"])
    except Exception:
        print("[EQUITY] 로그 없음 → PNG 생략")
        return
    if df.empty:
        print("[EQUITY] 데이터 비어있음 → PNG 생략")
        return

    df = df.sort_values("ts")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[EQUITY] matplotlib 불가 → PNG 생략: {e}")
        return

    plt.figure()
    plt.plot(df["ts"], df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (KRW)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[EQUITY] PNG 업데이트: {out_path}")

_last_equity_snap = 0.0
def maybe_snapshot_equity(access_token, app_key, app_secret, force: bool=False):
    """주기적으로 에쿼티 커브 저장 + PNG 갱신 + HEARTBEAT 로그"""
    global _last_equity_snap
    now_sec = time.time()
    if not force and (now_sec - _last_equity_snap) < EQUITY_SNAPSHOT_SEC:
        return

    cash, pv, eq = _portfolio_valuation(access_token, app_key, app_secret)
    try:
        df = pd.read_csv(EQUITY_CSV)
    except Exception:
        df = pd.DataFrame(columns=["ts","cash","positions_value","equity"])
    row = {
        "ts": now_kst().tz_localize(None),
        "cash": cash,
        "positions_value": pv,
        "equity": eq
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EQUITY_CSV, index=False, encoding="utf-8")
    _last_equity_snap = now_sec

    # 스냅샷 직후 PNG 갱신
    export_equity_png(os.path.join(RESULT_DIR, "equity_curve.png"))

    # 1시간마다 trades_log에도 HEARTBEAT 남김
    log_trade(side="SNAP", code="", qty=0, price=0.0, reason="HEARTBEAT", odno="",
              tp_px=None, sl_px=None, cash_after=cash)

# ========== 브로커 보유 맵/스테일 정리 ==========
def _broker_positions_map(access_token, app_key, app_secret) -> Dict[str, Tuple[int, float]]:
    """종목코드 → (보유수량, 매입단가)"""
    res1, _ = check_account(access_token, app_key, app_secret)
    mp: Dict[str, Tuple[int, float]] = {}
    if res1 is not None and not res1.empty:
        for _, r in res1.iterrows():
            mp[str(r["종목코드"]).zfill(6)] = (int(r["보유수량"]), float(r["매입단가"]))
    return mp

def _drop_stale_holdings_against_broker(df_hold: pd.DataFrame, pos_map: Dict[str, Tuple[int,float]], now_ts: pd.Timestamp) -> pd.DataFrame:
    """브로커에 없는 종목은 holdings에서 제거(단, 최근 추가 항목은 GRACE 유지)."""
    keep_rows = []
    for _, row in df_hold.iterrows():
        code = str(row["code"]).zfill(6)
        qty  = int(row.get("qty", 0) or 0)
        if qty <= 0:
            continue
        br_qty = pos_map.get(code, (0, 0.0))[0]
        if br_qty <= 0:
            last_upd = pd.to_datetime(row.get("last_update"))
            if pd.isna(last_upd):
                continue  # 오래된 행 → 제거
            age_sec = (now_ts.tz_localize(None) - last_upd).total_seconds()
            if age_sec <= CLEANUP_GRACE_SEC:
                keep_rows.append(row); continue
            print(f"[CLEANUP] 브로커 보유 0 & 유예초과 → holdings 제거: {code}")
            continue
        keep_rows.append(row)
    return pd.DataFrame(keep_rows, columns=df_hold.columns) if keep_rows else pd.DataFrame(columns=df_hold.columns)

# ========== 매수(시장가, 동적 슬롯 균등비중) ==========
def run_buy(tp_pct: float, sl_pct: float, k_max: int, horizon_d: int):
    """
    - today_recos.csv 상위 slots_to_use 종목만 집행
    - 보유 종목은 스킵, 남은 슬롯만 매수
    - 주문가능금액을 상한으로 수량 산정
    - 실패 시 '1주씩' 줄이며 재시도
    """
    ensure_files()

    _SLEEP_BETWEEN_BUYS = 1.5
    JITTER_SEC         = 0.4

    def _sleep(sec: float):
        if sec > 0:
            time.sleep(sec + (JITTER_SEC * 0.5))

    # 1) 토큰
    app_key, app_secret, access_token = get_auth_info()
    base_url = BASE_URL
    if not access_token:
        print("[BUY] 토큰 발급 실패 → 매수 중단")
        return

    # 2) 추천 로드
    if not os.path.exists(TODAY_RECOS):
        print(f"[BUY] 추천 파일 없음: {TODAY_RECOS}")
        return
    rec = pd.read_csv(TODAY_RECOS, parse_dates=["entry_date"])
    if rec.empty:
        print("[BUY] 추천 비어있음")
        return
    rec["code"] = rec["code"].astype(str).str.zfill(6)

    # 2.5) 현재 보유 종목/슬롯 파악 (브로커 기준)
    pos_map_now = _broker_positions_map(access_token, app_key, app_secret)
    held_codes = {c for c,(q,_) in pos_map_now.items() if q > 0}
    held_cnt   = len(held_codes)
    slots_left = max(0, k_max - held_cnt)
    if slots_left <= 0:
        print(f"[BUY] 남은 슬롯 0 (보유 {held_cnt}/{k_max}) → 매수 없음")
        return

    # 보유 중인 종목은 추천에서 제외
    rec = rec[~rec["code"].isin(held_codes)].reset_index(drop=True)
    if rec.empty:
        print("[BUY] 추천이 모두 보유 중인 종목 → 매수 없음")
        return

    slots_to_use = min(slots_left, len(rec))

    # 3) 주문가능금액 확인
    ord_cash_global = get_orderable_cash(access_token, app_key, app_secret)
    if ord_cash_global <= 0:
        print("[BUY] 주문가능금액 0원 → 매수 없음")
        return

    remaining_cash  = ord_cash_global
    taken = 0

    for _, r in rec.head(slots_to_use).iterrows():
        code = r["code"]

        # 현재가
        cur = get_current_price(access_token, app_key, app_secret, base_url, code)
        if (cur is None) or (cur <= 0):
            print(f"[SKIP] {code} 현재가 조회 실패/0원")
            _sleep(_SLEEP_BETWEEN_BUYS); continue
        if cur < MIN_PRICE_KRW:
            print(f"[SKIP] {code} 현재가 {cur:.0f} < {MIN_PRICE_KRW}원")
            _sleep(_SLEEP_BETWEEN_BUYS); continue

        # 동적 예산(참고) + 주문가능금액 상한
        remaining_slots = max(1, slots_to_use - taken)
        dyn_budget = max(0.0, remaining_cash) / remaining_slots

        LOCAL_CUSHION = 1.004
        ord_cash_now = get_orderable_cash(access_token, app_key, app_secret)
        budget_for_this = min(dyn_budget, ord_cash_now)
        qty = int(budget_for_this // (cur * LOCAL_CUSHION))

        if qty < 1:
            print(f"[SKIP] {code} 주문가능 {ord_cash_now:,.0f}원/동적 {dyn_budget:,.0f}원 → 1주 미만")
            _sleep(_SLEEP_BETWEEN_BUYS); continue

        print(f"[BUY] {code} 초기 수량 {qty}주 (현재가 {cur:.1f}, 동적 {dyn_budget:,.0f}, 주문가능 {ord_cash_now:,.0f})")

        # ---- 주문(1주씩 감소하며 재시도) ----
        odno = None
        attempts = 0
        MAX_ATTEMPTS = max(5, qty + 2)  # 너무 오래 끌지 않게 제한
        while attempts < MAX_ATTEMPTS and qty >= 1:
            attempts += 1

            # 주문 직전 주문가능금액 재확인 → 현재 수량이 안 되면 1주씩 줄임
            ord_cash_now = get_orderable_cash(access_token, app_key, app_secret)
            while qty >= 1 and (cur * qty * LOCAL_CUSHION) > (ord_cash_now - 1000):
                qty -= 1

            if qty < 1:
                print(f"[SKIP] {code} 주문가능금액 부족으로 수량 0")
                break

            odno = execute_order(
                stock_code=code, quantity=qty,
                order_type="매수", order_style="시장가",
                app_key=app_key, app_secret=app_secret,
                access_token=access_token, base_url=base_url
            )
            if odno:
                break

            # 실패 시 1주 감소 후 백오프
            qty -= 1
            if qty >= 1:
                _sleep(1.2)  # 소폭 백오프

        if not odno:
            print(f"[FAIL] 시장가 주문 접수 실패: {code}")
            _sleep(_SLEEP_BETWEEN_BUYS); continue

        # 실 주문가능금액 재조회(잔여)
        remaining_cash = get_orderable_cash(access_token, app_key, app_secret)

        # 포지션 기록
        entry_px_val = cur  # (체결가 대신 현재가 근사)
        entry_dt = pd.to_datetime(r.get("entry_date", now_kst().date()))
        add_position(code, qty, entry_dt, entry_px_val, tp_pct, sl_pct, horizon_d, order_id_buy=odno)

        # 거래 로그 & 에쿼티 스냅샷(즉시)
        log_trade("BUY", code, qty, entry_px_val, reason="INIT", odno=odno,
                  tp_px=entry_px_val*(1+tp_pct), sl_px=entry_px_val*(1-sl_pct),
                  cash_after=remaining_cash)
        maybe_snapshot_equity(access_token, app_key, app_secret, force=True)

        taken += 1
        _sleep(_SLEEP_BETWEEN_BUYS)

        if taken >= slots_to_use:
            break

    print(f"[BUY] 완료: 집행 {taken}종목 / 요청 {slots_to_use}종목, 잔여주문가능 {remaining_cash:,.0f}원")
    # 매수 루프 종료 후 한 번 더 스냅샷
    maybe_snapshot_equity(access_token, app_key, app_secret, force=True)

# ========== 매도(트리플 배리어) 루프 ==========
def run_sell_loop():
    ensure_files()

    app_key, app_secret, access_token = get_auth_info()
    if not access_token:
        print("[SELL] 토큰 발급 실패 → 매도 루프 중단")
        return
    base_url = BASE_URL

    # 잔고조회 캐시 (15초)
    pos_map = {}
    last_pos_fetch = 0.0
    POS_TTL = 15.0

    while True:
        now = now_kst()
        if now.time() >= dtime(CLOSE_HH, CLOSE_MM):
            break

        # 잔고 캐시 업데이트
        if (time.time() - last_pos_fetch) > POS_TTL:
            pos_map = _broker_positions_map(access_token, app_key, app_secret)
            last_pos_fetch = time.time()

        df = load_holdings()
        if not df.empty:
            df = _drop_stale_holdings_against_broker(df, pos_map, now)
            save_holdings(df)

        if df.empty:
            # 루프마다 1회 스냅샷(주기 조건 충족 시)
            maybe_snapshot_equity(access_token, app_key, app_secret, force=False)
            _sleep_with_jitter(IDLE_LOOP_SLEEP)
            continue

        changed = False
        for _, row in df.iterrows():
            code = str(row["code"]).zfill(6)

            # 브로커 실보유 수량으로만 매도 평가
            br_qty, br_avg = pos_map.get(code, (0, 0.0))
            if br_qty <= 0:
                continue

            # 방금 추가된 포지션은 MIN_HOLD_SEC 동안 매도평가 skip
            last_update = pd.to_datetime(row.get("last_update"))
            if pd.notna(last_update):
                age_sec = (now.tz_localize(None) - last_update).total_seconds()
                if age_sec < MIN_HOLD_SEC:
                    continue

            # 심볼 간 간격(글로벌 한도 완화)
            _sleep_with_jitter(0.20)

            cur = get_current_price(access_token, app_key, app_secret, base_url, code)
            if (cur is None) or (cur <= 0):
                print(f"[WARN] 가격 조회 실패/0원: {code}")
                continue

            try:
                tp = float(row["tp_px"]); sl = float(row["sl_px"])
            except Exception:
                print(f"[WARN] 배리어 파싱 실패: code={code}")
                continue
            if not (pd.notna(tp) and pd.notna(sl) and tp > 0 and sl > 0):
                print(f"[WARN] 잘못된 배리어 값: code={code} tp={tp} sl={sl} → 스킵")
                continue

            horizon_end = pd.to_datetime(row.get("horizon_end", pd.NaT))
            if pd.notna(horizon_end) and horizon_end.tzinfo is None:
                horizon_end = horizon_end.tz_localize("Asia/Seoul", nonexistent="NaT", ambiguous="NaT")

            reason = None
            if cur >= tp:
                reason = "TP"
            elif cur <= sl:
                reason = "SL"
            elif (pd.notna(horizon_end)) and (now >= horizon_end):
                reason = "TIME"

            if reason:
                print(f"[SELL] {code} x {br_qty} 시장가  reason={reason}  cur={cur:.1f} tp={tp:.1f} sl={sl:.1f}")
                odno = execute_order(
                    stock_code=code, quantity=br_qty,
                    order_type="매도", order_style="시장가",
                    app_key=app_key, app_secret=app_secret,
                    access_token=access_token, base_url=base_url
                )
                if odno:
                    # 거래 로그(현금은 재조회) & 스냅샷(즉시)
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

        # 루프마다 1회 스냅샷(주기 충족 시)
        maybe_snapshot_equity(access_token, app_key, app_secret, force=False)

        if not changed:
            _sleep_with_jitter(IDLE_LOOP_SLEEP)

# ========== CLI ==========
def main():
    import argparse
    ap = argparse.ArgumentParser(description="rule_3 추천 기반 트리플배리어 매매 봇 (VTS)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("buy", help="시장가 매수 실행 (동적 슬롯 균등비중; 보유 스킵)")
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
