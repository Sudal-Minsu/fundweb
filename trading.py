import os
import sys
import time
import json
import requests
import pandas as pd
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
import keyring

# ───────────── 설정 ─────────────
# 키링에서 불러오기. 최초에 키가 없다면 아래 DEFAULT_* 값으로 채워 넣는다.
APP_USER = "최진혁"
APP_KEY_SERVICE = "mock_app_key"
APP_SECRET_SERVICE = "mock_app_secret"

# 필요하다면 기본값으로 1회 세팅
DEFAULT_APP_KEY = "PSbWOQW9CsjVIq8MwF3oeHG9gY9JjLHJVu8t"
DEFAULT_APP_SECRET = (
    "uzxSVMytr/jWcbCYMBGcRMloeCM9A1fiTOur3Y3j30RY6gtvf3G0Bn1y/"
    "z6J2pa0CKKZRFf6OXpk/umYfxZaWQr4eVmoCJG6BX7wfQ/GOYlEDotyouzkMwevv7hjI06tzruSpPuN6EMS1nirtIeTnh8kxxN4LBS70XggdFevyM3KR87RG7k="
)

def _ensure_keyring():
    # 실행 환경에 키가 없으면 기본값으로 1회 등록 (이미 있으면 건너뜀)
    if keyring.get_password(APP_KEY_SERVICE, APP_USER) is None:
        keyring.set_password(APP_KEY_SERVICE, APP_USER, DEFAULT_APP_KEY)
    if keyring.get_password(APP_SECRET_SERVICE, APP_USER) is None:
        keyring.set_password(APP_SECRET_SERVICE, APP_USER, DEFAULT_APP_SECRET)

def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    _ensure_keyring()
    app_key = keyring.get_password(APP_KEY_SERVICE, APP_USER)
    app_secret = keyring.get_password(APP_SECRET_SERVICE, APP_USER)
    return app_key, app_secret

# 계좌 정보 (필요 시 수정)
ACCOUNT_INFO = {
    "CANO": "50139282",   # 계좌번호 앞 8자리
    "ACNT_PRDT_CD": "01", # 계좌번호 뒤 2자리
}

OUTPUT_DIR = "rule_2_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_BUY_BUDGET = 20_000_000    # 종목당 최대 매수 금액
TOP_N_TO_BUY   = 5             # 오늘 후보 중 최대 N개 매수
LOOP_SLEEP_SEC = 300           # 루프 대기 (초)

# 장 종료 시각(한국 주식): 15:30
MARKET_CLOSE_TIME = dtime(15, 30)

# 당일 기록 파일 (OUTPUT_DIR에 저장)
BOUGHT_TODAY_PATH       = os.path.join(OUTPUT_DIR, "bought_today.json")
NOT_TRADABLE_TODAY_PATH = os.path.join(OUTPUT_DIR, "not_tradable_today.json")

# 루프별 총 평가금액 CSV (OUTPUT_DIR 내부) — loop, tot_evlu_amt만 저장
PORTFOLIO_CSV = os.path.join(OUTPUT_DIR, "총평가금액.csv")

app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"  # 모의투자 VTS

# ───────────── 토큰 ─────────────
res = requests.post(
    f"{url_base}/oauth2/tokenP",
    headers={"content-type": "application/json"},
    data=json.dumps({
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    })
)
access_token = res.json().get("access_token", "")
if not access_token:
    print("❌ 액세스 토큰 발급 실패:", res.text, flush=True)
    sys.exit()
print("🔑 액세스 토큰 OK", flush=True)

# ───────────── 공통 유틸 ─────────────
def _num(x):
    if x is None: return None
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "null": return None
    try:
        return float(s)
    except:
        return None

def _num0(x):
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() == "null":
            return 0.0
        return float(s)
    except:
        return 0.0

def is_market_closed_msg(msg: str) -> bool:
    if not msg:
        return False
    m = msg.strip().lower()
    return ("장종료" in m) or ("장 종료" in m) or ("closed" in m) or ("market closed" in m)

# ───────────── 루프 시작번호: CSV에서 결정 ─────────────
def load_loop_counter_from_csv():
    """
    총평가금액.csv의 마지막 loop 값을 읽어 다음 시작 루프 번호를 반환.
    파일이 없거나 비어 있으면 1부터 시작.
    """
    try:
        if not os.path.exists(PORTFOLIO_CSV):
            return 1
        df = pd.read_csv(PORTFOLIO_CSV)
        if "loop" not in df.columns or df.empty:
            return 1
        last_loop = pd.to_numeric(df["loop"], errors="coerce").dropna()
        if last_loop.empty:
            return 1
        return int(last_loop.max()) + 1
    except Exception as e:
        print(f"⚠️ 루프 시작번호 계산 실패(기본 1로 시작): {e}", flush=True)
        return 1

# ───────────── 시세/주문 유틸 ─────────────
def get_quote(stock_code):
    """
    현재가/매도1호가/매수1호가 조회 (원시값 사용)
    반환: (current_price:int|None, best_ask:int|None, best_bid:int|None)
    """
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    res = requests.get(url, headers=headers, params=params)
    time.sleep(1.2)
    if res.status_code != 200 or 'output' not in res.json():
        print(f"❌ 시세 조회 실패: {stock_code} / {res.text}", flush=True)
        return None, None, None
    out = res.json()['output']

    def _to_int(x):
        try:
            return int(str(x).replace(",", "").strip())
        except:
            return None

    cur  = _to_int(out.get('stck_prpr'))
    ask1 = _to_int(out.get('askp1') or out.get('askp'))
    bid1 = _to_int(out.get('bidp1') or out.get('bidp'))
    return cur, ask1, bid1

def get_current_price(stock_code):
    cur, _, _ = get_quote(stock_code)
    return cur

def get_hashkey(data):
    url = f"{url_base}/uapi/hashkey"
    headers = {"Content-Type": "application/json", "appKey": app_key, "appSecret": app_secret}
    res = requests.post(url, headers=headers, data=json.dumps(data))
    time.sleep(1.2)
    return res.json().get("HASH", "")

def send_order(stock_code, price, qty, order_type="매수", ord_dvsn="00"):
    """
    고정 사용:
      - 매수: 04(최우선지정가) → ORD_UNPR=0
      - 매도: 01(시장가)      → ORD_UNPR=0
    """
    url = f"{url_base}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = "VTTC0802U" if order_type == "매수" else "VTTC0801U"

    price_free_types = {"01","03","04","11","12","13","14","15","16"}
    unpr = "0" if ord_dvsn in price_free_types else str(int(price))

    data = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "PDNO": stock_code,
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(qty),
        "ORD_UNPR": unpr
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
        j = res.json()
    except Exception:
        j = {"rt_cd": "-1", "msg1": res.text}
    return j

# ───────────── 실계좌 보유 조회(매 루프 강제 최신) ─────────────
def get_all_holdings():
    """
    [강제 최신화] 매 호출마다 API로 pchs_avg_pric·prpr를 가져옴.
    평균가는 'pchs_avg_pric'만, 현재가는 'prpr'만 사용.
    반환: { code: {"qty": int, "avg_price": float|None, "cur_price": float|None} }
    """
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
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
    merged = {}
    while True:
        res = requests.get(url, headers=headers, params=params)
        time.sleep(1.2)
        data = res.json()

        for item in data.get("output1", []):
            code = str(item.get("pdno", "")).zfill(6)
            qty  = _num(item.get("hldg_qty"))
            if not qty or qty <= 0:
                continue

            avg = _num(item.get("pchs_avg_pric"))  # 평균가
            cur = _num(item.get("prpr"))           # 현재가

            merged[code] = {
                "qty": int(qty),
                "avg_price": (avg if (avg and avg > 0) else None),
                "cur_price": (cur if (cur and cur > 0) else None),
            }

        if data.get("tr_cont", "F") != "M":
            break
        params["CTX_AREA_FK100"] = data.get("ctx_area_fk100", "")
        params["CTX_AREA_NK100"] = data.get("ctx_area_nk100", "")
    return merged

# ───────────── 당일 주문·체결 조회 ─────────────
def get_today_orders():
    """
    주식 일별 주문/체결 조회 (당일)
    - 모의투자(VTS) TR ID: VTTC0081R
    - 실전계좌는 TTTC0081R 참고
    반환: output1 리스트(주문내역들)
    """
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC0081R",
    }
    today = datetime.now().strftime("%Y%m%d")
    params = {
        "CANO": ACCOUNT_INFO["CANO"],
        "ACNT_PRDT_CD": ACCOUNT_INFO["ACNT_PRDT_CD"],
        "INQR_STRT_DT": today,
        "INQR_END_DT":  today,
        "SLL_BUY_DVSN_CD": "00",  # 전체
        "INQR_DVSN": "00",        # 전체
        "PDNO": "",               # 전체 종목
        "CCLD_DVSN": "00",        # 전체(체결/미체결)
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    items = []
    while True:
        res = requests.get(url, headers=headers, params=params)
        time.sleep(1.2)
        j = res.json()
        items.extend(j.get("output1", []) or [])
        if j.get("tr_cont", "F") != "M":
            break
        params["CTX_AREA_FK100"] = j.get("ctx_area_fk100", "")
        params["CTX_AREA_NK100"] = j.get("ctx_area_nk100", "")
    return items

def build_bought_today_set(today_orders):
    """
    오늘 '매수' 주문(체결/미체결 불문)이 1건이라도 있던 종목을 set으로 반환.
    → 당일 재매수 방지
    """
    bought = set()
    for o in today_orders:
        code = str(o.get("pdno", "")).zfill(6)
        side_txt = (o.get("sll_buy_dvsn_cd") or o.get("sll_buy_dvsn_name") or o.get("trad_dvsn_name") or "").strip()
        is_buy   = ("매수" in side_txt) or (str(side_txt) in ("02", "2"))
        if code and is_buy:
            bought.add(code)
    return bought

def has_open_orders(today_orders):
    """
    오늘자 주문 중 '진짜' 미체결 잔량이 있는지 판단.
    """
    def _text(o, *keys):
        for k in keys:
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    terminal_keywords = [
        "거부", "불가", "매매불가", "주문거절", "취소", "정정거부", "오류",
        "rejected", "reject", "cancel", "canceled", "cancelled", "error", "invalid"
    ]

    for o in today_orders:
        st = _text(o, "ordr_sttus_name", "ccld_dvsn_name", "ord_sttus")
        st_lower = st.lower()

        if any(k in st for k in terminal_keywords) or any(k in st_lower for k in terminal_keywords):
            continue
        if "미체결" in st:
            return True

        rmn = _num0(o.get("rmn_qty")) or _num0(o.get("unerc_qty"))
        if rmn > 0:
            return True

        ord_qty   = _num0(o.get("ord_qty"))
        ccld_qty1 = _num0(o.get("tot_ccld_qty"))
        ccld_qty2 = _num0(o.get("ccld_qty"))
        ccld_qty  = max(ccld_qty1, ccld_qty2)
        if ord_qty > 0 and ccld_qty < ord_qty:
            return True

    return False

# ───────────── 종목별 '매도 미체결 잔량' 계산 ─────────────
def get_open_sell_qty_for_code(today_orders, code: str) -> int:
    """
    당일 주문/체결 내역(today_orders)에서 특정 종목(code)의 '매도' 미체결 잔량 합계를 반환.
    - 상태가 거절/취소/오류 등 '종료'에 해당하면 제외
    - 잔량 필드는 rmn_qty 또는 unerc_qty 를 우선 사용
    - 보정: (주문수량 - 체결수량) 차이가 더 크면 그 값을 사용
    """
    code = str(code).zfill(6)

    def _text(o, *keys):
        for k in keys:
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    terminal_keywords = [
        "거부", "불가", "매매불가", "주문거절", "취소", "정정거부", "오류",
        "rejected", "reject", "cancel", "canceled", "cancelled", "error", "invalid"
    ]

    open_qty = 0
    for o in today_orders or []:
        pdno = str(o.get("pdno", "")).zfill(6)
        if pdno != code:
            continue

        # 매도 여부 판별
        side_txt = (o.get("sll_buy_dvsn_cd") or o.get("sll_buy_dvsn_name") or o.get("trad_dvsn_name") or "").strip()
        is_sell = ("매도" in side_txt) or (str(side_txt) in ("01", "1"))
        if not is_sell:
            continue

        # 종료 상태는 제외
        st = _text(o, "ordr_sttus_name", "ccld_dvsn_name", "ord_sttus")
        st_lower = st.lower()
        if any(k in st for k in terminal_keywords) or any(k in st_lower for k in terminal_keywords):
            continue

        # 미체결 잔량
        rmn = _num0(o.get("rmn_qty")) or _num0(o.get("unerc_qty"))
        if rmn > 0:
            open_qty += int(rmn)

        # 보정: 주문수량 - 체결수량
        ord_qty   = _num0(o.get("ord_qty"))
        ccld_qty1 = _num0(o.get("tot_ccld_qty"))
        ccld_qty2 = _num0(o.get("ccld_qty"))
        ccld_qty  = max(ccld_qty1, ccld_qty2)
        gap = max(0, int(round(ord_qty - ccld_qty)))
        if gap > open_qty:
            open_qty = gap

    return int(open_qty)

# ───────────── 오늘 매수/매매불가 이력 저장/복원 ─────────────
def load_bought_today(today_str):
    try:
        with open(BOUGHT_TODAY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == today_str:
            return set(data.get("codes", []))
    except Exception:
        pass
    return set()

def save_bought_today(today_str, codes_set):
    try:
        with open(BOUGHT_TODAY_PATH, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "codes": sorted(list(codes_set))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_not_tradable(today_str):
    try:
        with open(NOT_TRADABLE_TODAY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == today_str:
            return set(data.get("codes", []))
    except Exception:
        pass
    return set()

def save_not_tradable(today_str, codes_set):
    try:
        with open(NOT_TRADABLE_TODAY_PATH, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "codes": sorted(list(codes_set))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ───────────── 스냅샷 저장(요청 형식) ─────────────
def save_account_snapshot_to_json(holdings):
    """
    portfolio_2.json (루트 저장)
    {
      "017900": {"buy_price": 1930, "qty": 514},
      ...
    }
    """
    snapshot = {}
    for code, info in holdings.items():
        qty = int(info.get("qty", 0) or 0)
        avg = info.get("avg_price", None)
        if avg is not None:
            try:
                avg_int = int(round(float(avg)))
            except Exception:
                avg_int = avg
            buy_price = avg_int
        else:
            buy_price = None
        snapshot[code] = {"buy_price": buy_price, "qty": qty}

    with open("portfolio_2.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

# ───────────── 매수 직후 평단 재조회 ─────────────
def refresh_avg_after_buy(code, tries=3, delay=1.5):
    """
    매수 직후 체결/평균가가 잔고에 반영될 때까지 짧게 재조회.
    (평균가는 pchs_avg_pric만 사용)
    """
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

# ───────────── 매수/매도 규칙 ─────────────
def buy_candidates(holdings, buy_codes, loop_count, bought_today, not_tradable_today):
    """
    오늘 후보 상위 N개를 최우선지정가(04)만으로 매수.
    - 오늘 이미 매수/보유 이력(bought_today) 또는 오늘 매매불가 목록은 스킵
    - 수량 산정은 현재가(cur) 기준
    반환: market_closed_detected(bool)
    """
    print(f"[루프 {loop_count}] ▶ 매수 단계", flush=True)
    today_str = datetime.now().strftime("%Y%m%d")
    market_closed = False

    for code in buy_codes[:TOP_N_TO_BUY]:
        if code in not_tradable_today:
            print(f"  ↪ 오늘 매매불가 스킵: {code}", flush=True)
            continue
        if code in bought_today:
            print(f"  ↪ 오늘 이미 매수/보유 이력 있어 스킵: {code}", flush=True)
            continue

        cur = get_current_price(code)
        if not cur:
            print(f"  ❌ 현재가 조회 실패: {code}", flush=True)
            continue

        qty = MAX_BUY_BUDGET // cur
        if qty <= 0:
            print(f"  ❌ 예산 부족: {code} cur={cur}", flush=True)
            continue

        # 최우선지정가(04)만 사용 (가격 0)
        result = send_order(code, price=0, qty=qty, order_type="매수", ord_dvsn="04")
        msg = (result.get("msg1") or "").strip()
        print(f"  🟩 매수 04 요청: {code} x{qty} (기준가:{cur}) → {result.get('rt_cd')} {msg}", flush=True)
        log_trade(datetime.now(), code, cur, qty, "매수", result)

        # 시장 종료 감지 → 즉시 종료 신호
        if is_market_closed_msg(msg):
            print("⛔ 시장 종료 감지(매수 응답) → 루프 종료 예정", flush=True)
            market_closed = True
            break

        if str(result.get("rt_cd")) == "0":
            bought_today.add(code)
            save_bought_today(today_str, bought_today)
            refresh_avg_after_buy(code, tries=3, delay=1.5)
        else:
            # 매매 불가/거래정지류면 오늘 스킵 목록에 넣어 재시도 방지
            if any(k in msg for k in ["매매불가", "거래불가", "거래정지", "거래 금지", "주문거절"]):
                not_tradable_today.add(code)
                save_not_tradable(today_str, not_tradable_today)
                print(f"  ⛔ 매매불가 감지 → 오늘 스킵 등록: {code}", flush=True)

    return market_closed

def sell_rules_for_all(holdings, streaks, loop_count, today_orders):
    """
    매도 규칙:
    1) 수익률 < -1% 또는 > +2% : 시장가(01) 전량 매도
       → 단, '매도 미체결 잔량'을 제외한 '실매도 가능 수량(보유-미체결)'만 매도
    2) 수익률이 1% ~ 2% 구간에 2루프 연속 존재 : 시장가(01) 위와 동일
    3) 수익률이 0% ~ 1% 구간에 3루프 연속 존재 : 시장가(01) 위와 동일
    """
    print(f"[루프 {loop_count}] ▶ 매도 검사", flush=True)
    market_closed = False

    for code, pos in holdings.items():
        qty = pos.get('qty', 0)
        avg = pos.get('avg_price', None)
        if qty <= 0:
            streaks[code] = {"low":0, "mid":0}
            continue
        if avg is None:
            print(f"  • {code} 평균가 없음 → 매도 판단 보류", flush=True)
            streaks[code] = {"low":0, "mid":0}
            continue

        cur = pos.get('cur_price', None) or get_current_price(code)
        if not cur:
            print(f"  ❌ 현재가 조회 실패: {code}", flush=True)
            continue

        pnl_pct = (cur - avg) / avg * 100.0
        print(f"  • {code} 현재가:{cur} 평균가:{avg} 수익률:{pnl_pct:.2f}%", flush=True)

        # 공통: 현재 '매도 미체결 잔량' 조회 후, 실제 매도 가능 수량 계산
        open_sell_qty = get_open_sell_qty_for_code(today_orders, code)
        sellable_qty = max(0, int(qty) - int(open_sell_qty))

        # 규칙 1
        if pnl_pct < -1 or pnl_pct > 2.0:
            if sellable_qty <= 0:
                print(f"    → 규칙1 충족이나 매도 가능 수량 0 (보유:{qty}, 매도미체결:{open_sell_qty}) ▶ 스킵", flush=True)
            else:
                print(f"    → 규칙1 충족: {pnl_pct:.2f}% ▶ 시장가 매도 sellable={sellable_qty} (보유:{qty}, 미체결:{open_sell_qty})", flush=True)
                result = send_order(code, 0, sellable_qty, order_type="매도", ord_dvsn="01")
                msg = (result.get("msg1") or "").strip()
                log_trade(datetime.now(), code, cur, sellable_qty, "매도(규칙1)", result)
                print(f"      결과: {result.get('rt_cd')} {msg}", flush=True)
                if is_market_closed_msg(msg):
                    print("⛔ 시장 종료 감지 → 루프 종료 예정", flush=True)
                    market_closed = True
            streaks[code] = {"low":0, "mid":0}
            if market_closed:
                break
            continue

        # 규칙 2 (1% ~ 2% 구간 2루프 연속)
        if 1 < pnl_pct <= 2.0:
            streaks.setdefault(code, {"low":0, "mid":0})
            streaks[code]["mid"] += 1
        else:
            if code in streaks: streaks[code]["mid"] = 0

        if streaks.get(code, {}).get("mid", 0) >= 2:
            if sellable_qty <= 0:
                print(f"    → 규칙2 충족이나 매도 가능 수량 0 (보유:{qty}, 매도미체결:{open_sell_qty}) ▶ 스킵", flush=True)
            else:
                print(f"    → 규칙2 충족 ▶ 시장가 매도 sellable={sellable_qty} (보유:{qty}, 미체결:{open_sell_qty})", flush=True)
                result = send_order(code, 0, sellable_qty, order_type="매도", ord_dvsn="01")
                msg = (result.get("msg1") or "").strip()
                log_trade(datetime.now(), code, cur, sellable_qty, "매도(규칙2)", result)
                print(f"      결과: {result.get('rt_cd')} {msg}", flush=True)
                if is_market_closed_msg(msg):
                    print("⛔ 시장 종료 감지 → 루프 종료 예정", flush=True)
                    market_closed = True
                    break
            streaks[code]["mid"] = 0

        # 규칙 3 (0% ~ 1% 구간 3루프 연속)
        if 0.0 < pnl_pct <= 1:
            streaks.setdefault(code, {"low":0, "mid":0})
            streaks[code]["low"] += 1
        else:
            if code in streaks: streaks[code]["low"] = 0

        if streaks.get(code, {}).get("low", 0) >= 3:
            if sellable_qty <= 0:
                print(f"    → 규칙3 충족이나 매도 가능 수량 0 (보유:{qty}, 매도미체결:{open_sell_qty}) ▶ 스킵", flush=True)
            else:
                print(f"    → 규칙3 충족 ▶ 시장가 매도 sellable={sellable_qty} (보유:{qty}, 미체결:{open_sell_qty})", flush=True)
                result = send_order(code, 0, sellable_qty, order_type="매도", ord_dvsn="01")
                msg = (result.get("msg1") or "").strip()
                log_trade(datetime.now(), code, cur, sellable_qty, "매도(규칙3)", result)
                print(f"      결과: {result.get('rt_cd')} {msg}", flush=True)
                if is_market_closed_msg(msg):
                    print("⛔ 시장 종료 감지 → 루프 종료 예정", flush=True)
                    market_closed = True
                    break
            streaks[code]["low"] = 0

    return market_closed

# ───────────── 평가금액 요약(API) + CSV 저장 ─────────────
def get_account_summary():
    """
    잔고조회 API output2 요약 반환 (첫 페이지만으로 충분)
    - tot_evlu_amt : 총 평가금액 (현금 등 포함)
    - scts_evlu_amt: 유가증권 평가금액(= '평가금액' 의미, 주식 등)
    """
    url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
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
    j = res.json()
    out2 = (j.get("output2") or [{}])
    return out2[0] if out2 else {}

def append_portfolio_csv(loop_count, now_dt, summary: dict):
    """
    루프 단위로 총 평가금액만 CSV 저장
    - 같은 loop가 이미 존재하면 해당 행을 업데이트
    - 없으면 새 행을 추가
    컬럼: loop(int), 총평가금액(float)
    """
    row = {
        "loop": int(loop_count),
        "총평가금액": _num0(summary.get("tot_evlu_amt")),
    }
    p = Path(PORTFOLIO_CSV)
    if p.exists():
        df = pd.read_csv(p)
        if "loop" not in df.columns:
            df["loop"] = pd.Series(dtype="int")
        # 중복 loop 업데이트
        with pd.option_context('mode.chained_assignment', None):
            df["loop"] = pd.to_numeric(df["loop"], errors="coerce").astype("Int64")
            mask = df["loop"] == row["loop"]
            if mask.any():
                df.loc[mask, "총평가금액"] = row["총평가금액"]
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        # 정렬 및 정수 캐스팅
        df["loop"] = df["loop"].astype("Int64")
        df = df.sort_values("loop").reset_index(drop=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(p, index=False, encoding="utf-8-sig")

# ───────────── 로깅 (손익률 미저장, 종목코드 6자리 고정) ─────────────
def log_trade(timestamp, stock_code, price, qty, order_type, order_result, extra=None):
    """
    trade_log_2.csv는 루트에 저장.
    종목코드는 항상 문자열 6자리(앞자리 0 보존)로 저장.
    """
    log_file = Path("trade_log_2.csv")
    code_str = str(stock_code).zfill(6)

    log_entry = {
        "거래시간": timestamp,
        "종목코드": code_str,
        "현재가": price,
        "주문수량": qty,
        "주문종류": order_type,
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

# ───────────── 메인 루프 ─────────────
if __name__ == "__main__":
    buy_list_path = os.path.join(OUTPUT_DIR, "buy_list.csv")
    if not os.path.exists(buy_list_path):
        print("❌ buy_list.csv 없음", flush=True)
        sys.exit()

    top_candidates_df = pd.read_csv(buy_list_path, dtype={'종목코드': str})
    today_candidates = [row['종목코드'].zfill(6) for _, row in top_candidates_df.iterrows()]
    print(f"📋 오늘 후보 {len(today_candidates)}개, 상위 {TOP_N_TO_BUY} 매수 대상", flush=True)

    # ▶ 재시작 시 이어서: 시작 루프 번호를 CSV에서 결정
    loop_count = load_loop_counter_from_csv()
    print(f"▶ 시작 루프 번호: {loop_count}", flush=True)

    # ⬇️ 추가: 시작 루프를 고정 보관(종료 조건 가드에 사용)
    start_loop = loop_count

    portfolio_values = []  # 총 평가금액 로그(내부)
    streaks = defaultdict(lambda: {"low": 0, "mid": 0})

    # 날짜 및 당일 상태
    last_date = None
    bought_today = set()
    not_tradable_today = set()

    try:
        while True:
            now = datetime.now()
            today_str = now.strftime("%Y%m%d")

            # 날짜 바뀌면 당일 기록 로드
            if last_date != today_str:
                bought_today = load_bought_today(today_str)
                not_tradable_today = load_not_tradable(today_str)
                last_date = today_str

            print(f"\n────────── [LOOP {loop_count}] {now.strftime('%Y-%m-%d %H:%M:%S')} ──────────", flush=True)

            # 0) 계좌 보유 현황 최신화
            holdings = get_all_holdings()
            print(f"📦 실계좌 보유 {len(holdings)}종목 동기화", flush=True)

            # 0-1) 당일 주문/체결 조회 → 오늘 이미 '매수 주문' 종목을 누적
            try:
                today_orders = get_today_orders()
                api_bought = build_bought_today_set(today_orders)
                bought_today |= api_bought
            except Exception as e:
                print(f"⚠️ 당일 주문조회 실패: {e} (기존 bought_today 유지)", flush=True)
                today_orders = []  # 실패 시 안전 기본값

            # 0-2) **보유 중인 종목도 당일 재매수 금지에 포함**
            held_today = {c for c, v in holdings.items() if v.get("qty", 0) > 0}
            bought_today |= held_today

            # 0-3) 상태 저장 + 계좌 스냅샷(루트에 저장)
            save_bought_today(today_str, bought_today)
            save_not_tradable(today_str, not_tradable_today)
            save_account_snapshot_to_json(holdings)

            # 1) 매수: 보유/당일매수/매매불가 이력 제외하고 04로 매수
            market_closed = buy_candidates(holdings, today_candidates, loop_count, bought_today, not_tradable_today)
            if market_closed:
                print("✅ 시장 종료 감지 → 루프 즉시 종료", flush=True)
                break

            # 2) 매도 규칙: 전 종목 대상 01  (※ 매도 미체결 반영)
            holdings = get_all_holdings()  # 방금 매수 반영
            market_closed = sell_rules_for_all(holdings, streaks, loop_count, today_orders)
            if market_closed:
                print("✅ 시장 종료 감지 → 루프 즉시 종료", flush=True)
                break

            # 3) 평가금액 집계 (API 요약 사용) + CSV 저장
            try:
                summary = get_account_summary()
                total_eval_amount = _num0(summary.get("tot_evlu_amt"))   # 총 평가금액(현금 포함)
                eval_amount = _num0(summary.get("scts_evlu_amt"))        # 평가금액(주식 등 유가)
            except Exception as e:
                print(f"⚠️ 잔고 요약 조회 실패: {e}", flush=True)
                total_eval_amount = 0.0
                eval_amount = 0.0

            append_portfolio_csv(loop_count, now, summary)   # CSV에는 tot_evlu_amt만 저장
            portfolio_values.append(total_eval_amount)
            print(f"💰 (API) 총 평가금액: {total_eval_amount:,.0f} / 평가금액(주식): {eval_amount:,.0f}", flush=True)

            # 3-1) 종료 조건:
            #   A) 미체결 없음 AND 평가금액(주식) == 0
            #   B) 현재 시각 >= 15:30 (시장 종료 시간)  → 무조건 종료
            # ⬇️ 수정: 시작 직후 첫 루프는 종료 조건을 건너뛰고, 두 번째 루프부터 평가
            should_check_exit = (loop_count >= start_loop + 1)

            if should_check_exit:
                try:
                    no_open_orders = not has_open_orders(today_orders)
                except Exception as e:
                    no_open_orders = False
                    print(f"⚠️ 미체결 체크 실패: {e}", flush=True)

                reached_close_time = now.time() >= MARKET_CLOSE_TIME

                if (no_open_orders and eval_amount == 0) or reached_close_time:
                    if reached_close_time:
                        print("⏰ 15:30 도달 → 루프 종료", flush=True)
                    else:
                        print("🛑 미체결 주문 없음 + 평가금액(주식) 0원 → 루프 종료", flush=True)
                    break
            else:
                print("🛡️ 시작 보호 모드: 첫 루프는 종료 조건을 건너뜁니다.", flush=True)

            loop_count += 1
            time.sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        print("\n⏹ 사용자 중단", flush=True)

    finally:
        if portfolio_values:
            print(f"🧾 루프별 총 평가금액이 '{PORTFOLIO_CSV}'에 저장되었습니다.", flush=True)
        else:
            print("저장할 총 평가금액 데이터가 없습니다.", flush=True)
