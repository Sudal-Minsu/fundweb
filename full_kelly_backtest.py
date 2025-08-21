import os
import sys
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

"""
폴백 종목 6개만으로, buy_list 없이도 바로 도는 고정비율(시가->종가) 백테스트 v2
- ✅ KIS '기간별 시세' 엔드포인트 사용: inquire-daily-itemchartprice (TR: FHKST03010100)
- ✅ 시작일/종료일을 정확히 반영 (이전 '최근 30개' 제한 API 문제 해결)
- 기간이 길어 1회 100건 제한에 걸릴 수 있어, 내부적으로 날짜를 여러 구간으로 나눠 요청/병합
- 매일 자본의 일정 비율(INVEST_FRACTION_PER_DAY)을 폴백 종목에 균등 배분해 시가→종가 데이 트레이드
- 결과 CSV/PNG 저장
필요:
- config.py 안에 get_api_keys()가 있어야 함 (app_key, app_secret 리턴)
"""

# ================= 설정 =================
START_DATE = "20250601"   # YYYYMMDD
END_DATE   = "20250709"   # YYYYMMDD
OUTPUT_DIR = "rule_2_결과"
CACHE_DIR  = Path(OUTPUT_DIR) / "price_cache_api_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 10_000_000
INVEST_FRACTION_PER_DAY = 0.10   # 매일 자본의 10%

# 폴백 6종목 (원하면 교체)
CODES = ["005930", "000660", "035420", "051910", "068270", "105560"]

# 수수료/세금(검증 단계에서는 0으로 두고 먼저 형태 확인 추천)
FEE_RATE = 0.00015
TAX_RATE_SELL = 0.0025

# 한 번의 API 호출로 반환 가능한 최대 건 대비 여유를 둔 윈도우 크기(일수)
# (공식 제한 ~100건 기준, 영업일이 아닌 달력일로 90일 정도 권장)
WINDOW_DAYS = 90

# ================= KIS 인증 =================
from config import get_api_keys
app_key, app_secret = get_api_keys()
url_base = "https://openapivts.koreainvestment.com:29443"

SESSION = requests.Session()
SESSION.headers.update({"content-type": "application/json"})

def get_token():
    url = f"{url_base}/oauth2/tokenP"
    data = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
    res = SESSION.post(url, data=json.dumps(data), timeout=(3,10))
    tok = res.json().get("access_token", "")
    if not tok:
        print("❌ 토큰 발급 실패:", res.json()); sys.exit(1)
    print(f"액세스 토큰: {tok[:20]}...", flush=True)
    return tok

ACCESS_TOKEN = get_token()

# ================= 날짜 유틸 =================
def ymd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")

def chunk_date_ranges(start: str, end: str, window_days: int) -> list[tuple[str,str]]:
    s = parse_ymd(start)
    e = parse_ymd(end)
    ranges = []
    cur = s
    while cur <= e:
        nxt = cur + timedelta(days=window_days-1)
        if nxt > e:
            nxt = e
        ranges.append((ymd(cur), ymd(nxt)))
        cur = nxt + timedelta(days=1)
    return ranges

# ================= 데이터 로딩/캐시 =================
def api_fetch_itemchartprice(code: str, start: str, end: str) -> pd.DataFrame:
    """
    기간별 일봉 시세 API (정확한 날짜 범위 반영)
    - TR: FHKST03010100
    - 응답은 일반적으로 output2에 다건 시계열이 담김
    """
    url = f"{url_base}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST03010100",
        "Content-Type": "application/json",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",     # 주식/ETF/ETN
        "FID_INPUT_ISCD": str(code).zfill(6),
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
        "FID_PERIOD_DIV_CODE": "D",        # D=일, W=주, M=월, Y=년
        "FID_ORG_ADJ_PRC": "0",            # 0: 수정주가, 1: 원주가
    }
    res = SESSION.get(url, headers=headers, params=params, timeout=(3, 10))
    if res.status_code != 200:
        return pd.DataFrame(columns=["date","open","high","low","close"]).astype({"date": str})
    j = res.json()
    # 계정/버전에 따라 output1/output2 배치가 다를 수 있어 안전하게 처리
    out = j.get("output2") or j.get("output1") or []
    if not out:
        return pd.DataFrame(columns=["date","open","high","low","close"]).astype({"date": str})

    raw = pd.DataFrame(out)
    # 컬럼 매핑(필요시 print(raw.columns)로 확인)
    colmap = {
        "stck_bsop_date": "date",
        "stck_oprc": "open",
        "stck_hgpr": "high",
        "stck_lwpr": "low",
        "stck_clpr": "close",
    }
    use_cols = [c for c in colmap if c in raw.columns]
    if not use_cols:
        return pd.DataFrame(columns=["date","open","high","low","close"]).astype({"date": str})

    df = raw[use_cols].rename(columns=colmap)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(subset=["open","close"])
          .sort_values("date")
          .reset_index(drop=True)
    )
    # API가 과거->최근/최근->과거 정렬로 줄 수 있어도 위에서 sort로 정렬 보정
    # 반환
    return df

def fetch_daily_prices_range_cached(code: str, start: str, end: str) -> pd.DataFrame:
    """
    - 길이가 긴 구간은 WINDOW_DAYS로 쪼개 여러 번 호출해 병합
    - 병합 결과를 캐시에 저장
    """
    code = str(code).zfill(6)
    cache_path = CACHE_DIR / f"{code}_{start}_{end}.csv"
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path, dtype={"date": str})
        except Exception:
            pass

    ranges = chunk_date_ranges(start, end, WINDOW_DAYS)
    dfs = []
    for s, e in ranges:
        df = api_fetch_itemchartprice(code, s, e)
        if not df.empty:
            # 범위 밖 데이터가 섞여 올 가능성에 대비한 필터
            df = df[(df["date"] >= s) & (df["date"] <= e)]
            dfs.append(df)

    if not dfs:
        final_df = pd.DataFrame(columns=["date","open","high","low","close"]).astype({"date": str})
    else:
        final_df = (
            pd.concat(dfs, ignore_index=True)
              .drop_duplicates(subset=["date"])
              .sort_values("date")
              .reset_index(drop=True)
        )

    try:
        final_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    return final_df

# 날짜 합집합 캘린더(각 종목 일봉에서 날짜 모음)
def build_dates(codes, start, end):
    dates = set()
    for code in codes:
        px = fetch_daily_prices_range_cached(code, start, end)
        if not px.empty:
            dates.update(px["date"].tolist())
    return sorted([d for d in dates if (d >= start and d <= end)])

# ================= 백테스트(시가->종가 데이 트레이드) =================
def run_backtest():
    dates = build_dates(CODES, START_DATE, END_DATE)
    if not dates:
        print("❌ 시세 데이터가 없습니다. 폴백 종목/기간을 확인하세요.")
        sys.exit(0)

    capital = INITIAL_CASH
    track_dates = []
    capitals = []

    # 종목별 데이터 미리 적재(메모리)
    prices = {code: fetch_daily_prices_range_cached(code, START_DATE, END_DATE) for code in CODES}

    for d in dates:
        # 해당 날짜에 시세가 존재하는 폴백 종목만 필터
        tradables = []
        for code in CODES:
            df = prices[code]
            if df.empty:
                continue
            row = df[df["date"] == d]
            if row.empty:
                continue
            o = float(row.iloc[0]["open"])
            c = float(row.iloc[0]["close"])
            tradables.append((code, o, c))

        # 어떤 종목도 시세가 없으면 자본 그대로 유지
        if not tradables:
            track_dates.append(d)
            capitals.append(capital)
            continue

        invest_total = capital * INVEST_FRACTION_PER_DAY
        per_stock = invest_total / len(tradables)

        day_pnl = 0.0
        for code, o, c in tradables:
            if o <= 0:
                continue
            qty = per_stock / o
            # 수수료/세금 반영: 시가 매수 비용(+수수료), 종가 매도 수익(-수수료, -세금)
            buy_cost = o * qty * (1 + FEE_RATE)
            sell_rev = c * qty * (1 - FEE_RATE)
            sell_rev *= (1 - TAX_RATE_SELL)
            day_pnl += (sell_rev - buy_cost)
        capital += day_pnl

        track_dates.append(d)
        capitals.append(capital)

    return track_dates, capitals

# ================= 실행 & 저장 =================
if __name__ == "__main__":
    dates, capitals = run_backtest()

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    out_csv = outdir / f"equity_curve_fallback6_{START_DATE}_{END_DATE}.csv"
    pd.DataFrame({"date": dates, "capital": capitals}).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 그래프 저장
    plt.figure(figsize=(12,6))
    plt.plot(pd.to_datetime(dates), capitals)
    plt.title(f"Equity Curve — Fallback 6 (Open->Close) {START_DATE}~{END_DATE}")
    plt.xlabel("Date"); plt.ylabel("Capital (KRW)")
    plt.grid(True); plt.tight_layout()
    out_png = outdir / f"equity_curve_fallback6_{START_DATE}_{END_DATE}.png"
    plt.savefig(out_png, dpi=180)

    print(f"✅ 저장 완료: {out_csv}")
    print(f"✅ 저장 완료: {out_png}")
