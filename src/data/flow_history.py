"""
수급 이력 — 네이버 모바일 /api/stock/{code}/trend. KRX 자격증명 없이 2015년까지 확인(2026-09-23).

[규약 — 실측]
  pageSize 최대 60. bizdate=YYYYMMDD를 주면 그 날짜 '이전' 60거래일을 최신순으로 준다.
  상장 전 날짜는 빈 배열. 값은 "+375,676" 같은 문자열이고 **미조정 주식 수**다 —
  신호는 같은 출처 거래량으로 나눠 분할에 불변인 비율로 쓴다(signals/flow.py).
  호출당 약 0.6초. 종목별 캐시로 이어받는다.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NAVER_TREND = "https://m.stock.naver.com/api/stock/{code}/trend"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://m.stock.naver.com/"}
PAGE_SIZE = 60
PAUSE = 0.1
COLUMNS = ["foreign_net", "organ_net", "flow_volume"]


def _to_int(s) -> int:
    s = str(s if s is not None else "0").replace(",", "").replace("+", "").strip()
    return int(float(s)) if s not in ("", "-") else 0


def fetch_page(code: str, bizdate: str) -> list[dict]:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(NAVER_TREND.format(code=code), params={"pageSize": PAGE_SIZE, "bizdate": bizdate},
                             headers=HEADERS, timeout=15)
            if r.status_code == 200 and r.text.startswith("["):
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"수급 조회 실패 {code} {bizdate}: {last}")


def fetch_ticker_flow(code: str, start: str, end: str,
                      fetch: Optional[Callable[[str, str], list[dict]]] = None) -> pd.DataFrame:
    fetch = fetch or fetch_page
    stop = pd.Timestamp(start)
    cursor = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y%m%d")   # end 당일 포함
    rows: list[dict] = []
    while True:
        page = fetch(code, cursor)
        if not page:
            break
        for x in page:
            rows.append({"date": pd.Timestamp(x["bizdate"]),
                         "foreign_net": _to_int(x.get("foreignerPureBuyQuant")),
                         "organ_net": _to_int(x.get("organPureBuyQuant")),
                         "flow_volume": _to_int(x.get("accumulatedTradingVolume"))})
        oldest = page[-1]["bizdate"]
        if pd.Timestamp(oldest) <= stop or len(page) < PAGE_SIZE:
            break
        cursor = oldest
        if PAUSE:
            time.sleep(PAUSE)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    return df.loc[stop:pd.Timestamp(end), COLUMNS]


def load_flow_history(codes: list[str], start: str, end: str, cache_dir: Path = Path("data/backtest/flow"),
                      workers: int = 4, fetch: Optional[Callable[[str, str], list[dict]]] = None) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    def one(code: str):
        f = cache_dir / f"flow_{code}_{start}_{end}.pkl"
        if f.exists():
            # 이 함수가 아래에서 직접 쓴 로컬 캐시만 읽는다(외부 입력 아님) — backtest/data.py와 같은 방식
            return code, pd.read_pickle(f)
        try:
            df = fetch_ticker_flow(code, start, end, fetch=fetch)
        except Exception as e:
            logger.warning(f"[flow_history] {code} 실패: {e}")
            failed.append(code)
            return code, None
        df.to_pickle(f)
        return code, df

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        got = dict(ex.map(one, codes))
    ok = {c: d for c, d in got.items() if d is not None}
    fn = pd.DataFrame({c: d["foreign_net"] for c, d in ok.items()}).sort_index()
    fv = pd.DataFrame({c: d["flow_volume"] for c, d in ok.items()}).sort_index()
    return {"foreign_net": fn, "flow_volume": fv, "failed": sorted(failed)}
