"""
src/screening/stage1a_quant.py

Stage 1-A: pykrx 정량 스크린 (캐시 기반).

[설계 — 변경 사유]
  이전: get_market_ohlcv(start,end,ticker)를 110개 종목에 순차 호출
        → 누적 지연/타임아웃
  현재: data/mentions.db의 ohlcv_cache 테이블을 1차 소스로 사용
        → 매일 2회 호출(KOSPI + KOSDAQ)만으로 오늘 데이터 추가
        → 지표 계산은 캐시에서만 진행 (네트워크 0)

[흐름]
  1. ohlcv_cache 초기화
  2. 캐시가 비었거나 데이터 < MIN_BACKFILL_ROWS인 종목 → 초기 backfill
     (get_market_ohlcv 종목별 호출; 첫 실행 1회만 발생)
  3. 오늘(혹은 최근 거래일) 데이터가 캐시에 없으면 → 2회 호출로 갱신
     get_market_ohlcv_by_ticker(today, "KOSPI"/"KOSDAQ")
  4. 각 종목 시계열을 캐시에서 읽어 기존 지표 계산

[지표]
  price_change_5d  : 최근 5거래일 가격 변화율 (소수)
  volume_signal    : 최근 3일 평균 거래량 / 20일 평균 > 1.5
  ma_breakout      : 어제 종가가 20/60일선을 위로 돌파 (전전일 아래 → 어제 위)
  golden_cross     : 5일선이 20일선을 아래에서 위로 교차
  signal_count     : 위 4개 중 True 개수
                     (price_change_5d는 절댓값 PRICE_CHANGE_THRESHOLD 이상이면 1)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock as pykrx_stock

from src.data.ohlcv_cache import (
    init_ohlcv_cache,
    upsert_ohlcv_rows,
    get_ohlcv_series,
    cache_has_date,
    get_ticker_count,
    get_latest_cached_date,
    find_suspect_tickers,
)

logger = logging.getLogger(__name__)

# ── 지표 임계값 ─────────────────────────────────────────────
VOLUME_RATIO_THRESHOLD = 1.5
PRICE_CHANGE_THRESHOLD = 0.03   # 절댓값 3% 이상이면 신호로 카운트

# ── 캐시 정책 ───────────────────────────────────────────────
BACKFILL_DAYS      = 90   # 캘린더 일수 (≈ 60거래일 → MA60 가능)
MIN_BACKFILL_ROWS  = 50   # 종목별 최소 캐시 행 수. MA60(60일) 계산엔 60행이 이상적이지만,
                          # 휴장·신규상장 등으로 실제 거래일 수는 더 적을 수 있어 50으로 완화.
                          # 50 미만이면 backfill 대상 (정상 종목은 첫 실행 후 재진입 없음).
BACKFILL_WORKERS   = 1    # pykrx KRX 로그인이 스레드 안전하지 않아 병렬 시 데드락.
                          # 첫 1회만 발생하는 비용이므로 순차 실행이 안전.
MAX_CACHE_STALE_DAYS = 5  # 종목 캐시 최신 날짜가 이 일수를 넘게 뒤처지면 backfill 대상.
                          # 행 수만 보면(MIN_BACKFILL_ROWS) "행은 많은데 날짜가 멈춘"
                          # 캐시를 못 잡는다 — 2026-07-24~09-08 46일 정체가 그 사례.

# pykrx 컬럼 (한글 키)
_COL_OPEN, _COL_HIGH, _COL_LOW, _COL_CLOSE, _COL_VOL = (
    "시가", "고가", "저가", "종가", "거래량",
)

# 개별 pykrx 호출 외부 timeout (sync 함수라 asyncio.wait_for 불가 →
# ThreadPoolExecutor.submit().result(timeout=...) 사용).
_PYKRX_CALL_TIMEOUT = 30


def _call_with_timeout(fn, *args, timeout: float = _PYKRX_CALL_TIMEOUT, **kwargs):
    """
    동기 pykrx 호출을 데몬 스레드에서 실행하고 timeout 초과 시 TimeoutError 발생.

    [왜 ThreadPoolExecutor를 쓰지 않는가 — 두 가지 이유]

    ① `with ThreadPoolExecutor(...) as ex:` 는 timeout을 무효화한다.
       future.result(timeout=N)이 TimeoutError를 던지면 그 예외가 with를
       빠져나가고, 그 순간 __exit__이 shutdown(wait=True)를 호출해
       **매달린 worker가 끝날 때까지 블로킹**된다.
       실측: timeout=2.0s 선언 → 실제 소요 10.00s (worker의 sleep(10)을 끝까지 대기).

    ② with 없이 모듈 수준 executor를 써도 프로세스 종료가 막힌다.
       concurrent.futures가 atexit으로 worker를 join하기 때문에,
       매달린 스레드 하나가 인터프리터 종료를 영구히 붙잡는다.
       (CI 실행이 timeout-minutes까지 끌려간 원인이 될 수 있다.)

    → daemon=True 스레드는 인터프리터 종료 시 join되지 않는다.
      호출자는 timeout에 즉시 풀려나고, 매달린 스레드가 있어도 프로세스는 끝난다.
      OS 레벨 강제 종료는 파이썬에서 불가능하므로 스레드 자체는 남는다.
    """
    box: dict = {}

    def _runner() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:          # noqa: BLE001 — 호출자에게 그대로 전달
            box["error"] = e

    t = threading.Thread(target=_runner, daemon=True,
                         name=f"pykrx-{getattr(fn, '__name__', 'call')}")
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"pykrx call timed out after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ─────────────────────────────────────────────────────────
# 거래일 헬퍼
# ─────────────────────────────────────────────────────────

def _latest_trading_date_str() -> str:
    """가장 최근 거래일을 YYYYMMDD로 반환 — 한국 공휴일 인지.

    우선순위:
      1) 캐시 MAX(date) ≥ 오늘이면 그 날짜 (오늘분 이미 수집됨 → 추가 호출 0)
      2) pykrx.get_nearest_business_day_in_a_week(prev=True) — 한국 공휴일 캘린더 기반
      3) 단순 평일 룩백 (오프라인/예외 시 최종 폴백)
    """
    # (1) 캐시 우선
    latest_iso = get_latest_cached_date()
    if latest_iso:
        try:
            latest = datetime.strptime(latest_iso, "%Y-%m-%d").date()
            if latest >= datetime.now().date():
                return latest.strftime("%Y%m%d")
        except ValueError:
            pass

    # (2) pykrx 한국 영업일 헬퍼 — 짧은 timeout으로 호출 (네트워크 발생 가능)
    try:
        return _call_with_timeout(
            pykrx_stock.get_nearest_business_day_in_a_week, timeout=10
        )
    except Exception as e:
        logger.debug(f"[stage1a] get_nearest_business_day_in_a_week 실패: {e}")

    # (3) 평일 룩백 폴백
    today = datetime.now().date()
    for delta in range(7):
        d = today - timedelta(days=delta)
        if d.weekday() < 5:
            return d.strftime("%Y%m%d")
    return today.strftime("%Y%m%d")


def _yyyymmdd_to_iso(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


# ─────────────────────────────────────────────────────────
# (Step A) 초기 backfill — 종목별 get_market_ohlcv
# ─────────────────────────────────────────────────────────

def _backfill_one(ticker: str, days: int = BACKFILL_DAYS) -> int:
    """단일 종목의 최근 `days`일치 OHLCV를 캐시에 적재. 행 수 반환.

    days: 기본 BACKFILL_DAYS. 캐시 구멍이 그보다 크면 호출자가 늘려 넘긴다.
    """
    end_date   = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = _call_with_timeout(
            pykrx_stock.get_market_ohlcv, start_date, end_date, ticker
        )
    except Exception as e:
        logger.debug(f"[stage1a/backfill] {ticker}: {e}")
        return 0

    if df is None or df.empty:
        return 0

    rows: list[dict] = []
    for idx, r in df.iterrows():
        try:
            d_iso = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            rows.append({
                "ticker": ticker,
                "date":   d_iso,
                "open":   float(r[_COL_OPEN]),
                "high":   float(r[_COL_HIGH]),
                "low":    float(r[_COL_LOW]),
                "close":  float(r[_COL_CLOSE]),
                "volume": int(r[_COL_VOL]),
            })
        except Exception:
            continue
    return upsert_ohlcv_rows(rows)


def _stale_days(ticker: str, today) -> int | None:
    """종목 캐시의 최신 날짜가 며칠 뒤처졌는지. 데이터 없으면 None."""
    latest = get_latest_cached_date(ticker)
    if not latest:
        return None
    try:
        return (today - datetime.strptime(latest, "%Y-%m-%d").date()).days
    except ValueError:
        return None


def _backfill_missing(tickers: list[str]) -> dict[str, int]:
    """
    backfill이 필요한 종목을 골라 채운다.

    [선정 조건 — 2개]
      ① 행 수 < MIN_BACKFILL_ROWS      (첫 실행 / 신규 상장)
      ② 최신 날짜가 MAX_CACHE_STALE_DAYS 초과로 뒤처짐  ← 신규

    ②가 없으면 "행 수는 충분한데 날짜가 멈춘" 캐시가 영원히 방치된다.
    실제로 KRX 로그인이 막힌 2026-07-24 이후 46일간 그 상태였다:
    종목당 ~61행이라 ①(<50)에 걸리지 않아 backfill이 한 번도 돌지 않았고,
    _refresh_today()도 KRX 실패로 0건이라 캐시가 7/24에 얼어붙었다.
    그동안 정량 신호는 46일 묵은 가격으로 계산됐다.
    """
    today = datetime.now().date()
    need: list[str] = []
    gaps: dict[str, int] = {}
    n_rows = n_stale = 0

    for t in tickers:
        if get_ticker_count(t) < MIN_BACKFILL_ROWS:
            need.append(t)
            n_rows += 1
            continue
        gap = _stale_days(t, today)
        if gap is not None and gap > MAX_CACHE_STALE_DAYS:
            need.append(t)
            gaps[t] = gap
            n_stale += 1

    # 분할 미조정 의심 종목 — 행 수·날짜가 멀쩡해도 옛 행이 틀려 있다 (스펙 §1-2)
    try:
        wanted = set(tickers)
        suspects = [t for t in find_suspect_tickers() if t in wanted and t not in need]
    except Exception as e:
        logger.warning(f"[stage1a] 정합성 검사 실패(계속): {e}")
        suspects = []
    if suspects:
        need.extend(suspects)
        print(f"  [stage1a] 정합성 의심 {len(suspects)}종목 재백필: {suspects[:5]}")

    if not need:
        return {}

    logger.info(
        f"[stage1a] backfill 대상 {len(need)}개 "
        f"(행부족 {n_rows}, 날짜정체 {n_stale})"
    )
    print(
        f"  [stage1a] backfill {len(need)}개 종목 "
        f"(행부족 {n_rows} / 날짜정체 {n_stale}, workers={BACKFILL_WORKERS})"
    )
    if gaps:
        worst = max(gaps.values())
        print(f"    ⚠️ 캐시 정체 최대 {worst}일 — 그동안 정량 신호가 옛 가격으로 계산됐다")

    inserted: dict[str, int] = {}
    if BACKFILL_WORKERS <= 1:
        # 순차 — pykrx 스레드 안전 회피
        for i, tk in enumerate(need, 1):
            try:
                # 구멍이 BACKFILL_DAYS보다 크면 그만큼 더 거슬러 올라가 채운다.
                days = max(BACKFILL_DAYS, gaps.get(tk, 0) + 10)
                inserted[tk] = _backfill_one(tk, days=days)
            except Exception as e:
                logger.debug(f"[stage1a/backfill] {tk} 실패: {e}")
                inserted[tk] = 0
            if i % 20 == 0 or i == len(need):
                print(f"    backfill 진행: {i}/{len(need)}")
    else:
        with ThreadPoolExecutor(max_workers=BACKFILL_WORKERS) as ex:
            futures = {ex.submit(_backfill_one, t): t for t in need}
            for fut in as_completed(futures):
                tk = futures[fut]
                try:
                    inserted[tk] = fut.result()
                except Exception as e:
                    logger.debug(f"[stage1a/backfill] {tk} 실패: {e}")
                    inserted[tk] = 0
    return inserted


# ─────────────────────────────────────────────────────────
# (Step B) 오늘 갱신 — KOSPI/KOSDAQ 각 1회 bulk 호출
# ─────────────────────────────────────────────────────────

def _bulk_snapshot(date_yyyymmdd: str, market: str) -> pd.DataFrame | None:
    """get_market_ohlcv_by_ticker — 단일 거래일 전체 시장 스냅샷."""
    try:
        df = _call_with_timeout(
            pykrx_stock.get_market_ohlcv_by_ticker, date_yyyymmdd, market=market
        )
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"[stage1a/snapshot] {market} {date_yyyymmdd}: {e}")
        return None


def _refresh_today(tickers: list[str], today_yyyymmdd: str) -> int:
    """
    오늘 데이터가 캐시에 없으면 KOSPI + KOSDAQ bulk 스냅샷으로 한 번에 보충.
    캐시에 이미 오늘 데이터가 있으면 호출 자체를 스킵.

    Returns: 캐시에 upsert된 행 수
    """
    iso_today = _yyyymmdd_to_iso(today_yyyymmdd)
    if cache_has_date(iso_today):
        return 0

    universe = set(tickers)
    rows: list[dict] = []

    for market in ("KOSPI", "KOSDAQ"):
        df = _bulk_snapshot(today_yyyymmdd, market)
        if df is None:
            continue
        # ticker가 index에 들어있음. 유니버스 종목만 필터.
        subset = df.loc[df.index.intersection(universe)]
        for ticker, r in subset.iterrows():
            try:
                close = float(r[_COL_CLOSE])
                # 휴장일이면 가격이 0인 채로 들어옴 — 캐시에 넣지 않음
                if close <= 0:
                    continue
                rows.append({
                    "ticker": ticker,
                    "date":   iso_today,
                    "open":   float(r[_COL_OPEN]),
                    "high":   float(r[_COL_HIGH]),
                    "low":    float(r[_COL_LOW]),
                    "close":  close,
                    "volume": int(r[_COL_VOL]),
                })
            except Exception:
                continue

    n = upsert_ohlcv_rows(rows)
    if n:
        print(f"  [stage1a] 오늘 스냅샷 {n}건 캐시 갱신 ({iso_today})")
    else:
        # 여기가 조용히 흐르면 캐시가 멈춘 채 옛 데이터로 신호가 계산된다.
        # 2026-07-24 이후 KRX 로그인 실패로 46일간 이 상태였고, 로그에
        # warning 한 줄만 남아 아무도 알아채지 못했다. error로 올리고
        # stdout에도 찍어 CI 로그 상단에서 눈에 띄게 한다.
        latest = get_latest_cached_date()
        msg = (
            f"[stage1a] 오늘({iso_today}) 스냅샷 0건 — OHLCV 캐시가 갱신되지 않았다. "
            f"캐시 최신 날짜={latest}. KRX 로그인 실패(비밀번호 변경 요구 등)가 "
            f"가장 흔한 원인이며, 이 상태에서는 정량 신호가 옛 데이터로 계산된다."
        )
        logger.error(msg)
        print(f"  ⛔ {msg}")
    return n


# ─────────────────────────────────────────────────────────
# (Step C) 지표 계산 — 캐시 시계열만 사용
# ─────────────────────────────────────────────────────────

def _ma(values: list[float], window: int, offset_from_end: int = 0) -> float | None:
    if len(values) < window + offset_from_end:
        return None
    end = len(values) - offset_from_end
    return sum(values[end - window : end]) / window


def _compute_signals(series: list[dict]) -> dict | None:
    """캐시 시계열에서 4개 신호 + signal_count 계산."""
    closes  = [r["close"]  for r in series]
    volumes = [r["volume"] for r in series]

    if len(closes) < 21:  # MA20 + 5일선 비교 최소
        return None

    # price_change_5d
    price_5d_ago = closes[-6] if len(closes) >= 6 else closes[0]
    price_now    = closes[-1]
    price_change_5d = round((price_now - price_5d_ago) / price_5d_ago, 4) \
        if price_5d_ago else 0.0

    # volume_signal
    avg_vol_3d  = sum(volumes[-3:])  / 3  if len(volumes) >= 3  else 0.0
    avg_vol_20d = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0.0
    volume_signal = (
        avg_vol_20d > 0 and (avg_vol_3d / avg_vol_20d) > VOLUME_RATIO_THRESHOLD
    )

    # ma_breakout (20일선 또는 60일선)
    ma_breakout = False
    for window in (20, 60):
        ma_y  = _ma(closes, window, 0)
        ma_d  = _ma(closes, window, 1)
        if ma_y is None or ma_d is None:
            continue
        close_y = closes[-1]
        close_d = closes[-2] if len(closes) >= 2 else closes[-1]
        if close_d < ma_d and close_y > ma_y:
            ma_breakout = True
            break

    # golden_cross (5일선이 20일선 위로)
    ma5_y, ma5_d   = _ma(closes, 5,  0), _ma(closes, 5,  1)
    ma20_y, ma20_d = _ma(closes, 20, 0), _ma(closes, 20, 1)
    golden_cross = bool(
        ma5_y and ma5_d and ma20_y and ma20_d
        and ma5_d < ma20_d and ma5_y > ma20_y
    )

    signal_count = sum([
        abs(price_change_5d) >= PRICE_CHANGE_THRESHOLD,
        bool(volume_signal),
        bool(ma_breakout),
        bool(golden_cross),
    ])

    return {
        "price_change_5d": price_change_5d,
        "volume_signal":   bool(volume_signal),
        "ma_breakout":     bool(ma_breakout),
        "golden_cross":    bool(golden_cross),
        "signal_count":    signal_count,
    }


# ─────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────

def run_quant_screen(tickers: list[str]) -> dict[str, dict]:
    """
    Stage 1-A 진입점.

    동작:
      1. 캐시 초기화
      2. 캐시 부족 종목 → backfill (첫 실행 시 110회, 이후 0회)
      3. 오늘 데이터 없으면 → 2회 bulk 호출로 보충
      4. 캐시에서 종목별 시계열을 읽어 지표 계산
    """
    if not tickers:
        return {}

    init_ohlcv_cache()

    # Step A: 부족 종목 backfill
    try:
        _backfill_missing(tickers)
    except Exception as e:
        logger.warning(f"[stage1a] backfill 실패: {e}")

    # Step B: 오늘 데이터 갱신
    today = _latest_trading_date_str()
    try:
        _refresh_today(tickers, today)
    except Exception as e:
        logger.warning(f"[stage1a] 오늘 갱신 실패: {e}")

    # Step C: 캐시에서 지표 계산
    results: dict[str, dict] = {}
    skipped = 0
    for ticker in tickers:
        try:
            series = get_ohlcv_series(ticker, days=65)
            if len(series) < 21:
                skipped += 1
                continue
            metrics = _compute_signals(series)
            if metrics is not None:
                results[ticker] = metrics
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            logger.debug(f"[stage1a] {ticker} 지표 계산 실패: {e}")

    if skipped:
        logger.info(f"[stage1a] {skipped}/{len(tickers)}개 종목 데이터 부족")

    return results


# ─────────────────────────────────────────────────────────
# 단독 실행 (단위 테스트)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()

    test_tickers = ["005930", "000660", "005380", "035420", "051910"]
    print(f"[stage1a] 테스트 종목: {test_tickers}")
    print(f"[stage1a] 캐시 최신 날짜(전): {get_latest_cached_date()}")

    res = run_quant_screen(test_tickers)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[stage1a] 캐시 최신 날짜(후): {get_latest_cached_date()}")
    signaled = [t for t, m in res.items() if m["signal_count"] >= 1]
    print(f"신호 발생: {len(signaled)}/{len(test_tickers)}개")
