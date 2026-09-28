"""
src/universe/universe_builder.py

투자 유니버스 자동 구축 모듈

구조:
  Step 1: KOSPI 시총 상위 100개 → 2단계 필터 → KOSPI Pool
  Step 2: KOSDAQ 시총 상위 20개 → 2단계 필터 → KOSDAQ Pool
  Step 3: KOSPI Pool + KOSDAQ Pool → ticker 중복 제거 → 최종 유니버스

[핵심 설계]
  pykrx get_market_cap_by_ticker는 종목명 컬럼을 반환하지 않는다.
  → _add_ticker_names()로 종목명 컬럼을 직접 추가한 뒤 필터를 실행한다.

실행:
  python -m src.universe.universe_builder
  from src.universe.universe_builder import load_universe
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock as pykrx_stock

from src.universe.filters import apply_all_filters


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────

KOSPI_TOP_N  = 100
KOSDAQ_TOP_N = 20
KOSDAQ_MAX_N = 40

CONFIG_PATH = Path("src/config/universe_config.json")


def _get_latest_trading_date() -> str:
    for delta in range(5):
        d = datetime.now() - timedelta(days=delta)
        if d.weekday() < 5:
            return d.strftime("%Y%m%d")
    return datetime.now().strftime("%Y%m%d")


# ──────────────────────────────────────────
# 종목명 컬럼 추가 헬퍼
# ──────────────────────────────────────────

def _add_ticker_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    pykrx get_market_cap_by_ticker 결과에 '종목명' 컬럼을 추가합니다.

    get_market_ticker_name()을 각 ticker에 호출합니다.
    실패한 ticker는 ticker 코드 자체를 이름으로 사용합니다.
    """
    if "종목명" in df.columns:
        return df  # 이미 있으면 건너뜀

    names = {}
    for ticker in df.index:
        try:
            names[ticker] = pykrx_stock.get_market_ticker_name(ticker)
        except Exception:
            names[ticker] = ticker  # 실패 시 코드 자체 사용

    df = df.copy()
    df["종목명"] = df.index.map(names)
    return df


# ──────────────────────────────────────────
# Step 1: KOSPI Pool
# ──────────────────────────────────────────

def build_kospi_pool(date: str) -> pd.DataFrame:
    print(f"\n[Step 1] KOSPI 시총 상위 {KOSPI_TOP_N}개 조회 ({date})")

    df = pykrx_stock.get_market_cap_by_ticker(date, market="KOSPI")
    if "시가총액" in df.columns:
        df = df.sort_values("시가총액", ascending=False).head(KOSPI_TOP_N)
    else:
        df = df.head(KOSPI_TOP_N)

    df = _add_ticker_names(df)   # ← 종목명 추가
    print(f"  조회 완료: {len(df)}개")
    return apply_all_filters(df, market="KOSPI")


# ──────────────────────────────────────────
# Step 2: KOSDAQ Pool
# ──────────────────────────────────────────

def build_kosdaq_pool(date: str) -> pd.DataFrame:
    print(f"\n[Step 2] KOSDAQ 시총 상위 {KOSDAQ_TOP_N}개 조회 ({date})")

    df = pykrx_stock.get_market_cap_by_ticker(date, market="KOSDAQ")
    if "시가총액" in df.columns:
        df = df.sort_values("시가총액", ascending=False).head(KOSDAQ_TOP_N)
    else:
        df = df.head(KOSDAQ_TOP_N)

    df = _add_ticker_names(df)   # ← 종목명 추가
    print(f"  조회 완료: {len(df)}개")
    df = apply_all_filters(df, market="KOSDAQ")

    if len(df) < 5:
        print(f"  ⚠️ 잔여 {len(df)}개 < 5개 → 상위 {KOSDAQ_MAX_N}개로 확장")
        df_ext = pykrx_stock.get_market_cap_by_ticker(date, market="KOSDAQ")
        if "시가총액" in df_ext.columns:
            df_ext = df_ext.sort_values("시가총액", ascending=False).head(KOSDAQ_MAX_N)
        else:
            df_ext = df_ext.head(KOSDAQ_MAX_N)
        df_ext = _add_ticker_names(df_ext)
        df = apply_all_filters(df_ext, market="KOSDAQ")

    return df


# ──────────────────────────────────────────
# Step 3: Concat + 중복 제거
# ──────────────────────────────────────────

def merge_pools(kospi_df: pd.DataFrame, kosdaq_df: pd.DataFrame) -> list[str]:
    kospi_tickers  = list(kospi_df.index)
    kosdaq_tickers = list(kosdaq_df.index)

    print(f"\n[Step 3] 중복 제거")
    print(f"  KOSPI Pool:  {len(kospi_tickers)}개")
    print(f"  KOSDAQ Pool: {len(kosdaq_tickers)}개")

    seen:  set[str]  = set()
    final: list[str] = []
    for ticker in (kospi_tickers + kosdaq_tickers):
        if ticker not in seen:
            seen.add(ticker)
            final.append(ticker)

    print(f"  → 최종 유니버스: {len(final)}개")
    return final


# ──────────────────────────────────────────
# 저장 / 로드
# ──────────────────────────────────────────

def save_universe(tickers: list[str], date: str) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version":        "1.0",
        "built_at":       datetime.now().isoformat(),
        "reference_date": date,
        "ticker_count":   len(tickers),
        "tickers":        tickers,
    }
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✅ 저장 완료: {CONFIG_PATH} ({len(tickers)}개)")


_FALLBACK_TICKERS = ["005930", "000660", "005380", "035420", "051910"]

# 분기 갱신 대상 월 (2026-09-14 결정 3-4b)
QUARTER_MONTHS = (1, 4, 7, 10)


def load_universe() -> list[str]:
    """
    저장된 유니버스를 로드합니다.

    [2026-09-14 결정 3-4b] 진실의 원천은 DB(universe_snapshots, source='build')다.
      1순위 DB 최신 build 판 → 2순위 universe_config.json → 3순위 폴백 5개

    [왜 DB로 옮겼나]
      universe_config.json은 git 추적 파일이라 CI에서 갱신하면 커밋·푸시 권한이 필요하고
      충돌도 난다. DB는 Actions 캐시로 실행 간에 이어지므로 갱신 결과가 자연스럽게 남는다.
      JSON은 읽기 전용 폴백으로 남겨 둔다 — DB가 비어 있는 새 환경에서도 돌아야 한다.

    이 함수 하나만 바꾸면 되는 이유: 에이전트 5개·screener·daily_runner·security 화이트리스트가
    전부 이 함수를 거친다 (2026-09-14 전수 확인).
    """
    try:
        from src.evaluation.random_control import latest_built_universe
        _, tickers = latest_built_universe()
        if tickers:
            return tickers
    except Exception as e:            # DB 없음·스키마 없음 등 — JSON으로 떨어진다
        print(f"[universe] DB 조회 실패, JSON 폴백: {type(e).__name__}: {e}")

    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return config.get("tickers", _FALLBACK_TICKERS)
    return _FALLBACK_TICKERS


def universe_refresh_due(today: "datetime | None" = None) -> bool:
    """오늘 분기 갱신을 돌려야 하는가.

    조건: 이번 달이 분기 첫 달(1·4·7·10)이고, 마지막 build가 이번 분기 이전이다.
    월 안에서 날짜는 따지지 않는다 — cron 지연·휴장으로 1일에 못 돌 수 있으므로
    그 달 안에 한 번만 돌면 된다.
    """
    now = today or datetime.now()
    if now.month not in QUARTER_MONTHS:
        return False
    try:
        from src.evaluation.random_control import latest_built_universe
        last_date, tickers = latest_built_universe()
    except Exception:
        return True                    # DB를 못 읽으면 한 번 만들어 본다
    if not tickers or not last_date:
        return True
    return last_date[:7] < now.strftime("%Y-%m")


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────

def build_universe(date: str | None = None) -> list[str]:
    if date is None:
        date = _get_latest_trading_date()

    print("=" * 60)
    print(f"유니버스 빌더 — 기준일: {date}")
    print("=" * 60)

    kospi_df  = build_kospi_pool(date)
    kosdaq_df = build_kosdaq_pool(date)
    final     = merge_pools(kospi_df, kosdaq_df)

    save_universe(final, date)
    # 진실의 원천은 DB다 (결정 3-4b). JSON은 폴백으로만 남긴다.
    try:
        from datetime import date as _date
        from src.evaluation.random_control import record_universe_snapshot
        n = record_universe_snapshot(_date.today().isoformat(), final, source="build")
        print(f"✅ DB 저장 완료: universe_snapshots source='build' {n}행")
    except Exception as e:
        print(f"⚠️ DB 저장 실패 (JSON은 저장됨): {type(e).__name__}: {e}")
    return final


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    universe = build_universe()
    print(f"\n최종 유니버스 ({len(universe)}개):")
    for i, t in enumerate(universe, 1):
        name = pykrx_stock.get_market_ticker_name(t)
        print(f"  {i:3d}. {t}  {name}")