"""
tests/test_stage1a_backfill.py

backfill 선정 조건 회귀 테스트.

[배경]
  _backfill_missing()이 get_ticker_count()만 보고 backfill 대상을 골랐다.
  KRX 로그인이 막힌 2026-07-24 이후, 종목당 캐시가 ~61행이라
  MIN_BACKFILL_ROWS(50) 조건에 걸리지 않아 backfill이 한 번도 돌지 않았다.
  _refresh_today()도 KRX 실패로 0건이라, 캐시가 46일간 7/24에 얼어붙은 채
  정량 신호가 옛 가격으로 계산됐다.

  → 행 수뿐 아니라 "최신 날짜가 얼마나 뒤처졌는지"도 봐야 한다.
"""

from datetime import date, datetime, timedelta

import pytest

from src.screening import stage1a_quant as s1a


@pytest.fixture
def fake_cache(monkeypatch):
    """get_ticker_count / get_latest_cached_date 를 메모리 사전으로 대체."""
    state: dict[str, tuple[int, str | None]] = {}

    monkeypatch.setattr(s1a, "get_ticker_count", lambda t: state.get(t, (0, None))[0])
    monkeypatch.setattr(s1a, "get_latest_cached_date",
                        lambda t=None: state.get(t, (0, None))[1])

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(s1a, "_backfill_one",
                        lambda t, days=s1a.BACKFILL_DAYS: (calls.append((t, days)), 1)[1])

    return state, calls


def _iso(days_ago: int) -> str:
    return (date.today() - timedelta(days=days_ago)).isoformat()


# ─────────────────────────────────────────────────────────
# 1. 기존 조건 — 행 수 부족
# ─────────────────────────────────────────────────────────

def test_low_row_count_triggers_backfill(fake_cache):
    state, calls = fake_cache
    state["005930"] = (10, _iso(1))            # 행 부족
    s1a._backfill_missing(["005930"])
    assert [t for t, _ in calls] == ["005930"]


def test_healthy_cache_is_skipped(fake_cache):
    state, calls = fake_cache
    state["005930"] = (61, _iso(1))            # 행 충분 + 최신
    assert s1a._backfill_missing(["005930"]) == {}
    assert calls == []


# ─────────────────────────────────────────────────────────
# 2. 신규 조건 — 날짜 정체 (46일 사건 재현)
# ─────────────────────────────────────────────────────────

def test_stale_cache_triggers_backfill_even_with_enough_rows(fake_cache):
    """
    핵심 회귀 — 행은 61개로 충분하지만 날짜가 46일 멈춘 상태.
    이전 구현은 이걸 건너뛰어 46일간 방치했다.
    """
    state, calls = fake_cache
    state["005930"] = (61, _iso(46))

    s1a._backfill_missing(["005930"])

    assert [t for t, _ in calls] == ["005930"], "날짜 정체를 못 잡았다"


def test_backfill_window_covers_the_gap(fake_cache):
    """구멍이 BACKFILL_DAYS보다 크면 그만큼 더 거슬러 올라가야 한다."""
    state, calls = fake_cache
    state["005930"] = (61, _iso(200))          # 200일 구멍 > BACKFILL_DAYS(90)

    s1a._backfill_missing(["005930"])

    ticker, days = calls[0]
    assert days >= 200, f"구멍 200일인데 {days}일치만 요청했다"


def test_small_gap_is_tolerated(fake_cache):
    """주말·공휴일 정도의 지연은 backfill 대상이 아니다."""
    state, calls = fake_cache
    state["005930"] = (61, _iso(s1a.MAX_CACHE_STALE_DAYS))
    assert s1a._backfill_missing(["005930"]) == {}
    assert calls == []


def test_gap_just_over_threshold_triggers(fake_cache):
    state, calls = fake_cache
    state["005930"] = (61, _iso(s1a.MAX_CACHE_STALE_DAYS + 1))
    s1a._backfill_missing(["005930"])
    assert len(calls) == 1


# ─────────────────────────────────────────────────────────
# 3. 혼합 / 경계
# ─────────────────────────────────────────────────────────

def test_mixed_universe_selects_only_needy(fake_cache):
    state, calls = fake_cache
    state["005930"] = (61, _iso(1))            # 정상 → 제외
    state["000660"] = (61, _iso(46))           # 정체 → 포함
    state["005380"] = (10, _iso(1))            # 행부족 → 포함
    state["051910"] = (61, _iso(2))            # 정상 → 제외

    s1a._backfill_missing(["005930", "000660", "005380", "051910"])

    assert sorted(t for t, _ in calls) == ["000660", "005380"]


def test_no_date_info_does_not_crash(fake_cache):
    """캐시에 날짜가 없으면(행 수는 충분) 정체 판정을 하지 않는다."""
    state, calls = fake_cache
    state["005930"] = (61, None)
    assert s1a._backfill_missing(["005930"]) == {}


def test_malformed_date_is_ignored(fake_cache):
    state, calls = fake_cache
    state["005930"] = (61, "not-a-date")
    assert s1a._backfill_missing(["005930"]) == {}
