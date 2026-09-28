"""
tests/test_daily_runner_health.py

_assess_run_health() 회귀 테스트 — "성공 위장" 탐지.

[배경]
  2026-09-07 실행에서 OpenAI 401로 전문가 에이전트 7개가 전부 실패했다.
  그런데 base_agent._make_fallback()이 confidence=0.0 리포트를 만들고,
  Quality Gate 소프트 폴백이 그것을 통과시키고, 파이프라인은 끝까지 돌아
  GitHub Actions는 이 실행을 success로 기록했다.
  같은 날 리포트가 발행됐고 내용은 전부 "관망 0%"였으나 아무도 알아채지 못했다.

  동시에 OHLCV 캐시는 2026-07-24에서 멈춰 있었다(KRX 로그인 실패).
  46일간 정량 신호가 옛 데이터로 계산됐지만 logger.warning 한 줄만 남았다.

  _assess_run_health()는 이 두 상황을 healthy=False로 판정해야 하고,
  __main__이 그 결과로 exit(1) 하여 실행이 빨간불이 되게 한다.
"""

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest


def _stub_if_unimportable(name: str, **attrs) -> None:
    """
    실제로 import되지 않는 모듈만 최소 스텁으로 대체한다.

    daily_runner는 모듈 로드 시 screener → universe_builder → pykrx 체인을 타는데,
    pykrx는 import 시점에 KRX 로그인을 시도한다 (2026-08-19 / 08-30 실행이
    이 때문에 JSONDecodeError로 죽었다). 또 pipeline은 로드만 해도
    setup_feedback_system()으로 DB를 만든다.
    여기서 검증하려는 것은 순수 함수 _assess_run_health() 하나뿐이므로
    그 전체 스택을 요구하지 않는다.

    find_spec은 파일 존재만 확인하므로(의존성 누락을 못 잡는다) 실제 import를
    시도한다. 의존성이 갖춰진 환경에서는 import가 성공해 스텁이 설치되지 않고
    실제 모듈이 그대로 쓰인다.
    """
    try:
        importlib.import_module(name)
        return
    except Exception:
        sys.modules.pop(name, None)

    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_stub_if_unimportable("dotenv", load_dotenv=lambda *a, **k: None)
_stub_if_unimportable("src.schemas.agent_output", AnalysisReport=object)
_stub_if_unimportable("src.screening.screener", run_screening=lambda *a, **k: {})
_stub_if_unimportable("src.graph.pipeline", run_pipeline=None)

from src import daily_runner as dr  # noqa: E402


def _report(agent_name: str, confidence: float):
    """AnalysisReport 대용 — _assess_run_health는 두 속성만 읽는다."""
    return SimpleNamespace(agent_name=agent_name, confidence=confidence)


AGENTS = [
    "macro_economist", "kr_market_specialist", "us_market_specialist",
    "quant_analyst", "technical_analyst", "sentiment_analyst",
    "fundamental_analyst",
]


def _payload(confidences: list[float]):
    reports = [_report(n, c) for n, c in zip(AGENTS, confidences)]
    reports.append(_report("chief_strategist", 0.7))   # chief는 집계 대상 아님
    return {"ticker": "005930", "all_reports": reports}


@pytest.fixture(autouse=True)
def fresh_ohlcv(monkeypatch):
    """기본값: OHLCV 신선도 정상(0일). 개별 테스트에서 덮어쓴다."""
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: 0)


# ─────────────────────────────────────────────────────────
# 1. 전 에이전트 실패 탐지 — 2026-09-07 재현
# ─────────────────────────────────────────────────────────

def test_all_agents_dead_is_unhealthy():
    """confidence가 전부 0.0이면 폴백 리포트뿐이므로 실패로 판정해야 한다."""
    payloads = [_payload([0.0] * 7)]
    h = dr._assess_run_health(payloads, selected=["005930"])

    assert h["healthy"] is False
    assert h["live_agent_reports"] == 0
    assert h["total_agent_reports"] == 7
    assert any("전부 실패" in r for r in h["reasons"])


def test_chief_strategist_is_excluded_from_the_count():
    """chief는 폴백이어도 confidence를 채우므로 집계에서 빼야 오탐이 없다."""
    payloads = [_payload([0.0] * 7)]
    h = dr._assess_run_health(payloads, selected=["005930"])
    assert h["total_agent_reports"] == 7      # chief 제외 (8이 아님)


def test_partial_failure_is_still_healthy():
    """일부만 죽는 건 정상 범위 — 오탐으로 매일 빨간불이 되면 안 된다."""
    payloads = [_payload([0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8])]
    h = dr._assess_run_health(payloads, selected=["005930"])

    assert h["healthy"] is True
    assert h["live_agent_reports"] == 2


def test_normal_run_is_healthy():
    payloads = [_payload([0.6, 0.7, 0.8, 0.7, 0.85, 0.8, 0.7])]
    h = dr._assess_run_health(payloads, selected=["005930"])
    assert h["healthy"] is True
    assert h["live_agent_reports"] == 7


def test_multi_ticker_all_dead_is_unhealthy():
    """여러 종목을 돌려도 전부 죽었으면 실패."""
    payloads = [_payload([0.0] * 7) for _ in range(3)]
    h = dr._assess_run_health(payloads, selected=["005930", "000660", "005380"])
    assert h["healthy"] is False
    assert h["total_agent_reports"] == 21


# ─────────────────────────────────────────────────────────
# 2. OHLCV 정지 탐지 — 2026-07-24 재현
# ─────────────────────────────────────────────────────────

def test_stale_ohlcv_is_unhealthy(monkeypatch):
    """캐시가 46일 뒤처진 상태(실제 발생)를 잡아야 한다."""
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: 46)
    payloads = [_payload([0.7] * 7)]          # 에이전트는 멀쩡해도
    h = dr._assess_run_health(payloads, selected=["005930"])

    assert h["healthy"] is False
    assert h["ohlcv_stale_days"] == 46
    assert any("OHLCV" in r for r in h["reasons"])


def test_ohlcv_within_threshold_is_healthy(monkeypatch):
    """주말·공휴일로 며칠 뒤처지는 건 정상."""
    monkeypatch.setattr(dr, "_ohlcv_staleness_days",
                        lambda: dr.MAX_OHLCV_STALE_DAYS)
    h = dr._assess_run_health([_payload([0.7] * 7)], selected=["005930"])
    assert h["healthy"] is True


def test_unknown_ohlcv_staleness_does_not_trip(monkeypatch):
    """캐시 조회 자체가 실패(None)하면 그것만으로 실패 판정하지 않는다."""
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: None)
    h = dr._assess_run_health([_payload([0.7] * 7)], selected=["005930"])
    assert h["healthy"] is True
    assert h["ohlcv_stale_days"] is None


# ─────────────────────────────────────────────────────────
# 3. 경계 조건
# ─────────────────────────────────────────────────────────

def test_selected_but_no_results_is_unhealthy():
    """종목을 골랐는데 분석 결과가 0건이면 실패."""
    h = dr._assess_run_health([], selected=["005930"])
    assert h["healthy"] is False
    assert any("분석 결과가 0건" in r for r in h["reasons"])


def test_no_pick_day_is_healthy():
    """선정 종목 자체가 없는 날은 정상 — 빨간불이 되면 안 된다."""
    h = dr._assess_run_health([], selected=[])
    assert h["healthy"] is True
    assert h["reasons"] == []


def test_both_failures_reported_together():
    """두 장애가 동시에 나면 이유도 둘 다 담겨야 한다."""
    import types as _t
    dr_stale = dr._ohlcv_staleness_days
    try:
        dr._ohlcv_staleness_days = lambda: 46
        h = dr._assess_run_health([_payload([0.0] * 7)], selected=["005930"])
    finally:
        dr._ohlcv_staleness_days = dr_stale

    assert h["healthy"] is False
    assert len(h["reasons"]) == 2


# ─────────────────────────────────────────────────────────
# 4. 장중 실행 감지 (cron 지연 대응)
# ─────────────────────────────────────────────────────────

def _at(monkeypatch, y, m, d, hh, mm):
    """daily_runner가 보는 '지금'을 고정한다."""
    import datetime as _dt
    fixed = _dt.datetime(y, m, d, hh, mm, tzinfo=dr.KST)

    class _DT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed
        @classmethod
        def strptime(cls, *a, **k):
            return _dt.datetime.strptime(*a, **k)

    monkeypatch.setattr(dr, "datetime", _DT)


def test_intraday_run_is_flagged(monkeypatch):
    """2026-08-28 사례 재현 — 14:21 KST 발행."""
    _at(monkeypatch, 2026, 8, 28, 14, 21)          # 금요일 장중
    h = dr._assess_run_health([_payload([0.7] * 7)], selected=["005930"])

    assert h["healthy"] is True, "장중 실행은 실패가 아니라 경고다"
    assert any("장중 실행" in w for w in h["warnings"])
    assert any("확정 종가가 아니라" in w for w in h["warnings"])


def test_premarket_run_has_no_warning(monkeypatch):
    _at(monkeypatch, 2026, 9, 9, 6, 17)            # 수요일 06:17 — 장 전
    h = dr._assess_run_health([_payload([0.7] * 7)], selected=["005930"])
    assert h["warnings"] == []


def test_after_close_run_has_no_warning(monkeypatch):
    _at(monkeypatch, 2026, 9, 9, 16, 0)            # 15:30 이후
    h = dr._assess_run_health([_payload([0.7] * 7)], selected=["005930"])
    assert h["warnings"] == []


def test_weekend_run_has_no_warning(monkeypatch):
    _at(monkeypatch, 2026, 9, 12, 14, 0)           # 토요일 — 장 없음
    h = dr._assess_run_health([_payload([0.7] * 7)], selected=["005930"])
    assert h["warnings"] == []


def test_market_open_boundary(monkeypatch):
    _at(monkeypatch, 2026, 9, 9, 9, 0)             # 09:00 정각 — 개장
    assert dr._market_session_warning() is not None
    _at(monkeypatch, 2026, 9, 9, 8, 59)            # 08:59 — 장 전
    assert dr._market_session_warning() is None
