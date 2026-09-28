"""
tests/test_atr_stop.py

단계 3 — 손절폭을 종목 변동성(ATR)에 맞춘다 (doc/2026-09-21_agent-audit.md §2-5).

[왜]
  손절이 진입가 −4% 고정이었다. 분석 종목의 ATR14 중앙값은 7.3%다.
  즉 손절폭이 하루 평균 변동폭의 절반이다. 실측(ohlcv_cache 1,305 구간):

    고정 -4%        → 10거래일 내 손절 터치 84.9%
    2×ATR 상한 15%  → 46.4%

  000500이 그 사례다. D+1에 −7.09%로 손절됐고 D+5에는 +28.7%였다.

[같이 바꾸는 것 — 포지션]
  손절폭이 4% → 15%로 넓어지면 같은 포지션에서 손실이 3.75배가 된다.
  그래서 "한 거래 최대 손실 = 시드의 RISK_PER_TRADE%"로 포지션을 되돌린다.
  손절이 넓은 종목은 적게 산다.

유료 API 호출 없음.
"""

from __future__ import annotations

import pytest


# ── ATR 계산 ────────────────────────────────────────────────────────────

def _bars(*rows):
    """(high, low, close) 튜플 목록 → ohlcv_cache 행 모양."""
    return [{"date": f"2026-09-{i+1:02d}", "open": c, "high": h, "low": lo,
             "close": c, "volume": 1000}
            for i, (h, lo, c) in enumerate(rows)]


def test_atr_matches_hand_computed_wilder_average():
    from src.utils.atr import atr

    # 15봉: 첫 봉은 이전 종가가 없어 TR에 안 들어간다 → TR 14개 → 단순평균이 곧 ATR14
    bars = _bars(*[(110, 100, 105)] * 15)
    # 매 봉 high-low = 10, |high-prev_close| = 5, |low-prev_close| = 5 → TR = 10
    assert atr(bars, period=14) == pytest.approx(10.0)


def test_atr_needs_enough_bars():
    from src.utils.atr import atr

    assert atr(_bars(*[(110, 100, 105)] * 10), period=14) is None
    assert atr([], period=14) is None


def test_atr_uses_gap_in_true_range():
    """전일 종가 대비 갭이 당일 고저폭보다 크면 그 갭이 TR이다."""
    from src.utils.atr import atr

    flat = _bars(*[(101, 99, 100)] * 16)
    gapped = list(flat)
    gapped[-1] = {**gapped[-1], "high": 130, "low": 128, "close": 129}

    assert atr(gapped, period=14) > atr(flat, period=14)


# ── 손절폭 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("atr_pct, expected", [
    (7.3,  -14.6),   # 중앙값 종목 → 2×ATR
    (1.0,   -3.0),   # 저변동 → 하한 3%
    (10.0, -15.0),   # 고변동 → 상한 15%
    (0.1,   -3.0),
])
def test_stop_pct_from_atr_is_clamped(atr_pct, expected):
    from src.utils.atr import stop_pct_from_atr

    assert stop_pct_from_atr(atr_pct) == pytest.approx(expected)


def test_stop_pct_without_atr_falls_back_to_fixed():
    """ohlcv_cache에 봉이 모자란 종목은 종전 기본값을 그대로 쓴다."""
    from src.utils.atr import stop_pct_from_atr
    from src.agents.chief_python import DEFAULT_STOP_PCT

    assert stop_pct_from_atr(None) == DEFAULT_STOP_PCT


def test_stop_pct_for_ticker_reads_cache(monkeypatch):
    from src.utils import atr as atr_mod

    monkeypatch.setattr(atr_mod, "get_ohlcv_series",
                        lambda t, days=60: _bars(*[(110, 100, 105)] * 20))
    # ATR 10 / 종가 105 = 9.52% → 2배 19.0% → 상한 15%
    assert atr_mod.stop_pct_for("005930") == pytest.approx(-15.0)


def test_stop_pct_for_ticker_survives_db_failure(monkeypatch):
    from src.utils import atr as atr_mod
    from src.agents.chief_python import DEFAULT_STOP_PCT

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(atr_mod, "get_ohlcv_series", _boom)
    assert atr_mod.stop_pct_for("005930") == DEFAULT_STOP_PCT


# ── 거래 파라미터 ───────────────────────────────────────────────────────

def test_trade_params_accept_wide_atr_stop():
    """종전 clamp는 -3~-6%였다. ATR 손절이 그 안으로 잘리면 의미가 없다."""
    from src.agents.chief_python import compute_trade_params

    p = compute_trade_params(10_000, -12.0)

    assert p["stop_loss_pct"] == pytest.approx(-12.0)
    assert p["stop_loss"] == pytest.approx(8_800)
    assert p["take_profit_1"] == pytest.approx(10_000 + 1_200 * 2)
    assert p["rr_ratio"] == pytest.approx(2.0), "R:R 1:2는 손절폭이 바뀌어도 유지된다"


def test_trade_params_still_clamp_absurd_values():
    from src.agents.chief_python import compute_trade_params, MAX_STOP_PCT, MIN_STOP_PCT

    assert compute_trade_params(1000, -99.0)["stop_loss_pct"] == pytest.approx(MAX_STOP_PCT)
    assert compute_trade_params(1000, -0.1)["stop_loss_pct"] == pytest.approx(MIN_STOP_PCT)


# ── 포지션 리스크 스케일링 ──────────────────────────────────────────────

def test_position_shrinks_as_stop_widens():
    from src.agents.chief_python import risk_scaled_position, RISK_PER_TRADE_PCT

    wide   = risk_scaled_position(base_pct=5.0, stop_pct=-15.0)
    narrow = risk_scaled_position(base_pct=5.0, stop_pct=-4.0)

    assert wide < narrow, "손절이 넓으면 적게 산다"
    # 한 거래 최대 손실이 시드의 RISK_PER_TRADE_PCT를 넘지 않는다
    assert wide * 15.0 / 100 <= RISK_PER_TRADE_PCT + 1e-9


def test_position_never_exceeds_calibrated_base():
    """보정 표본이 쌓여 base가 작아지면 그쪽을 따른다 — 리스크 한도는 상한일 뿐이다."""
    from src.agents.chief_python import risk_scaled_position

    assert risk_scaled_position(base_pct=3.0, stop_pct=-4.0) == pytest.approx(3.0)


def test_position_has_floor():
    from src.agents.chief_python import risk_scaled_position, MIN_POSITION_PCT

    assert risk_scaled_position(base_pct=5.0, stop_pct=-99.0) == pytest.approx(MIN_POSITION_PCT)


# ── 회귀: 실측 기준 ─────────────────────────────────────────────────────

def test_median_ticker_gets_wider_stop_than_before():
    """분석 종목 중앙값(ATR 7.3%)이 종전 -4%보다 확실히 넓어진다."""
    from src.utils.atr import stop_pct_from_atr

    assert stop_pct_from_atr(7.3) < -10.0


# ── chief 통합: 손절이 실제로 ATR로 확정되는가 ──────────────────────────

def _report(rec="BUY", conf=0.7):
    from src.schemas.agent_output import AnalysisReport
    return AnalysisReport(
        agent_name="chief_strategist", recommendation=rec, confidence=conf,
        reasoning=["a", "b", "c"], data_sources=["x", "y"],
        prediction_basis=["p1", "p2"], risk_factors=["r"],
        stop_loss_pct=-4.0, stop_loss=9_600.0, position_size_pct=8.0,   # LLM이 제안한 값
    )


def test_apply_atr_stop_overrides_llm_values(monkeypatch, tmp_path):
    """LLM이 -4%를 제안해도 ATR이 넓으면 그 값으로 덮어쓴다."""
    from src.agents import chief_python as cp
    from src.utils import atr as atr_mod
    from src.evaluation import calibration as cal

    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")     # 보정 표본 없음 → base 5%
    monkeypatch.setattr(atr_mod, "stop_pct_for", lambda t: -12.0)

    r = _report()
    cp.apply_atr_stop(r, "005930", 10_000.0)

    assert r.stop_loss_pct == pytest.approx(-12.0), "LLM 제안 -4%가 아니라 ATR 값"
    assert r.stop_loss == pytest.approx(8_800.0)
    assert r.take_profit_1 == pytest.approx(12_400.0)
    # 손절 12% → 리스크 한도 0.5/12×100 = 4.2% < base 5%
    assert r.position_size_pct == pytest.approx(4.2)
    assert any("손절" in x and "ATR" in x for x in r.reasoning)


def test_apply_atr_stop_skips_non_buy(monkeypatch):
    from src.agents import chief_python as cp

    for rec in ("HOLD", "SELL"):
        r = _report(rec=rec)
        before = r.stop_loss_pct
        cp.apply_atr_stop(r, "005930", 10_000.0)
        assert r.stop_loss_pct == before, "BUY가 아니면 건드리지 않는다"


def test_apply_atr_stop_survives_atr_failure(monkeypatch, tmp_path):
    """ATR을 못 구해도 손절은 비어서는 안 된다 — 비면 position_tracker가 기본값으로 대체한다."""
    from src.agents import chief_python as cp
    from src.utils import atr as atr_mod
    from src.evaluation import calibration as cal

    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")

    def _boom(t):
        raise RuntimeError("cache down")

    monkeypatch.setattr(atr_mod, "stop_pct_for", _boom)

    r = _report()
    cp.apply_atr_stop(r, "005930", 10_000.0)

    assert r.stop_loss_pct == pytest.approx(cp.DEFAULT_STOP_PCT)
    assert r.stop_loss is not None and r.position_size_pct is not None


def test_stop_pct_for_reads_real_cache_shape(monkeypatch):
    """ohlcv_cache가 실제로 반환하는 dict 모양으로 끝까지 도는지."""
    from src.utils import atr as atr_mod

    bars = [{"date": f"2026-09-{i+1:02d}", "open": 100.0, "high": 104.0,
             "low": 98.0, "close": 100.0, "volume": 10}
            for i in range(20)]
    monkeypatch.setattr(atr_mod, "get_ohlcv_series", lambda t, days=60: bars)

    # TR = max(6, |104-100|, |98-100|) = 6 → ATR 6 / 종가 100 = 6% → 2배 -12%
    assert atr_mod.stop_pct_for("005930") == pytest.approx(-12.0)
