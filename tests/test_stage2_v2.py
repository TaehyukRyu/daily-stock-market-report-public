"""
tests/test_stage2_v2.py — 스크리닝 v2 Stage 2. 순수 함수, 외부 호출 없음.
"""
from __future__ import annotations

import pytest

from src.screening import stage2_scorer as s2


def _q(ret5=0.0):        return {"signal_count": 0, "price_change_5d": ret5}
def _e(burst=None, dart=0, sent=None, titles=None):
    return {"news_burst": burst, "news_recent": 0, "dart_event": dart, "dart_titles": titles or [], "sent_delta": sent}
def _n(m=False):         return {"news_match": m, "reason": ""}


def _run(tickers, bursts=None, darts=(), ret5=None, news=(), regime=None, missing_quant=()):
    bursts = bursts or {}
    ret5 = ret5 or {}
    quant = {t: _q(ret5.get(t, 0.0)) for t in tickers if t not in missing_quant}
    event = {t: _e(bursts.get(t), 1 if t in darts else 0) for t in tickers}
    newsr = {t: _n(t in news) for t in tickers}
    return s2.run_scoring(tickers, quant, event, newsr, regime=regime)


def test_no_legacy_symbols():
    for name in ("MIN_GUARANTEED_OPTIONAL", "SCORE_CONFIRMED_MIN", "confirmed_min_for"):
        assert not hasattr(s2, name)


def test_percentile_ranks_basic():
    assert s2.percentile_ranks({"A": 1.0, "B": 2.0, "C": 3.0}) == {"A": 0.0, "B": 0.5, "C": 1.0}
    assert s2.percentile_ranks({"A": 5.0}) == {"A": 1.0}
    assert s2.percentile_ranks({}) == {}


def test_percentile_ties_share_rank():
    assert s2.percentile_ranks({"A": 1.0, "B": 1.0, "C": 2.0}) == {"A": 0.0, "B": 0.0, "C": 1.0}


def test_top20_burst_becomes_candidate():
    tickers = list("ABCDE")
    out = _run(tickers, bursts={t: float(i) for i, t in enumerate(tickers, 1)})
    assert out["confirmed"] == ["E"]
    assert out["scores"]["E"]["news_burst_pct"] == 1.0 and out["scores"]["E"]["is_candidate"] == 1
    assert out["scores"]["D"]["news_burst_pct"] == 0.75 and out["scores"]["D"]["is_candidate"] == 0
    assert out["optional"] == []


def test_equal_bursts_yield_no_burst_candidates():
    tickers = list("ABCDE")
    out = _run(tickers, bursts={t: 1.0 for t in tickers})
    assert out["confirmed"] == []
    assert all(d["news_burst_pct"] == 0.0 for d in out["scores"].values())


def test_dart_event_alone_makes_candidate():
    out = _run(["A", "B"], bursts={"A": 1.0, "B": 1.0}, darts=("B",))
    assert out["confirmed"] == ["B"] and out["scores"]["B"]["score"] == pytest.approx(0.5)   # (pct 0 + dart 1) / 2


def test_already_moved_filter_blocks_candidate():
    out = _run(["A"], darts=("A",), ret5={"A": 0.12})
    assert out["confirmed"] == []
    assert out["scores"]["A"]["filtered_reason"] == "already_moved" and out["scores"]["A"]["is_candidate"] == 0


def test_missing_ohlcv_filter():
    out = _run(["A"], darts=("A",), missing_quant=("A",))
    assert out["scores"]["A"]["filtered_reason"] == "insufficient_ohlcv" and out["confirmed"] == []


def test_ranking_by_score_then_burst():
    tickers = list("ABCDEFGHIJ")
    bursts = {t: float(i) for i, t in enumerate(tickers, 1)}     # J 최고
    out = _run(tickers, bursts=bursts, darts=("B", "I"))          # B: pct .11 + dart, I: pct .89 + dart
    # 후보 = I(공시), B(공시), J(pct 1.0). score: I (0.889+1)/2=.944, B (0.111+1)/2=.556, J (1+0)/2=.5
    assert out["confirmed"] == ["I", "B", "J"]
    assert [out["scores"][t]["rank"] for t in out["confirmed"]] == list(range(1, len(out["confirmed"]) + 1))
    assert out["scores"]["A"]["rank"] is None


@pytest.mark.parametrize("regime, cap", [
    ("bull", 8), ("neutral", 8), ("sideways", 8), (None, 8), ("unknown", 8), ("mars", 8),
    ("bear", 5), ("volatile", 5), ("BEAR", 5),
])
def test_cap_by_regime(regime, cap):
    assert s2.cap_for(regime) == cap
    out = _run(["A"], regime=regime)
    assert out["cap"] == cap and out["regime"] == regime


def test_warmup_when_bursts_missing():
    out = _run(["A", "B", "C"], darts=("A",))                      # bursts 전부 None
    assert out["warmup"] is True
    assert out["scores"]["A"]["score"] == 1.0 and out["confirmed"] == ["A"]   # dart만으로 점수
    assert "워밍업" in out["summary"]


def test_not_warmup_when_most_bursts_present():
    out = _run(["A", "B", "C"], bursts={"A": 1.0, "B": 2.0})
    assert out["warmup"] is False


def test_news_match_is_recorded_but_not_scored():
    out = _run(["A", "B"], bursts={"A": 1.0, "B": 1.0}, news=("A",))
    assert out["scores"]["A"]["news_match"] == 1 and out["confirmed"] == []


def test_dart_titles_carried_into_scores():
    quant = {"A": _q()}
    event = {"A": _e(dart=1, titles=["유상증자결정"])}
    out = s2.run_scoring(["A"], quant, event, {"A": _n()})
    assert out["scores"]["A"]["dart_titles"] == ["유상증자결정"]
