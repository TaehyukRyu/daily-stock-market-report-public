"""
tests/test_screener_v2.py — run_screening 조립. 각 stage는 목. 외부 호출 없음.
"""
from __future__ import annotations

import pytest

from src.screening import screener as sc


@pytest.fixture
def stages(monkeypatch):
    monkeypatch.setattr(sc, "_init_mention_db", lambda: None)
    monkeypatch.setattr(sc, "_init_ohlcv_cache", lambda: None)
    monkeypatch.setattr(sc, "run_quant_screen", lambda u: {t: {"signal_count": 0, "price_change_5d": 0.0} for t in u})
    monkeypatch.setattr(sc, "news_burst_ready", lambda today: True)
    monkeypatch.setattr(sc, "run_event_screen",
        lambda u, today=None, news_ready=True: {
            t: {"news_burst": float(i), "news_recent": i, "dart_event": 1 if t == "B" else 0,
                "dart_titles": ["유상증자결정"] if t == "B" else [], "sent_delta": None}
            for i, t in enumerate(u, 1)})
    monkeypatch.setattr(sc, "run_news_screen", lambda u: {t: {"news_match": t == "A", "reason": ""} for t in u})
    monkeypatch.setattr(sc._s1c, "LAST_HEADLINE_COUNT", 30)
    return sc


def test_run_screening_shape(stages):
    out = stages.run_screening(list("ABCDE"), regime="bull")
    assert set(out) >= {"confirmed", "optional", "scores", "summary", "headline_count", "cap", "warmup", "stage_results"}
    assert out["confirmed"] == ["B", "E"]                 # B: 공시 (p0.25+1)/2=0.625, E: burst 최고 (p1.0+0)/2=0.5
    assert out["headline_count"] == 30 and out["cap"] == 8 and out["warmup"] is False
    assert out["stage_results"]["universe"] == list("ABCDE")
    assert out["stage_results"]["dart"] == {"B": ["유상증자결정"]}
    assert set(out["scores"]) == set("ABCDE")


def test_run_screening_stage_failure_is_isolated(stages, monkeypatch):
    def boom(*a, **k): raise RuntimeError("dart down")
    monkeypatch.setattr(stages, "run_event_screen", boom)
    out = stages.run_screening(list("ABC"), regime="bear")
    assert out["confirmed"] == [] and out["cap"] == 5 and out["warmup"] is True


def test_run_screening_empty_universe(stages):
    out = stages.run_screening([], regime=None)
    assert out["confirmed"] == [] and out["scores"] == {} and out["stage_results"]["universe"] == []
