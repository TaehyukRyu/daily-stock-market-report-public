"""
src/evaluation/screen_factors.py

스크리닝 v2 팩터별 D+5/D+20 초과수익 — "이 팩터로 고른 종목이 같은 날 유니버스 평균보다 나았는가".
스펙: docs/superpowers/specs/2026-09-21-screening-v2-design.md §6

  입력   screen_candidates (110행/일, 팩터 컬럼) × ohlcv_cache (종가)
  수익률 run_date 이후(포함) 첫 거래일 종가 진입 → H거래일 뒤 종가 청산
  초과   같은 run_date 유니버스 평균을 뺀 값 (지수 대비가 아니다 — 스펙 §1)
  비교   팩터별 A/B 집단의 날짜별 평균 차이 → 평균·t (겹치는 창을 날짜 단위로 접는다)
  게이트 날짜 ≥ 30 AND |t| ≥ 1.96 (prediction_logger.has_weight_evidence와 같은 규칙)

가중치를 자동으로 바꾸지 않는다. 표를 보고 사람이 결정한다.
실행: python -m src.evaluation.screen_factors
"""

from __future__ import annotations

import sqlite3
import statistics as st
from bisect import bisect_left
from pathlib import Path
from typing import Callable, Optional

from src.evaluation.outcomes import DB_PATH, init_outcome_tables

HORIZONS       = (5, 20)
MIN_DATES      = 30
SIGNIFICANCE_Z = 1.96
BURST_TOP      = 0.8      # news_burst_pct 상위 20%
SENT_TOP       = 0.8      # sent_delta 날짜별 백분위 상위 20%
SENT_BOTTOM    = 0.2


def has_evidence(n_dates: int, t: Optional[float]) -> bool:
    return n_dates >= MIN_DATES and t is not None and abs(t) >= SIGNIFICANCE_Z


# ── 분할 규칙: 행 → 'A' | 'B' | None ─────────────────────────

def _split_selected(r):
    return "A" if r["analyzed"] == 1 else "B"


def _split_burst(r):
    if r["news_burst_pct"] is None:
        return None
    return "A" if r["news_burst_pct"] >= BURST_TOP else "B"


def _split_dart(r):
    return "A" if r["dart_event"] == 1 else "B"


def _split_news(r):
    return "A" if r["news_match"] == 1 else "B"


def _split_sent(r):
    p = r.get("sent_pct")
    if p is None:
        return None
    if p >= SENT_TOP:
        return "A"
    if p <= SENT_BOTTOM:
        return "B"
    return None


SPLITS: dict[str, tuple[str, str, Callable[[dict], Optional[str]]]] = {
    "selected":   ("분석(analyzed)", "미선정",     _split_selected),
    "news_burst": ("급증 상위20%",  "나머지",      _split_burst),
    "dart_event": ("공시 있음",     "없음",        _split_dart),
    "news_match": ("1-C 매칭",      "비매칭",      _split_news),
    "sent_delta": ("감성 상위20%",  "하위20%",     _split_sent),
}


# ── 데이터 ────────────────────────────────────────────────

def _rows(db_path: Path | None) -> list[dict]:
    """v2 행(is_candidate NOT NULL)만. sent_delta는 날짜별 백분위(sent_pct)를 붙인다."""
    init_outcome_tables(db_path)
    with sqlite3.connect(db_path or DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT run_date, ticker, analyzed, is_candidate, news_burst_pct, dart_event, news_match, sent_delta "
            "FROM screen_candidates WHERE is_candidate IS NOT NULL")]
    by_date: dict[str, list[float]] = {}
    for r in rows:
        if r["sent_delta"] is not None:
            by_date.setdefault(r["run_date"], []).append(float(r["sent_delta"]))
    for r in rows:
        r["sent_pct"] = None
        vals = sorted(by_date.get(r["run_date"], []))
        if r["sent_delta"] is not None and len(vals) > 1:
            r["sent_pct"] = bisect_left(vals, float(r["sent_delta"])) / (len(vals) - 1)
    return rows


def _closes(db_path: Path | None) -> tuple[list[str], dict[str, dict[str, float]]]:
    with sqlite3.connect(db_path or DB_PATH) as conn:
        data = conn.execute("SELECT ticker, date, close FROM ohlcv_cache").fetchall()
    closes: dict[str, dict[str, float]] = {}
    dates: set[str] = set()
    for t, d, c in data:
        closes.setdefault(t, {})[d] = c
        dates.add(d)
    return sorted(dates), closes


def forward_excess(rows: list[dict], horizon: int, trading: list[str],
                   closes: dict[str, dict[str, float]]) -> list[dict]:
    """각 행에 ret(진입→청산)과 excess(같은 날 유니버스 평균 대비)를 붙인다. 창이 안 찬 행은 뺀다."""
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        i = bisect_left(trading, r["run_date"])          # run_date 이후(포함) 첫 거래일
        if i + horizon >= len(trading):
            continue
        d0, d1 = trading[i], trading[i + horizon]
        c = closes.get(r["ticker"], {})
        if not c.get(d0) or d1 not in c:
            continue
        by_date.setdefault(r["run_date"], []).append({**r, "ret": c[d1] / c[d0] - 1.0})
    out: list[dict] = []
    for lst in by_date.values():
        m = st.mean(x["ret"] for x in lst)
        out.extend({**x, "excess": x["ret"] - m} for x in lst)
    return out


def group_stats(rows: list[dict], split: Callable[[dict], Optional[str]]) -> dict:
    """날짜별 (A 평균 − B 평균) → 그 차이들의 평균과 t. 둘 다 있는 날짜만 센다."""
    by_date: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        side = split(r)
        if side:
            by_date.setdefault(r["run_date"], {"A": [], "B": []})[side].append(r["excess"])
    diffs = [st.mean(v["A"]) - st.mean(v["B"]) for v in by_date.values() if v["A"] and v["B"]]
    n = len(diffs)
    mean = st.mean(diffs) if diffs else None
    t = None
    if n >= 3:
        sd = st.stdev(diffs)
        t = (mean / (sd / n ** 0.5)) if sd > 0 else None
    return {"n_dates": n, "mean_diff": mean, "t": t, "evidence": has_evidence(n, t)}


def summary(db_path: Path | None = None) -> dict[int, dict[str, dict]]:
    rows = _rows(db_path)
    trading, closes = _closes(db_path)
    out: dict[int, dict[str, dict]] = {}
    for h in HORIZONS:
        fx = forward_excess(rows, h, trading, closes)
        out[h] = {name: group_stats(fx, fn) for name, (_, _, fn) in SPLITS.items()}
    return out


def format_summary(db_path: Path | None = None) -> str:
    s = summary(db_path)
    lines = ["[screen_factors] 팩터별 유니버스 대비 초과수익 (A − B, 날짜별 평균) — 게이트: 날짜 ≥ 30 AND |t| ≥ 1.96"]
    for h in HORIZONS:
        lines.append(f"\n  D+{h}")
        lines.append(f"  {'factor':12s} {'A':>14s} vs {'B':<12s} {'날짜':>4s} {'차이%':>8s} {'t':>6s}  게이트")
        for name, (la, lb, _) in SPLITS.items():
            g = s[h][name]
            diff = f"{100 * g['mean_diff']:8.2f}" if g["mean_diff"] is not None else f"{'n/a':>8s}"
            tt   = f"{g['t']:6.2f}" if g["t"] is not None else f"{'n/a':>6s}"
            lines.append(f"  {name:12s} {la:>14s} vs {lb:<12s} {g['n_dates']:4d} {diff} {tt}  {'통과' if g['evidence'] else '-'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_summary())
