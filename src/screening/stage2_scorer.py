"""
src/screening/stage2_scorer.py

Stage 2 (v2): 팩터 백분위 → 점수 → 후보 정의 → 필터 → 레짐 상한.
스펙: docs/superpowers/specs/2026-09-21-screening-v2-design.md §3
감사: doc/2026-09-20_screening-audit.md (왜 점수 합산·임계값 구조를 버렸나)

  점수      = 반영 팩터의 단순 평균 (news_burst_pct, dart_event). 균등 가중, 자동 갱신 없음.
  후보      = news_burst_pct ≥ 0.8  OR  dart_event == 1
  필터      = OHLCV 21행 미만(stage1a가 quant_results에서 뺀 종목) / |5일 수익률| ≥ 10%  (후보에서 제외, 사유 기록)
  선정 순서 = score ↓ → news_burst 원값 ↓ → 코드 ↑
  상한      = 레짐별 (bear·volatile 5, 그 외 8). daily_runner가 MAX_CONFIRMED_TICKERS와 min을 취한다.

'신호 없음'은 정상 출력이다. 후보 0개면 confirmed=[]가 그대로 나간다 (G5).
"""

from __future__ import annotations

import bisect
import logging

logger = logging.getLogger(__name__)

NEWS_BURST_PCT_MIN = 0.8      # 유니버스 상위 20%
MAX_ABS_RET5       = 0.10     # 이미 크게 움직인 종목 제외. 근거 없는 초기값 — 기록만 하고 시작 (스펙 §3)
DEFAULT_CAP        = 8
CAP_BY_REGIME: dict[str, int] = {"bear": 5, "volatile": 5}


def cap_for(regime: str | None) -> int:
    return CAP_BY_REGIME.get((regime or "").lower(), DEFAULT_CAP)


def percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """값 → 유니버스 내 백분위 0~1 = (자기보다 작은 값의 수) / (n − 1). 동률은 같은 값. n=1이면 1.0."""
    if not values:
        return {}
    vals = sorted(values.values())
    n = len(vals)
    if n == 1:
        return {k: 1.0 for k in values}
    return {k: bisect.bisect_left(vals, v) / (n - 1) for k, v in values.items()}


def _score_one(quant_row: dict | None, event_row: dict | None, news_row: dict | None,
               pct: float | None) -> dict:
    event_row = event_row or {}
    news_row  = news_row or {}
    dart      = int(bool(event_row.get("dart_event")))
    parts     = [v for v in (pct, float(dart)) if v is not None]
    d = {
        "score":           round(sum(parts) / len(parts), 4) if parts else 0.0,
        "news_burst":      event_row.get("news_burst"),
        "news_burst_pct":  pct,
        "dart_event":      dart,
        "news_match":      int(bool(news_row.get("news_match"))),
        "sent_delta":      event_row.get("sent_delta"),
        "ret5":            None if not quant_row else quant_row.get("price_change_5d"),
        "is_candidate":    0,
        "rank":            None,
        "filtered_reason": None,
        "dart_titles":     list(event_row.get("dart_titles") or []),
    }
    if not quant_row:
        d["filtered_reason"] = "insufficient_ohlcv"
    elif d["ret5"] is not None and abs(float(d["ret5"])) >= MAX_ABS_RET5:
        d["filtered_reason"] = "already_moved"
    elif (pct is not None and pct >= NEWS_BURST_PCT_MIN) or dart == 1:
        d["is_candidate"] = 1
    return d


def _log_distribution(scores: dict[str, dict], cands: list[str]) -> None:
    n_burst = sum(1 for d in scores.values() if (d["news_burst_pct"] or 0.0) >= NEWS_BURST_PCT_MIN)
    n_dart  = sum(1 for d in scores.values() if d["dart_event"])
    n_filt  = sum(1 for d in scores.values() if d["filtered_reason"])
    print(f"  [Stage2] 뉴스급증 상위20% {n_burst}개 / 공시 {n_dart}개 / 필터 제외 {n_filt}개 → 후보 {len(cands)}개")
    for t in cands[:10]:
        d = scores[t]
        tags = []
        if d["news_burst"] is not None:
            tags.append(f"burst {d['news_burst']:.1f}배(p{d['news_burst_pct']:.2f})")
        if d["dart_event"]:
            tags.append("공시:" + " / ".join(d["dart_titles"][:2]))
        if d["news_match"]:
            tags.append("1-C")
        print(f"    {d['rank']:2d}. {t} : {d['score']:.2f}  [{', '.join(tags) or '-'}]")


def run_scoring(
    tickers:       list[str],
    quant_results: dict[str, dict],
    event_results: dict[str, dict],
    news_results:  dict[str, dict],
    regime:        str | None = None,
) -> dict:
    """
    Returns:
      confirmed  후보 전부, 선정 순서 (상한 적용 전 — daily_runner가 cap과 MAX_CONFIRMED_TICKERS로 자른다)
      optional   [] 고정
      scores     유니버스 전 종목 {ticker: detail}
      summary    "후보 N개 / 필터 제외 K개 / 상한 C (레짐 R[, 워밍업])"
      cap        레짐 상한
      warmup     news_burst가 절반 이상 None이면 True
      regime     입력 그대로
    """
    bursts = {t: float(v) for t in tickers
              if (v := (event_results.get(t) or {}).get("news_burst")) is not None}
    pcts   = percentile_ranks(bursts)

    scores = {t: _score_one(quant_results.get(t), event_results.get(t), news_results.get(t), pcts.get(t))
              for t in tickers}

    cands = [t for t in tickers if scores[t]["is_candidate"]]
    cands.sort(key=lambda t: (-scores[t]["score"], -(scores[t]["news_burst"] or 0.0), t))
    for i, t in enumerate(cands, 1):
        scores[t]["rank"] = i

    cap    = cap_for(regime)
    warmup = (len(bursts) < len(tickers) / 2) if tickers else True
    n_filt = sum(1 for d in scores.values() if d["filtered_reason"])
    summary = (f"후보 {len(cands)}개 / 필터 제외 {n_filt}개 / 상한 {cap} "
               f"(레짐 {regime or 'n/a'}{', 워밍업' if warmup else ''})")

    _log_distribution(scores, cands)
    return {"confirmed": cands, "optional": [], "scores": scores, "summary": summary,
            "cap": cap, "warmup": warmup, "regime": regime}
