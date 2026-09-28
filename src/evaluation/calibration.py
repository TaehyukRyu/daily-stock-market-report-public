"""
src/evaluation/calibration.py

confidence 보정 (G3) — 자기보고 확신도를 실제 적중률로 번역한다.

[왜 필요한가]
  S1 4-2·5-3: position_size_pct = confidence × 10. LLM 자기보고 확신도가 그대로 시드 배분이다.
  문헌(arXiv 2505.02151)은 이 값이 체계적으로 부풀려진다고 말한다. 0.8이 80% 적중을 뜻한다는
  검증이 없다.

[발상만 가져온 것] prism-insight (AGPL) cores/llm/features/forecast_stats.py:
  "같은 점수 구간의 과거 완료 결과 분포를 base rate로, 표본 부족이면 넓은 코호트로 폴백,
   tier를 보고해 얼마나 구체적인 코호트인지 정직하게 표시". 코드는 새로 씀.

[데이터] prediction_log(에이전트·confidence·recommendation) ⨝ prediction_outcomes(D+10 확정 행).
  2026-09-11: D+1 채점(eval_score)에서 **D+10(G1 holding_days=10)**으로 바꿨다 — 표방 포지션이
  1~4주인데 1일 정확도로 보정하면 지평이 안 맞는다(S1 4-4). 적중 = BUY면 raw_return > +1.5%,
  SELL이면 < -1.5%, HOLD면 |raw| ≤ 1.5% (outcomes.HIT_THRESHOLD[10]).

[규칙 — 사용자 지시]
  표본 30건 미만이면 position_size_pct 고정 5%. 표는 비워 둔다(코드·스키마만).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/mentions.db")

MIN_SAMPLES          = 30      # 이 미만이면 보정값을 쓰지 않는다
HORIZON_DAYS         = 10      # G1 holding_days 중 2주 (outcomes.HORIZONS에 있어야 한다)
DEFAULT_POSITION_PCT = 5.0     # 표본 부족 시 고정
MAX_POSITION_PCT     = 15.0    # chief 프롬프트의 상한과 동일
MIN_POSITION_PCT     = 2.0

# confidence 구간. Quality Gate 컷(0.6) 아래는 애초에 투표에 안 들어간다.
BINS: tuple[tuple[float, float], ...] = ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))


def bin_of(confidence: float) -> tuple[float, float]:
    for lo, hi in BINS:
        if lo <= confidence < hi:
            return (lo, hi)
    return BINS[-1]


def calibration_table(db_path: Path | None = None) -> dict:
    """{(agent, (lo,hi)): {"n": int, "hit_rate": float}} + {("*", (lo,hi)): …} (전체 코호트)."""
    path = db_path or DB_PATH
    table: dict = {}
    if not path.exists():
        return table
    from src.evaluation.outcomes import HIT_THRESHOLD
    thr = HIT_THRESHOLD.get(HORIZON_DAYS, 0.015)
    with sqlite3.connect(path) as conn:
        try:
            rows = conn.execute(
                "SELECT p.agent_name, p.confidence, p.recommendation, o.raw_return "
                "FROM prediction_log p JOIN prediction_outcomes o "
                "  ON o.pred_date = p.pred_date AND o.ticker = p.ticker "
                "WHERE p.ticker != '' AND o.status = 'resolved' AND o.horizon_days = ? "
                "  AND o.raw_return IS NOT NULL",
                (HORIZON_DAYS,),
            ).fetchall()
        except sqlite3.OperationalError:
            return table
    acc: dict = {}
    for agent, conf, rec, raw in rows:
        raw = float(raw)
        hit = (raw > thr) if rec == "BUY" else (raw < -thr) if rec == "SELL" else (abs(raw) <= thr)
        score = 1.0 if hit else 0.0
        b = bin_of(float(conf))
        for key in ((agent, b), ("*", b)):
            a = acc.setdefault(key, [0, 0.0])
            a[0] += 1
            a[1] += float(score)
    return {k: {"n": n, "hit_rate": s / n} for k, (n, s) in acc.items()}


def calibrated_hit_rate(confidence: float, agent: Optional[str] = None,
                        table: Optional[dict] = None, db_path: Path | None = None) -> tuple[Optional[float], str]:
    """(적중률, tier). tier: 'agent' | 'all' | 'insufficient'.
    에이전트별 코호트가 MIN_SAMPLES 이상이면 그것, 아니면 전체 코호트, 그것도 부족하면 None."""
    table = table if table is not None else calibration_table(db_path)
    b = bin_of(confidence)
    if agent:
        cell = table.get((agent, b))
        if cell and cell["n"] >= MIN_SAMPLES:
            return cell["hit_rate"], "agent"
    cell = table.get(("*", b))
    if cell and cell["n"] >= MIN_SAMPLES:
        return cell["hit_rate"], "all"
    return None, "insufficient"


def position_size_for(confidence: float, agent: Optional[str] = None,
                      table: Optional[dict] = None, db_path: Path | None = None) -> tuple[float, str]:
    """포지션 크기(%)와 근거 문자열. 표본 부족이면 DEFAULT_POSITION_PCT."""
    hit, tier = calibrated_hit_rate(confidence, agent, table, db_path)
    if hit is None:
        return DEFAULT_POSITION_PCT, f"보정 표본 부족(<{MIN_SAMPLES}) → 고정 {DEFAULT_POSITION_PCT:.0f}%"
    size = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, round(hit * 10.0, 1)))
    return size, f"보정 적중률 {hit:.2f} ({tier} 코호트, confidence {confidence:.2f}) → {size:.1f}%"


def format_table(db_path: Path | None = None) -> str:
    t = calibration_table(db_path)
    if not t:
        return f"[calibration] D+{HORIZON_DAYS} 확정 예측 없음 — 표 비어 있음 (포지션 고정 5%)"
    lines = [f"[calibration] confidence 구간별 적중률 (D+{HORIZON_DAYS} 방향 적중, prediction_outcomes)"]
    for (agent, (lo, hi)), v in sorted(t.items(), key=lambda kv: (kv[0][0] != "*", kv[0][0], kv[0][1])):
        flag = "" if v["n"] >= MIN_SAMPLES else f"  (표본 부족 <{MIN_SAMPLES})"
        lines.append(f"  {agent:<24} [{lo:.1f},{hi:.1f})  n={v['n']:<4} 적중 {v['hit_rate']:.2f}{flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_table())
