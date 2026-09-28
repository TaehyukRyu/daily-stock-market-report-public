"""
src/evaluation/llm_contribution.py

"LLM 층이 규칙층보다 무엇을 더했는가" 집계 (2026-09-14 결정 3-5 (C)).

[왜 이것부터인가]
  본선 잣대 (A) 선택 능력은 D+10 확정 175거래일이 쌓여야 답이 나온다(2027-05경).
  그때까지 기다리지 않고 **지금 바로 잴 수 있는 유일한 것**이 이 비교다.
  같은 날 같은 종목에 대해 규칙 투표자(quant_rule_agent)와 최종 종합(chief_strategist)이
  이미 둘 다 prediction_log에 남고 있어서, 별도 적재 없이 집계만 만들면 된다.

[무엇을 비교하나]
  (pred_date, ticker)마다 두 판단을 짝지어 prediction_outcomes의 실현 수익에 붙인다.
    - 같은 방향으로 갔을 때: LLM 층이 규칙을 확인만 한 것
    - 갈렸을 때: LLM 층이 규칙을 뒤집은 것 — 이때 누가 맞았나가 기여의 핵심이다
  판정은 방향 적중(HIT_THRESHOLD)과 실현 알파 둘 다로 본다.

[표본 단위]
  outcomes와 같이 **하루**다 (CR-8 / 결정 Q6-a). 단 "갈린 건수"는 종목 단위로도 센다 —
  하루에 몇 번 뒤집었는지가 그 자체로 의미 있는 수라서다.

[한계]
  chief는 규칙 투표자를 **입력으로 포함**해 종합한다. 둘은 독립 판단이 아니다.
  그래서 이 수치는 "LLM이 규칙을 얼마나·어느 방향으로 조정했고 그 조정이 맞았나"이지
  "LLM 단독 vs 규칙 단독"이 아니다. 해석할 때 이 점을 빼놓으면 안 된다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from src.evaluation.outcomes import DB_PATH, HIT_THRESHOLD, HORIZONS

RULE_AGENT  = "quant_rule_agent"
CHIEF_AGENT = "chief_strategist"


def _hit(recommendation: str, ret: float, thr: float) -> bool:
    """방향 적중. outcomes.summary의 판정과 같은 규칙."""
    if recommendation == "BUY":
        return ret > thr
    if recommendation == "SELL":
        return ret < -thr
    return abs(ret) <= thr


def paired_rows(horizon: int, db_path: Path | None = None) -> list[dict]:
    """규칙·chief 판단이 모두 있고 실현까지 끝난 (pred_date, ticker) 행."""
    path = db_path or DB_PATH
    if not Path(path).exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT o.pred_date, o.ticker, o.raw_return, o.net_return, o.alpha_vs_kospi,
                       r.recommendation AS rule_rec, r.confidence AS rule_conf,
                       c.recommendation AS chief_rec, c.confidence AS chief_conf
                FROM prediction_outcomes o
                JOIN prediction_log r
                  ON r.pred_date = o.pred_date AND r.ticker = o.ticker AND r.agent_name = ?
                JOIN prediction_log c
                  ON c.pred_date = o.pred_date AND c.ticker = o.ticker AND c.agent_name = ?
                WHERE o.status = 'resolved' AND o.horizon_days = ? AND o.raw_return IS NOT NULL
                """,
                (RULE_AGENT, CHIEF_AGENT, horizon),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


def summary(horizon: int, db_path: Path | None = None) -> dict:
    """지평별 기여 집계. 표본 단위는 하루, 불일치 건수는 종목 단위."""
    thr = HIT_THRESHOLD.get(horizon, 0.015)
    out: dict = {
        "horizon": horizon, "n_days": 0, "n_pairs": 0,
        "agree_n": 0, "disagree_n": 0, "disagree_rate": None,
        "rule_hit_rate": None, "chief_hit_rate": None,
        "disagree_rule_hit_rate": None, "disagree_chief_hit_rate": None,
        "avg_alpha_agree": None, "avg_alpha_disagree": None,
    }
    rows = paired_rows(horizon, db_path)
    if not rows:
        return out

    out["n_pairs"] = len(rows)
    out["n_days"]  = len({r["pred_date"] for r in rows})

    agree    = [r for r in rows if r["rule_rec"] == r["chief_rec"]]
    disagree = [r for r in rows if r["rule_rec"] != r["chief_rec"]]
    out["agree_n"], out["disagree_n"] = len(agree), len(disagree)
    out["disagree_rate"] = len(disagree) / len(rows)

    def hit_rate(subset: list[dict], key: str) -> Optional[float]:
        if not subset:
            return None
        return sum(_hit(r[key], float(r["raw_return"]), thr) for r in subset) / len(subset)

    out["rule_hit_rate"]  = hit_rate(rows, "rule_rec")
    out["chief_hit_rate"] = hit_rate(rows, "chief_rec")
    # 갈린 건만 따로 — 여기가 LLM 층이 실제로 판단을 바꾼 지점이다
    out["disagree_rule_hit_rate"]  = hit_rate(disagree, "rule_rec")
    out["disagree_chief_hit_rate"] = hit_rate(disagree, "chief_rec")

    def avg_alpha(subset: list[dict]) -> Optional[float]:
        vals = [float(r["alpha_vs_kospi"]) for r in subset if r["alpha_vs_kospi"] is not None]
        return sum(vals) / len(vals) if vals else None

    out["avg_alpha_agree"]    = avg_alpha(agree)
    out["avg_alpha_disagree"] = avg_alpha(disagree)
    return out


def format_summary(db_path: Path | None = None) -> str:
    lines = ["[llm_contribution] 규칙 투표자 vs 최종 종합 (같은 날·같은 종목, 표본 단위=하루)"]
    def rate(x):
        return "n/a" if x is None else f"{x*100:.0f}%"
    def pct(x):
        return "n/a" if x is None else f"{x*100:+.2f}%"
    for h in HORIZONS:
        s = summary(h, db_path)
        if not s["n_pairs"]:
            lines.append(f"  D+{h}: 짝지을 표본 없음")
            continue
        lines.append(
            f"  D+{h}: {s['n_days']}일 / {s['n_pairs']}쌍  "
            f"불일치 {s['disagree_n']}건({rate(s['disagree_rate'])})  "
            f"적중 규칙 {rate(s['rule_hit_rate'])} vs 종합 {rate(s['chief_hit_rate'])}"
        )
        if s["disagree_n"]:
            lines.append(
                f"      갈린 건만: 규칙 {rate(s['disagree_rule_hit_rate'])} vs "
                f"종합 {rate(s['disagree_chief_hit_rate'])}  "
                f"알파 일치 {pct(s['avg_alpha_agree'])} / 불일치 {pct(s['avg_alpha_disagree'])}"
            )
    lines.append("  ※ chief는 규칙 투표자를 입력으로 포함한다. 독립 비교가 아니라 '조정의 방향과 적중'이다.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_summary())
