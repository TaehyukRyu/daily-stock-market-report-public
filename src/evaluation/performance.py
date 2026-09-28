"""
src/evaluation/performance.py

실현 성과 집계 — 승률 / 실현 손익 / KOSPI 대비.

[왜 이 모듈이 없었나]
  positions 테이블에 realized_pnl 컬럼은 있었지만 close_position()의 호출자가
  CLI 하나뿐이라 포지션이 닫히지 않았다 (2026-09-08 기준 positions 0행).
  닫힌 포지션이 없으니 집계할 대상도 없었고, 그래서 집계 코드도 없었다.
  auto_close_triggered_positions()가 생기면서 비로소 집계가 성립한다.

[한계 — 반드시 알고 읽을 것]
  1. 이 시스템은 주문을 내지 않는다. 청산가는 신호 도달일의 종가이며
     실제 체결가가 아니다. 슬리피지·수수료·세금이 반영되지 않았다.
  2. 표본이 작으면 승률은 거의 의미가 없다. MIN_SAMPLES_FOR_WINRATE
     미만이면 승률을 None으로 반환한다.
  3. KOSPI 대비는 "같은 기간 지수 수익률"과의 단순 차이다. 베타 조정도
     리스크 조정도 하지 않았으므로 알파가 아니다.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/mentions.db")

# 이 미만이면 승률을 숫자로 내놓지 않는다. 3승 1패를 "75% 승률"로 부르면
# 표본 크기를 숨긴 채 성과를 과장하게 된다.
MIN_SAMPLES_FOR_WINRATE = 10


def _fetch_closed() -> list[dict]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT ticker, ticker_name, entry_date, close_date,
                       entry_price, close_price, realized_pnl_pct,
                       allocation_pct, close_reason
                FROM positions
                WHERE status='closed' AND realized_pnl_pct IS NOT NULL
                ORDER BY close_date
                """
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"[performance] positions 조회 실패: {e}")
            return []
    return [dict(r) for r in rows]


def _kospi_return_pct(start: str, end: str) -> Optional[float]:
    """기간 KOSPI 수익률(%). 조회 실패 시 None."""
    try:
        import yfinance as yf
        hist = yf.Ticker("^KS11").history(start=start, end=end)
        if hist.empty or len(hist) < 2:
            return None
        first, last = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        return round((last / first - 1.0) * 100, 2) if first else None
    except Exception as e:
        logger.warning(f"[performance] KOSPI 조회 실패: {e}")
        return None


def get_performance_summary() -> dict:
    """
    닫힌 포지션 전체를 집계한다.

    Returns:
      {
        "closed_count":        int,
        "win_count":           int,
        "win_rate_pct":        float | None,   # 표본 부족 시 None
        "sample_warning":      str | None,
        "avg_pnl_pct":         float | None,   # 건당 단순 평균
        "cumulative_seed_pct": float | None,   # 시드 대비 누적 (배분 가중)
        "best": dict | None, "worst": dict | None,
        "by_reason":           {reason: count},
        "period":              (first_entry, last_close) | None,
        "kospi_return_pct":    float | None,
        "excess_vs_kospi_pct": float | None,
      }
    """
    rows = _fetch_closed()
    out: dict = {
        "closed_count": len(rows), "win_count": 0, "win_rate_pct": None,
        "sample_warning": None, "avg_pnl_pct": None, "cumulative_seed_pct": None,
        "best": None, "worst": None, "by_reason": {}, "period": None,
        "kospi_return_pct": None, "excess_vs_kospi_pct": None,
    }
    if not rows:
        out["sample_warning"] = "닫힌 포지션이 없어 집계할 수 없다."
        return out

    pnls = [r["realized_pnl_pct"] for r in rows]
    out["win_count"]   = sum(1 for p in pnls if p > 0)
    out["avg_pnl_pct"] = round(sum(pnls) / len(pnls), 2)

    if len(rows) >= MIN_SAMPLES_FOR_WINRATE:
        out["win_rate_pct"] = round(out["win_count"] / len(rows) * 100, 1)
    else:
        out["sample_warning"] = (
            f"표본 {len(rows)}건 < {MIN_SAMPLES_FOR_WINRATE}건 — "
            f"승률을 계산하지 않는다 (숫자가 오해를 부른다)."
        )

    # 시드 대비 누적: 배분 비중을 곱해야 실제 포트폴리오 손익에 가깝다.
    # allocation_pct가 없는 옛 레코드는 집계에서 제외한다.
    weighted = [
        r["allocation_pct"] * r["realized_pnl_pct"] / 100.0
        for r in rows if r["allocation_pct"]
    ]
    if weighted:
        out["cumulative_seed_pct"] = round(sum(weighted), 2)

    best  = max(rows, key=lambda r: r["realized_pnl_pct"])
    worst = min(rows, key=lambda r: r["realized_pnl_pct"])
    for key, r in (("best", best), ("worst", worst)):
        out[key] = {
            "ticker": r["ticker"], "name": r["ticker_name"],
            "pnl_pct": round(r["realized_pnl_pct"], 2), "reason": r["close_reason"],
        }

    for r in rows:
        reason = r["close_reason"] or "unknown"
        out["by_reason"][reason] = out["by_reason"].get(reason, 0) + 1

    entries = [r["entry_date"] for r in rows if r["entry_date"]]
    closes  = [r["close_date"] for r in rows if r["close_date"]]
    if entries and closes:
        start, end = min(entries), max(closes)
        out["period"] = (start, end)
        kospi = _kospi_return_pct(start, end)
        out["kospi_return_pct"] = kospi
        if kospi is not None and out["cumulative_seed_pct"] is not None:
            out["excess_vs_kospi_pct"] = round(out["cumulative_seed_pct"] - kospi, 2)

    return out


def format_performance_section(summary: Optional[dict] = None) -> str:
    """리포트에 붙일 마크다운 조각."""
    s = summary if summary is not None else get_performance_summary()

    if s["closed_count"] == 0:
        return "## 📈 실현 성과\n\n청산된 포지션이 없습니다.\n"

    lines = ["## 📈 실현 성과", ""]
    lines.append(f"- 청산 {s['closed_count']}건 (수익 {s['win_count']}건)")

    if s["win_rate_pct"] is not None:
        lines.append(f"- 승률 **{s['win_rate_pct']}%**")
    else:
        lines.append(f"- 승률: 표본 부족 — {s['sample_warning']}")

    if s["avg_pnl_pct"] is not None:
        lines.append(f"- 건당 평균 {s['avg_pnl_pct']:+.2f}%")
    if s["cumulative_seed_pct"] is not None:
        lines.append(f"- 시드 대비 누적 **{s['cumulative_seed_pct']:+.2f}%**")
    if s["kospi_return_pct"] is not None:
        lines.append(f"- 같은 기간 KOSPI {s['kospi_return_pct']:+.2f}%")
    if s["excess_vs_kospi_pct"] is not None:
        lines.append(f"- KOSPI 대비 **{s['excess_vs_kospi_pct']:+.2f}%p**")

    if s["best"]:
        b, w = s["best"], s["worst"]
        lines.append(f"- 최고 {b['ticker']} {b['pnl_pct']:+.2f}% / 최저 {w['ticker']} {w['pnl_pct']:+.2f}%")
    if s["by_reason"]:
        detail = ", ".join(f"{k} {v}건" for k, v in sorted(s["by_reason"].items()))
        lines.append(f"- 청산 사유: {detail}")

    lines.append("")
    lines.append(
        "> 청산가는 신호 도달일 종가이며 실제 체결가가 아니다. "
        "슬리피지·수수료·세금 미반영."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(format_performance_section())
