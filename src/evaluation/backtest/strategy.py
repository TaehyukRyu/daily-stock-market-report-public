"""
strategy.py — 선정 규칙 (사전 등록 §2-3).

후보 = 그날 유니버스 ∩ quant_rule_agent BUY
순위 = signal_count 내림차순 → price_change_5d 내림차순 → 코드 오름차순
상위 N. N 미만이면 채우지 않는다 (G5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_candidates(date_: pd.Timestamp, universe: list[str], buy_mask: pd.DataFrame,
                    frames: dict[str, pd.DataFrame], tradable: set[str]) -> list[str]:
    """정렬된 후보 코드 리스트 (N으로 자르기 전)."""
    if date_ not in buy_mask.index:
        return []
    row = buy_mask.loc[date_]
    cands = [c for c in universe if c in row.index and bool(row[c]) and c in tradable]
    if not cands:
        return []
    cnt = frames["signal_count"].loc[date_]
    pc = frames["price_change_5d"].loc[date_]
    cands.sort(key=lambda c: (-float(cnt[c]), -float(pc[c]), c))
    return cands


def select_top_n(cands: list[str], n: int) -> list[str]:
    return cands[:n]


def random_select(universe: list[str], tradable: set[str], n: int, rng: np.random.Generator) -> list[str]:
    """무작위 대조군: 같은 유니버스·같은 N. 거래 가능한 종목 중 무작위 N개 (부족하면 있는 만큼)."""
    pool = sorted(c for c in universe if c in tradable)
    if len(pool) <= n:
        return pool
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]
