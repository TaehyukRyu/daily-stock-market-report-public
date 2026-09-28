"""
signals.py — Stage 1-A 4개 지표의 벡터화.

stage1a_quant._compute_signals(series)는 "마지막 봉" 하나에 대해 계산한다. 백테스트는 매일·매종목
값이 필요하므로 같은 정의를 pandas rolling으로 옮겼다. 동일성은 tests/test_backtest.py가
_compute_signals와 무작위 시계열에서 비교해 고정한다.

원본 정의 (stage1a_quant.py:366-424):
  price_change_5d = close[-1]/close[-6] - 1
  volume_signal   = mean(vol[-3:]) / mean(vol[-20:]) > VOLUME_RATIO_THRESHOLD
  ma_breakout     = ∃ w∈{20,60}: close[-2] < MA_w(prev) and close[-1] > MA_w(now)
  golden_cross    = MA5(prev) < MA20(prev) and MA5(now) > MA20(now)
  signal_count    = (|pc5| ≥ PRICE_CHANGE_THRESHOLD) + volume + breakout + golden
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalParams:
    price_change_threshold: float = 0.03   # stage1a PRICE_CHANGE_THRESHOLD
    volume_ratio_threshold: float = 1.5    # stage1a VOLUME_RATIO_THRESHOLD
    ma_short: int = 20                     # 돌파 창 1 + 골든크로스의 장기선
    ma_long: int = 60                      # 돌파 창 2
    golden_fast: int = 5                   # 골든크로스 단기선 (고정)

    def key(self) -> str:
        return f"pc{self.price_change_threshold}_vr{self.volume_ratio_threshold}_ma{self.ma_short}-{self.ma_long}"


def compute_signal_frames(adj_close: pd.DataFrame, volume: pd.DataFrame,
                          p: SignalParams = SignalParams()) -> dict[str, pd.DataFrame]:
    """각 (date × code)에 대한 지표. NaN은 워밍업 부족."""
    c, v = adj_close, volume.reindex_like(adj_close)
    pc5 = c / c.shift(5) - 1.0
    vol_ratio = v.rolling(3).mean() / v.rolling(20).mean()
    volume_signal = vol_ratio > p.volume_ratio_threshold

    def breakout(w: int) -> pd.DataFrame:
        ma = c.rolling(w).mean()
        return (c.shift(1) < ma.shift(1)) & (c > ma)

    ma_breakout = breakout(p.ma_short) | breakout(p.ma_long)
    fast, slow = c.rolling(p.golden_fast).mean(), c.rolling(p.ma_short).mean()
    golden = (fast.shift(1) < slow.shift(1)) & (fast > slow)

    price_sig = pc5.abs() >= p.price_change_threshold
    # 워밍업: 원본은 len(closes) < 21이면 None. 60일선은 값이 없으면 그 창만 건너뛴다.
    warm = c.rolling(21).count() >= 21
    count = (price_sig.astype(int) + volume_signal.astype(int) + ma_breakout.astype(int) + golden.astype(int))
    count = count.where(warm)
    return {
        "price_change_5d": pc5, "volume_signal": volume_signal & warm,
        "ma_breakout": ma_breakout & warm, "golden_cross": golden & warm,
        "signal_count": count,
    }


def rule_buy(frames: dict[str, pd.DataFrame], p: SignalParams = SignalParams()) -> pd.DataFrame:
    """quant_rule_agent.build_report의 BUY 규칙: (골든크로스 or MA돌파) and pc5 >= 0."""
    return (frames["golden_cross"] | frames["ma_breakout"]) & (frames["price_change_5d"] >= 0) \
        & frames["signal_count"].notna()


def rule_sell(frames: dict[str, pd.DataFrame], p: SignalParams = SignalParams()) -> pd.DataFrame:
    return (frames["price_change_5d"] <= -p.price_change_threshold) \
        & ~(frames["golden_cross"] | frames["ma_breakout"]) & frames["signal_count"].notna()


def signals_from_series(series: list[dict], p: SignalParams = SignalParams()) -> dict | None:
    """단일 시계열(list[dict])에 대해 마지막 봉 값 — 테스트에서 stage1a._compute_signals와 대조."""
    if len(series) < 21:
        return None
    df = pd.DataFrame(series)
    c = pd.DataFrame({"x": df["close"].astype(float).values})
    v = pd.DataFrame({"x": df["volume"].astype(float).values})
    f = compute_signal_frames(c, v, p)
    i = len(df) - 1
    return {
        "price_change_5d": round(float(f["price_change_5d"].iloc[i, 0]), 4),
        "volume_signal": bool(f["volume_signal"].iloc[i, 0]),
        "ma_breakout": bool(f["ma_breakout"].iloc[i, 0]),
        "golden_cross": bool(f["golden_cross"].iloc[i, 0]),
        "signal_count": int(f["signal_count"].iloc[i, 0]),
    }
