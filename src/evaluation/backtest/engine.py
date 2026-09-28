"""
engine.py — H거래일 리밸런싱 시뮬레이션 (사전 등록 §2-3·§2-4).

시간 규약
  t      = 신호일 (종가 기준 지표)
  t+1    = 조정 시가 진입
  t+H    = 조정 종가 청산  (다음 신호일 = t+H, 다음 진입 = t+H+1 시가 → 포지션 겹침 없음)
비중   동일가중 1/N. 후보가 N 미만이면 나머지는 현금(수익 0).
비용   매수: 수수료+제비용+슬리피지 / 매도: 수수료+제비용+슬리피지+거래세(매도일 연도별)
자산곡선 일별. 보유 구간은 조정 종가로 마크, 청산일에 매도 비용 반영.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.evaluation.backtest.strategy import rank_candidates, select_top_n, random_select


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.000140527     # 한국투자증권 뱅키스 온라인, KRX (2025-10-27 기준)
    exchange_fee: float = 0.000036396   # 유관기관 제비용, KRX
    slippage: float = 0.001             # 가정 0.10% (편도)
    # 증권거래세(매도). 연도별. 코스피 2026 = 거래세 0.05 + 농특세 0.15.
    tax_by_year: dict = field(default_factory=lambda: {2023: 0.0018, 2024: 0.0018, 2025: 0.0015, 2026: 0.0020})

    def buy_cost(self) -> float:
        return self.commission + self.exchange_fee + self.slippage

    def sell_cost(self, sell_date: pd.Timestamp) -> float:
        tax = self.tax_by_year.get(sell_date.year, 0.0020)
        return self.commission + self.exchange_fee + self.slippage + tax

    def with_slippage_mult(self, m: float) -> "CostModel":
        return CostModel(self.commission, self.exchange_fee, self.slippage * m, dict(self.tax_by_year))

    def key(self) -> str:
        return f"c{self.commission}_f{self.exchange_fee}_s{self.slippage}"


@dataclass
class SimResult:
    equity: pd.Series                 # 일별 자산 (시작 1.0)
    bench_equity: pd.Series           # B2 — 벤치마크 buy&hold 100% (시작 1.0), 같은 인덱스
    trades: pd.DataFrame              # 거래 원장
    periods: pd.DataFrame             # 리밸런싱 구간별 (signal_date, entry, exit, n_picked, ret)
    params: dict
    # B1 — 노출 일치 벤치마크: 지수를 exposure(=weight×n_picks)만큼 buy&hold, 나머지는 현금(0%).
    # S5 사전 등록 §2. B1만 보면 "지는 게임에 덜 참여한 것"이 개선으로 읽히므로 B2와 항상 같이 본다.
    bench_equity_matched: Optional[pd.Series] = None


def simulate(
    prices: dict[str, pd.DataFrame],
    bench: pd.DataFrame,
    universes: dict[pd.Timestamp, list[str]],
    buy_mask: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    start: str, end: str,
    n_picks: int = 5, horizon: int = 10,
    costs: CostModel = CostModel(),
    random_pick: bool = False, seed: Optional[int] = None,
    weight: Optional[float] = None,
    selector: Optional[Callable[[pd.Timestamp, list[str], set[str]], list[str]]] = None,
) -> SimResult:
    """weight: 종목당 비중. None이면 1/n_picks(=풀투자, 기존 동작).

    S5(사전 등록 `doc/2026-09-15_s5-prereg.md`): 운영은 종목당 5%이고 백테스트는 20%였다.
    weight를 n_picks와 분리해 운영 비중으로 돌릴 수 있게 한다.
    **weight=None이면 기존 결과가 비트 단위로 재현되어야 한다** (R0 회귀).
    selector: 규칙 파이프라인(스펙 §1-2)이 주입하는 선정 함수. None이면 기존 quant_rule 경로.
    """
    from src.evaluation.backtest.data import universe_on

    adj_close, adj_open = prices["adj_close"], prices["adj_open"]
    idx = adj_close.index
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) < horizon + 2:
        raise ValueError("기간이 보유기간보다 짧다")
    pos = {d: i for i, d in enumerate(adj_close.index)}
    rng = np.random.default_rng(seed) if random_pick else None

    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades: list[dict] = []
    periods: list[dict] = []
    nav = 1.0
    k = 0
    last_mark_end = None
    while True:
        t = idx[k]
        i = pos[t]
        if i + horizon >= len(adj_close.index):
            break
        entry_d, exit_d = adj_close.index[i + 1], adj_close.index[i + horizon]
        if exit_d > idx[-1]:
            break
        universe = universe_on(t, universes)
        tradable = {c for c in universe
                    if c in adj_open.columns and pd.notna(adj_open.at[entry_d, c]) and pd.notna(adj_close.at[exit_d, c])
                    and adj_open.at[entry_d, c] > 0}
        if random_pick:
            picks = random_select(universe, tradable, n_picks, rng)
        elif selector is not None:
            picks = [c for c in selector(t, universe, tradable) if c in tradable][:n_picks]
        else:
            picks = select_top_n(rank_candidates(t, universe, buy_mask, frames, tradable), n_picks)

        w = (1.0 / n_picks) if weight is None else float(weight)
        buy_c = costs.buy_cost()
        sell_c = costs.sell_cost(exit_d)
        b_entry = float(bench["open"].get(entry_d, np.nan)) if entry_d in bench.index else np.nan
        b_exit = float(bench["close"].get(exit_d, np.nan)) if exit_d in bench.index else np.nan
        bench_r = (b_exit / b_entry - 1.0) if (b_entry and not np.isnan(b_entry) and not np.isnan(b_exit)) else np.nan

        # 일별 마크: 보유 구간 [entry_d, exit_d]
        hold_days = adj_close.index[i + 1: i + horizon + 1]
        day_vals = pd.Series(1.0 - w * len(picks), index=hold_days)   # 현금 몫
        period_ret = 0.0
        for c in picks:
            e = float(adj_open.at[entry_d, c]) * (1 + buy_c)
            path = adj_close.loc[hold_days, c].astype(float) / e
            path.iloc[-1] = path.iloc[-1] * (1 - sell_c)
            day_vals = day_vals.add(w * path.ffill(), fill_value=0.0)
            x = float(adj_close.at[exit_d, c])
            r_gross = x / float(adj_open.at[entry_d, c]) - 1.0
            r_net = x * (1 - sell_c) / e - 1.0
            period_ret += w * r_net
            trades.append({
                "signal_date": t, "entry_date": entry_d, "exit_date": exit_d, "code": c,
                "entry": float(adj_open.at[entry_d, c]), "exit": x,
                "r_gross": r_gross, "r_net": r_net, "bench_r": bench_r,
                "excess": (r_net - bench_r) if not np.isnan(bench_r) else np.nan,
                "horizon": horizon, "random": random_pick,
            })
        equity.loc[hold_days] = nav * day_vals.values
        nav = nav * (1.0 + period_ret)
        periods.append({"signal_date": t, "entry_date": entry_d, "exit_date": exit_d,
                        "n_picked": len(picks), "ret": period_ret, "bench_r": bench_r})
        last_mark_end = exit_d
        k += horizon
        if k >= len(idx):
            break

    equity = equity.loc[:last_mark_end] if last_mark_end is not None else equity
    equity.iloc[0] = 1.0 if np.isnan(equity.iloc[0]) else equity.iloc[0]
    equity = equity.ffill()
    b = bench["close"].reindex(equity.index).ffill()
    bench_equity = b / float(b.iloc[0])
    w_eff = (1.0 / n_picks) if weight is None else float(weight)
    exposure = w_eff * n_picks
    bench_equity_matched = 1.0 + exposure * (bench_equity - 1.0)
    return SimResult(
        equity=equity, bench_equity=bench_equity,
        trades=pd.DataFrame(trades), periods=pd.DataFrame(periods),
        params={"start": start, "end": end, "n_picks": n_picks, "horizon": horizon,
                "random": random_pick, "seed": seed, "costs": asdict(costs),
                "weight": w_eff, "exposure": exposure},
        bench_equity_matched=bench_equity_matched,
    )
