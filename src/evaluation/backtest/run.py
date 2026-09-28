"""
run.py — 사전 등록 §4의 실행 계획을 그대로 수행하고 기록한다.

  python -m src.evaluation.backtest.run --all            # R1 + R2 + R3
  python -m src.evaluation.backtest.run --r1             # 본 실행만
  python -m src.evaluation.backtest.run --r2 --n-random 1000
  python -m src.evaluation.backtest.run --r3

기록
  doc/backtest/backtest_runs.csv           run_id, 시각, 파라미터 해시, 기간, 결과 요약 (모든 실행)
  doc/backtest/results/R1_H{H}.json        본 실행 요약·walk-forward·국면·비용·유의성
  doc/backtest/results/R1_H{H}_trades.csv  거래 원장
  doc/backtest/results/R2_random.json      무작위 대조군
  doc/backtest/results/R3_sensitivity.json 민감도
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import data as D
from src.evaluation.backtest import controls as C
from src.evaluation.backtest import metrics as M
from src.evaluation.backtest.engine import CostModel, simulate
from src.evaluation.backtest.signals import SignalParams, compute_signal_frames, rule_buy

logger = logging.getLogger(__name__)

OUT_DIR = Path("doc/backtest")
RES_DIR = OUT_DIR / "results"
RUNS_CSV = OUT_DIR / "backtest_runs.csv"

# ── 사전 등록 §1·§2·§4 값 ─────────────────────────────────────────────
# [S5 2026-09-15] 기간 3년 → 10.7년. 사전 등록 doc/2026-09-15_s5-prereg.md §4.
#   H=10 구간 71 → 약 261. N2가 계산한 M1 필요 표본 237구간을 처음으로 넘긴다.
#   ⚠️ IS_END/OOS_START는 상위 사전 등록의 사용자 지정값이라 바꾸지 않는다.
#      따라서 늘어나는 것은 IS와 full이고 OOS는 29구간 그대로다.
FETCH_START = "2015-09-01"      # 워밍업(60거래일)용 여유
START, END  = "2016-01-04", "2026-09-11"
IS_END, OOS_START = "2025-06-30", "2025-07-01"
# 모델 컷오프(gpt-4o-mini 2023-10-01) 정렬 하위 기간 — 기존 3년 창. 계속 같이 보고한다.
SUB_START = "2023-10-04"
HORIZONS = (5, 10, 20)
N_PICKS = 5
# [S5] 종목당 비중. 운영 calibration.DEFAULT_POSITION_PCT(5.0%)와 맞춘다.
#   백테스트는 1/N_PICKS = 20%였고 운영은 5%였다 — 4배 차이.
#   exposure = WEIGHT × N_PICKS = 25%. 나머지 75%는 현금(수익 0%, 사전 등록 §3).
WEIGHT = 0.05
BASE_SIGNAL = SignalParams()
BASE_COST = CostModel()
SENS_GRID = {
    "price_change_threshold": [0.018, 0.024, 0.030, 0.036, 0.042],
    "volume_ratio_threshold": [0.9, 1.2, 1.5, 1.8, 2.1],
    "ma_short": [12, 16, 20, 24, 28],
    "ma_long":  [36, 48, 60, 72, 84],
    "n_picks":  [3, 4, 5, 6, 7],
}


def param_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]


def record_run(run_id: str, params: dict, period: str, summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "timestamp", "param_hash", "period", "horizon", "n_picks", "cagr", "bench_cagr",
                        "excess_cagr", "mdd", "sharpe", "n_trades", "win_rate", "note"])
        w.writerow([run_id, datetime.now().isoformat(timespec="seconds"), param_hash(params), period,
                    params.get("horizon"), params.get("n_picks"),
                    _r(summary.get("cagr")), _r(summary.get("bench_cagr")), _r(summary.get("excess_cagr")),
                    _r(summary.get("mdd")), _r(summary.get("sharpe")), summary.get("n_trades"),
                    _r(summary.get("win_rate")), summary.get("note", "")])


def _r(x):
    return None if x is None else round(float(x), 5)


class Context:
    """데이터 한 번 로드 → 여러 실행이 공유."""

    def __init__(self, refresh: bool = False):
        t0 = time.time()
        pool = D.apply_universe_filters(D.fetch_pool(refresh=refresh))
        self.pool = pool
        self.prices = D.fetch_prices(pool, FETCH_START, END, refresh=refresh)
        self.bench = D.fetch_benchmark(FETCH_START, END, refresh=refresh)
        self.universes = D.monthly_universes(pool, self.prices["adj_close"])
        self._frames_cache: dict[str, tuple] = {}
        logger.info(f"[backtest] 데이터 준비 {time.time()-t0:.1f}s — 풀 {len(pool)}, 가격 {self.prices['adj_close'].shape}, 유니버스 월 {len(self.universes)}")

    def frames(self, sp: SignalParams):
        k = sp.key()
        if k not in self._frames_cache:
            fr = compute_signal_frames(self.prices["adj_close"], self.prices["volume"], sp)
            self._frames_cache[k] = (fr, rule_buy(fr, sp))
        return self._frames_cache[k]

    def run(self, start: str, end: str, horizon: int, n_picks: int = N_PICKS,
            sp: SignalParams = BASE_SIGNAL, costs: CostModel = BASE_COST,
            random_pick: bool = False, seed: int | None = None,
            weight: float | None = None):
        fr, buy = self.frames(sp)
        return simulate(self.prices, self.bench, self.universes, buy, fr, start, end,
                        n_picks=n_picks, horizon=horizon, costs=costs, random_pick=random_pick,
                        seed=seed, weight=weight)


def _params(horizon: int, n_picks: int = N_PICKS, sp: SignalParams = BASE_SIGNAL, costs: CostModel = BASE_COST,
            **extra) -> dict:
    # [S5] weight/exposure를 넣어 param_hash가 비중 변경 전후를 구분하게 한다.
    return {"horizon": horizon, "n_picks": n_picks, **asdict(sp), "costs": costs.key(),
            "weight": WEIGHT, "exposure": WEIGHT * n_picks, "start": START, **extra}


# ── R1 ─────────────────────────────────────────────────────────────────

def run_r1(ctx: Context) -> dict:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    labels = C.regime_labels(ctx.bench["close"])
    all_out = {}
    for h in HORIZONS:
        out: dict = {"horizon": h, "params": _params(h)}
        full = ctx.run(START, END, h, weight=WEIGHT)
        out["full"] = M.summarize(full)
        out["in_sample"] = M.summarize(ctx.run(START, IS_END, h, weight=WEIGHT))
        out["out_of_sample"] = M.summarize(ctx.run(OOS_START, END, h, weight=WEIGHT))
        # [S5 §5] PRE = 이번에 새로 추가된 7.55년(한 번도 안 봤다) / SUB = 기존 3년 창
        out["pre"] = M.summarize(ctx.run(START, SUB_START, h, weight=WEIGHT))
        out["sub"] = M.summarize(ctx.run(SUB_START, END, h, weight=WEIGHT))
        out["walk_forward"] = C.walk_forward(lambda s, e: ctx.run(s, e, h, weight=WEIGHT), START, END)
        out["regime"] = C.regime_split(full.trades, labels)
        oos_trades = full.trades[full.trades["signal_date"] >= pd.Timestamp(OOS_START)]
        out["regime_oos"] = C.regime_split(oos_trades, labels)
        out["cost_sensitivity"] = {}
        for m in (1, 2, 3):
            cm = BASE_COST.with_slippage_mult(m)
            out["cost_sensitivity"][f"slippage_x{m}"] = {
                "full_excess_cagr": M.summarize(ctx.run(START, END, h, costs=cm, weight=WEIGHT))["excess_cagr"],
                "oos_excess_cagr": M.summarize(ctx.run(OOS_START, END, h, costs=cm, weight=WEIGHT))["excess_cagr"],
                "pre_excess_cagr": M.summarize(ctx.run(START, SUB_START, h, costs=cm, weight=WEIGHT))["excess_cagr"],
            }
        out["significance_full"] = C.significance(full.trades)
        out["significance_oos"] = C.significance(oos_trades)
        full.trades.to_csv(RES_DIR / f"R1_H{h}_trades.csv", index=False, encoding="utf-8")
        (RES_DIR / f"R1_H{h}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        for period, s in (("full", out["full"]), ("IS", out["in_sample"]), ("OOS", out["out_of_sample"]),
                          ("PRE", out["pre"]), ("SUB", out["sub"])):
            record_run(f"R1s5_H{h}_{period}", _params(h), period, s)
        all_out[h] = out
        print(f"[R1] H={h}  PRE B1 {M.fmt_pct(out['pre']['excess_cagr_matched'])} / B2 {M.fmt_pct(out['pre']['excess_cagr'])}"
              f" | SUB B1 {M.fmt_pct(out['sub']['excess_cagr_matched'])} / B2 {M.fmt_pct(out['sub']['excess_cagr'])}"
              f" | full B1 {M.fmt_pct(out['full']['excess_cagr_matched'])} / B2 {M.fmt_pct(out['full']['excess_cagr'])}"
              f" | trades {out['full']['n_trades']}"
              f" | WF {out['walk_forward']['positive']}+/{out['walk_forward']['negative']}-")
    return all_out


# ── R2 ─────────────────────────────────────────────────────────────────

def run_r2(ctx: Context, n_random: int = 1000) -> dict:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for h in HORIZONS:
        strat_full = M.summarize(ctx.run(START, END, h, weight=WEIGHT))["excess_cagr"]
        strat_oos = M.summarize(ctx.run(OOS_START, END, h, weight=WEIGHT))["excess_cagr"]
        rc_full = C.random_control(lambda s: ctx.run(START, END, h, random_pick=True, seed=s, weight=WEIGHT), strat_full, n_random)
        rc_oos = C.random_control(lambda s: ctx.run(OOS_START, END, h, random_pick=True, seed=10_000 + s, weight=WEIGHT), strat_oos, n_random)
        out[h] = {"full": rc_full, "oos": rc_oos}
        record_run(f"R2s5_H{h}_random{n_random}_full", _params(h, random=True, n_random=n_random), "full",
                   {"excess_cagr": rc_full.get("random_p50"), "note": f"random p50; strategy pct {rc_full.get('strategy_percentile')}"})
        record_run(f"R2s5_H{h}_random{n_random}_OOS", _params(h, random=True, n_random=n_random), "OOS",
                   {"excess_cagr": rc_oos.get("random_p50"), "note": f"random p50; strategy pct {rc_oos.get('strategy_percentile')}"})
        print(f"[R2] H={h}  전략 백분위 full {rc_full.get('strategy_percentile'):.1f}% / OOS {rc_oos.get('strategy_percentile'):.1f}%")
    (RES_DIR / "R2_random.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


# ── R3 ─────────────────────────────────────────────────────────────────

def run_r3(ctx: Context) -> dict:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for h in HORIZONS:
        base = {"price_change_threshold": BASE_SIGNAL.price_change_threshold,
                "volume_ratio_threshold": BASE_SIGNAL.volume_ratio_threshold,
                "ma_short": BASE_SIGNAL.ma_short, "ma_long": BASE_SIGNAL.ma_long, "n_picks": N_PICKS}

        def run_with(params: dict, period=("OOS",)):
            sp = SignalParams(price_change_threshold=params["price_change_threshold"],
                              volume_ratio_threshold=params["volume_ratio_threshold"],
                              ma_short=int(params["ma_short"]), ma_long=int(params["ma_long"]))
            return ctx.run(OOS_START, END, h, n_picks=int(params["n_picks"]), sp=sp, weight=WEIGHT)

        def run_full(params: dict):
            sp = SignalParams(price_change_threshold=params["price_change_threshold"],
                              volume_ratio_threshold=params["volume_ratio_threshold"],
                              ma_short=int(params["ma_short"]), ma_long=int(params["ma_long"]))
            return ctx.run(START, END, h, n_picks=int(params["n_picks"]), sp=sp, weight=WEIGHT)

        sens_oos = C.sensitivity(run_with, base, SENS_GRID)
        sens_full = C.sensitivity(run_full, base, SENS_GRID)
        verdict = {p: C.plateau_or_peak(rows, base[p]) for p, rows in sens_full.items()}
        out[h] = {"base": base, "oos": sens_oos, "full": sens_full, "verdict_full": verdict}
        for p, rows in sens_full.items():
            for r in rows:
                params = dict(base); params[p] = r["value"]
                record_run(f"R3_H{h}_{p}={r['value']}", _params(h, n_picks=int(params["n_picks"]),
                           sp=SignalParams(params["price_change_threshold"], params["volume_ratio_threshold"],
                                           int(params["ma_short"]), int(params["ma_long"]))), "full",
                           {"excess_cagr": r.get("excess_cagr"), "cagr": r.get("cagr"), "mdd": r.get("mdd"),
                            "n_trades": r.get("n_trades"), "note": "sensitivity"})
        print(f"[R3] H={h}  " + ", ".join(f"{p}:{v}" for p, v in verdict.items()))
    (RES_DIR / "R3_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--r1", action="store_true")
    ap.add_argument("--r2", action="store_true")
    ap.add_argument("--r3", action="store_true")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ctx = Context(refresh=a.refresh)
    if a.all or a.r1:
        run_r1(ctx)
    if a.all or a.r2:
        run_r2(ctx, a.n_random)
    if a.all or a.r3:
        run_r3(ctx)


if __name__ == "__main__":
    main()
