"""
controls.py — 과적합 검사 ③~⑧ (사전 등록 §5).

  walk_forward      6개월 창을 3개월씩 밀며 각 창 초과수익 부호
  random_control    같은 유니버스·같은 N·같은 날짜로 무작위 선정 K회 → 전략의 백분위
  sensitivity       one-at-a-time 그리드 (고원 vs 봉우리)
  regime_split      KOSPI200 60거래일 수익률로 강세/약세/횡보 → 국면별 거래 초과수익
  cost_sensitivity  슬리피지 ×1/×2/×3
  significance      거래별 초과수익 t-검정 + 부트스트랩 95% CI
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.backtest import metrics as M


def walk_forward_windows(start: str, end: str, window_months: int = 6, step_months: int = 3) -> list[tuple[str, str]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = s
    while True:
        w_end = cur + pd.DateOffset(months=window_months) - pd.Timedelta(days=1)
        if w_end > e:
            break
        out.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = cur + pd.DateOffset(months=step_months)
    return out


def walk_forward(run: Callable[[str, str], object], start: str, end: str) -> dict:
    rows = []
    for ws, we in walk_forward_windows(start, end):
        try:
            res = run(ws, we)
            s = M.summarize(res)
            rows.append({"start": ws, "end": we, "excess_cagr": s["excess_cagr"], "cagr": s["cagr"],
                         "bench_cagr": s["bench_cagr"], "n_trades": s["n_trades"], "mdd": s["mdd"]})
        except Exception as e:
            rows.append({"start": ws, "end": we, "error": str(e)[:80]})
    df = pd.DataFrame(rows)
    ex = df["excess_cagr"].dropna() if "excess_cagr" in df else pd.Series(dtype=float)
    pos, neg = int((ex > 0).sum()), int((ex <= 0).sum())
    return {"windows": rows, "n": int(len(ex)), "positive": pos, "negative": neg,
            "sign_consistency": (max(pos, neg) / len(ex)) if len(ex) else None}


def random_control(run_random: Callable[[int], object], strategy_excess: float, n_runs: int = 1000, seed0: int = 0) -> dict:
    vals = []
    for s in range(n_runs):
        try:
            res = run_random(seed0 + s)
            v = M.summarize(res)["excess_cagr"]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        except Exception:
            continue
    arr = np.array(vals, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    pct = float((arr < strategy_excess).mean() * 100.0)
    return {"n": int(arr.size), "strategy_excess_cagr": strategy_excess,
            "random_mean": float(arr.mean()), "random_std": float(arr.std()),
            "random_p5": float(np.percentile(arr, 5)), "random_p50": float(np.percentile(arr, 50)),
            "random_p95": float(np.percentile(arr, 95)), "strategy_percentile": pct,
            "pass_top5pct": pct >= 95.0}


def sensitivity(run_with: Callable[[dict], object], base: dict, grid: dict[str, list]) -> dict:
    """grid = {param: [values...]} one-at-a-time. 반환: {param: [(value, excess_cagr, n_trades), ...]}"""
    out = {}
    for p, values in grid.items():
        rows = []
        for v in values:
            params = dict(base); params[p] = v
            try:
                s = M.summarize(run_with(params))
                rows.append({"value": v, "excess_cagr": s["excess_cagr"], "cagr": s["cagr"], "n_trades": s["n_trades"], "mdd": s["mdd"]})
            except Exception as e:
                rows.append({"value": v, "error": str(e)[:80]})
        out[p] = rows
    return out


def plateau_or_peak(rows: list[dict], base_value) -> str:
    """기준값 결과가 이웃보다 눈에 띄게(0.05 초과) 높고 이웃끼리는 비슷하면 '봉우리', 아니면 '고원'."""
    vals = [(r["value"], r.get("excess_cagr")) for r in rows if r.get("excess_cagr") is not None]
    if len(vals) < 3:
        return "판정불가"
    vals.sort(key=lambda x: x[0])
    xs = [v for _, v in vals]
    base = next((v for k, v in vals if k == base_value), None)
    if base is None:
        return "판정불가"
    others = [v for k, v in vals if k != base_value]
    if base - max(others) > 0.05:
        return "봉우리(과적합 의심)"
    diffs = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    return "고원" if max(diffs) <= 0.10 else "불규칙"


def regime_labels(bench_close: pd.Series, lookback: int = 60, thr: float = 0.05) -> pd.Series:
    r = bench_close / bench_close.shift(lookback) - 1.0
    lab = pd.Series("횡보", index=bench_close.index)
    lab[r > thr] = "강세"
    lab[r < -thr] = "약세"
    lab[r.isna()] = "n/a"
    return lab


def regime_split(trades: pd.DataFrame, labels: pd.Series) -> dict:
    if trades is None or trades.empty:
        return {}
    t = trades.copy()
    t["regime"] = t["signal_date"].map(labels).fillna("n/a")
    out = {}
    for g, d in t.groupby("regime"):
        ex = d["excess"].dropna()
        out[g] = {"n": int(len(d)), "avg_excess": float(ex.mean()) if len(ex) else None,
                  "win_rate": float((d["r_net"] > 0).mean()), "avg_r_net": float(d["r_net"].mean())}
    return out


def significance(trades: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    ex = trades["excess"].dropna().astype(float).values if (trades is not None and not trades.empty) else np.array([])
    n = int(ex.size)
    if n < 2:
        return {"n": n, "verdict": "표본 부족"}
    mean, sd = float(ex.mean()), float(ex.std(ddof=1))
    t_stat = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    except Exception:
        from math import erfc
        p = float(erfc(abs(t_stat) / math.sqrt(2)))
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(ex, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    if n < 100:
        verdict = "표본 부족 (<100건) — 검정 결과보다 앞선다"
    elif p <= 0.05 or (lo > 0 or hi < 0):
        verdict = "유의 (p≤0.05 또는 CI가 0 제외)"
    else:
        verdict = "구분 불가 (p>0.05, CI가 0 포함)"
    return {"n": n, "mean_excess": mean, "t": t_stat, "p": p, "boot_ci95": [lo, hi], "verdict": verdict}


# ── 게이트용 날짜 단위 통제 (스펙 §2-2·§2-4, 2026-09-23) ─────────────────
# 기존 walk_forward·random_control·significance는 과거 R1 재현용으로 그대로 둔다.
# 게이트는 아래 함수만 쓴다. 전부 "같은 창 벤치마크 대비 거래 초과"만 입력으로 받아
# 현금 비율·시장 방향(초과 CAGR 오염)이 끼어들 자리가 없다.

def _date_means(trades: pd.DataFrame) -> pd.Series:
    if trades is None or len(trades) == 0 or "excess" not in trades or "signal_date" not in trades:
        return pd.Series(dtype=float)
    df = trades[["signal_date", "excess"]].copy()
    df["excess"] = pd.to_numeric(df["excess"], errors="coerce")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df.dropna()
    return df.groupby("signal_date")["excess"].mean()


def significance_by_date(trades: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    """리밸런스 날짜별 평균 초과로 묶어 t·부트스트랩 CI. 보유 창이 겹치지 않아 날짜끼리 독립이다."""
    by = _date_means(trades)
    ok = trades is not None and len(trades) > 0 and "excess" in trades
    n_trades = int(pd.to_numeric(trades["excess"], errors="coerce").notna().sum()) if ok else 0
    out: dict = {"n_dates": int(len(by)), "n_trades": n_trades}
    if len(by) < 2:
        return out
    x = by.values.astype(float)
    n = len(x)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd > 0:
        t = mean / (sd / math.sqrt(n))
    else:
        t = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    out.update({"mean": mean, "sd": sd, "t": t,
                "boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]})
    return out


def walk_forward_from_trades(trades: pd.DataFrame, start: str, end: str) -> dict:
    """전체 기간 거래를 6개월 창(3개월 보폭)으로 잘라 창별 날짜 평균 초과의 부호를 센다."""
    by = _date_means(trades)
    rows = []
    for ws, we in walk_forward_windows(start, end):
        seg = by[(by.index >= pd.Timestamp(ws)) & (by.index <= pd.Timestamp(we))]
        if len(seg):
            rows.append({"start": ws, "end": we, "mean_excess": float(seg.mean()), "n_dates": int(len(seg))})
    pos = sum(1 for r in rows if r["mean_excess"] > 0)
    n = len(rows)
    return {"windows": rows, "n": n, "positive": pos, "negative": n - pos,
            "positive_ratio": (pos / n) if n else None}


def date_mean(trades: pd.DataFrame) -> float:
    by = _date_means(trades)
    return float(by.mean()) if len(by) else float("nan")


def control_percentile(random_values: list[float], strategy_value: float) -> dict:
    """무작위 선정의 날짜 평균 초과 분포에서 전략이 몇 백분위인가. 통과선(100−5/m)은 판정부가 정한다."""
    arr = np.array([v for v in random_values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "strategy_mean_excess": float(strategy_value),
            "random_mean": float(arr.mean()), "random_std": float(arr.std()),
            "random_p5": float(np.percentile(arr, 5)), "random_p50": float(np.percentile(arr, 50)),
            "random_p95": float(np.percentile(arr, 95)),
            "strategy_percentile": float((arr < strategy_value).mean() * 100.0)}
