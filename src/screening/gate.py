"""
게이트 — 규칙이 살아나는 조건. 스펙 §2.

판정은 리밸런스 날짜 단위 거래 초과(같은 창 벤치마크 대비, 비용 차감)를 전체 기간에서 잰다.
OOS는 붕괴 확인에만 쓴다 — OOS 14개월로 유의성을 요구하면 연 45%p가 필요하다(§2-4).
THRESHOLDS는 게이트를 한 번 돌린 뒤 바꾸지 않는다. 바꾸면 그 뒤 규칙은 다른 자로 잰 것이다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.backtest import controls as C
from src.evaluation.backtest.engine import CostModel, simulate
from src.screening.gate_data import pool_hash, universe_ew_bench_r, with_bench
from src.screening.rules import registry as R
from src.screening.rules.spec import RuleSpec, bonferroni_t, parse_spec
from src.screening.signals import SIGNAL_NEEDS, apply_rule

THRESHOLDS = {
    "min_history_months":   18,
    "min_dates":            60,
    "min_oos_dates":        10,
    "oos_t_floor":          -1.0,
    "wf_min_windows":       5,
    "wf_positive_ratio":    0.60,
    "control_top_pct_base": 5.0,       # 상위 5%/m
}
CHECK_ORDER = ["min_history", "min_dates", "min_oos_dates",
               "mean_index", "t_index", "ci_index", "mean_uni", "t_uni", "ci_uni",
               "oos_index", "oos_uni", "control_index", "control_uni", "wf_index", "wf_uni"]


def _num(d: dict, key: str) -> float:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def verdict(sig_index: dict, sig_uni: dict, oos_index: dict, oos_uni: dict,
            ctrl_index: dict, ctrl_uni: dict, wf_index: dict, wf_uni: dict,
            m: int, history_months: float) -> dict:
    T = THRESHOLDS
    m = max(1, int(m))
    t_star = bonferroni_t(m)
    need_pct = 100.0 - T["control_top_pct_base"] / m
    c: dict[str, dict] = {}
    c["min_history"] = {"ok": history_months >= T["min_history_months"], "months": history_months}
    c["min_dates"] = {"ok": int(sig_index.get("n_dates", 0)) >= T["min_dates"]
                      and int(sig_uni.get("n_dates", 0)) >= T["min_dates"],
                      "n_dates": int(sig_index.get("n_dates", 0))}
    c["min_oos_dates"] = {"ok": int(oos_index.get("n_dates", 0)) >= T["min_oos_dates"],
                          "n_dates": int(oos_index.get("n_dates", 0))}
    for name, sig in (("index", sig_index), ("uni", sig_uni)):
        mean, t = _num(sig, "mean"), _num(sig, "t")
        lo = float((sig.get("boot_ci95") or [float("nan")])[0])
        c[f"mean_{name}"] = {"ok": mean > 0, "mean": mean}
        c[f"t_{name}"] = {"ok": t >= t_star, "t": t, "t_star": t_star}
        c[f"ci_{name}"] = {"ok": lo > 0, "ci_low": lo}
    for name, oos in (("index", oos_index), ("uni", oos_uni)):
        t = _num(oos, "t")
        c[f"oos_{name}"] = {"ok": (not math.isnan(t)) and t > T["oos_t_floor"], "t": t}
    for name, ctrl in (("index", ctrl_index), ("uni", ctrl_uni)):
        pct = _num(ctrl, "strategy_percentile")
        c[f"control_{name}"] = {"ok": pct >= need_pct, "percentile": pct, "need": need_pct}
    for name, wf in (("index", wf_index), ("uni", wf_uni)):
        n, ratio = int(wf.get("n", 0)), wf.get("positive_ratio")
        c[f"wf_{name}"] = {"ok": n >= T["wf_min_windows"] and ratio is not None and ratio >= T["wf_positive_ratio"],
                           "n": n, "positive_ratio": ratio}
    failed = [k for k in CHECK_ORDER if not c[k]["ok"]]
    return {"passed": not failed, "t_star": t_star, "m": m, "checks": c, "failed": failed}


# ── 실행부 (스펙 §2-3) ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
DATA_START = "2015-09-01"
HISTORY_START = "2016-01-01"
GLOBAL_OOS_START = "2025-07-01"


def _build_inputs(spec: RuleSpec, data_start: str, end: str, cache_dir: Optional[Path]) -> dict:
    """외부 데이터를 모아 게이트 입력으로. 테스트는 이 함수를 통째로 바꾼다.
    풀에 우선주·바이오 필터를 미리 걸지 않는다 — 순위는 전 종목 기준(라이브 pykrx와 같다),
    제외는 규칙의 exclude가 한다."""
    from src.evaluation.backtest import data as D
    from src.screening.gate_data import (GATE_KOSDAQ_PAGES, GATE_KOSPI_PAGES, monthly_snapshots,
                                         monthly_universes_for)
    cd = Path(cache_dir) if cache_dir else D.CACHE_DIR
    pool = D.fetch_pool(cd, kospi_pages=GATE_KOSPI_PAGES, kosdaq_pages=GATE_KOSDAQ_PAGES, stocks_only=True)
    prices = D.fetch_prices(pool, data_start, end, cd, tag=pool_hash(pool))
    unis = monthly_universes_for(spec.universe, monthly_snapshots(pool, prices))
    extra: dict = {}
    if SIGNAL_NEEDS.get(spec.signal["name"]):
        from src.data.flow_history import load_flow_history
        codes = sorted({c for members in unis.values() for c in members})
        fl = load_flow_history(codes, data_start, end, cache_dir=cd / "flow")
        if len(fl["failed"]) > 0.10 * max(1, len(codes)):
            raise RuntimeError(f"수급 이력 실패 {len(fl['failed'])}/{len(codes)}종목 — 게이트 중단")
        extra = {"foreign_net": fl["foreign_net"], "flow_volume": fl["flow_volume"]}
    return {"pool": pool, "prices": prices, "bench_index": D.fetch_benchmark(data_start, end, cd),
            "universes": unis, "extra": extra}


def _frames(inp: dict) -> dict:
    p = inp["prices"]
    fr = {"adj_close": p["adj_close"], "open": p["open"], "close": p["close"], "volume": p["volume"]}
    fr.update(inp.get("extra") or {})
    return fr


def _selector_for(spec: RuleSpec, frames: dict):
    """선택자 하나에 난수 생성기 하나 — 날짜마다 이어 써서 null_random이 매번 다른 종목을 고른다."""
    rng = None
    if spec.signal["name"] == R.NULL_RULE_SIGNAL:
        rng = np.random.default_rng(int((spec.signal.get("params") or {}).get("seed", 0)))

    def sel(t: pd.Timestamp, universe: list[str], tradable: set[str]) -> list[str]:
        return [c for c, _ in apply_rule(spec, t, [u for u in universe if u in tradable], frames, rng=rng)]
    return sel


def _sim(spec: RuleSpec, inp: dict, start: str, end: str, random_pick: bool = False,
         seed: Optional[int] = None, frames: Optional[dict] = None):
    """라이브와 같은 청산(스펙 §3-5): t+1 시가 진입, t+H 종가 청산, 손절·익절 없음. 벤치마크 KOSPI200."""
    sel = None if random_pick else _selector_for(spec, frames if frames is not None else _frames(inp))
    return simulate(inp["prices"], inp["bench_index"], inp["universes"], buy_mask=None, frames=None,
                    start=start, end=end, n_picks=spec.n_picks, horizon=spec.horizon,
                    costs=CostModel(), random_pick=random_pick, seed=seed, selector=sel)


def _control_key(spec: RuleSpec, inp: dict, start: str, end: str, n_random: int) -> str:
    body = {"universe": spec.universe, "horizon": spec.horizon, "n_picks": spec.n_picks,
            "start": start, "end": end, "pool": pool_hash(inp["pool"]), "n_random": n_random,
            "costs": asdict(CostModel())}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _control_values(spec: RuleSpec, inp: dict, start: str, end: str, n_random: int,
                    cache_dir: Optional[Path], bench_r_uni: pd.Series) -> dict:
    """무작위 n_random회의 날짜 평균 초과 (KOSPI200 기준, 유니버스 EW 기준). 키가 같으면 캐시를 읽는다.
    무작위 선정은 규칙의 필터·신호를 쓰지 않으니 같은 유니버스·H·N·기간이면 분포가 같다."""
    from src.evaluation.backtest.data import CACHE_DIR
    f = Path(cache_dir or CACHE_DIR) / f"control_{_control_key(spec, inp, start, end, n_random)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    vals: dict[str, list[float]] = {"index": [], "uni": []}
    for s in range(n_random):
        tr = _sim(spec, inp, start, end, random_pick=True, seed=s).trades
        vals["index"].append(C.date_mean(tr))
        vals["uni"].append(C.date_mean(with_bench(tr, bench_r_uni)))
        if (s + 1) % 100 == 0:
            logger.info(f"[gate] 대조군 {s + 1}/{n_random}")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(vals), encoding="utf-8")
    return vals


def _months_between(a: str, b: str) -> float:
    da, dbb = pd.Timestamp(a), pd.Timestamp(b)
    return (dbb.year - da.year) * 12 + (dbb.month - da.month) + (dbb.day - da.day) / 30.0


def run_gate(rule_id: str, *, start: str = HISTORY_START, end: Optional[str] = None,
             oos_start: str = GLOBAL_OOS_START, n_random: int = 1000,
             db_path: Optional[Path] = None, cache_dir: Optional[Path] = None,
             result_dir: Path = Path("doc/backtest/results")) -> dict:
    rule = R.get_rule(rule_id, db_path)
    if rule is None:
        raise ValueError(f"규칙 없음: {rule_id}")
    if rule["status"] != "draft":
        raise ValueError(f"draft만 게이트를 돈다 (현재 {rule['status']})")
    spec = parse_spec(rule["spec"])
    end = end or date.today().isoformat()
    m = 1 if spec.signal["name"] == R.NULL_RULE_SIGNAL else R.fix_quarter_m(rule["quarter"], db_path)
    data_start = min(DATA_START, (pd.Timestamp(start) - pd.DateOffset(months=4)).date().isoformat())
    inp = _build_inputs(spec, data_start, end, cache_dir)

    first = max(inp["prices"]["adj_close"].dropna(how="all").index[0], pd.Timestamp(start))
    hist = _months_between(str(first.date()), end)
    if first > pd.Timestamp(oos_start):                      # 데이터가 전역 경계 뒤에 시작 → 2/3 : 1/3
        oos_start = str((first + (pd.Timestamp(end) - first) * 2 / 3).date())

    full = _sim(spec, inp, start, end)
    bench_r_uni = universe_ew_bench_r(inp["prices"], inp["universes"], full.periods)
    trades = {"index": full.trades, "uni": with_bench(full.trades, bench_r_uni)}
    rand = _control_values(spec, inp, start, end, n_random, cache_dir, bench_r_uni)

    sig, oos, ctrl, wf, summ = {}, {}, {}, {}, {}
    for name, tr in trades.items():
        sig[name] = C.significance_by_date(tr)
        oos_tr = tr[pd.to_datetime(tr["signal_date"]) >= pd.Timestamp(oos_start)] if len(tr) else tr
        oos[name] = C.significance_by_date(oos_tr)
        me = sig[name].get("mean", float("nan"))
        ctrl[name] = C.control_percentile(rand[name], me)
        wf[name] = C.walk_forward_from_trades(tr, start, end)
        summ[name] = {"n_trades": int(len(tr)), "n_dates": sig[name].get("n_dates", 0), "mean_excess": me}

    v = verdict(sig["index"], sig["uni"], oos["index"], oos["uni"], ctrl["index"], ctrl["uni"],
                wf["index"], wf["uni"], m=m, history_months=hist)
    out = {"rule_id": rule_id, "spec_hash": rule["spec_hash"], "quarter": rule["quarter"],
           "period": {"start": start, "end": end, "oos_start": oos_start, "first_bar": str(first.date())},
           "summary": summ, "significance": sig, "oos": oos, "control": ctrl,
           "walk_forward": wf, **v}
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"gate_{rule_id}_{rule['spec_hash']}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    R.transition(rule_id, "gated" if v["passed"] else "rejected",
                 reason="gate pass" if v["passed"] else "gate fail: " + ",".join(v["failed"]),
                 metrics=out, db_path=db_path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="스크리닝 규칙 게이트 (스펙 §2)")
    ap.add_argument("rule_id", nargs="?")
    ap.add_argument("--all-draft", action="store_true")
    ap.add_argument("--register-dir", default=None)
    ap.add_argument("--quarter", default=None)
    ap.add_argument("--declare-m", type=int, default=None, help="첫 게이트 전에 --quarter의 m을 선언")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if a.declare_m is not None:
        if not a.quarter:
            ap.error("--declare-m에는 --quarter가 필요하다")
        R.declare_quarter_m(a.quarter, a.declare_m)
        print(f"{a.quarter} m={a.declare_m} 선언 (t*={bonferroni_t(a.declare_m):.2f})")
        return
    if a.register_dir:
        q = a.quarter or R.quarter_of(date.today())
        print("등록:", R.register_from_dir(Path(a.register_dir), quarter=q), "분기", q)
        return
    ids = [r["rule_id"] for r in R.list_rules("draft")] if a.all_draft else [a.rule_id]
    for rid in ids:
        out = run_gate(rid, end=a.end, n_random=a.n_random)
        s = out["summary"]["uni"]
        print(f"{rid}: {'PASS' if out['passed'] else 'FAIL ' + ','.join(out['failed'])}  "
              f"t*={out['t_star']:.2f} m={out['m']}  날짜 {s['n_dates']}  "
              f"평균초과(유니버스EW) {100 * s['mean_excess']:+.2f}%p")


if __name__ == "__main__":
    main()
