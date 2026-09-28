"""
src/evaluation/random_control.py

G1 forward 무작위 대조군 (B-3 C3용).

매일 실행 때 그날 스크리닝 유니버스를 스냅샷으로 저장하고, 같은 유니버스에서 무작위 5종목을
N_SAMPLES(100)회 뽑아 prediction_outcomes와 같은 지평(D+5/10/20)으로 등록한다.
확정(resolve)은 outcomes._resolve_one을 그대로 써서 실제 예측과 같은 자로 잰다.

왜: 백테스트(doc/2026-09-13_backtest-and-ops.md R2)에서 "무작위 5종목도 지수에 크게 뒤진다"가
확인됐다. forward에서도 실제 선정이 무작위 분포의 어디에 있는지를 봐야 "LLM 층이 무엇을 더했는가"를
말할 수 있다. 사전 등록 §3의 판정선: 무작위 분포 상위 5% 안.
2026-09-14 결정 Q2(c): 리포트에는 지수 대비를 쓰고 **승격 판정선은 이 무작위 분포**에 건다.

시드: run_date 숫자(YYYYMMDD). 같은 날 재실행해도 같은 표본 → 멱등.
판단 로직에는 관여하지 않는다(기록 전용).
"""

from __future__ import annotations

import logging
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.evaluation import outcomes as oc

logger = logging.getLogger(__name__)

N_SAMPLES = 100
N_PICKS   = 5


def init_tables(db_path: Path | None = None) -> None:
    with sqlite3.connect(db_path or oc.DB_PATH) as conn:
        conn.executescript("""
            -- 유니버스의 진실의 원천 (2026-09-14 결정 3-4b).
            --   source='build' : 분기 재구축이 만든 판. load_universe()가 이 중 최신을 읽는다.
            --   source='daily' : 그날 실제로 쓴 목록. 대조군 표본 추출에 쓴다.
            -- 둘을 한 표에 두는 이유: "갱신일"과 "그날 쓴 목록"이 같은 축(run_date)에 있어야
            -- 성과를 갱신 구간별로 나눠 볼 수 있다 (결정 Q5b).
            CREATE TABLE IF NOT EXISTS universe_snapshots (
                run_date   TEXT NOT NULL,
                ticker     TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT 'daily',
                created_at TEXT NOT NULL,
                UNIQUE (run_date, ticker)
            );
            CREATE TABLE IF NOT EXISTS random_control_outcomes (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                pred_date        TEXT NOT NULL,
                sample_id        INTEGER NOT NULL,
                ticker           TEXT NOT NULL,
                horizon_days     INTEGER NOT NULL,
                price_at_pred    REAL,
                price_at_res     REAL,
                raw_return       REAL,
                kospi_return     REAL,
                alpha_vs_kospi   REAL,
                resolution_date  TEXT,
                net_return       REAL,                       -- 결정 Q7: 비용 차감
                benchmark        TEXT,                       -- CR-7: 이 행을 무엇과 비교했나
                status           TEXT NOT NULL DEFAULT 'pending',
                created_at       TEXT NOT NULL,
                updated_at       TEXT,
                UNIQUE (pred_date, sample_id, ticker, horizon_days)
            );
            CREATE INDEX IF NOT EXISTS idx_rc_status ON random_control_outcomes (status, horizon_days);
        """)
        # 기존 DB에 열 추가 (outcomes._migrate와 같은 관례)
        for _ddl in ("ALTER TABLE random_control_outcomes ADD COLUMN benchmark TEXT",
                     "ALTER TABLE random_control_outcomes ADD COLUMN net_return REAL",
                     "ALTER TABLE universe_snapshots ADD COLUMN source TEXT DEFAULT 'daily'"):
            try:
                conn.execute(_ddl)
            except sqlite3.OperationalError:
                pass
        try:
            conn.execute(
                "UPDATE random_control_outcomes SET benchmark=? WHERE benchmark IS NULL AND status='resolved'",
                (oc.LEGACY_BENCHMARK,),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()


def record_universe_snapshot(run_date: str, tickers: list[str], db_path: Path | None = None,
                             source: str = "daily") -> int:
    """유니버스를 저장. 멱등(INSERT OR IGNORE). 반환: 새로 들어간 행 수.

    source='build'는 분기 재구축이 만든 판이고, load_universe()가 이 중 최신을 읽는다.
    같은 날짜에 build가 먼저 들어가면 그날의 daily는 INSERT OR IGNORE로 무시된다 —
    의도한 동작이다(재구축한 날의 판은 build로 남아야 한다).
    """
    if not tickers:
        return 0
    init_tables(db_path)
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path or oc.DB_PATH) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO universe_snapshots (run_date, ticker, source, created_at) "
            "VALUES (?,?,?,?)",
            [(run_date, t, source, now) for t in sorted(set(tickers))],
        )
        conn.commit()
        return conn.total_changes - before


def latest_built_universe(db_path: Path | None = None) -> tuple[Optional[str], list[str]]:
    """가장 최근 분기 재구축(source='build') 판. (기준일, 종목목록). 없으면 (None, [])."""
    path = db_path or oc.DB_PATH
    if not Path(path).exists():
        return None, []
    with sqlite3.connect(path) as conn:
        try:
            row = conn.execute(
                "SELECT MAX(run_date) FROM universe_snapshots WHERE source='build'").fetchone()
        except sqlite3.OperationalError:
            return None, []
        if not row or not row[0]:
            return None, []
        run_date = row[0]
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM universe_snapshots WHERE run_date=? AND source='build' ORDER BY ticker",
            (run_date,))]
    return run_date, tickers


def build_dates(db_path: Path | None = None) -> list[str]:
    """유니버스가 갱신된 날짜 목록 (결정 Q5b: 성과를 갱신 구간별로 나눠 보기 위해)."""
    path = db_path or oc.DB_PATH
    if not Path(path).exists():
        return []
    with sqlite3.connect(path) as conn:
        try:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT run_date FROM universe_snapshots WHERE source='build' ORDER BY run_date")]
        except sqlite3.OperationalError:
            return []


def register_random_control(run_date: str, tickers: list[str],
                            n_samples: int = N_SAMPLES, n_picks: int = N_PICKS,
                            db_path: Path | None = None) -> int:
    """무작위 n_picks종목 × n_samples 표본 × HORIZONS를 pending으로 등록. 반환: 새 행 수."""
    pool = sorted(set(tickers))
    if len(pool) < n_picks:
        logger.warning(f"[random_control] 유니버스 {len(pool)}종목 < {n_picks} — 등록 생략")
        return 0
    init_tables(db_path)
    rng = random.Random(int(run_date.replace("-", "")))      # 날짜 고정 시드 → 재실행해도 같은 표본
    now = datetime.now().isoformat()
    rows = []
    for s in range(n_samples):
        for t in rng.sample(pool, n_picks):
            for h in oc.HORIZONS:
                rows.append((run_date, s, t, h, now))
    with sqlite3.connect(db_path or oc.DB_PATH) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO random_control_outcomes "
            "(pred_date, sample_id, ticker, horizon_days, status, created_at) VALUES (?,?,?,?,'pending',?)",
            rows,
        )
        conn.commit()
        return conn.total_changes - before


def resolve_pending(db_path: Path | None = None) -> dict:
    """outcomes._resolve_one으로 창이 찬 행을 확정. {"resolved": n, "still_pending": m}"""
    init_tables(db_path)
    oc._KOSPI_CACHE = None
    with sqlite3.connect(db_path or oc.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        pending = [dict(r) for r in conn.execute(
            "SELECT * FROM random_control_outcomes WHERE status='pending' ORDER BY pred_date, ticker, horizon_days"
        )]
    now = datetime.now().isoformat()
    updates = []
    for row in pending:
        out = oc._resolve_one(row, db_path)
        if out is None:
            continue
        updates.append((out["price_at_pred"], out["price_at_res"], out["raw_return"], out["kospi_return"],
                        out["alpha_vs_kospi"], out["resolution_date"], out["benchmark"],
                        out["net_return"], now, row["id"]))
    if updates:
        with sqlite3.connect(db_path or oc.DB_PATH) as conn:
            conn.executemany(
                "UPDATE random_control_outcomes SET price_at_pred=?, price_at_res=?, raw_return=?, "
                "kospi_return=?, alpha_vs_kospi=?, resolution_date=?, benchmark=?, net_return=?, "
                "status='resolved', updated_at=? WHERE id=?",
                updates,
            )
            conn.commit()
    return {"resolved": len(updates), "still_pending": len(pending) - len(updates)}


def summary(horizon: int, db_path: Path | None = None) -> dict:
    """지평별: 표본(sample_id)마다 평균 알파 → 분포 p5/p50/p95, 실제 예측(prediction_outcomes)의 백분위.

    같은 pred_date 집합끼리만 비교한다(실제 예측이 없는 날의 무작위 표본은 뺀다)."""
    out: dict = {"horizon": horizon, "n_days": 0, "n_samples": 0,
                 "random_p5": None, "random_p50": None, "random_p95": None,
                 "strategy_alpha": None, "strategy_percentile": None}
    path = db_path or oc.DB_PATH
    if not path.exists():
        return out
    with sqlite3.connect(path) as conn:
        try:
            strat = conn.execute(
                "SELECT pred_date, AVG(alpha_vs_kospi) FROM prediction_outcomes "
                "WHERE status='resolved' AND horizon_days=? AND alpha_vs_kospi IS NOT NULL GROUP BY pred_date",
                (horizon,)).fetchall()
            days = {d for d, _ in strat}
            if not days:
                return out
            rc = conn.execute(
                "SELECT pred_date, sample_id, AVG(alpha_vs_kospi) FROM random_control_outcomes "
                "WHERE status='resolved' AND horizon_days=? AND alpha_vs_kospi IS NOT NULL "
                "GROUP BY pred_date, sample_id", (horizon,)).fetchall()
        except sqlite3.OperationalError:
            return out
    rc = [(d, s, a) for d, s, a in rc if d in days]
    if not rc:
        return out
    by_sample: dict[int, list[float]] = {}
    for _, s, a in rc:
        by_sample.setdefault(s, []).append(a)
    sample_means = sorted(sum(v) / len(v) for v in by_sample.values())
    strat_alpha = sum(a for _, a in strat) / len(strat)
    n = len(sample_means)
    def q(p: float) -> float:
        return sample_means[min(n - 1, int(p * n))]
    out.update({
        "n_days": len(days), "n_samples": n,
        "random_p5": q(0.05), "random_p50": q(0.50), "random_p95": q(0.95),
        "strategy_alpha": strat_alpha,
        "strategy_percentile": 100.0 * sum(m < strat_alpha for m in sample_means) / n,
    })
    return out


def format_summary(db_path: Path | None = None) -> str:
    lines = [f"[random_control] 실제 선정 vs 같은 유니버스 무작위 5종목 (일평균 알파, {oc.BENCHMARK_LABEL} 대비)"]
    def pct(x):
        return "n/a" if x is None else f"{x*100:+.2f}%"
    for h in oc.HORIZONS:
        s = summary(h, db_path)
        if not s["n_days"]:
            lines.append(f"  D+{h}: 비교 가능한 날 0일")
            continue
        lines.append(f"  D+{h}: {s['n_days']}일  실제 {pct(s['strategy_alpha'])}  "
                     f"무작위 p5/p50/p95 {pct(s['random_p5'])}/{pct(s['random_p50'])}/{pct(s['random_p95'])}  "
                     f"백분위 {s['strategy_percentile']:.0f}% (n={s['n_samples']})")
    return "\n".join(lines)
