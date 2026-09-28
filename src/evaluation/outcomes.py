"""
src/evaluation/outcomes.py

예측 → 실현 수익 귀속 (G1). D+5 / D+10 / D+20 세 지평(1주·2주·4주), KOSPI200(069500.KS) 대비 알파.
표본 단위는 **하루**다 (2026-09-14 결정 Q6).

[가져온 것] TradingAgents (Apache-2.0) tradingagents/agents/utils/memory.py +
  graph/trading_graph.py:_fetch_returns 의 pending→resolved 패턴:
    - 예측은 pending으로 등록되고, 보유 창(holding_days)이 다 찰 때까지 미확정으로 남는다
      ("Require the full holding window… rather than settling on a premature partial return", #1169)
    - 확정되면 raw_return / alpha(raw - benchmark) / holding_days / resolution_date를 기록한다
    - resolution_date = "결과가 알려진 날" = 마지막으로 쓴 봉의 날짜 (#1251)
  TradingAgents는 벤치마크를 접미사로 자동 매핑하지만 .KS/.KQ가 없어 SPY로 떨어진다.
  여기서는 KOSPI200 ETF(069500.KS) 고정 — 백테스트와 같은 자를 쓴다 (CR-7).

[발상만 가져온 것] prism-insight (AGPL) performance_feedback: "후보(미진입) vs 실제" 분리 집계.
  → screen_candidates 테이블에 확정 종목 전부(분석된 것과 MAX_CONFIRMED_TICKERS로 잘린 것)를
    같이 등록해 컷오프 자체를 평가할 수 있게 한다.
    2026-09-21 v2부터는 유니버스 전 종목(110행/일)의 팩터값을 남긴다 — src/evaluation/screen_factors.py가 읽는다.

[형식만 참고] LLM-Trading-Lab (라이선스 없음) csv_files/ 열 구성 → export_csv.

[데이터 원천]
  - 종목 종가: ohlcv_cache (stage1a가 매일 110종목을 채운다). pykrx 재호출 없음.
  - 벤치마크: yfinance 069500.KS (auto_adjust 기본값이라 분배금 반영). 실패하면 alpha=None, raw만 기록.

[한계]
  체결가가 아니라 종가다. raw_return에는 비용이 없고, net_return이 왕복 비용을 뺀 값이다(결정 Q7).
  진입 시점 차이(백테스트는 t+1 시가, 여기는 t 종가)는 남아 있다.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH   = Path("data/mentions.db")
LEDGER_DIR = Path("data/ledger")

HORIZONS  = (5, 10, 20)      # 거래일 = 1주·2주·4주. D+1 채점(feedback_evaluator)과 별개 — 표방 포지션이 1~4주다

# 벤치마크 (CR-7, 2026-09-14 결정 Q2).
#
# [왜 바꿨나]
#   백테스트는 069500.KS(KODEX 200 ETF)로 쟀는데 여기는 ^KS11(KOSPI 전체)이었다.
#   같은 시스템의 성과를 서로 다른 자로 재면 나란히 놓을 수 없다.
#   ^KS200은 yfinance에서 2026-07-16 이후 결측이라 못 쓴다(doc/2026-09-13_change-requests.md CR-1).
#   yfinance history()는 auto_adjust=True가 기본이라 Close가 이미 분배금 조정가다.
#
# [이전 행과 무엇이 달라지나]
#   2026-09-14 이전에 확정된 행의 kospi_return은 ^KS11 수익률이다. 열 이름은 유지하되
#   각 행의 benchmark 열이 무엇으로 쟀는지 스스로 밝힌다. 전환 시점은 eval_method_log에 남는다.
BENCHMARK       = "069500.KS"
BENCHMARK_LABEL = "KOSPI200 (KODEX 200 ETF 069500.KS, 조정가)"
LEGACY_BENCHMARK = "^KS11"

# 방향 적중 판정용. D+1 채점의 ±1%와 구분하기 위해 지평별로 둔다.
HIT_THRESHOLD = {5: 0.01, 10: 0.015, 20: 0.02}

# 리포트에 성적을 수치로 내보일 최소 표본(일). 이 미만이면 "표본 N일 (최소 M)"만 쓴다.
#
# [왜 30인가]
#   calibration.MIN_SAMPLES(30건)와 같은 자리수로 맞췄다. 승격 판정선인 175거래일
#   (doc/2026-09-13_backtest-and-ops.md B-3 C1)과는 다른 값이다 — 그건 "우위가 있다"를
#   통계적으로 말하는 선이고, 이것은 "수치를 보여줘도 되는" 선이다.
MIN_DAYS_FOR_TREND = 30

# 거래 비용 (결정 Q7, 2026-09-14). raw_return은 그대로 두고 net_return을 따로 쌓는다.
#
# [왜 지금 넣나]
#   (a) 신호 생성기 전환을 판단할 때는 비용을 반드시 넣어야 한다. 그때 자를 바꾸면
#   이전 행과 비교가 안 되므로, 표본이 쌓이기 전인 지금 열을 만들어 둔다.
#
# [값의 출처]
#   src/evaluation/backtest/engine.py CostModel과 같은 값이다. 백테스트 패키지를
#   import하면 pandas·numpy가 딸려오는데 feedback.yml은 최소 의존성만 설치하므로
#   여기서는 상수를 복제한다. 어긋나지 않도록 tests/test_eval_method.py가 두 값을 대조한다.
COMMISSION   = 0.000140527   # 한국투자증권 뱅키스 온라인, KRX (2025-10-27 기준)
EXCHANGE_FEE = 0.000036396   # 유관기관 제비용
SLIPPAGE     = 0.001         # 가정 0.10% (편도)
TAX_BY_YEAR  = {2023: 0.0018, 2024: 0.0018, 2025: 0.0015, 2026: 0.0020}
DEFAULT_TAX  = 0.0020


def net_of_costs(raw_return: Optional[float], sell_date: Optional[str]) -> Optional[float]:
    """왕복 비용을 뺀 수익률. 매도 연도의 증권거래세를 적용한다.

    net = (1 + raw) × (1 − 매도비용) ÷ (1 + 매수비용) − 1
    (진입가는 비용만큼 비싸게, 청산가는 비용만큼 싸게 잡힌다)
    """
    if raw_return is None:
        return None
    year = int(sell_date[:4]) if sell_date and len(sell_date) >= 4 else None
    tax  = TAX_BY_YEAR.get(year, DEFAULT_TAX)
    buy  = COMMISSION + EXCHANGE_FEE + SLIPPAGE
    sell = COMMISSION + EXCHANGE_FEE + SLIPPAGE + tax
    return (1.0 + float(raw_return)) * (1.0 - sell) / (1.0 + buy) - 1.0


def roundtrip_cost(year: Optional[int] = None) -> float:
    """참고용 왕복 비용 근사치 (손익분기 승률 계산에 쓴다)."""
    tax = TAX_BY_YEAR.get(year, DEFAULT_TAX)
    return 2 * (COMMISSION + EXCHANGE_FEE + SLIPPAGE) + tax


# ── DDL ─────────────────────────────────────────────────────────────────

def init_outcome_tables(db_path: Path | None = None) -> None:
    with sqlite3.connect(db_path or DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                pred_date        TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                horizon_days     INTEGER NOT NULL,
                source           TEXT NOT NULL,           -- prediction | candidate
                price_at_pred    REAL,
                price_at_res     REAL,
                raw_return       REAL,
                kospi_return     REAL,
                alpha_vs_kospi   REAL,
                resolution_date  TEXT,
                status           TEXT NOT NULL DEFAULT 'pending',   -- pending | resolved
                created_at       TEXT NOT NULL,
                updated_at       TEXT,
                UNIQUE (pred_date, ticker, horizon_days)
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_status ON prediction_outcomes (status, horizon_days);

            CREATE TABLE IF NOT EXISTS screen_candidates (
                run_date   TEXT NOT NULL,
                ticker     TEXT NOT NULL,
                analyzed   INTEGER NOT NULL,     -- 1: 심층 분석함, 0: 상한으로 잘림
                created_at TEXT NOT NULL,
                UNIQUE (run_date, ticker)
            );

            -- 평가 방법이 바뀐 지점을 남긴다 (2026-09-14 결정).
            -- 이 표가 없으면 "이 수치는 어느 자로 잰 것인가"를 나중에 되짚을 수 없다.
            CREATE TABLE IF NOT EXISTS eval_method_log (
                change_key   TEXT PRIMARY KEY,   -- 'benchmark' | 'sample_unit' | 'net_return' ...
                applied_date TEXT NOT NULL,      -- 적용 시작일 (KST)
                detail       TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );
        """)
        _migrate(conn)
        conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """기존 DB에 새 열을 붙인다. 이미 있으면 조용히 넘어간다 (position_tracker와 같은 관례)."""
    for ddl in (
        # 각 행이 "무엇과 비교했는지"를 스스로 밝힌다. CR-7 전환 전 행은 ^KS11이다.
        "ALTER TABLE prediction_outcomes ADD COLUMN benchmark TEXT",
        # 결정 Q7: 비용 차감 수익률. raw_return은 그대로 보존한다.
        "ALTER TABLE prediction_outcomes ADD COLUMN net_return REAL",
        # [스크리닝 v2 §4] 유니버스 전 종목의 팩터값. 과거 행은 NULL로 남는다.
        "ALTER TABLE screen_candidates ADD COLUMN is_candidate INTEGER",
        "ALTER TABLE screen_candidates ADD COLUMN cand_rank INTEGER",
        "ALTER TABLE screen_candidates ADD COLUMN score REAL",
        "ALTER TABLE screen_candidates ADD COLUMN news_burst REAL",
        "ALTER TABLE screen_candidates ADD COLUMN news_burst_pct REAL",
        "ALTER TABLE screen_candidates ADD COLUMN dart_event INTEGER",
        "ALTER TABLE screen_candidates ADD COLUMN news_match INTEGER",
        "ALTER TABLE screen_candidates ADD COLUMN sent_delta REAL",
        "ALTER TABLE screen_candidates ADD COLUMN ret5 REAL",
        "ALTER TABLE screen_candidates ADD COLUMN filtered_reason TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    # 전환 이전에 확정된 행에 레거시 벤치마크를 명시한다.
    try:
        conn.execute(
            "UPDATE prediction_outcomes SET benchmark=? WHERE benchmark IS NULL AND status='resolved'",
            (LEGACY_BENCHMARK,),
        )
    except sqlite3.OperationalError:
        pass


def record_method_change(change_key: str, detail: str, applied_date: str | None = None,
                         db_path: Path | None = None) -> None:
    """평가 방법 변경 1건을 기록한다. 같은 key는 최초 1회만 남는다(적용 시작일 보존)."""
    init_outcome_tables(db_path)
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path or DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO eval_method_log (change_key, applied_date, detail, created_at) "
            "VALUES (?,?,?,?)",
            (change_key, applied_date or date.today().isoformat(), detail, now),
        )
        conn.commit()


SCREEN_V2_KEY = "screen_v2"


def method_applied_date(change_key: str, db_path: Path | None = None) -> str | None:
    """eval_method_log에 기록된 적용 시작일. 없으면 None."""
    init_outcome_tables(db_path)
    with sqlite3.connect(db_path or DB_PATH) as conn:
        row = conn.execute(
            "SELECT applied_date FROM eval_method_log WHERE change_key = ?", (change_key,)).fetchone()
    return row[0] if row else None


# ── 등록 ────────────────────────────────────────────────────────────────

def record_screen_candidates(run_date: str, rows: list[dict], analyzed: list[str],
                             db_path: Path | None = None) -> int:
    """유니버스 전 종목의 팩터값을 남긴다 (스크리닝 v2 §4). 같은 날 재실행은 첫 행을 지킨다(INSERT OR IGNORE).

    rows: stage2 scores[t]에 "ticker"를 붙인 dict 리스트. analyzed: 심층 분석한 ticker.
    """
    if not rows:
        return 0
    init_outcome_tables(db_path)
    now = datetime.now().isoformat()
    an = set(analyzed)
    with sqlite3.connect(db_path or DB_PATH) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO screen_candidates "
            "(run_date, ticker, analyzed, created_at, is_candidate, cand_rank, score, news_burst, "
            " news_burst_pct, dart_event, news_match, sent_delta, ret5, filtered_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run_date, r["ticker"], 1 if r["ticker"] in an else 0, now,
              int(r.get("is_candidate") or 0), r.get("rank"), r.get("score"), r.get("news_burst"),
              r.get("news_burst_pct"), int(r.get("dart_event") or 0), int(r.get("news_match") or 0),
              r.get("sent_delta"), r.get("ret5"), r.get("filtered_reason"))
             for r in rows],
        )
        conn.commit()
    return len(rows)


def register_pending(db_path: Path | None = None) -> int:
    """prediction_log(ticker 있는 행)와 screen_candidates에서 아직 등록 안 된
    (pred_date, ticker) × HORIZONS 를 pending으로 만든다. 멱등."""
    init_outcome_tables(db_path)
    now = datetime.now().isoformat()
    inserted = 0
    with sqlite3.connect(db_path or DB_PATH) as conn:
        keys: dict[tuple[str, str], tuple[str, Optional[float]]] = {}
        try:
            for d, t, p in conn.execute(
                "SELECT pred_date, ticker, MAX(price_at_pred) FROM prediction_log "
                "WHERE ticker != '' GROUP BY pred_date, ticker"
            ):
                keys[(d, t)] = ("prediction", p)
        except sqlite3.OperationalError:
            pass
        # v2: 후보이거나 분석된 것만 D+5/10/20 채점에 올린다. 미선정 100여 종목의 수익률은
        # screen_factors가 ohlcv_cache에서 직접 계산한다. is_candidate NULL = v2 이전 행(전부 확정분).
        for d, t in conn.execute(
            "SELECT run_date, ticker FROM screen_candidates "
            "WHERE analyzed = 1 OR is_candidate = 1 OR is_candidate IS NULL"
        ):
            keys.setdefault((d, t), ("candidate", None))

        for (d, t), (src, p) in keys.items():
            for h in HORIZONS:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO prediction_outcomes "
                    "(pred_date,ticker,horizon_days,source,price_at_pred,status,created_at) "
                    "VALUES (?,?,?,?,?,'pending',?)",
                    (d, t, h, src, p, now),
                )
                inserted += cur.rowcount
        conn.commit()
    return inserted


# ── 가격 ────────────────────────────────────────────────────────────────

def _bars_from(ticker: str, pred_date: str, db_path: Path | None = None) -> list[tuple[str, float]]:
    """ohlcv_cache에서 pred_date 이후(포함) 봉을 오래된순으로."""
    with sqlite3.connect(db_path or DB_PATH) as conn:
        try:
            rows = conn.execute(
                "SELECT date, close FROM ohlcv_cache WHERE ticker=? AND date>=? ORDER BY date ASC",
                (ticker, pred_date),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [(d, float(c)) for d, c in rows if c is not None]


_KOSPI_CACHE: dict[str, float] | None = None


def _kospi_closes(start: str, end: str) -> dict[str, float]:
    """{'YYYY-MM-DD': close}. 실패 시 {} — alpha만 비고 raw는 기록된다."""
    global _KOSPI_CACHE
    if _KOSPI_CACHE is not None:
        return _KOSPI_CACHE
    try:
        import yfinance as yf
        from datetime import timedelta
        end_plus = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
        hist = yf.Ticker(BENCHMARK).history(start=start, end=end_plus)
        _KOSPI_CACHE = {idx.strftime("%Y-%m-%d"): float(v) for idx, v in hist["Close"].items()}
    except Exception as e:
        logger.warning(f"[outcomes] KOSPI 조회 실패 — alpha 없이 기록: {e}")
        _KOSPI_CACHE = {}
    return _KOSPI_CACHE


def _resolve_one(row: dict, db_path: Path | None) -> Optional[dict]:
    """창이 다 찼으면 결과 dict, 아니면 None(그대로 pending)."""
    h = int(row["horizon_days"])
    bars = _bars_from(row["ticker"], row["pred_date"], db_path)
    if not bars:
        return None
    # 시작가: pred_date 봉이 있으면 그 종가, 없으면 price_at_pred
    if bars[0][0] == row["pred_date"]:
        start_close = bars[0][1]
        after = bars[1:]
    else:
        start_close = row["price_at_pred"]
        after = bars
    if not start_close or len(after) < h:
        return None                     # TradingAgents #1169: 창이 안 찼으면 확정하지 않는다
    res_date, res_close = after[h - 1]
    raw = res_close / start_close - 1.0

    kospi = _kospi_closes(row["pred_date"], res_date)
    k0, k1 = kospi.get(row["pred_date"]), kospi.get(res_date)
    kret  = (k1 / k0 - 1.0) if (k0 and k1) else None
    alpha = (raw - kret) if kret is not None else None
    return {
        "price_at_pred": start_close, "price_at_res": res_close, "raw_return": raw,
        "kospi_return": kret, "alpha_vs_kospi": alpha, "resolution_date": res_date,
        "benchmark": BENCHMARK, "net_return": net_of_costs(raw, res_date),
    }


def resolve_pending(db_path: Path | None = None) -> dict:
    """pending 전부를 시도한다. 반환: {"resolved": n, "still_pending": m}"""
    global _KOSPI_CACHE
    _KOSPI_CACHE = None
    init_outcome_tables(db_path)
    with sqlite3.connect(db_path or DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        pending = [dict(r) for r in conn.execute(
            "SELECT * FROM prediction_outcomes WHERE status='pending' ORDER BY pred_date, ticker, horizon_days"
        )]
    resolved = 0
    now = datetime.now().isoformat()
    for row in pending:
        out = _resolve_one(row, db_path)
        if out is None:
            continue
        with sqlite3.connect(db_path or DB_PATH) as conn:
            conn.execute(
                "UPDATE prediction_outcomes SET price_at_pred=?, price_at_res=?, raw_return=?, "
                "kospi_return=?, alpha_vs_kospi=?, resolution_date=?, benchmark=?, net_return=?, "
                "status='resolved', updated_at=? WHERE id=?",
                (out["price_at_pred"], out["price_at_res"], out["raw_return"], out["kospi_return"],
                 out["alpha_vs_kospi"], out["resolution_date"], out["benchmark"], out["net_return"],
                 now, row["id"]),
            )
            conn.commit()
        resolved += 1
    return {"resolved": resolved, "still_pending": len(pending) - resolved}


# ── 집계 ────────────────────────────────────────────────────────────────

def _day_mean(rows: list[dict], key: str) -> tuple[Optional[float], int]:
    """행을 pred_date로 묶어 하루 평균을 내고, 그 평균들의 평균과 일 수를 돌려준다.

    [왜 하루가 표본 1개인가 — CR-8 / 결정 Q6(a)]
      하루에 추천하는 5종목은 같은 시장 충격을 받아 서로 독립이 아니다. 종목을
      독립 표본으로 세면 표본 수가 부풀고 유의성이 실제보다 강하게 나온다.
      random_control.summary()도 같은 단위를 쓴다 — 둘이 달라 백분위 비교가
      어긋나던 것을 맞췄다.
    """
    by_day: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(key)
        if v is not None:
            by_day.setdefault(r["pred_date"], []).append(float(v))
    if not by_day:
        return None, 0
    day_means = [sum(v) / len(v) for v in by_day.values()]
    return sum(day_means) / len(day_means), len(day_means)


def no_pick_days(db_path: Path | None = None) -> int:
    """확정 종목 0개였던 실행 일수 (결정 Q8: 표본에서 빼되 일수는 따로 센다).

    universe_snapshots는 확정 0개인 날에도 기록되고 screen_candidates는 확정이
    있을 때만 기록되므로, 둘의 차이가 no-pick 일수다.
    """
    path = db_path or DB_PATH
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT run_date FROM universe_snapshots "
                "WHERE run_date NOT IN (SELECT DISTINCT run_date FROM screen_candidates))"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def summary(horizon: int, db_path: Path | None = None) -> dict:
    """지평별 집계. **표본 단위는 하루**(CR-8). 종목 행 수는 n_predictions로 따로 둔다."""
    thr = HIT_THRESHOLD.get(horizon, 0.01)
    out: dict = {"horizon": horizon, "n_days": 0, "n_predictions": 0,
                 "avg_raw": None, "avg_alpha": None, "avg_net": None,
                 "hit_rate": None, "breakeven_win_rate": None,
                 "benchmark": BENCHMARK_LABEL, "no_pick_days": no_pick_days(db_path),
                 "by_agent": {}, "candidates": {}}
    path = db_path or DB_PATH
    if not path.exists():
        return out
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            res = [dict(r) for r in conn.execute(
                "SELECT * FROM prediction_outcomes WHERE status='resolved' AND horizon_days=?", (horizon,))]
        except sqlite3.OperationalError:
            return out
        if not res:
            return out
        out["n_predictions"] = len(res)
        out["avg_raw"], out["n_days"] = _day_mean(res, "raw_return")
        out["avg_alpha"], _           = _day_mean(res, "alpha_vs_kospi")
        out["avg_net"], _             = _day_mean(res, "net_return")      # 결정 Q7

        # 결정 Q4: 적중률과 손익분기 승률을 같이 낸다. 승률 하나만으로는
        # 좋다·나쁘다를 말할 수 없다 — 손익분기선이 이익/손실 크기에 따라 달라진다.
        raws = [float(r["raw_return"]) for r in res if r["raw_return"] is not None]
        if raws:
            wins   = [x for x in raws if x >  thr]
            losses = [-x for x in raws if x < -thr]
            out["hit_rate"] = len(wins) / len(raws)
            if wins and losses:
                w_bar, l_bar = sum(wins) / len(wins), sum(losses) / len(losses)
                cost = roundtrip_cost()
                if (w_bar + l_bar) > 0:
                    # 백테스트 metrics.trade_stats와 같은 식: (L̄ + 왕복비용) / (W̄ + L̄)
                    out["breakeven_win_rate"] = (l_bar + cost) / (w_bar + l_bar)

        by_key = {(r["pred_date"], r["ticker"]): r for r in res}
        try:
            preds = conn.execute(
                "SELECT pred_date, ticker, agent_name, recommendation, confidence FROM prediction_log WHERE ticker != ''"
            ).fetchall()
        except sqlite3.OperationalError:
            preds = []
        agg: dict[str, dict] = {}
        for p in preds:
            r = by_key.get((p["pred_date"], p["ticker"]))
            if r is None or r["raw_return"] is None:
                continue
            rec, ret = p["recommendation"], r["raw_return"]
            hit = (ret > thr) if rec == "BUY" else (ret < -thr) if rec == "SELL" else (abs(ret) <= thr)
            a = agg.setdefault(p["agent_name"], {"n": 0, "hits": 0, "buy_n": 0, "buy_alpha_pos": 0})
            a["n"] += 1
            a["hits"] += int(hit)
            if rec == "BUY" and r["alpha_vs_kospi"] is not None:
                a["buy_n"] += 1
                a["buy_alpha_pos"] += int(r["alpha_vs_kospi"] > 0)
        out["by_agent"] = {
            k: {"n": v["n"], "hit_rate": v["hits"] / v["n"],
                "buy_alpha_pos_rate": (v["buy_alpha_pos"] / v["buy_n"]) if v["buy_n"] else None}
            for k, v in agg.items()
        }

        try:
            cands = conn.execute("SELECT run_date, ticker, analyzed FROM screen_candidates").fetchall()
        except sqlite3.OperationalError:
            cands = []
        buckets = {"analyzed": [], "cut": []}
        for c in cands:
            r = by_key.get((c["run_date"], c["ticker"]))
            if r is None or r["raw_return"] is None:
                continue
            buckets["analyzed" if c["analyzed"] else "cut"].append(r["raw_return"])
        out["candidates"] = {k: {"n": len(v), "avg_raw": (sum(v) / len(v)) if v else None}
                             for k, v in buckets.items()}
    return out


def format_summary(db_path: Path | None = None) -> str:
    lines = [f"[outcomes] 예측 → 실현 수익 (종가 기준, 표본 단위=하루, {BENCHMARK_LABEL} 대비)"]
    def pct(x):
        return "n/a" if x is None else f"{x*100:+.2f}%"
    def rate(x):
        return "n/a" if x is None else f"{x*100:.0f}%"
    for h in HORIZONS:
        s = summary(h, db_path)
        lines.append(
            f"  D+{h}: {s['n_days']}일 ({s['n_predictions']}건)  평균 {pct(s['avg_raw'])}  "
            f"알파 {pct(s['avg_alpha'])}  비용차감 {pct(s['avg_net'])}  적중 {rate(s['hit_rate'])} "
            f"(손익분기 {rate(s['breakeven_win_rate'])})"
        )
        for agent, v in sorted(s["by_agent"].items()):
            bap = "n/a" if v["buy_alpha_pos_rate"] is None else f"{v['buy_alpha_pos_rate']*100:.0f}%"
            lines.append(f"      {agent:<24} n={v['n']:<4} 방향적중 {v['hit_rate']*100:.0f}%  BUY알파>0 {bap}")
        c = s["candidates"]
        if c:
            lines.append(f"      후보 분석됨 n={c['analyzed']['n']} {pct(c['analyzed']['avg_raw'])} / "
                         f"상한으로 잘림 n={c['cut']['n']} {pct(c['cut']['avg_raw'])}")
    return "\n".join(lines)


# ── CSV 내보내기 (LLM-Trading-Lab 로그 형식 참고) ────────────────────────

def export_csv(db_path: Path | None = None, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or LEDGER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = db_path or DB_PATH
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        specs = {
            "outcomes.csv": ("SELECT pred_date, ticker, horizon_days, source, price_at_pred, price_at_res, "
                             "raw_return, net_return, kospi_return, alpha_vs_kospi, benchmark, resolution_date, status "
                             "FROM prediction_outcomes ORDER BY pred_date, ticker, horizon_days"),
            "predictions.csv": ("SELECT pred_date, ticker, ticker_name, agent_name, recommendation, confidence, "
                                "regime, price_at_pred, evaluated, eval_score FROM prediction_log "
                                "WHERE ticker != '' ORDER BY pred_date, ticker, agent_name"),
        }
        for name, sql in specs.items():
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.OperationalError:
                continue
            p = out_dir / name
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if rows:
                    w.writerow(rows[0].keys())
                    w.writerows([tuple(r) for r in rows])
            written.append(p)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# 지평 비교 (S5 D-4 "지금 할 것", 2026-09-15)
#
# [왜 나란히 놓나]
#   S5 조사 결론: 검증된 전략들은 회전율이 낮고 보유기간이 길다. AlphaFlow가
#   구조를 바꿔야 하는지 판단하려면 "H가 길수록 성과가 개선되는가"를 봐야 하는데,
#   지금 summary()는 지평별로 따로 출력해 비교가 어렵다.
#   백테스트에서는 H=20이 셋 중 가장 덜 나빴다(전체 −36.7%p vs H=5 −75.5%p).
#   forward에서도 같은 방향이면 보유기간 연장의 근거가 되고, 지평과 무관하게
#   음수면 수익원 자체(S5 B4)가 문제라는 뜻이다.
#
#   리포트에는 넣지 않는다 — 사람이 매일 볼 수치가 아니라 분기에 한 번 볼 수치다.
#   CI 로그와 `python -m src.evaluation.outcomes`에서만 나온다.
# ─────────────────────────────────────────────────────────────────────────────

def horizon_comparison(db_path: Path | None = None) -> dict:
    """세 지평을 같은 자로 나란히. 반환: {horizon: summary dict} + 추세 판정."""
    rows = {h: summary(h, db_path) for h in HORIZONS}
    alphas = [(h, rows[h]["avg_alpha"]) for h in HORIZONS if rows[h]["avg_alpha"] is not None]
    trend = "표본 부족"
    if len(alphas) == len(HORIZONS):
        vals = [a for _, a in alphas]
        if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            trend = "H가 길수록 개선 — 보유기간 연장 근거"
        elif all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            trend = "H가 길수록 악화 — 단기 쪽이 낫다"
        elif all(v < 0 for v in vals):
            trend = "전 지평 음수 — 지평이 아니라 수익원이 문제 (S5 B4)"
        else:
            trend = "일관된 방향 없음"
    return {"by_horizon": rows, "trend": trend}


def format_horizon_comparison(db_path: Path | None = None) -> str:
    """지평 비교 표. 표본이 모자라면 그렇다고 쓴다."""
    res = horizon_comparison(db_path)
    rows = res["by_horizon"]
    def pct(x):
        return "   n/a " if x is None else f"{x*100:+6.2f}%"
    def rate(x):
        return " n/a" if x is None else f"{x*100:3.0f}%"
    lines = [
        f"[horizon] 지평 비교 (표본 단위=하루, {BENCHMARK_LABEL} 대비)",
        "  지평 |   일수 |   평균 |   알파 | 비용차감 | 적중 | 손익분기",
        "  -----+--------+--------+--------+----------+------+---------",
    ]
    for h in HORIZONS:
        s = rows[h]
        flag = "" if s["n_days"] >= MIN_DAYS_FOR_TREND else f"  (최소 {MIN_DAYS_FOR_TREND}일 미달)"
        lines.append(
            f"  D+{h:<2} | {s['n_days']:>5}일 | {pct(s['avg_raw'])} | {pct(s['avg_alpha'])} "
            f"| {pct(s['avg_net'])}  | {rate(s['hit_rate'])} | {rate(s['breakeven_win_rate'])}{flag}"
        )
    lines.append(f"  → 추세: {res['trend']}")
    lines.append("  ※ S5 D-4: H가 길수록 개선되면 보유기간 연장 근거, 전 지평 음수면 수익원 문제.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # 평가 방법 전환 지점을 DB에 남긴다 (최초 1회만 기록된다).
    record_method_change("benchmark",
                         f"^KS11 → {BENCHMARK} ({BENCHMARK_LABEL}). 백테스트와 자를 통일 (CR-7)")
    record_method_change("sample_unit",
                         "summary 표본 단위를 종목-행 → 하루로 변경 (CR-8 / 결정 Q6-a)")
    record_method_change("net_return",
                         f"비용 차감 net_return 열 추가. 왕복 {roundtrip_cost()*100:.3f}% "
                         f"(수수료·제비용·슬리피지 0.1%×2 + 매도 연도별 거래세). raw_return은 보존 (결정 Q7)")
    n = register_pending()
    r = resolve_pending()
    print(f"[outcomes] pending 등록 {n}건 / 이번에 확정 {r['resolved']}건 / 미확정 {r['still_pending']}건")
    print(format_summary())
    from src.evaluation import random_control as _rc
    rr = _rc.resolve_pending()
    print(f"[random_control] 이번에 확정 {rr['resolved']}건 / 미확정 {rr['still_pending']}건")
    print(_rc.format_summary())
    # 결정 3-5 (C): 지금 바로 잴 수 있는 유일한 잣대. 별도 적재 없이 집계만 돈다.
    from src.evaluation import llm_contribution as _lc
    print(_lc.format_summary())
    # S5 D-4: 세 지평을 나란히 — "H가 길수록 개선되는가"를 한눈에
    print(format_horizon_comparison())
    for p in export_csv():
        print(f"[outcomes] CSV: {p}")
