"""
prediction_logger.py
────────────────────
Task C — 피드백 루프 (1/3): 예측 원장 저장

테이블 3개 관리:
  1. prediction_log  — 일별 에이전트 예측 원장
  2. agent_weights   — 에이전트별·레짐별 EMA 가중치
  3. regime_history  — 일별 레짐 이력 (동적 α 계산용) ← 신규
"""

import sqlite3
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/mentions.db")

SCORED_AGENTS = [
    "kr_market_specialist", "quant_analyst", "technical_analyst",
    "fundamental_analyst", "macro_economist", "us_market_specialist", "sentiment_analyst",
    "quant_rule_agent",   # G4: LLM 없는 규칙 투표자. 기존 7개 seed 행은 INSERT OR IGNORE라 유지된다
]

# 최종 종합. **투표자가 아니라 시스템의 출력**이다.
CHIEF_AGENT = "chief_strategist"

# 원장(prediction_log)에 남기는 이름 = 투표자 8명 + chief.
#
# [왜 SCORED_AGENTS와 나누나] (doc/2026-09-21_agent-audit.md §2-6)
#   SCORED_AGENTS 하나가 "저장 대상"과 "가중치 대상"을 겸하고 있었다. chief가 SCORED_AGENTS에
#   없으니 원장에도 한 행이 안 남았고, 그래서 "이 시스템의 BUY를 따르면 버는가"를 잴 표본이
#   1,761행 중 0이었다. llm_contribution은 chief 행을 조인하는데 매칭이 0건이었고,
#   calibration의 chief 코호트도 비어 포지션이 5%에 고정돼 있었다.
#   chief를 SCORED_AGENTS에 넣으면 INITIAL_WEIGHT가 1/9이 되고 chief가 EMA 가중치를 받는다 —
#   자기 자신을 가중해 자기를 종합하는 꼴이다. 그래서 목록을 둘로 나눈다.
LOGGED_AGENTS = SCORED_AGENTS + [CHIEF_AGENT]

REGIMES = ["Bull", "Bear", "Sideways", "Volatile", "Neutral"]
# 레짐 무관 합산 행. 레짐별로 쪼개면 표본이 8명×5레짐=40칸으로 흩어져 각 칸이 30개 차는 데
# 너무 오래 걸린다. 레짐별 증거가 부족하면 이 행의 가중치로 대신한다 (계층 폴백).
# regime_detector 출력에는 없고 agent_weights에만 존재한다.
ALL_REGIME = "All"
_REGIME_NORMALIZE_MAP = {r.lower(): r for r in REGIMES + [ALL_REGIME]}

INITIAL_WEIGHT    = round(1.0 / len(SCORED_AGENTS), 6)  # ≈ 0.142857
# 2026-09-17: 10 → 30. 동전 10번 중 7번 앞면은 17% 확률로 나온다 — 그 정도 운을
# 실력으로 오인해 가중치가 움직였다. 30개는 아래 z-검정이 성립하는 최소 크기.
WARMUP_SAMPLE_COUNT = 30
MIN_WEIGHT        = 0.05
# 가중치가 균등(INITIAL_WEIGHT)에서 벗어나려면 표본 수만으로는 부족하다.
# "평균 점수가 0.5(동전 던지기)와 통계적으로 다르다"까지 필요하다. 양측 95%.
SIGNIFICANCE_Z    = 1.96


def _normalize_regime(regime: str) -> str:
    """'bull'|'BULL'|'Bull' → 'Bull'. 알 수 없는 값 → 'Neutral'."""
    normalized = _REGIME_NORMALIZE_MAP.get(regime.lower())
    if normalized is None:
        logger.warning(f"[prediction_logger] 알 수 없는 레짐 '{regime}' → 'Neutral'로 대체")
        return "Neutral"
    return normalized


# ── DDL ───────────────────────────────────────────────────────────────────────

def init_feedback_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pred_date       TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                ticker_name     TEXT,
                agent_name      TEXT NOT NULL,
                recommendation  TEXT NOT NULL,
                confidence      REAL NOT NULL,
                regime          TEXT NOT NULL,
                price_at_pred   REAL,
                evaluated       INTEGER NOT NULL DEFAULT 0,
                eval_score      REAL,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pred_log_date_eval
                ON prediction_log (pred_date, evaluated);
            CREATE INDEX IF NOT EXISTS idx_pred_log_agent
                ON prediction_log (agent_name, evaluated);

            CREATE TABLE IF NOT EXISTS agent_weights (
                agent_name      TEXT NOT NULL,
                regime          TEXT NOT NULL,
                weight          REAL NOT NULL DEFAULT 0.142857,
                sample_count    INTEGER NOT NULL DEFAULT 0,
                last_score      REAL,
                last_updated    TEXT NOT NULL,
                PRIMARY KEY (agent_name, regime)
            );

            -- 레짐 이력: 파이프라인 실행마다 오늘 레짐을 1행으로 기록
            -- feedback_evaluator가 최근 N일을 조회해 레짐 전환 여부 판단 → 동적 α 결정
            CREATE TABLE IF NOT EXISTS regime_history (
                record_date     TEXT PRIMARY KEY,   -- 'YYYY-MM-DD'
                regime          TEXT NOT NULL,      -- 'Bull' | 'Bear' | ...
                created_at      TEXT NOT NULL
            );
        """)
        conn.commit()
    logger.info("[prediction_logger] 테이블 초기화 완료 (prediction_log, agent_weights, regime_history)")


def seed_agent_weights() -> None:
    now  = datetime.now().isoformat()
    rows = [(agent, regime, INITIAL_WEIGHT, 0, None, now)
            for agent in SCORED_AGENTS for regime in REGIMES + [ALL_REGIME]]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO agent_weights (agent_name,regime,weight,sample_count,last_score,last_updated) VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    logger.info(f"[prediction_logger] agent_weights 초기 seed: {len(rows)}개 조합 보장")


# ── 레짐 이력 ── 신규 ─────────────────────────────────────────────────────────

def record_regime(regime: str, record_date: Optional[str] = None) -> None:
    """
    오늘 레짐을 regime_history에 기록 (파이프라인 log_predictions_node에서 호출).

    INSERT OR REPLACE: 같은 날 파이프라인을 여러 번 돌려도 덮어쓰기만 함.
    """
    if record_date is None:
        record_date = date.today().isoformat()
    regime = _normalize_regime(regime)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO regime_history (record_date, regime, created_at) VALUES (?,?,?)",
            (record_date, regime, datetime.now().isoformat()),
        )
        conn.commit()
    logger.debug(f"[prediction_logger] 레짐 이력 기록: {record_date}={regime}")


def get_recent_regimes(days: int = 6) -> list[dict]:
    """
    최근 N일간의 레짐 이력 반환 (오래된 순 → 최신 순).

    Args:
        days: 조회할 최근 일수. 기본 6 — 주말 포함해도 5거래일 커버.

    Returns:
        [{'record_date': '2025-05-06', 'regime': 'Bull'}, ...]

    호출처: feedback_evaluator.get_dynamic_alpha()
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT record_date, regime FROM regime_history ORDER BY record_date DESC LIMIT ?",
            (days,),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]  # 오래된 순 → 최신 순


# ── 예측 저장 ──────────────────────────────────────────────────────────────────

def log_agent_predictions(
    qualified_reports: list,
    regime: str,
    pred_date: Optional[str] = None,
    price_at_pred: Optional[float] = None,
) -> int:
    """
    에이전트 예측을 원장에 저장.

    price_at_pred: 예측 시점 종가. 넘기면 그대로 저장하고, 없으면 NULL로 두어
        feedback_evaluator가 나중에 pykrx로 되짚어 조회한다. 넘기는 편이 낫다 —
        되짚기는 네트워크 호출이 늘고 KRX 장애 시 채점 자체가 불가능해진다.
    """
    if pred_date is None:
        pred_date = date.today().isoformat()
    regime = _normalize_regime(regime)

    if not qualified_reports:
        logger.warning("[prediction_logger] qualified_reports가 비어있어 저장 건너뜀")
        return 0

    now     = datetime.now().isoformat()
    records = []
    for report in qualified_reports:
        r = report.__dict__ if hasattr(report, "__dict__") else (report if isinstance(report, dict) else None)
        if r is None:
            continue
        agent_name = r.get("agent_name", "unknown")
        if agent_name not in LOGGED_AGENTS:   # 투표자 8명 + chief. 그 밖은 저장하지 않는다
            continue
        records.append((
            pred_date, r.get("ticker") or "", r.get("ticker_name") or "",
            agent_name, r.get("recommendation","HOLD"), float(r.get("confidence",0.5)),
            regime, price_at_pred, 0, None, now,
        ))

    empty_ticker = sum(1 for rec in records if not rec[1])
    if empty_ticker:
        # 빈 ticker는 채점 단계에서 전부 건너뛰어진다 — 조용히 넘어가면 안 된다.
        logger.error(
            f"[prediction_logger] ticker가 비어 있는 예측 {empty_ticker}/{len(records)}건. "
            f"이 레코드들은 D+1 채점에서 제외되어 EMA 가중치에 반영되지 않는다."
        )

    if not records:
        logger.info("[prediction_logger] 저장할 유효 레코드 없음")
        return 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,confidence,regime,price_at_pred,evaluated,eval_score,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()
    logger.info(f"[prediction_logger] {len(records)}개 예측 저장 완료 (날짜={pred_date}, 레짐={regime})")
    return len(records)


def backfill_chief_from_decisions() -> int:
    """chief_decisions에만 있던 과거 chief 판단을 prediction_log로 옮긴다. 멱등.

    2026-09-18~09-20 사이 chief 판단은 decision_memory(chief_decisions)에만 기록됐다.
    같은 판단을 원장에도 남겨야 outcomes·calibration·llm_contribution이 읽는다.
    이미 같은 (pred_date, ticker) chief 행이 있으면 건너뛴다. 옮긴 행 수를 반환.
    """
    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            rows = conn.execute(
                "SELECT pred_date, ticker, ticker_name, recommendation, confidence, regime, price_at_pred "
                "FROM chief_decisions WHERE ticker != '' ORDER BY pred_date, ticker"
            ).fetchall()
        except sqlite3.OperationalError:
            logger.info("[prediction_logger] chief_decisions 테이블 없음 — 이관 생략")
            return 0

        existing = {
            (d, t) for d, t in conn.execute(
                "SELECT pred_date, ticker FROM prediction_log WHERE agent_name=?", (CHIEF_AGENT,)
            )
        }
        new_rows = [
            (d, t, name or "", CHIEF_AGENT, rec, float(conf), _normalize_regime(rg), price, 0, None, now)
            for d, t, name, rec, conf, rg, price in rows
            if (d, t) not in existing
        ]
        if new_rows:
            conn.executemany(
                "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
                "confidence,regime,price_at_pred,evaluated,eval_score,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                new_rows,
            )
            conn.commit()
    logger.info(f"[prediction_logger] chief_decisions → prediction_log 이관 {len(new_rows)}건")
    return len(new_rows)


# ── 조회 ──────────────────────────────────────────────────────────────────────

def get_unevaluated_predictions(target_date: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id,ticker,ticker_name,agent_name,recommendation,confidence,regime,price_at_pred FROM prediction_log WHERE pred_date=? AND evaluated=0 AND ticker != '' ORDER BY agent_name,ticker",
            (target_date,),
        ).fetchall()
    return [dict(row) for row in rows]


def has_weight_evidence(agent_name: str, regime: str) -> tuple[bool, str]:
    """
    이 에이전트의 가중치가 균등에서 벗어나도 되는가.

    조건 (AND):
      ① 채점된 표본이 WARMUP_SAMPLE_COUNT 이상
      ② 평균 eval_score가 0.5(동전 던지기)와 통계적으로 다르다 — |z| ≥ SIGNIFICANCE_Z
         z = (평균 − 0.5) / (표본표준편차 / √n). 좋은 쪽이든 나쁜 쪽이든 "운이 아니다"면 통과.

    Returns:
        (통과 여부, 근거 문자열 — 로그용)
    """
    regime = _normalize_regime(regime)
    sql    = "SELECT eval_score FROM prediction_log WHERE agent_name=? AND evaluated=1 AND eval_score IS NOT NULL"
    params: tuple = (agent_name,)
    if regime != ALL_REGIME:            # All은 레짐 필터 없이 전부 합산
        sql   += " AND regime=?"
        params = (agent_name, regime)
    with sqlite3.connect(DB_PATH) as conn:
        scores = [r[0] for r in conn.execute(sql, params)]
    n = len(scores)
    if n < WARMUP_SAMPLE_COUNT:
        return False, f"표본 {n}/{WARMUP_SAMPLE_COUNT}"
    mean = sum(scores) / n
    var  = sum((s - mean) ** 2 for s in scores) / (n - 1)
    se   = (var / n) ** 0.5
    if se == 0.0:
        # 전부 같은 점수. 0.5가 아니면 분명한 증거, 0.5면 정확히 동전.
        return mean != 0.5, f"n={n} 평균 {mean:.2f} 분산 0"
    z = (mean - 0.5) / se
    return abs(z) >= SIGNIFICANCE_Z, f"n={n} 평균 {mean:.2f} z={z:+.2f}"


def get_agent_weights(regime: str) -> dict[str, float]:
    """
    에이전트별 가중치 (합이 1이 되도록 정규화).

    에이전트마다 3단 폴백:
      ① 이 레짐의 증거가 있으면 → 레짐별 가중치
      ② 없고 레짐 무관(All) 증거가 있으면 → All 가중치
      ③ 둘 다 없으면 → INITIAL_WEIGHT (균등)
    """
    regime = _normalize_regime(regime)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT agent_name, regime, weight FROM agent_weights WHERE regime IN (?, ?)",
            (regime, ALL_REGIME),
        ).fetchall()
    regime_w = {name: w for name, rg, w in rows if rg == regime}
    all_w    = {name: w for name, rg, w in rows if rg == ALL_REGIME}
    if not regime_w:
        logger.warning(f"[prediction_logger] agent_weights에 '{regime}' 레짐 데이터 없음. 균등 가중치 사용")
        return {agent: INITIAL_WEIGHT for agent in SCORED_AGENTS}

    weights = {}
    for name, w in regime_w.items():
        if has_weight_evidence(name, regime)[0]:
            weights[name] = max(w, MIN_WEIGHT)
        elif name in all_w and has_weight_evidence(name, ALL_REGIME)[0]:
            weights[name] = max(all_w[name], MIN_WEIGHT)
        else:
            weights[name] = INITIAL_WEIGHT
    total = sum(weights.values())
    return {k: round(v / total, 6) for k, v in weights.items()} if total > 0 else weights


def get_weight_summary() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT agent_name,regime,weight,sample_count,last_score,last_updated FROM agent_weights ORDER BY regime, weight DESC",
        ).fetchall()
    return [dict(row) for row in rows]


# ── 업데이트 ──────────────────────────────────────────────────────────────────

def mark_evaluated(record_ids: list[int], scores: list[float]) -> None:
    assert len(record_ids) == len(scores)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "UPDATE prediction_log SET evaluated=1, eval_score=? WHERE id=?",
            [(s, i) for i, s in zip(record_ids, scores)],
        )
        conn.commit()


def update_agent_weight(agent_name: str, regime: str, new_weight: float, latest_score: float) -> None:
    regime = _normalize_regime(regime)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE agent_weights SET weight=?, sample_count=sample_count+1, last_score=?, last_updated=? WHERE agent_name=? AND regime=?",
            (round(max(new_weight, MIN_WEIGHT), 6), latest_score, datetime.now().isoformat(), agent_name, regime),
        )
        conn.commit()


# ── 초기화 ────────────────────────────────────────────────────────────────────

# ticker가 빈 문자열인 예측은 영원히 채점할 수 없다 (AnalysisReport에 ticker 필드가
# 없던 시절, 2026-07-23~09-07 사이에 쌓인 행). evaluated=1로 바꾸면 "채점됨"으로
# 오인되므로 별도 값으로 표시해 채점 대상·집계 양쪽에서 제외한다.
EVALUATED_LEGACY = -1

# D+1 방향으로 매긴 옛 점수. 채점 지평이 D+5로 바뀌면서(2026-09-21) 같은 열에
# 서로 다른 자로 잰 값이 섞이게 됐다. has_weight_evidence는 evaluated=1만 세므로
# 이 값으로 표시하면 증거 집계에서 빠지고, D+5가 확정되면 다시 채점된다
# (feedback_evaluator._resolved_unscored가 0과 이 값을 모두 대상으로 삼는다).
EVALUATED_D1_LEGACY = -2


def retire_d1_scores() -> int:
    """D+1로 매긴 점수(evaluated=1)를 EVALUATED_D1_LEGACY로 표시한다. 멱등.

    점수(eval_score)는 지우지 않고 이력으로 남긴다. 표시만 바꿔 증거 집계에서 빼고,
    D+5가 확정되면 같은 행이 새 자로 다시 채점된다.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE prediction_log SET evaluated=? WHERE evaluated=1", (EVALUATED_D1_LEGACY,))
        conn.commit()
    if cur.rowcount:
        logger.info(f"[prediction_logger] D+1 점수 {cur.rowcount}건을 이력으로 표시 (D+5 재채점 대상)")
    return cur.rowcount


def retire_legacy_predictions() -> int:
    """ticker=''인 미채점 행을 EVALUATED_LEGACY로 표시. 멱등. 표시한 행 수 반환."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE prediction_log SET evaluated=? WHERE ticker='' AND evaluated=0",
            (EVALUATED_LEGACY,),
        )
        conn.commit()
        n = cur.rowcount
    if n:
        logger.warning(f"[prediction_logger] ticker 없는 레거시 예측 {n}건을 채점 대상에서 제외(evaluated={EVALUATED_LEGACY})")
    return n


def setup_feedback_system() -> None:
    """피드백 루프 전체 초기화. 파이프라인 시작 시 1회 호출."""
    init_feedback_tables()
    seed_agent_weights()
    retire_legacy_predictions()
    backfill_chief_from_decisions()   # 멱등. 원장에 없는 과거 chief 판단만 옮긴다
    logger.info("[prediction_logger] 피드백 시스템 초기화 완료")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 60)
    print("prediction_logger 초기화 테스트")
    print("=" * 60)
    setup_feedback_system()
    record_regime("Bull")

    summary = get_weight_summary()
    print(f"\n📊 가중치 현황 ({len(summary)}개 조합):")
    print(f"{'에이전트':<30} {'레짐':<12} {'가중치':>8} {'샘플수':>6}")
    print("-" * 60)
    for row in summary:
        print(f"{row['agent_name']:<30} {row['regime']:<12} {row['weight']:>8.4f} {row['sample_count']:>6}")

    bull_weights = get_agent_weights("Bull")
    print(f"\n🔍 Bull 레짐 가중치 합계: {sum(bull_weights.values()):.4f}")

    history = get_recent_regimes(6)
    print(f"\n📅 레짐 이력: {history}")
    print("\n✅ 테스트 완료")