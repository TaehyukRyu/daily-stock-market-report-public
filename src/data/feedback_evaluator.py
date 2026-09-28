"""
feedback_evaluator.py
──────────────────────
Task C — 피드백 루프 (2/3): D+5 채점 + EMA 가중치 갱신

[채점 지평] 2026-09-21에 D+1 → D+5로 바꿨다. 표방 포지션이 1~4주인데
  하루 등락으로 매긴 점수가 EMA를 움직이고 있었다. 근거는 prediction_outcomes의
  확정 행이므로 feedback.yml에서 outcomes가 먼저 돌아야 한다.

[동적 EMA α — 계획서 9.2절]
  기존: 레짐별 고정 α (Bull=0.1, Bear=0.3, ...)
  변경: 레짐 이력을 보고 전환 여부에 따라 α를 동적 결정

  동작 규칙:
    ① Volatile 레짐: 항상 α=0.5 (레짐 자체가 고변동성)
    ② 레짐이 최근 5거래일 내 전환됨: α=0.5 (빠른 적응)
    ③ 5거래일 연속 동일 레짐: α=base_alpha (안정 복귀)
    ④ 이력 없음 (초기): 레짐별 base_alpha 사용

  콘솔 출력 예시:
    [동적 α] Bull / 안정(5일+) → α=0.1
    [동적 α] Bull→Bear 전환 감지 → α=0.5 (빠른 적응)
"""

import logging
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pykrx import stock as krx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    logger.warning("[feedback_evaluator] pykrx 없음")
    PYKRX_AVAILABLE = False

from src.data.prediction_logger import (
    DB_PATH, SCORED_AGENTS, MIN_WEIGHT, INITIAL_WEIGHT, ALL_REGIME,
    _normalize_regime,
    has_weight_evidence, EVALUATED_D1_LEGACY, retire_d1_scores,
    mark_evaluated, update_agent_weight,
    get_recent_regimes,
)


# ── 레짐별 기본 α (안정 상태일 때 사용) ───────────────────────────────────────
BASE_REGIME_ALPHA = {
    "Bull":      0.1,
    "Bear":      0.3,
    "Sideways":  0.2,
    "Volatile":  0.5,   # Volatile은 항상 최대 α
    "Neutral":   0.1,
}

TRANSITION_ALPHA     = 0.5   # 레짐 전환 직후 α (계획서 9.2절)
STABLE_DAYS_THRESHOLD = 5    # 이 일수 이상 동일 레짐이면 안정으로 판단

# ── 채점 지평 ──────────────────────────────────────────────────────────────────
# 표방 포지션이 1~4주인데 D+1 방향으로 채점하고 있었다 (doc/2026-09-21_agent-audit.md §2-4).
# prediction_outcomes가 이미 D+5/10/20을 확정하므로 그 값을 읽는다 — 가격을 다시 조회하지 않는다.
SCORING_HORIZON_DAYS = 5

# ── 채점 임계값 ────────────────────────────────────────────────────────────────
CORRECT_THRESHOLD  = 0.01
PARTIAL_THRESHOLD  = 0.01
HOLD_CORRECT_RANGE = 0.01
# HOLD가 임계를 살짝 넘겼을 때 주는 부분점수 구간의 배수.
# BUY/SELL은 "반대로 가지 않았으면" 0.5를 받는데 HOLD만 그 구간이 없었다.
HOLD_PARTIAL_MULTIPLE = 2.0

# ── 조정 한도 (계획서 11.4절) ──────────────────────────────────────────────────
MAX_WEIGHT_CHANGE_RATIO = 0.30


# ── 동적 α 계산 ── 핵심 신규 함수 ─────────────────────────────────────────────

def get_dynamic_alpha(current_regime: str) -> tuple[float, str]:
    """
    레짐 이력 기반 동적 α 계산.

    Args:
        current_regime: 오늘 레짐 (정규화 전 문자열 허용)

    Returns:
        (alpha, reason_str)
        예: (0.1, "Bull/안정(5일+)")
            (0.5, "Bull→Bear 전환 감지")
            (0.5, "Volatile 레짐")

    판단 로직:
        1. Volatile → 항상 0.5 (즉시 반환)
        2. regime_history에서 최근 6일 이력 조회
        3. 이력 없음 → base_alpha (초기 상태)
        4. 최근 이력에 오늘 레짐과 다른 레짐 존재 → 전환 감지 → 0.5
        5. 최근 5거래일 모두 동일 레짐 → 안정 → base_alpha
        6. 전환 후 아직 5일 미만 → TRANSITION_ALPHA

    설계 이유:
        달력 기준 6일을 조회하는 이유:
          주말(토/일)은 파이프라인 미실행 → 이력 없음.
          6일을 조회하면 주말 2일을 제외해도 5거래일이 커버됨.
    """
    current_regime = _normalize_regime(current_regime)
    base_alpha     = BASE_REGIME_ALPHA.get(current_regime, 0.1)

    # ① Volatile: 항상 최대 α
    if current_regime == "Volatile":
        return (0.5, "Volatile 레짐")

    # ② 레짐 이력 조회
    history = get_recent_regimes(days=6)  # 오래된 순 → 최신 순

    if not history:
        return (base_alpha, f"{current_regime}/이력 없음(초기)")

    # 이력에서 레짐만 추출 (최신 순으로 뒤집기)
    recent_regimes = [h["regime"] for h in history]

    # ③ 최근 이력에 다른 레짐이 있으면 전환 감지
    unique_recent = set(recent_regimes)
    if len(unique_recent) > 1:
        # 전환이 있었음 → 가장 최근에 어디서 왔는지 찾기
        prev_regime = None
        for regime in reversed(recent_regimes):
            if regime != current_regime:
                prev_regime = regime
                break
        reason = f"{prev_regime}→{current_regime} 전환 감지"
        return (TRANSITION_ALPHA, reason)

    # ④ 모두 동일 레짐 → 안정 일수 계산
    stable_days = len(recent_regimes)  # 최근 N일 모두 동일
    if stable_days >= STABLE_DAYS_THRESHOLD:
        return (base_alpha, f"{current_regime}/안정({stable_days}일+)")
    else:
        # 동일하긴 한데 5일 미만 → 아직 전환 직후일 수 있음
        return (TRANSITION_ALPHA, f"{current_regime}/안정({stable_days}일, {STABLE_DAYS_THRESHOLD}일 미달)")


# ── 주가 조회 ──────────────────────────────────────────────────────────────────

def _get_closing_prices(tickers: list[str], target_date: str) -> dict[str, float]:
    if not PYKRX_AVAILABLE:
        logger.error("[feedback_evaluator] pykrx 없음")
        return {}
    result = {}
    for ticker in tickers:
        try:
            df = krx_stock.get_market_ohlcv_by_date(fromdate=target_date, todate=target_date, ticker=ticker)
            if not df.empty and "종가" in df.columns:
                result[ticker] = float(df["종가"].iloc[-1])
        except Exception as e:
            logger.warning(f"[feedback_evaluator] {ticker} 종가 조회 실패: {e}")
    return result


# ── 채점 로직 ──────────────────────────────────────────────────────────────────

def _score_prediction(recommendation: str, price_at_pred: Optional[float], actual_price: float) -> float:
    """BUY/SELL/HOLD 채점. 1.0=적중, 0.5=애매, 0.0=오답.

    [세 방향을 같은 모양으로] doc/2026-09-21_agent-audit.md §2-4
      종전에는 HOLD만 부분점수가 없었다. BUY/SELL은 ±1% 안에 머물면 0.5를 받는데
      HOLD는 ±1%를 벗어나면 바로 0.0이었다. 일 변동성이 6%인 종목군에서 HOLD가
      1.0을 받을 확률은 낮다. 그 결과 "통과한 22표가 전부 SELL"인 macro_economist가
      평균 0.841로 1등, HOLD 68%인 technical_analyst가 0.263이었다.
      표본이 차면 EMA가 "항상 한 방향"을 학습하게 된다.

      이제 HOLD도 임계의 2배 안이면 0.5를 받는다. 셋 다 적중/애매/오답 3단이다.

    price_at_pred가 없으면 0.5(중립) — 채점할 수 없다는 뜻이지 틀렸다는 뜻이 아니다.
    """
    if price_at_pred is None or price_at_pred <= 0:
        return 0.5

    change = (actual_price / price_at_pred) - 1.0
    rec    = recommendation.upper()

    if rec == "BUY":
        return 1.0 if change > CORRECT_THRESHOLD else (0.0 if change < -PARTIAL_THRESHOLD else 0.5)
    if rec == "SELL":
        return 1.0 if change < -CORRECT_THRESHOLD else (0.0 if change > PARTIAL_THRESHOLD else 0.5)
    if rec == "HOLD":
        if abs(change) <= HOLD_CORRECT_RANGE:
            return 1.0
        return 0.5 if abs(change) <= HOLD_CORRECT_RANGE * HOLD_PARTIAL_MULTIPLE else 0.0
    return 0.5


# ── EMA 계산 ───────────────────────────────────────────────────────────────────

def _apply_ema(old_weight: float, new_score: float, alpha: float) -> float:
    """EMA 공식 + ±30% Clipping + MIN_WEIGHT 하한선."""
    raw = alpha * new_score + (1.0 - alpha) * old_weight
    clipped = max(old_weight * (1 - MAX_WEIGHT_CHANGE_RATIO),
                  min(old_weight * (1 + MAX_WEIGHT_CHANGE_RATIO), raw))
    return round(max(clipped, MIN_WEIGHT), 6)


# ── 일회성 마이그레이션 ────────────────────────────────────────────────────────

SCORING_D5_KEY = "scoring_d5"


def _retire_d1_once() -> None:
    """옛 D+1 점수를 이력으로 표시한다. **딱 한 번만** 돈다.

    매 실행 돌면 D+5로 새로 매긴 점수(evaluated=1)까지 도로 이력으로 바꿔버린다.
    eval_method_log에 적용일을 남겨 두 번째부터는 건너뛴다.
    """
    try:
        from src.evaluation.outcomes import method_applied_date, record_method_change
        if method_applied_date(SCORING_D5_KEY):
            return
        n = retire_d1_scores()
        record_method_change(
            SCORING_D5_KEY,
            f"채점 지평 D+1 → D+{SCORING_HORIZON_DAYS}. 옛 D+1 점수 {n}건을 증거 집계에서 제외 "
            f"(EVALUATED_D1_LEGACY). HOLD 부분점수 추가로 세 방향 점수표가 대칭이 됐다.",
        )
        logger.info(f"[feedback_evaluator] D+1 점수 {n}건 이력 처리 — 이후 D+{SCORING_HORIZON_DAYS}로 재채점")
    except Exception as e:
        logger.warning(f"[feedback_evaluator] D+1 점수 이력 처리 실패(계속): {e}")


# ── 채점 대상 조회 ─────────────────────────────────────────────────────────────

def _resolved_unscored(target_date: Optional[str] = None) -> list[dict]:
    """D+5가 확정됐는데 아직 채점 안 된 prediction_log 행.

    evaluated=0(미채점)과 EVALUATED_D1_LEGACY(옛 D+1 점수)를 모두 대상으로 한다.
    D+1로 매겼던 행도 D+5가 확정되면 같은 자로 다시 매긴다.
    ticker가 빈 행(EVALUATED_LEGACY)은 애초에 채점할 수 없어 제외된다.
    """
    sql = (
        "SELECT p.id, p.ticker, p.agent_name, p.recommendation, p.regime, o.raw_return "
        "FROM prediction_log p "
        "JOIN prediction_outcomes o "
        "  ON o.pred_date = p.pred_date AND o.ticker = p.ticker "
        "WHERE p.ticker != '' AND p.evaluated IN (0, ?) "
        "  AND o.horizon_days = ? AND o.status = 'resolved' AND o.raw_return IS NOT NULL"
    )
    params: list = [EVALUATED_D1_LEGACY, SCORING_HORIZON_DAYS]
    if target_date:
        sql += " AND p.pred_date = ?"
        params.append(target_date)
    sql += " ORDER BY p.agent_name, p.ticker"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, params)]
        except sqlite3.OperationalError as e:
            logger.warning(f"[feedback_evaluator] 채점 대상 조회 실패: {e}")
            return []


# ── 메인: D+5 채점 ────────────────────────────────────────────────────────────

def evaluate_predictions(target_date: Optional[str] = None) -> dict:
    """D+5 채점 + EMA 가중치 갱신.

    [무엇을 읽나] doc/2026-09-21_agent-audit.md §2-4
      prediction_outcomes에서 horizon_days=5 이고 status='resolved' 인 행을 읽는다.
      그 행은 outcomes.resolve_pending()이 ohlcv_cache로 확정한 실현 수익이다.
      여기서 가격을 다시 조회하지 않는다 — 두 곳이 서로 다른 값을 쓰는 일을 막는다.

    [target_date]
      None이면 **확정된 것 전부**를 채점한다. D+5는 예측일로부터 5거래일이 지나야
      확정되므로 "어제 것만" 보면 영원히 0건이다. feedback.yml은 인자 없이 부른다.
      날짜를 주면 그 예측일만 채점한다 (재실행·백필용).

    [동적 α]
      레짐 전환이 감지되면 α=0.5로 스파이크, 5거래일 안정 후 base_alpha 복귀.
    """
    result = {"target_date": target_date or "all-resolved", "evaluated_count": 0,
              "agent_scores": {}, "weight_changes": {}, "errors": []}

    _retire_d1_once()

    rows = _resolved_unscored(target_date)
    if not rows:
        logger.info(f"[feedback_evaluator] 채점할 D+{SCORING_HORIZON_DAYS} 확정 예측 없음. 종료")
        return result

    logger.info(f"[feedback_evaluator] D+{SCORING_HORIZON_DAYS} 확정 미채점 {len(rows)}건 채점 시작")

    agent_regime_scores: dict[tuple, list] = defaultdict(list)
    record_ids_to_mark, scores_to_mark = [], []

    for row in rows:
        raw = row["raw_return"]
        if raw is None:
            result["errors"].append(f"{row['ticker']}: raw_return 없음")
            continue
        # raw_return은 이미 (확정가/진입가 - 1)이다. 같은 자로 채점하려고 100 기준으로 환산한다.
        score = _score_prediction(row["recommendation"], 100.0, 100.0 * (1.0 + float(raw)))

        agent_regime_scores[(row["agent_name"], row["regime"])].append(score)
        record_ids_to_mark.append(row["id"])
        scores_to_mark.append(score)

    if record_ids_to_mark:
        mark_evaluated(record_ids_to_mark, scores_to_mark)
        result["evaluated_count"] = len(record_ids_to_mark)

    # ── EMA 가중치 갱신 (동적 α 적용) ────────────────────────────────────────
    # mark_evaluated 이후에 돌므로 has_weight_evidence는 오늘 점수까지 포함해 판단한다.
    # 레짐별 행과 레짐 무관(All) 행을 같이 갱신한다. All은 에이전트당 하루 1회.
    agent_all_scores: dict[str, list] = defaultdict(list)
    for (agent_name, regime), scores in agent_regime_scores.items():
        avg_score = sum(scores) / len(scores)
        # chief는 원장에 남고 채점도 되지만 **투표자가 아니라서** EMA 가중치를 받지 않는다
        # (doc/2026-09-21_agent-audit.md §2-6). 자기 가중치로 자기를 종합하는 순환을 막는다.
        if agent_name in SCORED_AGENTS:
            _update_one_weight(agent_name, regime, avg_score, regime, result)
            agent_all_scores[agent_name].extend(scores)
        result["agent_scores"].setdefault(agent_name, []).append(avg_score)

    day_regime = next(iter(agent_regime_scores))[1] if agent_regime_scores else "Neutral"
    for agent_name, scores in agent_all_scores.items():
        _update_one_weight(agent_name, ALL_REGIME, sum(scores) / len(scores), day_regime, result)

    result["agent_scores"] = {
        agent: round(sum(s) / len(s), 4)
        for agent, s in result["agent_scores"].items()
    }

    logger.info(
        f"[feedback_evaluator] 채점 완료: {result['evaluated_count']}건, "
        f"가중치 변경 {len(result['weight_changes'])}개"
    )
    return result


def _get_weight_row(agent_name: str, regime: str) -> tuple[float, int]:
    """agent_weights의 원시 (weight, sample_count). 행이 없으면 (INITIAL_WEIGHT, 0)."""
    regime = _normalize_regime(regime)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT weight, sample_count FROM agent_weights WHERE agent_name=? AND regime=?",
            (agent_name, regime),
        ).fetchone()
    return (row[0], row[1]) if row else (INITIAL_WEIGHT, 0)


def _update_one_weight(agent_name: str, target_regime: str, avg_score: float,
                       day_regime: str, result: dict) -> None:
    """
    (agent, target_regime) 가중치 행 1개를 갱신한다.

    증거 부족(has_weight_evidence 실패)이면 가중치를 INITIAL_WEIGHT로 두고 표본만 누적한다.
    한때 유의했다가 표본이 쌓이며 유의성을 잃으면 다시 균등으로 돌아간다 —
    "증거 없음 = 균등"을 매일 다시 판정하는 것이지, 한 번 통과하면 영구가 아니다.

    α는 target_regime이 아니라 그날의 레짐(day_regime)으로 계산한다.
    """
    old_weight, _       = _get_weight_row(agent_name, target_regime)
    significant, reason = has_weight_evidence(agent_name, target_regime)

    if not significant:
        update_agent_weight(agent_name, target_regime, INITIAL_WEIGHT, avg_score)
        logger.info(
            f"[feedback_evaluator] {agent_name}/{target_regime} 증거 부족 ({reason}) → 균등 유지"
        )
        return

    alpha, alpha_reason = get_dynamic_alpha(day_regime)
    new_weight          = _apply_ema(old_weight, avg_score, alpha)
    update_agent_weight(agent_name, target_regime, new_weight, avg_score)

    result["weight_changes"][f"{agent_name}/{target_regime}"] = {
        "old": round(old_weight, 4), "new": round(new_weight, 4),
        "delta": round(new_weight - old_weight, 4),
        "score": round(avg_score, 4),
        "alpha": alpha, "alpha_reason": alpha_reason,
        "evidence": reason,
    }
    logger.info(
        f"[feedback_evaluator] 가중치 갱신: {agent_name}/{target_regime} "
        f"{old_weight:.4f}→{new_weight:.4f} "
        f"(점수={avg_score:.2f}, α={alpha} [{alpha_reason}], {reason})"
    )


# ── 주간 종합 평가 ─────────────────────────────────────────────────────────────

def evaluate_weekly(week_dates: Optional[list[str]] = None) -> dict:
    if week_dates is None:
        today, week_dates = date.today(), []
        d = today - timedelta(days=1)
        while len(week_dates) < 5:
            if d.weekday() < 5:
                week_dates.append(d.isoformat())
            d -= timedelta(days=1)
        week_dates.reverse()

    logger.info(f"[feedback_evaluator] 주간 평가: {week_dates[0]} ~ {week_dates[-1]}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT agent_name, eval_score FROM prediction_log WHERE pred_date IN ({','.join('?'*len(week_dates))}) AND evaluated=1 AND eval_score IS NOT NULL",
            week_dates,
        ).fetchall()

    if not rows:
        return {"week": _week_label(week_dates), "agent_weekly_avg": {}, "total": 0}

    agent_scores: dict[str, list] = defaultdict(list)
    for row in rows:
        agent_scores[row["agent_name"]].append(row["eval_score"])

    return {
        "week": _week_label(week_dates),
        "agent_weekly_avg": {a: round(sum(s)/len(s), 4) for a, s in agent_scores.items()},
        "total": len(rows),
    }


def _week_label(dates: list[str]) -> str:
    if dates:
        d = date.fromisoformat(dates[0])
        return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"
    return "Unknown"


# ── 단독 실행 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.prediction_logger import setup_feedback_system

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 60)
    print("feedback_evaluator 테스트")
    print("=" * 60)

    setup_feedback_system()

    # 동적 α 테스트 (레짐 이력이 있을 경우)
    print("\n📐 동적 α 테스트:")
    for regime in ["Bull", "Bear", "Volatile", "Sideways"]:
        alpha, reason = get_dynamic_alpha(regime)
        print(f"  {regime:<10} → α={alpha}  [{reason}]")

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    print(f"\n📅 채점 대상: {yesterday}")
    result = evaluate_predictions(yesterday)

    print(f"\n📊 채점 결과:")
    print(f"  - 채점 건수: {result['evaluated_count']}")
    for agent, score in result["agent_scores"].items():
        print(f"    {agent:<30} {score:.4f}")
    for key, change in result["weight_changes"].items():
        print(f"    {key:<35} {change['old']:.4f}→{change['new']:.4f} (α={change['alpha']} [{change['alpha_reason']}])")

    # ── 종료 코드 판정 ────────────────────────────────────────────────
    # 채점 대상이 있었는데 한 건도 못 매긴 경우는 실패다.
    # (2026-07-23~09-07: ticker가 빈 문자열이라 1,357건이 전부 스킵됐는데
    #  errors만 쌓이고 exit 0으로 끝나 아무도 알아채지 못했다.)
    # 예측 자체가 없는 날(errors도 비어 있음)은 정상이다.
    import sys

    if result["evaluated_count"] == 0 and result["errors"]:
        print(
            f"\n⛔ 채점 대상 {len(result['errors'])}건이 있었으나 0건 채점됨. "
            f"EMA 가중치가 갱신되지 않는다."
        )
        for e in result["errors"][:5]:
            print(f"    - {e}")
        if len(result["errors"]) > 5:
            print(f"    ... 외 {len(result['errors']) - 5}건")
        sys.exit(1)

    if result["evaluated_count"] == 0:
        print("\n💡 채점할 예측이 없습니다 (예측 미생성일 — 정상).")

    print("\n✅ 채점 완료")