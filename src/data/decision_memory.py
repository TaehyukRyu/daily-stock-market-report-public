"""
decision_memory.py
──────────────────
chief_strategist 판단의 문장 교훈 메모리 (2026-09-18, TradingAgents memory.py/reflection.py 방식).

왜 있나: EMA 가중치는 표본 30개 + z-검정이 차야 균등에서 벗어난다. 그때까지 chief는 매일 백지에서
판단한다. 문장 교훈은 표본 1개부터 작동한다. **chief에게만** 주입한다 — 투표자 8명에게 보여주면
llm_contribution(LLM vs 규칙) 비교가 오염된다.

흐름:
  pipeline.log_predictions_node   → store_decision()    LLM 호출 없음
  feedback.yml (장 마감 후)        → resolve_pending()   gpt-4o-mini 1콜/건
  pipeline.chief_strategist_node  → get_past_context()  SELECT만

시점 차단: get_past_context(as_of)는 outcome_date <= as_of인 교훈만 돌려준다.
과거 날짜로 돌릴 때 "그 날짜 이후에 확정된 교훈"이 프롬프트에 들어가면 미래를 보고 판단하는 셈이다.

`CREATE TABLE IF NOT EXISTS`는 신규 테이블만 만든다. 컬럼을 추가하면 운영 DB(39MB)는 ALTER 없이는
`no such column`으로 죽는다 — 테스트는 늘 새 DB라 못 잡는다.
`as_of`는 UTC 발행일 기준이다. `outcome_date == as_of`인 교훈도 보이므로 KST 거래일 기준으로 백테스트할 때는 전 거래일을 넘겨라.
"""
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("MENTIONS_DB_PATH", "data/mentions.db"))

REFLECT_MODEL   = "gpt-4o-mini"
SAME_TICKER_N   = 3
CROSS_TICKER_N  = 3


# ── DDL ───────────────────────────────────────────────────────────────────────

def init_decision_table() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chief_decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pred_date       TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                ticker_name     TEXT,
                recommendation  TEXT NOT NULL,
                confidence      REAL NOT NULL,
                regime          TEXT NOT NULL,
                reasoning       TEXT NOT NULL,
                price_at_pred   REAL,
                outcome_date    TEXT,
                actual_price    REAL,
                return_pct      REAL,
                score           REAL,
                lesson          TEXT,
                resolved        INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                UNIQUE (pred_date, ticker)
            );
            CREATE INDEX IF NOT EXISTS idx_chief_dec_ticker_outcome
                ON chief_decisions (ticker, resolved, outcome_date);
        """)
        conn.commit()


# ── 기록 (파이프라인, LLM 없음) ───────────────────────────────────────────────

def store_decision(
    pred_date: str,
    ticker: str,
    ticker_name: str,
    recommendation: str,
    confidence: float,
    regime: str,
    reasoning: list[str],
    price_at_pred: Optional[float],
) -> bool:
    """같은 (pred_date, ticker)가 아직 미채점(resolved=0)이면 최신 판단으로 덮어쓴다 —
    같은 날 재실행 시 first-wins로 남으면 발행된 리포트(마지막 판단)와 원장이 어긋난다.
    이미 채점됐으면(resolved=1) 덮어쓰지 않고 False. confidence<=0(의견 없음)은 기록 자체를 생략한다."""
    if confidence <= 0:
        logger.info(f"[decision_memory] {ticker} confidence<=0 — 기록 생략 (의견 없음)")
        return False
    init_decision_table()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO chief_decisions "
            "(pred_date,ticker,ticker_name,recommendation,confidence,regime,reasoning,price_at_pred,resolved,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,0,?) "
            "ON CONFLICT(pred_date, ticker) DO UPDATE SET "
            "ticker_name=excluded.ticker_name, recommendation=excluded.recommendation, "
            "confidence=excluded.confidence, regime=excluded.regime, reasoning=excluded.reasoning, "
            "price_at_pred=excluded.price_at_pred, created_at=excluded.created_at "
            "WHERE chief_decisions.resolved=0",
            (pred_date, ticker, ticker_name or "", recommendation.upper(), float(confidence),
             regime, " / ".join(reasoning), price_at_pred, datetime.now().isoformat()),
        )
        conn.commit()
        return cur.rowcount == 1


# ── 조회 (파이프라인 → chief 프롬프트) ────────────────────────────────────────

def _label(score: float) -> str:
    if score >= 1.0:
        return "적중"
    if score >= 0.5:
        return "부분"
    return "오답"


def _fetch_resolved(conn, where: str, params: tuple, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT pred_date,ticker,ticker_name,recommendation,confidence,return_pct,score,lesson "
        f"FROM chief_decisions WHERE resolved=1 AND lesson<>'' AND outcome_date<=? AND {where} "
        "ORDER BY pred_date DESC, id DESC LIMIT ?",
        params + (limit,),
    ).fetchall()


def _format_line(r: sqlite3.Row, with_ticker: bool) -> str:
    who = f"{r['ticker']} " if with_ticker else ""
    lesson = " ".join((r["lesson"] or "").split())   # 줄바꿈이 프롬프트에 주입되지 않도록 한 줄로 압축
    return (
        f"    - {r['pred_date']} {who}{r['recommendation']}({r['confidence']:.2f}) → "
        f"D+1 {r['return_pct']:+.1f}% ({_label(r['score'])}). 교훈: {lesson}"
    )


def get_past_context(
    ticker: str,
    as_of: str,
    same_n: int = SAME_TICKER_N,
    cross_n: int = CROSS_TICKER_N,
) -> str:
    """교훈이 하나도 없으면 "" — chief 프롬프트에 섹션 자체가 안 붙는다."""
    init_decision_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        same  = _fetch_resolved(conn, "ticker=?",  (as_of, ticker), same_n)
        cross = _fetch_resolved(conn, "ticker!=?", (as_of, ticker), cross_n)

    if not same and not cross:
        return ""

    lines = ["[과거 판단 교훈 — chief_strategist 전용]"]
    if same:
        lines.append(f"  같은 종목({ticker}) 최근 판단:")
        lines += [_format_line(r, with_ticker=False) for r in same]
    if cross:
        lines.append("  다른 종목 최근 교훈:")
        lines += [_format_line(r, with_ticker=True) for r in cross]
    lines.append(
        "  ✅ 교훈은 참고일 뿐입니다. 오늘 데이터가 우선이며, 과거 오답 하나로 방향을 뒤집지 마십시오."
    )
    return "\n".join(lines)


# ── 복기 (feedback.yml, 장 마감 후) ───────────────────────────────────────────

def get_pending(pred_date: str) -> list[dict]:
    init_decision_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id,pred_date,ticker,ticker_name,recommendation,confidence,regime,reasoning,price_at_pred,"
            "score,return_pct "
            "FROM chief_decisions WHERE pred_date=? AND resolved=0 ORDER BY ticker",
            (pred_date,),
        ).fetchall()
    return [dict(r) for r in rows]


_REFLECT_SYSTEM = "[REDACTED] Proprietary prompt engineering"


async def reflect_with_openai(row: dict, return_pct: float, score: float) -> str:
    from openai import AsyncOpenAI   # feedback.yml 최소 설치 환경에서만 필요. 모듈 import 시점엔 안 부른다.

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user = (
        f"종목: {row['ticker_name']}({row['ticker']})\n"
        f"판단일: {row['pred_date']}  레짐: {row['regime']}\n"
        f"판단: {row['recommendation']} (확신도 {row['confidence']:.2f})\n"
        f"근거: {row['reasoning']}\n"
        f"D+1 수익률: {return_pct:+.2f}%\n"
        f"채점: {_label(score)}"
    )
    response = await client.chat.completions.create(
        model=REFLECT_MODEL,
        temperature=0.3,
        max_tokens=300,
        messages=[{"role": "system", "content": _REFLECT_SYSTEM},
                  {"role": "user",   "content": user}],
    )
    return (response.choices[0].message.content or "").strip()


ReflectFn = Callable[[dict, float, float], Awaitable[str]]
PriceFn   = Callable[[list[str], str], dict[str, float]]


async def resolve_pending(
    target_date: Optional[str] = None,
    reflect_fn: Optional[ReflectFn] = None,
    price_fn: Optional[PriceFn] = None,
) -> dict:
    """target_date(기본: 어제)에 기록된 chief 판단을 오늘 종가로 채점하고 교훈을 쓴다.

    feedback_evaluator.evaluate_predictions와 같은 날짜 규약(예측일=어제, 종가=오늘).
    예측일 종가(price_at_pred)가 비어 있으면 price_fn으로 백필을 시도하고, 그래도 못 구하면
    "+0.00%"를 날조하지 않고 복기 자체를 생략한 채 pending으로 남긴다.
    resolved는 "교훈이 쓰였다"를 뜻한다. 복기 LLM이 실패하면 채점 값(return_pct/score)만 저장하고
    resolved=0으로 남겨 다음 실행 때 재시도한다 — 이미 채점된 행은 가격을 다시 조회하지 않는다.
    """
    from src.data.feedback_evaluator import _get_closing_prices, _score_prediction

    reflect_fn = reflect_fn or reflect_with_openai
    price_fn   = price_fn or _get_closing_prices

    today = date.today()
    target_date = target_date or (today - timedelta(days=1)).isoformat()
    result: dict[str, Any] = {"target_date": target_date, "resolved_count": 0, "errors": []}

    pending = get_pending(target_date)
    if not pending:
        logger.info(f"[decision_memory] {target_date} 복기 대상 없음")
        return result

    # 이미 채점된(score IS NOT NULL) 행은 복기 재시도일 뿐이므로 가격을 다시 조회하지 않는다.
    unscored = [row for row in pending if row["score"] is None]

    # F1: 예측일 종가가 비어 있는 행만 price_fn으로 백필하고 DB에도 반영한다.
    missing_pred_tickers = list({row["ticker"] for row in unscored if not row["price_at_pred"]})
    if missing_pred_tickers:
        pred_date_str = date.fromisoformat(target_date).strftime("%Y%m%d")
        backfilled = price_fn(missing_pred_tickers, pred_date_str)
        if backfilled:
            with sqlite3.connect(DB_PATH) as conn:
                for row in unscored:
                    if not row["price_at_pred"] and row["ticker"] in backfilled:
                        row["price_at_pred"] = backfilled[row["ticker"]]
                        conn.execute(
                            "UPDATE chief_decisions SET price_at_pred=? WHERE id=?",
                            (row["price_at_pred"], row["id"]),
                        )
                conn.commit()

    d1_tickers = [row["ticker"] for row in unscored if row["price_at_pred"] and row["price_at_pred"] > 0]
    prices = price_fn(d1_tickers, today.strftime("%Y%m%d")) if d1_tickers else {}
    outcome_date = today.isoformat()

    for row in pending:
        ticker = row["ticker"]
        already_scored = row["score"] is not None

        if already_scored:
            return_pct = row["return_pct"]
            score = row["score"]
        else:
            if not row["price_at_pred"] or row["price_at_pred"] <= 0:
                result["errors"].append(f"{ticker}: 예측일 종가 없음 — 복기 생략")
                continue
            if ticker not in prices:
                result["errors"].append(f"{ticker}: D+1 종가 없음")
                continue
            actual = prices[ticker]
            score  = _score_prediction(row["recommendation"], row["price_at_pred"], actual)
            return_pct = ((actual / row["price_at_pred"]) - 1.0) * 100

        try:
            lesson = await reflect_fn(row, return_pct, score)
            resolved = 1
        except Exception as e:
            logger.error(f"[decision_memory] {ticker} 복기 실패: {e}")
            result["errors"].append(f"{ticker}: 복기 실패 — {e}")
            lesson = ""
            resolved = 0

        with sqlite3.connect(DB_PATH) as conn:
            if already_scored:
                conn.execute(
                    "UPDATE chief_decisions SET resolved=?, lesson=? WHERE id=?",
                    (resolved, lesson, row["id"]),
                )
            else:
                conn.execute(
                    "UPDATE chief_decisions SET resolved=?, outcome_date=?, actual_price=?, return_pct=?, "
                    "score=?, lesson=? WHERE id=?",
                    (resolved, outcome_date, prices[ticker], return_pct, score, lesson, row["id"]),
                )
            conn.commit()
        if resolved:
            result["resolved_count"] += 1

    logger.info(f"[decision_memory] 복기 완료: {result['resolved_count']}건, 오류 {len(result['errors'])}건")
    return result


if __name__ == "__main__":
    import asyncio
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else None
    out = asyncio.run(resolve_pending(target))
    print(f"복기 {out['resolved_count']}건 (예측일 {out['target_date']})")
    for err in out["errors"]:
        print(f"  ⚠️ {err}")
    # 종가 없는 날(휴장)은 정상이라 exit 0. 복기 LLM 자체가 실패한 경우만 job 실패로 드러낸다.
    if any("복기 실패" in err for err in out["errors"]):
        sys.exit(1)
