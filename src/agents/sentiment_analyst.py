"""
Sentiment Analyst — 종목 이벤트 해석자 (v3, 2026-09-21)

[v2 → v3 변경 이유] doc/2026-09-21_agent-audit.md §1-6 · §2-2
  v2는 `search_news(categories=economy/finance/industry)`로 **시장 전체 헤드라인 30건**을
  읽고 종목 뉴스는 언급 '건수'만 봤다. 그런데 스크리닝 v2는 "이 종목에 오늘 뉴스가
  급증했다 / 공시가 났다"로 종목을 고른다. 고른 이유를 아무도 읽지 않는 구조였다.
  실측: 35표 중 BUY 0건(0%), HOLD 66%. 프롬프트의 "긍정 80% 이상 → BUY" 조건이
  시장 전체 헤드라인에서는 사실상 도달 불가였다.

  v3는 **그 종목에 무슨 일이 났는지**를 읽는다:
    ① 스크리닝이 넘겨준 선정 사유 (공시 제목 · 뉴스 급증 배수 · 1-C 헤드라인 해석)
    ② 그 종목 헤드라인 원문 (mentions 테이블, 발행일 기준)
    ③ 최근 5일 수익률 — "이미 가격에 반영됐는가"를 묻기 위한 대조값

  시장 전체 분위기는 macro_economist·us_market_specialist가 이미 본다. 중복을 없앴다.

[이름을 유지하는 이유]
  하는 일은 바뀌었지만 agent_name은 `sentiment_analyst` 그대로다. 바꾸면
  agent_weights 행과 prediction_log 251행의 이력이 끊긴다. 의미가 달라진 시점은
  eval_method_log(`sentiment_v3`)에 남긴다.

[비용] gpt-4o-mini 1콜. MCP 뉴스 수집 호출이 빠져 v2보다 줄었다.
"""

import asyncio
import logging
from datetime import date, timedelta

from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.base_agent import create_structured_agent
from src.schemas.agent_output import AnalysisReport
from src.rag.context_injection import get_context_for_agent, inject_context_into_prompt
from src.screening.screen_context import selection_reason

logger = logging.getLogger(__name__)


SENTIMENT_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


# 헤드라인 조회 창. 스크리닝의 news_burst 최근 창과 같은 자를 쓴다.
FALLBACK_WINDOW_DAYS = 7      # 최근 창에 기사가 없을 때만 넓힌다
MAX_HEADLINES        = 25
DATA_COLLECT_TIMEOUT = 30


def _window(today: date) -> tuple[str, str]:
    """stage1b가 news_burst를 잰 것과 같은 '최근 창'(직전 평일 ~ 오늘)."""
    try:
        from src.screening.stage1b_events import recent_window
        days = recent_window(today)
        return days[0], days[-1]
    except Exception as e:
        logger.debug(f"[sentiment] recent_window 실패, 3일 창으로 대체: {e}")
        return (today - timedelta(days=3)).isoformat(), today.isoformat()


def _fetch_headlines(ticker: str, today: date) -> tuple[list[dict], str]:
    """(헤드라인, 창 설명). 최근 창이 비면 7일로 한 번만 넓힌다."""
    from src.data.mention_db import get_recent_mentions, init_db

    try:
        init_db()
    except Exception as e:
        logger.warning(f"[sentiment] mention_db init 실패 (계속): {e}")

    start, end = _window(today)
    rows = get_recent_mentions(ticker, start, end, limit=MAX_HEADLINES)
    if rows:
        return rows, f"{start}~{end}"

    wide_start = (today - timedelta(days=FALLBACK_WINDOW_DAYS)).isoformat()
    rows = get_recent_mentions(ticker, wide_start, end, limit=MAX_HEADLINES)
    return rows, f"{wide_start}~{end} (최근 창 0건이라 {FALLBACK_WINDOW_DAYS}일로 확대)"


async def _collect(ticker: str, today: date) -> dict:
    """헤드라인 조회. DB 읽기라 빠르지만 잠금 대기를 대비해 타임아웃을 건다."""
    try:
        rows, window = await asyncio.wait_for(
            asyncio.to_thread(_fetch_headlines, ticker, today),
            timeout=DATA_COLLECT_TIMEOUT,
        )
        return {"headlines": rows, "window": window}
    except Exception as e:
        logger.warning(f"[sentiment] {ticker} 헤드라인 조회 실패: {e}")
        return {"headlines": [], "window": "조회 실패"}


def _sentiment_counts(rows: list[dict]) -> dict[str, int]:
    out = {"positive": 0, "neutral": 0, "negative": 0}
    for r in rows:
        key = str(r.get("sentiment") or "neutral")
        if key in out:
            out[key] += 1
    return out


def _format_prompt(ticker: str, data: dict, ctx: dict) -> str:
    # [REDACTED] Proprietary data formatting logic
    return ""


async def run_sentiment_analyst(target_ticker: str | None = None,
                                screen_context: dict | None = None) -> AnalysisReport:
    """종목 이벤트 해석 에이전트 진입점.

    target_ticker:  분석 대상 종목. None이면 헤드라인 없이 '이벤트 없음'으로 답한다.
    screen_context: 오늘의 선정 사유 (src/screening/screen_context.py). 없으면 빈 dict.
    """
    ctx    = dict(screen_context or {})
    ticker = target_ticker or ctx.get("ticker") or ""

    data      = await _collect(ticker, date.today()) if ticker else {"headlines": [], "window": "종목 미지정"}
    formatted = _format_prompt(ticker or "(종목 미지정)", data, ctx)

    rag_context = get_context_for_agent(
        agent_name="sentiment_analyst",
        state_vars={"ticker": ticker, "date": date.today().isoformat()},
    )
    system_content = inject_context_into_prompt(SENTIMENT_SYSTEM_PROMPT, rag_context)

    agent  = create_structured_agent(model="gpt-4o-mini")
    report: AnalysisReport = await agent.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ])
    report.agent_name = "sentiment_analyst"

    # 종목 단위 판단이므로 항상 채워져 있어야 한다 (LLM이 Optional 필드를 빠뜨리는 일이 있다).
    if not report.selection_rationale:
        report.selection_rationale = (
            report.reasoning[-1] if report.reasoning else "이벤트 근거 부족으로 판단 보류"
        )
    return report
