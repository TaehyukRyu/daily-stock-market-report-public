"""
src/agents/quant_rule_agent.py

LLM 없는 투표자 — 순수 수식으로 AnalysisReport를 만든다 (G4).

[가져온 것] ai-hedge-fund (MIT) hedge_fund/signals/base.py의 설계 원칙만:
  "An alpha model is anything that produces a forecast… both quant signals (PEAD, regime) and
   LLM investor agents are alpha models — they all implement this interface" —
  수식 모델과 LLM 모델이 **같은 출력 인터페이스**로 투표한다. 코드는 우리 스키마(AnalysisReport)에
  맞춰 새로 썼다 (ai-hedge-fund의 Signal/blend 코드는 langchain-anthropic 의존이 있는 패키지라
  60줄 순수 함수 원칙으로 발상만).

[왜 필요한가]
  S1 5-1·6-2④: 7개 중 5개가 같은 모델(gpt-4o-mini)이고 3개가 같은 OHLCV를 본다.
  오류 상관이 0인 투표자는 LLM이 아닌 것뿐이다. 이 에이전트는 stage1a가 이미 계산하는
  4개 지표(5일 수익률·거래량비·MA 돌파·골든크로스)를 그대로 읽어 규칙으로 판단한다.
  LLM 호출 0, 비용 0, 결정론.

[규칙 — 임시값. G1 성과 데이터로 조정한다]
  BUY : (골든크로스 or MA 돌파) and 5일 수익률 ≥ 0
  SELL: 5일 수익률 ≤ -3% and 골든크로스·MA 돌파 없음
  HOLD: 그 외
  confidence = 0.6 + 0.1 × signal_count (최대 0.9). 데이터 부족(봉 < 21) → data_sufficient=False, 0.0
"""

from __future__ import annotations

import logging

from src.data.ohlcv_cache import get_ohlcv_series
from src.schemas.agent_output import AnalysisReport
from src.screening.stage1a_quant import PRICE_CHANGE_THRESHOLD, _compute_signals

logger = logging.getLogger(__name__)

AGENT_NAME = "quant_rule_agent"


def build_report(ticker: str, signals: dict | None) -> AnalysisReport:
    """signals = stage1a._compute_signals() 출력. None이면 데이터 부족 리포트."""
    if not signals:
        return AnalysisReport(
            agent_name=AGENT_NAME, confidence=0.0, recommendation="HOLD", data_sufficient=False,
            reasoning=["ohlcv_cache에 21거래일 미만 — 지표 계산 불가"],
            data_sources=["ohlcv_cache", "stage1a_quant"],
            prediction_basis=["데이터 부족"], risk_factors=["지표 없음"],
        )

    pc   = float(signals["price_change_5d"])
    gc   = bool(signals["golden_cross"])
    mb   = bool(signals["ma_breakout"])
    vol  = bool(signals["volume_signal"])
    cnt  = int(signals["signal_count"])

    if (gc or mb) and pc >= 0:
        rec = "BUY"
    elif pc <= -PRICE_CHANGE_THRESHOLD and not (gc or mb):
        rec = "SELL"
    else:
        rec = "HOLD"

    conf = min(0.9, 0.6 + 0.1 * cnt)
    trend = "상승" if pc > 0 else "하락" if pc < 0 else "보합"
    return AnalysisReport(
        agent_name=AGENT_NAME,
        confidence=conf,
        recommendation=rec,
        reasoning=[
            f"5일 수익률 {pc*100:+.2f}% ({trend}), 거래량 신호 {'있음' if vol else '없음'}",
            f"MA20/60 돌파 {'있음' if mb else '없음'}, 5-20 골든크로스 {'있음' if gc else '없음'} → 신호 {cnt}/4",
            f"규칙: {'추세 전환 + 비하락 → BUY' if rec == 'BUY' else '급락 + 추세 신호 없음 → SELL' if rec == 'SELL' else '조건 미충족 → HOLD'}",
        ],
        data_sources=["ohlcv_cache", "stage1a_quant"],
        prediction_basis=[
            f"price_change_5d={pc:+.4f}",
            f"golden_cross={gc}, ma_breakout={mb}, volume_signal={vol}, signal_count={cnt}",
        ],
        risk_factors=["일봉 공개 지표만 사용 — 정보 우위 없음 (S1 2-3). 다양성 확보용 투표자"],
        selection_rationale=f"{ticker}: 규칙 기반 {rec}",
    )


async def run_quant_rule_agent(target_ticker: str | None = None) -> AnalysisReport:
    if not target_ticker:
        return build_report("", None)
    try:
        series = get_ohlcv_series(target_ticker, days=65)
        signals = _compute_signals(series) if series else None
    except Exception as e:
        logger.warning(f"[quant_rule_agent] {target_ticker} 지표 계산 실패: {e}")
        signals = None
    report = build_report(target_ticker, signals)
    report.ticker = target_ticker
    return report
