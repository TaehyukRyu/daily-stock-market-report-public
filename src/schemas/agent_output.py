from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


class AnalysisReport(BaseModel):
    """8개 전문가 에이전트가 공통으로 출력하는 분석 보고서 스키마.

    LangChain의 with_structured_output()으로 LLM 출력을 강제한다.
    Quality Gate (계획서 8.2절 Pattern 2): confidence ≥ 0.6, reasoning ≥ 3단계,
    data_sources ≥ 2개 인용이 통과 조건이다.
    """

    agent_name: str = Field(
        description="에이전트 식별자. 예: 'macro_economist', 'kr_market_specialist'"
    )

    # ── 분석 대상 종목 (LLM이 채우지 않음 — 파이프라인이 사후 주입) ────────────
    # 이 필드가 없어서 prediction_logger가 r.get("ticker","")로 항상 빈 문자열을
    # 저장했고, feedback_evaluator가 빈 ticker를 전부 건너뛰어 D+1 채점이
    # 1,357건 중 0건 수행됐다 (2026-07-23 ~ 09-07). agent_weights도 seed
    # 0.142857에 고정된 채였다. 파이프라인이 종목을 확정하므로 LLM 출력이 아니라
    # 파이프라인에서 주입한다 (default=None → 구조화 출력 스키마에 영향 없음).
    ticker: Optional[str] = Field(
        default=None,
        description="분석 대상 종목 코드. 파이프라인이 주입하며 LLM은 채우지 않는다.",
    )

    ticker_name: Optional[str] = Field(
        default=None,
        description="분석 대상 종목명. 파이프라인이 주입하며 LLM은 채우지 않는다.",
    )

    needs_review: bool = Field(
        default=False,
        description="파이프라인이 설정. LLM 출력을 해석할 수 없어 사람 확인이 필요함(REVIEW). "
                    "HOLD로 날조하지 않기 위한 표식 (TradingAgents rating.REVIEW 관례). LLM은 채우지 않는다.",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="이 분석의 확신도. 0.0(전혀 확신 없음) ~ 1.0(매우 확신). "
                    "근거가 약하거나 데이터가 충돌할 때는 반드시 0.6 미만으로 낮출 것."
    )

    recommendation: Literal['BUY', 'SELL', 'HOLD'] = Field(
        description="투자 판단. BUY=매수 우호, SELL=매도 우호, HOLD=관망. "
                    "거시 에이전트는 시장 환경 판단을 BUY(리스크 ON)/SELL(리스크 OFF)/HOLD로 표현."
    )

    data_sufficient: bool = Field(
        default=True,
        description="분석에 필요한 데이터가 실제로 제공되었는지. "
                    "입력에 '수집 실패'나 error가 있어 근거로 쓸 수치가 없으면 false로 두고 "
                    "confidence를 0.2 이하로 낮출 것. "
                    "false일 때는 reasoning 1개, prediction_basis 1개만 써도 되며, "
                    "**없는 수치를 지어내지 말 것**. 데이터가 없다는 사실 자체를 쓰면 된다."
    )

    reasoning: list[str] = Field(
        min_length=1,
        description="Chain-of-Thought 형식의 사고 흐름을 단계별로 작성. "
                    "각 항목은 한 단계의 추론. data_sufficient=true면 최소 3단계 필수. "
                    "예: ['관찰 1', '관찰 1로부터 도출되는 함의', '결론']"
    )

    data_sources: list[str] = Field(
        default_factory=list,
        description="이 분석에 실제로 사용한 데이터 출처. "
                    "예: ['FRED DGS10 (4월 30일)', 'BOK ECOS 기준금리 (4월 25일)']. "
                    "data_sufficient=true면 단일 출처 의존을 막기 위해 최소 2개 이상 필수. "
                    "false면 비워도 된다."
    )

    selection_rationale: Optional[str] = Field(
        default=None,
        description="특정 종목을 선별한 이유. 종목 단위로 판단하는 에이전트만 작성. "
                    "거시 에이전트처럼 시장 전체를 보는 경우 None으로 둘 것."
    )

    prediction_basis: list[str] = Field(
        default_factory=list,
        description="예측을 뒷받침하는 구체적·정량적 근거. reasoning이 사고 과정이라면 "
                    "이것은 그 사고의 출발점이 된 실제 숫자/사실. "
                    "예: ['DXY 102.5, 전월 대비 -2.3%', '미 10Y 금리 4.0%, 60일 평균 대비 -20bp']. "
                    "data_sufficient=true면 최소 2개 필수. "
                    "false면 비워도 된다 (지어내지 말 것)."
    )

    risk_factors: list[str] = Field(
        min_length=1,
        description="이 예측이 틀릴 수 있는 시나리오나 리스크 요인. "
                    "Bull/Bear 토론에서 Bear 측이 활용하는 핵심 입력. "
                    "예: ['미 CPI 재가속 시 금리 재상승 가능', '중국 디플레 리스크 확산 시 원화 약세']. "
                    "최소 1개 이상 필수."
    )

    # ── 거래 파라미터 (chief_strategist 전용, 나머지 에이전트는 None) ──────────
    entry_price: Optional[float] = Field(
        default=None,
        description="추천 진입가. 현재가 기준, BUY 시에만 입력."
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="손절가. 기술적 지지선 또는 현재가 × (1 + stop_loss_pct/100)."
    )
    stop_loss_pct: Optional[float] = Field(
        default=None,
        description="손절 비율(%). 음수. 예: -4.0 → -4%. 범위: -3 ~ -6."
    )
    take_profit_1: Optional[float] = Field(
        default=None,
        description="1차 익절가. 손절폭 × 2 기준 (R:R 1:2 최소 보장)."
    )
    take_profit_2: Optional[float] = Field(
        default=None,
        description="2차 익절가. 손절폭 × 3 기준 (R:R 1:3)."
    )
    rr_ratio: Optional[float] = Field(
        default=None,
        description="Risk:Reward 비율. (take_profit_1 - entry_price) / (entry_price - stop_loss)."
    )
    position_size_pct: Optional[float] = Field(
        default=None,
        description="포지션 사이즈 (시드 대비 %). confidence × 10, 최대 15."
    )
    holding_period_weeks: Optional[int] = Field(
        default=None,
        description="예상 보유 기간 (주 단위)."
    )
    entry_strategy: Optional[str] = Field(
        default=None,
        description="진입 전략. '시장가' / '분할매수' / '지정가대기' 중 하나."
    )

    # ── 조건부 최소 길이 검증 ─────────────────────────────────────────────
    # [왜 Field(min_length=)에서 옮겼나]
    #   reasoning≥3 / prediction_basis≥2를 무조건 강제하면, MCP 수집이 실패해
    #   입력이 비었을 때도 LLM이 2개의 "정량적 근거"를 만들어내야만 스키마를
    #   통과할 수 있다. with_structured_output이 통과할 때까지 재시도하므로
    #   재시도 루프가 날조를 압박한다.
    #   프롬프트로는 "지어내지 마"라고 하면서 스키마로는 "채워라"라고 한 셈이다.
    #
    #   → data_sufficient=False라는 탈출구를 만들고, 그때만 길이 요구를 푼다.
    #     정상 분석(data_sufficient=True)의 품질 기준은 그대로 유지된다.
    @model_validator(mode="after")
    def _require_depth_when_data_exists(self):
        if not self.data_sufficient:
            return self
        if len(self.reasoning) < 3:
            raise ValueError(
                "data_sufficient=true면 reasoning은 3단계 이상이어야 한다. "
                "근거가 없다면 data_sufficient=false로 두고 지어내지 말 것."
            )
        if len(self.prediction_basis) < 2:
            raise ValueError(
                "data_sufficient=true면 prediction_basis는 2개 이상이어야 한다. "
                "제공된 수치가 없다면 data_sufficient=false로 둘 것."
            )
        if len(self.data_sources) < 2:
            raise ValueError(
                "data_sufficient=true면 data_sources는 2개 이상이어야 한다. "
                "출처를 2개 댈 수 없다면 data_sufficient=false로 둘 것."
            )
        return self


# 7개 전문가 에이전트의 시스템 프롬프트 끝에 붙는 공통 규칙 (base_agent.create_structured_agent).
# 2026-09-11 실행: quant·fundamental이 data_sufficient=true인 채 근거 1개로 스키마 위반 →
# 같은 프롬프트 3회 반복 → 폴백. 규칙을 명시하고 위반 시 1회만 피드백 재요청한다.
ABSTAIN_RULE = (
    "\n\n[출력 규칙 — 반드시 지킬 것]\n"
    "근거(prediction_basis)와 출처(data_sources)를 각각 2개 이상 댈 수 없으면 "
    "data_sufficient=false로 답하라. 그때는 두 목록을 비워도 된다. "
    "없는 수치나 출처를 지어내지 마라. data_sufficient=true로 답하려면 둘 다 2개 이상이어야 한다."
)