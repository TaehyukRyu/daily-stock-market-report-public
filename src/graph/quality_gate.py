"""
Quality Gate 노드 — 계획서 8.2절 Pattern 2

parallel_analysis가 끝난 뒤, debate/chief_strategist로 넘기기 전에
품질 기준 미달 에이전트 보고서를 걸러냅니다.

통과 조건 (AND — 세 조건 모두 충족해야 통과):
  ① confidence   ≥ 0.6   (LLM 자체 확신도)
  ② len(reasoning)    ≥ 3    (CoT 최소 3단계)
  ③ len(data_sources) ≥ 2    (데이터 출처 최소 2개)

소프트 폴백 (Graceful Degradation):
  통과 에이전트가 MIN_QUALIFIED(2개) 미만이면 필터를 적용하지 않고
  원본 전체를 qualified_reports에 담아 내려보냅니다.
  → 계획서 KPI "데일리 리포트 자동 생성률 > 95%" 유지 목적.
"""

from src.schemas.graph_state import GraphState
from src.schemas.agent_output import AnalysisReport

# ──────────────────────────────────────────
# 임계값 (계획서 8.2절 Pattern 2 고정값)
# ──────────────────────────────────────────
CONFIDENCE_MIN   = 0.6
REASONING_MIN    = 3
DATA_SOURCES_MIN = 2
MIN_QUALIFIED    = 2   # 이 미만이면 소프트 폴백 발동


# ──────────────────────────────────────────
# 단일 리포트 검사 함수
# ──────────────────────────────────────────

def _check_report(report: AnalysisReport) -> tuple[bool, list[str]]:
    """
    단일 AnalysisReport의 품질 기준 통과 여부를 반환합니다.

    Returns:
        (passed: bool, failure_reasons: list[str])
        passed가 True이면 failure_reasons는 빈 리스트.
    """
    failures: list[str] = []

    if report.confidence < CONFIDENCE_MIN:
        failures.append(
            f"confidence {report.confidence:.2f} < {CONFIDENCE_MIN}"
        )

    if len(report.reasoning) < REASONING_MIN:
        failures.append(
            f"reasoning {len(report.reasoning)}단계 < {REASONING_MIN}단계"
        )

    if len(report.data_sources) < DATA_SOURCES_MIN:
        failures.append(
            f"data_sources {len(report.data_sources)}개 < {DATA_SOURCES_MIN}개"
        )

    return (len(failures) == 0), failures


# ──────────────────────────────────────────
# Quality Gate 노드
# ──────────────────────────────────────────

async def quality_gate_node(state: GraphState) -> dict:
    """
    LangGraph 노드 함수.

    위치: parallel_analysis → [quality_gate] → debate → chief_strategist

    동작 순서:
      1. analysis_reports에서 chief_strategist 제외 (아직 실행 전)
      2. 각 에이전트 보고서에 _check_report() 적용
      3. 통과 에이전트 목록을 qualified_reports에 저장
      4. 통과 < MIN_QUALIFIED이면 소프트 폴백
    """
    print(f"\n[2.5/5] quality_gate — 품질 검사")

    # chief_strategist는 필터 대상에서 제외 (아직 실행되지 않았음)
    candidates = [
        r for r in state.analysis_reports
        if r.agent_name != "chief_strategist"
    ]

    passed:      list[AnalysisReport] = []
    failed_logs: list[str]            = []

    for report in candidates:
        ok, failures = _check_report(report)
        if ok:
            passed.append(report)
            print(f"  ✅ {report.agent_name:<28} confidence={report.confidence:.2f}  통과")
        else:
            reason = ", ".join(failures)
            print(f"  ❌ {report.agent_name:<28} 탈락 ({reason})")
            failed_logs.append(f"[QualityGate] {report.agent_name} 탈락: {reason}")

    # ── 소프트 폴백 ──────────────────────────────
    # G3: confidence 0.0은 "의견 없음(abstain)"이지 "관망 의견"이 아니다. 폴백에서도 투표에
    # 넣지 않는다 (ai-hedge-fund construction.py: "'no opinion' must not masquerade as
    # 'opinion: neutral'"). 전부 abstain이면 빈 목록 → chief가 HOLD 폴백을 낸다.
    if len(passed) < MIN_QUALIFIED:
        voting   = [r for r in candidates if r.confidence > 0.0]
        abstained = len(candidates) - len(voting)
        print(
            f"  ⚠️  통과 {len(passed)}/{len(candidates)}개 < {MIN_QUALIFIED}개 "
            f"→ 소프트 폴백: 원본 {len(voting)}개 사용 (abstain {abstained}개 제외)"
        )
        failed_logs.append(
            f"[QualityGate] 소프트 폴백 발동 — "
            f"통과 {len(passed)}/{len(candidates)}개, 원본 {len(voting)}개 사용, abstain {abstained}개 제외"
        )
        qualified = voting
    else:
        qualified = passed

    print(f"  → 최종 통과: {len(qualified)}/{len(candidates)}개 에이전트\n")

    return {
        "qualified_reports": qualified,
        "error_log":         failed_logs,   # operator.add 리듀서 → 기존 로그에 append
    }