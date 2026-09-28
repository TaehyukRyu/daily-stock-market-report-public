"""
src/data/daily_mention_stats.py

일별 언급량 통계 집계 및 sentiment_analyst 입력 데이터 생성

[역할]
  - mentions DB에서 일별 통계 조회
  - 변화율(5일 전 대비) 계산
  - '후행 가능성' 경고 생성 (언급량 급증 + RSI 과매수 조합 시)
  - sentiment_analyst가 사용할 구조화 데이터 반환
"""

from datetime import date
from src.data.mention_db import get_daily_stats, get_mention_change_rate, init_db


# ─────────────────────────────────────────────────
# 신호 품질 판단
# ─────────────────────────────────────────────────

def _assess_signal_quality(
    change_rate: float | None,
    positive_ratio: float,
    negative_ratio: float,
) -> str:
    """
    언급량 변화율 + 감성 비율로 신호 품질 판단.

    언급량이 갑자기 많아지는 건 선행/후행 양쪽 가능성 있음.
    단독 판단 금지 — RSI/수급과 교차 확인 권고 메시지 포함.
    """
    if change_rate is None:
        return "데이터 부족 (5일 미만)"

    if change_rate >= 1.5:   # 150% 이상 급증
        if positive_ratio >= 0.6:
            return "급증_긍정우세 (선행 가능성, RSI/수급 교차 확인 필요)"
        elif negative_ratio >= 0.5:
            return "급증_부정우세 (악재 노출 — SELL 신호 강화)"
        else:
            return "급증_혼조 (후행 가능성 — 고점 주의)"
    elif change_rate >= 0.5:  # 50~150% 증가
        return "완만한_증가 (추세 확인 중)"
    elif change_rate <= -0.3: # 30% 이상 감소
        return "감소 (관심도 하락)"
    else:
        return "보합 (변화 미미)"


# ─────────────────────────────────────────────────
# 단일 종목 요약
# ─────────────────────────────────────────────────

def get_ticker_mention_summary(
    ticker: str,
    today: str | None = None,
    lookback_days: int = 7,
) -> dict:
    """
    단일 종목의 언급량 요약 데이터 반환.

    sentiment_analyst가 이 데이터를 입력으로 받아 분석합니다.

    Returns:
        {
            "ticker": "005930",
            "today": "2026-05-11",
            "mention_count_today": 45,
            "mention_count_5d_avg": 28,
            "change_rate": 0.61,
            "positive_ratio": 0.42,
            "neutral_ratio": 0.35,
            "negative_ratio": 0.23,
            "signal_quality": "완만한_증가 (추세 확인 중)",
            "recent_trend": [{"date": ..., "mention_count": ..., "positive_ratio": ...}, ...]
            "data_available": True
        }
    """
    target_date = today or date.today().isoformat()
    stats = get_daily_stats(ticker, days=lookback_days)

    if not stats:
        return {
            "ticker":              ticker,
            "today":               target_date,
            "mention_count_today": 0,
            "mention_count_5d_avg": 0,
            "change_rate":         None,
            "positive_ratio":      0.0,
            "neutral_ratio":       1.0,
            "negative_ratio":      0.0,
            "signal_quality":      "데이터 없음 (크롤링 필요)",
            "recent_trend":        [],
            "data_available":      False,
        }

    # 오늘 데이터 (가장 최신)
    today_stats = next(
        (s for s in reversed(stats) if s["date"] == target_date),
        stats[-1],  # 오늘 데이터 없으면 가장 최신 날짜 사용
    )

    # 5일 평균 언급량
    mention_counts = [s["mention_count"] for s in stats]
    avg_5d = round(sum(mention_counts) / len(mention_counts), 1)

    # 변화율 (DB에서 직접 조회)
    change_rate = get_mention_change_rate(ticker, compare_days=5)

    # 신호 품질
    signal_quality = _assess_signal_quality(
        change_rate     = change_rate,
        positive_ratio  = today_stats.get("positive_ratio", 0.0),
        negative_ratio  = today_stats.get("negative_ratio", 0.0),
    )

    # 최근 추세 (날짜, 언급수, 긍정비율만)
    recent_trend = [
        {
            "date":           s["date"],
            "mention_count":  s["mention_count"],
            "positive_ratio": s["positive_ratio"],
            "negative_ratio": s["negative_ratio"],
        }
        for s in stats
    ]

    return {
        "ticker":               ticker,
        "today":                target_date,
        "mention_count_today":  today_stats.get("mention_count", 0),
        "mention_count_5d_avg": avg_5d,
        "change_rate":          change_rate,
        "positive_ratio":       today_stats.get("positive_ratio", 0.0),
        "neutral_ratio":        today_stats.get("neutral_ratio", 1.0),
        "negative_ratio":       today_stats.get("negative_ratio", 0.0),
        "signal_quality":       signal_quality,
        "recent_trend":         recent_trend,
        "data_available":       True,
    }


# ─────────────────────────────────────────────────
# 여러 종목 일괄 요약
# ─────────────────────────────────────────────────

def get_universe_mention_summary(
    tickers: list[str],
    today: str | None = None,
) -> list[dict]:
    """
    유니버스 전체 종목의 언급량 요약 반환.
    데이터 있는 종목만 포함 (data_available=True).
    """
    summaries = [get_ticker_mention_summary(t, today=today) for t in tickers]
    return [s for s in summaries if s["data_available"]]


# ─────────────────────────────────────────────────
# sentiment_analyst용 포맷팅
# ─────────────────────────────────────────────────

def format_mention_context(tickers: list[str], today: str | None = None) -> str:
    """
    sentiment_analyst 프롬프트에 삽입할 언급량 컨텍스트 문자열 생성.

    데이터가 없으면 빈 문자열 반환 (하위 호환 유지).
    """
    summaries = get_universe_mention_summary(tickers, today=today)
    if not summaries:
        return ""

    lines = ["=== 종목별 언급량 추세 (최근 5일) ===\n"]
    for s in summaries:
        change_str = (
            f"{s['change_rate']:+.0%}" if s["change_rate"] is not None else "N/A"
        )
        lines.append(
            f"[{s['ticker']}] 오늘 {s['mention_count_today']}건 "
            f"(5일평균 {s['mention_count_5d_avg']}건, 변화율 {change_str}) | "
            f"긍정 {s['positive_ratio']:.0%} / 부정 {s['negative_ratio']:.0%} | "
            f"신호: {s['signal_quality']}"
        )
    lines.append(
        "\n⚠️ 언급량 급증은 선행/후행 양쪽 가능성 있음 — RSI/수급 지표와 반드시 교차 확인"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────
# 단독 실행 (테스트용)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    init_db()

    test_tickers = ["005930", "000660"]
    print(format_mention_context(test_tickers))
