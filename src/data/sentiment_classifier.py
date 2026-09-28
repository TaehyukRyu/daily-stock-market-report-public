"""
src/data/sentiment_classifier.py

뉴스 헤드라인 3-class 감성 분류기

[분류 클래스]
  positive — 주가에 긍정적 (실적 호조, 수주, 투자 확대 등)
  neutral  — 중립적 (단순 현황 보도, 인사 등)
  negative — 주가에 부정적 (실적 악화, 소송, 규제 등)

[설계 원칙]
  - 배치 처리: 기사 여러 건을 한 번에 분류 (API 호출 횟수 최소화)
  - JSON 응답 강제: 형식 이탈 최소화
  - gpt-4o-mini 사용 (비용 절감)
"""

import json
import os
from openai import AsyncOpenAI


CLASSIFIER_MODEL = "gpt-4o-mini"
BATCH_SIZE = 20   # 한 번 API 호출에 처리할 최대 기사 수

CLASSIFIER_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


async def classify_headlines(headlines: list[str]) -> list[dict]:
    """
    뉴스 헤드라인 목록을 배치로 분류.

    Args:
        headlines: 분류할 헤드라인 목록

    Returns:
        [{"index": 0, "sentiment": "positive", "score": 0.85}, ...]
        실패 시 neutral 0.5로 폴백
    """
    if not headlines:
        return []

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []

    # 배치 단위로 처리
    for batch_start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[batch_start : batch_start + BATCH_SIZE]

        # 번호 붙여서 프롬프트 구성
        numbered = "\n".join(
            f"{i}. {title}" for i, title in enumerate(batch)
        )
        human_prompt = f"다음 뉴스 헤드라인들을 분류해주세요:\n\n{numbered}"

        try:
            response = await client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user",   "content": human_prompt},
                ],
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            batch_results = parsed.get("results", [])

            # index를 전체 리스트 기준으로 보정
            for item in batch_results:
                item["index"] = item["index"] + batch_start
                results.append(item)

        except Exception as e:
            print(f"  ⚠️ 감성 분류 실패 (배치 {batch_start}~): {e}")
            # 폴백: 해당 배치 전체 neutral 처리
            for i in range(len(batch)):
                results.append({
                    "index":     batch_start + i,
                    "sentiment": "neutral",
                    "score":     0.5,
                })

    # index 순 정렬 후 반환
    results.sort(key=lambda x: x["index"])
    return results


async def classify_single(headline: str) -> dict:
    """단일 헤드라인 분류 (테스트/개별 호출용)."""
    results = await classify_headlines([headline])
    if results:
        return results[0]
    return {"index": 0, "sentiment": "neutral", "score": 0.5}
