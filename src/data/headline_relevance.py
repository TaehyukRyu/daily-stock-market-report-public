"""
종목 헤드라인 관련도. 판정서(doc/2026-09-22_screening-verdict.md §3-1): NAVER 종목명 검색이
시장 기사를 함께 돌려줘 저장된 헤드라인의 70%가 그 종목 얘기가 아니었다.
제목에 종목명·약칭이 없으면 버린다. news_burst와 sentiment v3가 같은 입력을 쓴다.
"""
from __future__ import annotations

import re

_SUFFIXES = ("홀딩스", "지주", "에너빌리티", "에어로스페이스", "솔루션", "바이오로직스")
_PREFIXES = ("SK", "LG", "HD", "DB", "KB", "CJ", "GS", "LS", "NH")
# 영문 사명의 기사 속 한글 표기 (유니버스 110 중 한글로 더 자주 쓰는 것, 2026-09-23 확인)
_HANGUL = {"NAVER": "네이버", "POSCO": "포스코", "S-OIL": "에쓰오일", "LSELECTRIC": "LS일렉트릭"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").upper()


def aliases(ticker_name: str) -> set[str]:
    n = _norm(ticker_name)
    out = {n}
    for suf in _SUFFIXES:
        if n.endswith(suf) and len(n) - len(suf) >= 2:
            out.add(n[: -len(suf)])
    for pre in _PREFIXES:
        if n.startswith(pre) and len(n) - len(pre) >= 3:
            out.add(n[len(pre):])
    for a in list(out):
        for en, ko in _HANGUL.items():
            if en in a:
                out.add(a.replace(en, ko))
    # 앞 2~3글자 자르기는 하지 않는다 — "기업은행"→"기업", "SK하이닉스"→"하이"가 남의 기사를 다 잡는다
    return {a for a in out if len(a) >= 2}


def is_relevant(title: str, ticker_name: str) -> bool:
    t = _norm(title)
    return any(a in t for a in aliases(ticker_name))
