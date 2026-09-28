"""종목 헤드라인 관련도 — 판정서 §3-1: 관련도 중앙값 30%, 로보티즈 4%."""
import pytest

from src.data.headline_relevance import aliases, is_relevant


@pytest.mark.parametrize("title, name, ok", [
    ("삼성물산, 1.8조원 성수3지구 재개발 시공사 선정", "삼성물산", True),
    ("외인·기관 '사자'에 코스닥 1%대 상승", "로보티즈", False),
    ("에버랜드, 무안경 3D로 호러 체험 강화", "삼성물산", False),
    ("SK 하이닉스 HBM4 양산", "SK하이닉스", True),
    ("하이닉스 주가 급등", "SK하이닉스", True),
    ("신한금융, 4분기 배당 확대", "신한지주", True),
    ("두산에너빌리티 원전 수주", "두산에너빌리티", True),
    ("코스피 6715 마감", "기업은행", False),
    ("중소기업 대출 증가", "기업은행", False),
    ("삼성전자 HBM 공급", "삼성물산", False),
    ("하이브 신인 데뷔", "SK하이닉스", False),
])
def test_is_relevant(title, name, ok):
    assert is_relevant(title, name) is ok


def test_aliases_are_at_least_two_chars():
    a = aliases("LG에너지솔루션")
    assert "LG에너지솔루션" in a and all(len(x) >= 2 for x in a)


def test_mention_tracker_drops_irrelevant():
    from src.data import mention_tracker as mt
    kept = mt._filter_relevant([("삼성물산 수주", "2026-09-21"), ("코스피 마감", "2026-09-21")], "삼성물산")
    assert kept == [("삼성물산 수주", "2026-09-21")]


def test_mention_tracker_keeps_all_when_name_unknown():
    from src.data import mention_tracker as mt
    fresh = [("아무 기사", "2026-09-21")]
    assert mt._filter_relevant(fresh, "") == fresh


@pytest.mark.parametrize("title, name", [
    ("네이버 웹툰 美 상장 추진", "NAVER"),
    ("포스코홀딩스 2분기 실적 발표", "POSCO홀딩스"),
    ("에쓰오일 정제마진 개선", "S-Oil"),
    ("LS일렉트릭 북미 수주", "LS ELECTRIC"),
    ("LS ELECTRIC 북미 수주", "LS ELECTRIC"),
])
def test_latin_names_match_hangul_spelling(title, name):
    """영문 사명은 기사에서 한글로 더 자주 쓴다 — 유니버스 110 중 NAVER·POSCO홀딩스·S-Oil·LS ELECTRIC."""
    assert is_relevant(title, name) is True
