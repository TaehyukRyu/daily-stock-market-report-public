"""
src/screening — 종목 스크리닝 파이프라인.

유니버스 110개를 4단계로 좁혀 "오늘 새 정보가 있는 종목 최대 8개(없으면 0개)"를 선정한다.

  Stage 1-A  stage1a_quant.py    pykrx 정량 지표 (무료, 빠름)
  Stage 1-B  stage1b_events.py   발행일 기준 뉴스 급증 + DART 공시 (무료)
  Stage 1-C  stage1c_news.py     매크로 뉴스 LLM 1회 호출 (유료)
  Stage 2    stage2_scorer.py    스코어링 & 선정
  진입점     screener.py         run_screening()
"""
