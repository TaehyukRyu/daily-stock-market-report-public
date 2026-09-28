"""
src/evaluation/backtest — Baseline 백테스트 (LLM 0콜).

사전 등록: doc/2026-09-13_backtest-prereg.md  (실행 전 확정. 여기 코드는 그 문서를 구현한다)
실행:      python -m src.evaluation.backtest.run --all
기록:      doc/backtest/backtest_runs.csv (모든 실행), doc/backtest/results/*.json

모듈
  data.py      가격(yfinance)·벤치마크(^KS200)·후보 풀(네이버 시총 순위)·월별 유니버스 복원
  signals.py   Stage 1-A 4개 지표를 벡터화 (stage1a._compute_signals와 동일 결과, 테스트로 고정)
  strategy.py  quant_rule_agent BUY 규칙 + 순위 + 상위 N 선정
  engine.py    H거래일 리밸런싱 시뮬레이션, 비용, 일별 자산곡선, 거래 원장
  metrics.py   CAGR·MDD·Sharpe·승률·손익비·초과수익·손익분기 승률
  controls.py  무작위 대조군·민감도·walk-forward·국면·비용 민감도·유의성
  run.py       CLI, 실행 기록
"""
