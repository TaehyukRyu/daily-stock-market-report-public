"""
tests/test_position_tracker.py

add_position()의 손절가·목표가 전파 회귀 테스트.

[배경]
  이전 동작: add_position()에 stop_loss 파라미터가 아예 없어
             stop_loss_price = entry_price × (1 - 0.07)로 무조건 계산됨.
             chief_strategist가 -4%를 계산해 리포트에 표시해도
             DB에는 -7%가 저장되고, update_current_prices()는 그 -7%를 감시했다.
             → 리포트가 보여준 손절선과 시스템이 감시하는 손절선이 달랐다.

  현재 동작: stop_loss / target_price 인자를 받아 그대로 저장.
             인자가 없을 때만 비율 기본값으로 대체하고 경고를 남긴다.

핵심 테스트는 test_pipeline_shaped_call_matches_report — 파이프라인이 실제로
넘기는 형태로 호출했을 때 DB 감시값이 리포트 표시값과 일치하는지 확인한다.
"""

import sqlite3
import sys
import types
from pathlib import Path

import pytest

from src.data import position_tracker as pt


# ─────────────────────────────────────────────────────────
# 픽스처 — 테스트마다 격리된 임시 DB 사용
# ─────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    """position_tracker.DB_PATH를 임시 파일로 바꾸고 테이블을 만든다."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(pt, "DB_PATH", path)
    pt.init_position_table()
    return path


def _row(db_path: Path, ticker: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT * FROM positions WHERE ticker=? ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    return dict(r) if r else {}


# ─────────────────────────────────────────────────────────
# 1. 인자로 준 값이 그대로 저장되는가
# ─────────────────────────────────────────────────────────

def test_stop_loss_argument_is_stored_verbatim(db):
    """chief가 계산한 손절가가 그대로 stop_loss_price에 들어가야 한다."""
    pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                    stop_loss=67_200)          # -4%
    assert _row(db, "005930")["stop_loss_price"] == 67_200


def test_target_price_argument_is_stored_verbatim(db):
    pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                    stop_loss=67_200, target_price=75_600)
    assert _row(db, "005930")["target_price"] == 75_600


def test_stop_loss_is_not_overwritten_by_ratio_default(db):
    """회귀 방지 — 인자를 줬는데 기본 비율이 덮어쓰면 실패.

    기본값(-4%)과 **다른** 값을 줘야 변별력이 있다. chief가 허용하는 -6%를 쓴다.
    """
    entry = 70_000
    ratio_default = round(entry * (1 - pt.STOP_LOSS_RATIO), 0)   # -4% → 67,200
    pt.add_position(ticker="000660", entry_price=entry, quantity=0,
                    stop_loss=65_800)                            # -6%
    stored = _row(db, "000660")["stop_loss_price"]
    assert stored == 65_800
    assert stored != ratio_default


# ─────────────────────────────────────────────────────────
# 2. 인자가 없을 때의 폴백 (CLI 수동 입력 경로 보존)
# ─────────────────────────────────────────────────────────

def test_stop_loss_falls_back_to_ratio_when_omitted(db):
    """CLI 수동 입력 등 stop_loss 미지정 시 기본 비율을 쓴다.

    [2026-09-14 CR-6] 기본값이 -7% → -4%로 바뀌었다. chief 규칙과 같은 값이어야
    리포트에 표시된 손절선과 감시 손절선이 어긋나지 않는다.
    """
    pt.add_position(ticker="005380", entry_price=100_000, quantity=0)
    assert _row(db, "005380")["stop_loss_price"] == 96_000


def test_stop_loss_fallback_emits_warning(db, caplog):
    """폴백은 조용히 넘어가면 안 된다 — 경고 로그가 남아야 한다."""
    with caplog.at_level("WARNING"):
        pt.add_position(ticker="035420", entry_price=100_000, quantity=0)
    assert any("stop_loss 미지정" in r.message for r in caplog.records)


def test_target_price_prefers_take_profit_1_over_ratio(db):
    """target_price가 없어도 take_profit_1이 있으면 그것을 감시값으로 쓴다."""
    pt.add_position(ticker="051910", entry_price=100_000, quantity=0,
                    stop_loss=94_000, take_profit_1=112_000)     # -6% → R:R 2.0
    row = _row(db, "051910")
    assert row["target_price"] == 112_000
    assert row["target_price"] != round(100_000 * (1 + pt.DEFAULT_TARGET_RATIO), 0)


def test_target_price_falls_back_to_ratio_when_nothing_given(db):
    """[CR-6] +15% → +8% (R:R 2.0 기준 리스크 4% × 2)."""
    pt.add_position(ticker="207940", entry_price=100_000, quantity=0,
                    stop_loss=96_000)
    assert _row(db, "207940")["target_price"] == 108_000


# ─────────────────────────────────────────────────────────
# 3. 핵심 회귀 테스트 — 파이프라인이 넘기는 형태
# ─────────────────────────────────────────────────────────

def test_pipeline_shaped_call_matches_report(db):
    """
    pipeline.chief_strategist_node가 실제로 호출하는 형태를 그대로 재현한다.

    chief_strategist의 거래 파라미터 계산 규칙:
      entry=70,000 / stop_loss_pct=-6% → stop_loss=65,800
      take_profit_1 = entry + (entry - stop_loss) × 2 = 78,400  (R:R 1:2)
      take_profit_2 = entry + (entry - stop_loss) × 3 = 82,600  (R:R 1:3)

    리포트에 표시되는 값은 stop_loss=65,800 / take_profit_1=78,400 이므로,
    update_current_prices()가 읽는 stop_loss_price / target_price도
    같은 값이어야 한다.

    [CR-6] 기본 비율이 -4%/+8%로 바뀌어 기본값과 겹치지 않는 -6% 시나리오를 쓴다.
    """
    entry, stop, tp1, tp2 = 70_000, 65_800, 78_400, 82_600

    pt.add_position(
        ticker               = "005930",
        entry_price          = entry,
        quantity             = 0,
        stop_loss            = stop,
        target_price         = tp1,
        allocation_pct       = 7.2,
        take_profit_1        = tp1,
        take_profit_2        = tp2,
        holding_period_weeks = 3,
        rr_ratio             = 2.0,
        entry_rationale      = "테스트",
    )

    row = _row(db, "005930")

    # 감시 컬럼 == 리포트 표시값
    assert row["stop_loss_price"] == stop, "감시 손절가가 리포트 손절가와 다름"
    assert row["target_price"]    == tp1,  "감시 목표가가 리포트 1차 익절가와 다름"

    # 참조 컬럼도 보존
    assert row["take_profit_1"] == tp1
    assert row["take_profit_2"] == tp2
    assert row["rr_ratio"] == 2.0
    assert row["allocation_pct"] == 7.2

    # 하드코딩 기본값이 끼어들지 않았는지
    assert row["stop_loss_price"] != round(entry * (1 - pt.STOP_LOSS_RATIO), 0)
    assert row["target_price"]    != round(entry * (1 + pt.DEFAULT_TARGET_RATIO), 0)


def test_alert_thresholds_read_the_propagated_values(db, monkeypatch):
    """
    update_current_prices()의 손절·목표 판정이 전달된 값을 기준으로 도는지 확인.

    전달 손절가 65,800(-6%), 기본 비율 손절가 67,200(-4%) 사이인 66,500원을 현재가로 둔다.
      - 전달값 65,800 기준 → 아직 위 → 경보 없음 (기대)
      - 기본값 67,200을 보고 있으면 → 이하 → 경보 발생 (예전 버그)
    경보가 "없어야" 전달값을 실제로 읽는다는 뜻이다.
    """
    pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                    stop_loss=65_800, target_price=78_400, take_profit_1=78_400)

    # update_current_prices()는 함수 안에서 `from pykrx import stock`을 하므로
    # sys.modules에 가짜 모듈을 심어야 가로챌 수 있다.
    class _Series:
        iloc = [66_500.0]          # 현재가 66,500원

    class _FakeDF:
        empty = False
        columns = ["종가"]
        def __getitem__(self, key):
            return _Series()

    fake_stock = types.SimpleNamespace(
        get_market_ohlcv_by_date=lambda *a, **k: _FakeDF()
    )
    fake_pykrx = types.ModuleType("pykrx")
    fake_pykrx.stock = fake_stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)
    monkeypatch.setitem(sys.modules, "pykrx.stock", fake_stock)

    results = pt.update_current_prices()
    assert len(results) == 1
    assert results[0]["alert"] is None, (
        "전달된 손절가 65,800 기준이면 66,500은 아직 위라 경보가 없어야 한다. "
        "기본 비율 67,200을 보고 있으면 stop_loss 경보가 뜬다."
    )


# ─────────────────────────────────────────────────────────
# 4. CR-14 — 중복 오픈 포지션 (2026-09-14)
# ─────────────────────────────────────────────────────────

def test_same_day_rerun_does_not_insert_or_mutate(db):
    """같은 날 재실행은 새 행을 만들지도, 기존 행을 바꾸지도 않는다 (CR-14 + S1)."""
    first = pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                            entry_date="2026-09-14", stop_loss=67_200,
                            take_profit_1=75_600, allocation_pct=5.0)
    second = pt.add_position(ticker="005930", entry_price=71_000, quantity=0,
                             entry_date="2026-09-14", stop_loss=68_160,
                             take_profit_1=76_680, allocation_pct=5.0)
    assert first == second, "새 행을 만들지 않는다"
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT entry_price, stop_loss_price FROM positions WHERE status='open'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 70_000 and rows[0][1] == 67_200, "진입 시점 값이 유지된다"


def test_later_day_does_not_reregister(db):
    """다른 날이면 기존 포지션을 유지한다 — 최초 추천 시점 기록이 사라지면 안 된다."""
    first = pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                            entry_date="2026-09-14", stop_loss=67_200, allocation_pct=5.0)
    again = pt.add_position(ticker="005930", entry_price=80_000, quantity=0,
                            entry_date="2026-09-20", stop_loss=76_800, allocation_pct=5.0)
    assert again == first
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT entry_date, entry_price FROM positions WHERE status='open'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "2026-09-14" and rows[0][1] == 70_000, "최초 진입 기록 보존"


def test_invested_pct_does_not_inflate_on_rerun(db):
    """CR-14의 실제 증상: 투자중 비율이 부풀지 않는다."""
    for _ in range(3):
        pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                        entry_date="2026-09-14", stop_loss=67_200, allocation_pct=5.0)
    assert pt.get_portfolio_pct_summary()["invested_pct"] == pytest.approx(5.0)


def test_dedupe_removes_extras_keeping_earliest(db):
    """방어책 이전에 쌓인 중복을 정리한다. 남기는 것은 가장 이른 행."""
    now = "2026-09-14T00:00:00"
    with sqlite3.connect(db) as conn:
        for price in (70_000, 71_000, 72_000):
            conn.execute(
                "INSERT INTO positions (ticker,ticker_name,entry_date,entry_price,quantity,"
                "stop_loss_price,target_price,status,created_at,updated_at,allocation_pct) "
                "VALUES ('005930','삼성','2026-09-14',?,0,67200,75600,'open',?,?,5.0)",
                (price, now, now))
        conn.execute(
            "INSERT INTO positions (ticker,ticker_name,entry_date,entry_price,quantity,"
            "stop_loss_price,target_price,status,created_at,updated_at,allocation_pct) "
            "VALUES ('000660','하이닉스','2026-09-14',100000,0,96000,108000,'open',?,?,5.0)",
            (now, now))
        conn.commit()

    removed = pt.dedupe_open_positions()
    assert len(removed) == 2
    assert {r["ticker"] for r in removed} == {"005930"}
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT ticker, entry_price FROM positions WHERE status='open' ORDER BY ticker").fetchall()
    assert rows == [("000660", 100000.0), ("005930", 70000.0)], "가장 이른 행만 남는다"
    assert pt.dedupe_open_positions() == [], "멱등"


def test_dedupe_ignores_closed_positions(db):
    """청산된 행은 건드리지 않는다 — 같은 종목을 두 번 거래했을 수 있다."""
    now = "2026-09-14T00:00:00"
    with sqlite3.connect(db) as conn:
        for status in ("closed", "closed", "open"):
            conn.execute(
                "INSERT INTO positions (ticker,ticker_name,entry_date,entry_price,quantity,"
                "stop_loss_price,target_price,status,created_at,updated_at) "
                "VALUES ('005930','삼성','2026-09-01',70000,0,67200,75600,?,?,?)",
                (status, now, now))
        conn.commit()
    assert pt.dedupe_open_positions() == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 3


# ─────────────────────────────────────────────────────────
# 5. 리포트 표시와 자동 청산이 같은 선을 보는가
# ─────────────────────────────────────────────────────────

def test_report_alert_and_auto_close_agree(db, monkeypatch):
    """리포트가 "도달"이라 쓴 포지션과 자동 청산이 닫는 포지션이 같아야 한다.

    두 경로는 update_current_prices()가 세운 alert 필드 하나를 공유한다.
      리포트 — report_formatter.position_alert() → ("reached", 문구)
      청산   — auto_close_triggered_positions() → close_position()

    한쪽이 자체 판정식을 갖게 되면 리포트에는 "손절선 도달"이라 적히고도
    포지션은 열린 채 남는다. 그 분기를 막는 회귀 테스트다.
    """
    from src.graph import report_formatter as rf

    # 손절 67,200(-4%) / 목표 75,600(+8%). 종목마다 현재가만 다르게 둔다.
    for ticker in ("005930", "000660", "012450", "051910"):
        pt.add_position(ticker=ticker, entry_price=70_000, quantity=0,
                        stop_loss=67_200, target_price=75_600)

    priced = [
        {"ticker": "005930", "current_price": 66_000.0,      # 손절 도달 (≤ 67,200)
         "stop_loss_price": 67_200.0, "target_price": 75_600.0, "alert": "stop_loss"},
        {"ticker": "000660", "current_price": 76_000.0,      # 목표 도달 (≥ 75,600)
         "stop_loss_price": 67_200.0, "target_price": 75_600.0, "alert": "target"},
        {"ticker": "012450", "current_price": 71_000.0,      # 어느 선에도 닿지 않음
         "stop_loss_price": 67_200.0, "target_price": 75_600.0, "alert": None},
        {"ticker": "051910", "current_price": 68_000.0,      # 손절선까지 1.2% — near
         "stop_loss_price": 67_200.0, "target_price": 75_600.0, "alert": None},
    ]
    monkeypatch.setattr(pt, "update_current_prices", lambda: priced)

    reached = set()
    for p in priced:
        hit = rf.position_alert(p)
        if hit and hit[0] == "reached":
            reached.add(p["ticker"])

    closed = {c["ticker"] for c in pt.auto_close_triggered_positions()}

    assert reached == {"005930", "000660"}, "리포트가 도달로 표시하는 집합"
    assert closed == reached, "표시와 청산이 갈라졌다"

    # near·미도달은 표시만 되고 닫히지 않는다
    assert rf.position_alert(priced[3])[0] == "near"
    assert rf.position_alert(priced[2]) is None
    with sqlite3.connect(db) as conn:
        still_open = {r[0] for r in conn.execute(
            "SELECT ticker FROM positions WHERE status='open'")}
    assert still_open == {"012450", "051910"}


# ─────────────────────────────────────────────────────────
# 6. S1 — 진입 시점 리스크 파라미터 동결 (2026-09-15)
# ─────────────────────────────────────────────────────────

def test_same_day_rerun_does_not_move_stop_loss(db):
    """S1: 같은 날 재실행이 손절선을 아래로 밀지 못한다.

    재현 시나리오 — 오전 BUY 기록 후 오후 재실행 때 주가가 내려가 있으면
    chief가 더 낮은 진입가·손절가를 계산해 넘긴다. 예전에는 그대로 덮어써서
    손실 구간에서 손절선이 자동으로 밀렸다(7-G-2 리플 사례와 같은 구조).
    """
    pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                    entry_date="2026-09-14", stop_loss=67_200,
                    take_profit_1=75_600, allocation_pct=5.0)
    pt.add_position(ticker="005930", entry_price=68_000, quantity=0,
                    entry_date="2026-09-14", stop_loss=65_280,
                    take_profit_1=73_440, allocation_pct=9.0)

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT entry_price, stop_loss_price, target_price, allocation_pct "
            "FROM positions WHERE status='open'").fetchall()
    assert len(rows) == 1, "행이 늘지 않는다 (CR-14 목적 유지)"
    entry, stop, target, alloc = rows[0]
    assert stop == 67_200, "손절선은 진입 시점 값 그대로여야 한다"
    assert entry == 70_000, "진입가도 그대로여야 한다 — 낮춰 기록하면 수익률이 부풀려진다"
    assert target == 75_600, "목표가도 진입 시점 값"
    assert alloc == 5.0, "비중도 진입 시점 값"
