"""
position_tracker.py
───────────────────
포지션 추적 모듈 (계획서 12장)

역할:
  - AI 리포트 기반으로 실제 매수한 포지션을 SQLite에 기록
  - 매일 파이프라인 실행 시 현재가·손익 자동 업데이트
  - 손절가(-4%, CR-6) 도달 시 경고 생성
  - report_formatter에 포지션 현황 섹션 제공

설계 원칙:
  - 실제 증권사 API 연동 없음 (수동 입력 기본값 — 계획서 12.9.2절)
  - 진입/청산은 사람이 직접, 시스템은 추적만 담당
  - Notion 발행 시 포지션 현황 자동 포함

DB: data/mentions.db (기존 DB에 positions 테이블 추가)

CLI 사용법:
  # 포지션 추가 (매수 후 기록)
  python -m src.data.position_tracker add 005930 78500 12

  # 포지션 청산 (매도 후 기록)
  python -m src.data.position_tracker close 005930 82000

  # 현재 보유 포지션 조회
  python -m src.data.position_tracker list
"""

import sqlite3
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/mentions.db")

# 리스크 파라미터 기본값.
#
# [2026-09-14 CR-6] 0.07 / 0.15 → 0.04 / 0.08 로 맞춤.
#   chief_strategist 프롬프트와 chief_python.compute_trade_params가 쓰는 규칙은
#   손절 -4%(범위 -3~-6), 1차 익절 = 진입가 + 리스크×2 = -4% 기준 +8%다.
#   여기 기본값이 -7%/+15%라 chief가 값을 채우지 않으면 리포트에 표시된 손절선과
#   시스템이 감시하는 손절선이 약 2배 어긋났다(pipeline.py가 경고만 찍었다).
#   기본값이 쓰이는 것은 chief가 값을 안 줄 때뿐이고, 그때도 규칙과 같은 값이어야 한다.
STOP_LOSS_RATIO      = 0.04   # 진입가 대비 -4% 손절 (chief 기본 stop_loss_pct와 동일)
DEFAULT_TARGET_RATIO = 0.08   # 진입가 대비 +8% 목표 (R:R 2.0 → 리스크 4% × 2)

# 보유 경과 알림 기준 (CR-5, 2026-09-14 결정).
#
# [왜 자동 청산이 아닌가]
#   설계는 중타 1~2주지만 (b) 리서치 도구에서는 청산 시점을 사람이 정한다.
#   그래서 시간이 지나도 포지션을 닫지 않고 "지났다"고 알리기만 한다.
#   자동 청산은 여전히 손절가·목표가 도달 시에만 일어난다(auto_close_triggered_positions).
#
# 14는 달력일이다. outcomes.HORIZONS의 D+10은 거래일 10일이고 주말을 끼면 약 14일이 된다.
# 공휴일이 끼면 실제로는 조금 더 걸리므로 이 알림은 "대략 2주 지났다"는 뜻이다.
HOLDING_ALERT_DAYS = 14


# ── DDL ───────────────────────────────────────────────────────────────────────

def init_position_table() -> None:
    """
    positions 테이블 생성 (없으면 생성, 있으면 그대로).

    계획서 12.9.1절 추적 항목 전부 포함.
    추가: 시드 대비 % 추적용 컬럼 (allocation_pct, take_profit_1/2,
          holding_period_weeks, rr_ratio) — ALTER TABLE 마이그레이션.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 종목 정보
                ticker           TEXT NOT NULL,
                ticker_name      TEXT,

                -- 진입 정보 (수동 입력)
                entry_date       TEXT NOT NULL,    -- 'YYYY-MM-DD'
                entry_price      REAL NOT NULL,    -- 평균 진입가 (원)
                quantity         INTEGER NOT NULL, -- 보유 수량

                -- 리스크 관리 (자동 계산)
                stop_loss_price  REAL NOT NULL,    -- 진입가 × (1 - 0.07)
                target_price     REAL NOT NULL,    -- 진입가 × (1 + 0.15)

                -- 근거 (AI 리포트 참조)
                entry_rationale  TEXT,             -- 진입 당시 AI 판단 이유

                -- 상태
                status           TEXT NOT NULL DEFAULT 'open',  -- 'open' | 'closed'
                close_date       TEXT,             -- 청산일
                close_price      REAL,             -- 청산가
                realized_pnl     REAL,             -- 실현 손익 (원)
                realized_pnl_pct REAL,             -- 실현 손익률 (%)
                close_reason     TEXT,             -- 'stop_loss' | 'target' | 'manual' | 'signal'

                -- 메타
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_positions_status
                ON positions (status, ticker);
        """)

        # 시드 % 추적용 컬럼 안전 마이그레이션
        for col_def in [
            "ALTER TABLE positions ADD COLUMN allocation_pct REAL",
            "ALTER TABLE positions ADD COLUMN take_profit_1 REAL",
            "ALTER TABLE positions ADD COLUMN take_profit_2 REAL",
            "ALTER TABLE positions ADD COLUMN holding_period_weeks INTEGER",
            "ALTER TABLE positions ADD COLUMN rr_ratio REAL",
        ]:
            try:
                conn.execute(col_def)
            except sqlite3.OperationalError:
                pass  # 이미 존재하는 컬럼이면 무시

        conn.commit()
    logger.info("[position_tracker] positions 테이블 초기화 완료")


# ── 포지션 추가 ────────────────────────────────────────────────────────────────

def dedupe_open_positions() -> list[dict]:
    """같은 종목의 오픈 포지션이 2개 이상이면 **가장 먼저 등록된 것만 남기고 삭제**한다 (CR-14).

    [왜 삭제인가]
      중복 행은 실제 매매가 아니라 같은 추천이 두 번 기록된 것이다(quantity=0,
      진입가·진입일까지 동일). 청산 처리하면 없던 거래가 청산 이력에 남아
      성과 집계를 오염시킨다. 남기는 것은 가장 이른 행 — 그것이 최초 추천 시점이다.

    멱등하다. 반환: 삭제한 행 정보 목록 (보고용).
    """
    removed: list[dict] = []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, ticker, entry_date, entry_price, allocation_pct FROM positions "
                "WHERE status='open' ORDER BY ticker, id"
            ).fetchall()
        except sqlite3.OperationalError:
            return removed
        seen: set[str] = set()
        for r in rows:
            if r["ticker"] in seen:
                removed.append(dict(r))
            else:
                seen.add(r["ticker"])
        if removed:
            conn.executemany("DELETE FROM positions WHERE id=?", [(r["id"],) for r in removed])
            conn.commit()
            for r in removed:
                logger.warning(
                    f"[position_tracker] 중복 오픈 포지션 삭제: id={r['id']} {r['ticker']} "
                    f"진입일 {r['entry_date']} 배분 {r['allocation_pct']}%"
                )
            logger.warning(f"[position_tracker] CR-14 중복 정리 — 총 {len(removed)}건 삭제")
    return removed


def add_position(
    ticker: str,
    entry_price: float,
    quantity: int,
    ticker_name: str = "",
    entry_rationale: str = "",
    entry_date: Optional[str] = None,
    target_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    allocation_pct: Optional[float] = None,
    take_profit_1: Optional[float] = None,
    take_profit_2: Optional[float] = None,
    holding_period_weeks: Optional[int] = None,
    rr_ratio: Optional[float] = None,
) -> int:
    """
    신규 포지션 추가.

    Args:
        ticker: 종목 코드 ('005930')
        entry_price: 평균 매수가 (원)
        quantity: 매수 수량 (시드 % 기반 추적 시 0 가능)
        ticker_name: 종목명 (선택)
        entry_rationale: 진입 근거 (AI 리포트 기반으로 직접 입력)
        entry_date: 매수일 (기본값: 오늘)
        target_price: 감시할 목표가. None이면 take_profit_1 → 비율 기본값 순으로 대체
        stop_loss: 감시할 손절가. None이면 비율 기본값으로 대체
        allocation_pct: 시드 대비 배분 % (예: 20.0)
        take_profit_1: 1차 익절가
        take_profit_2: 2차 익절가
        holding_period_weeks: 예상 보유 기간 (주)
        rr_ratio: R:R 비율

    Returns:
        생성된 포지션 id

    손절가·목표가 결정 순서:
        stop_loss    : 인자 → (없으면) entry_price × (1 - STOP_LOSS_RATIO)
        target_price : 인자 → take_profit_1 → entry_price × (1 + DEFAULT_TARGET_RATIO)

        stop_loss_price / target_price 컬럼은 update_current_prices()가 손절·목표
        도달을 판정할 때 읽는 값이다. 따라서 리포트에 표시한 값을 그대로 넘겨야
        "리포트가 보여준 손절선"과 "시스템이 감시하는 손절선"이 일치한다.
        인자가 비어 비율 기본값으로 대체될 때는 조용히 넘어가지 않고 경고를 남긴다.
    """
    if entry_date is None:
        entry_date = date.today().isoformat()

    if stop_loss is None:
        stop_loss = round(entry_price * (1 - STOP_LOSS_RATIO), 0)
        logger.warning(
            f"[position_tracker] {ticker} stop_loss 미지정 → 기본값 "
            f"-{STOP_LOSS_RATIO:.0%} 적용 ({stop_loss:,.0f}원). "
            f"리포트에 다른 손절가를 표시했다면 값이 어긋난다."
        )
    else:
        stop_loss = round(float(stop_loss), 0)

    if target_price is None:
        if take_profit_1 is not None:
            target_price = round(float(take_profit_1), 0)
        else:
            target_price = round(entry_price * (1 + DEFAULT_TARGET_RATIO), 0)
            logger.warning(
                f"[position_tracker] {ticker} target_price·take_profit_1 모두 미지정 "
                f"→ 기본값 +{DEFAULT_TARGET_RATIO:.0%} 적용 ({target_price:,.0f}원)."
            )
    else:
        target_price = round(float(target_price), 0)

    now = datetime.now().isoformat()

    # ── 기존 오픈 포지션 처리 (CR-14 2026-09-14 → S1 2026-09-15) ────────────
    #
    # [무엇이 문제였나 — CR-14]
    #   예전에는 경고만 찍고 그대로 INSERT했다. 같은 종목 BUY가 반복되면 행이 쌓여
    #   invested_pct가 부풀고 available_pct가 줄었다(실측 11.2% → 16.2%).
    #   하루 1회 cron에서는 안 드러나고 같은 날 재실행에서 드러났다.
    #
    # [정책 — 오픈 포지션이 있으면 같은 날이든 다른 날이든 아무것도 바꾸지 않는다]
    #   기존 행의 진입일·진입가는 "시스템이 언제 처음 추천했는가"의 기록이다.
    #   덮어쓰면 그 기록이 사라지고, 새 행을 더하면 사람이 하나만 들고 있는데
    #   비중이 두 배로 잡힌다. 추가 매수 판단은 리포트 액션 플랜의
    #   "🟢 추가 매수 검토"가 이미 사람에게 넘기고 있다.
    #
    # [S1 — 같은 날 UPDATE를 없앤 이유 두 가지]
    #   CR-14는 같은 날 재실행을 "같은 판단의 반복"으로 보고 갱신했다. 두 가지가 깨진다.
    #
    #   (1) 손절 원칙이 깨진다.
    #       오전에 진입가 70,000 / 손절 67,200으로 기록한 뒤 오후 재실행 때 주가가
    #       68,000이면 chief가 손절 65,280을 계산해 넘긴다. 갱신하면 손실 구간에서
    #       손절선이 자동으로 아래로 밀린다. 사람이 미루는 것보다 나쁘다 — 코드가
    #       대신 미뤄주고 로그에도 "갱신"으로만 남아 흔적이 안 보인다.
    #       근거: 사람이 가장 이성적인 순간은 무포지션 상태이므로, 그때 정한 손절
    #       기준을 손실 구간에서 바꾸면 안 된다.
    #
    #   (2) 성과 측정의 정직성이 깨진다.
    #       UPDATE는 손절선만이 아니라 entry_price도 덮어쓴다. 주가가 내려간 뒤
    #       진입가를 낮춰 기록하면 나중 실현 수익률이 실제보다 좋게 찍힌다
    #       (close_position이 realized_pnl_pct를 entry_price 기준으로 계산한다).
    #       이는 백테스트에서 막는 룩어헤드 편향과 같은 종류의 왜곡이 운영 원장에서
    #       일어나는 것이다. 결과를 알고 난 뒤 진입 조건을 유리하게 고치는 셈이다.
    #
    #   CR-14가 실제로 고치려던 증상은 "행이 쌓여 invested_pct가 부푼다"였고,
    #   그건 새 행을 안 만드는 것만으로 해결된다. 갱신은 그 목적에 필요하지 않다.
    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute(
            "SELECT id, entry_date FROM positions WHERE ticker=? AND status='open' ORDER BY id",
            (ticker,),
        ).fetchone()
        if existing:
            exist_id, exist_date = existing[0], existing[1]
            logger.warning(
                f"[position_tracker] {ticker} 이미 오픈 포지션 존재 "
                f"(id={exist_id}, 진입일 {exist_date}) → 진입 시점 값을 유지하고 "
                f"아무것도 갱신하지 않는다 (S1). "
                f"추가 매수는 리포트를 보고 사람이 판단한다."
            )
            return exist_id

        cursor = conn.execute(
            """
            INSERT INTO positions
                (ticker, ticker_name, entry_date, entry_price, quantity,
                 stop_loss_price, target_price, entry_rationale,
                 status, created_at, updated_at,
                 allocation_pct, take_profit_1, take_profit_2,
                 holding_period_weeks, rr_ratio)
            VALUES (?,?,?,?,?,?,?,?,  'open',?,?,  ?,?,?,?,?)
            """,
            (ticker, ticker_name, entry_date, entry_price, quantity,
             stop_loss, target_price, entry_rationale, now, now,
             allocation_pct, take_profit_1, take_profit_2,
             holding_period_weeks, rr_ratio),
        )
        conn.commit()
        pos_id = cursor.lastrowid

    logger.info(
        f"[position_tracker] 포지션 추가: {ticker} {quantity}주 "
        f"@{entry_price:,.0f}원 (손절={stop_loss:,.0f}, 목표={target_price:,.0f})"
    )
    return pos_id


# ── 자동 청산 ─────────────────────────────────────────────────────────────────

def auto_close_triggered_positions() -> list[dict]:
    """
    손절가·목표가에 도달한 오픈 포지션을 자동으로 청산한다.

    [왜 필요한가]
      update_current_prices()는 alert 필드만 세우고 청산하지 않았다.
      close_position()의 호출자는 CLI 하나뿐이라 사람이 직접 치지 않으면
      포지션이 영원히 열린 채 남는다. 실현 손익이 기록되지 않으니
      승률·수익률 집계가 원천적으로 불가능했다 (positions 테이블 0행).

    [경계]
      감시 기준은 stop_loss_price / target_price 컬럼이며, 이 값은
      add_position()이 chief_strategist가 계산한 값을 그대로 받은 것이다.
      즉 리포트에 표시된 손절선과 동일한 선에서 청산된다.

      청산가는 현재가(종가)를 쓴다. 장중 실제 체결가와는 다르므로
      집계는 근사치다 — 이 시스템은 주문을 내지 않고 기록만 한다.

    Returns:
        청산된 포지션 결과 리스트. 도달한 것이 없으면 빈 리스트.
    """
    triggered = [
        p for p in update_current_prices()
        if p.get("alert") in ("stop_loss", "target") and p.get("current_price")
    ]
    if not triggered:
        return []

    closed = []
    for p in triggered:
        try:
            result = close_position(
                ticker       = p["ticker"],
                close_price  = float(p["current_price"]),
                close_reason = p["alert"],
            )
            if result:
                closed.append(result)
                logger.info(
                    f"[position_tracker] 자동 청산: {p['ticker']} "
                    f"[{p['alert']}] {p['current_price']:,.0f}원 "
                    f"({result['pnl_pct']:+.1f}%)"
                )
        except Exception as e:
            logger.error(f"[position_tracker] {p['ticker']} 자동 청산 실패: {e}")

    return closed


# ── 포지션 청산 ────────────────────────────────────────────────────────────────

def close_position(
    ticker: str,
    close_price: float,
    close_reason: str = "manual",
    close_date: Optional[str] = None,
) -> Optional[dict]:
    """
    포지션 청산 처리 (실현 손익 계산).

    Args:
        ticker: 종목 코드
        close_price: 청산가 (원)
        close_reason: 'stop_loss' | 'target' | 'manual' | 'signal'
        close_date: 청산일 (기본값: 오늘)

    Returns:
        청산 결과 dict 또는 None (포지션 없을 때)
    """
    if close_date is None:
        close_date = date.today().isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        pos = conn.execute(
            "SELECT * FROM positions WHERE ticker=? AND status='open' ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()

        if not pos:
            logger.warning(f"[position_tracker] {ticker} 오픈 포지션 없음")
            return None

        pos = dict(pos)
        realized_pnl     = (close_price - pos["entry_price"]) * pos["quantity"]
        realized_pnl_pct = ((close_price / pos["entry_price"]) - 1.0) * 100

        conn.execute(
            """
            UPDATE positions
            SET status='closed', close_date=?, close_price=?,
                realized_pnl=?, realized_pnl_pct=?, close_reason=?, updated_at=?
            WHERE id=?
            """,
            (close_date, close_price, round(realized_pnl, 0), round(realized_pnl_pct, 2),
             close_reason, datetime.now().isoformat(), pos["id"]),
        )
        conn.commit()

    result = {
        "ticker":         ticker,
        "entry_price":    pos["entry_price"],
        "close_price":    close_price,
        "quantity":       pos["quantity"],
        "realized_pnl":   round(realized_pnl, 0),
        "pnl_pct":        round(realized_pnl_pct, 2),
        "close_reason":   close_reason,
    }
    pnl_sign = "+" if realized_pnl >= 0 else ""
    logger.info(
        f"[position_tracker] 포지션 청산: {ticker} "
        f"{pnl_sign}{realized_pnl:,.0f}원 ({pnl_sign}{realized_pnl_pct:.1f}%) [{close_reason}]"
    )
    return result


# ── 현재가 업데이트 + 손절 체크 ────────────────────────────────────────────────

def update_current_prices() -> list[dict]:
    """
    오픈 포지션의 현재가를 pykrx로 조회하고 손절/목표 도달 경고 생성.

    Returns:
        [{'ticker': ..., 'current_price': ..., 'unrealized_pnl': ...,
          'pnl_pct': ..., 'alert': None | 'stop_loss' | 'target'}, ...]

    호출처: pipeline.py의 report_formatter (파이프라인 실행마다)

    ⚠️ 자동 청산은 하지 않음.
       alert 필드만 설정 → 사람이 Notion 리포트를 보고 직접 판단.
    """
    positions = get_open_positions(with_price=False)
    if not positions:
        return []

    try:
        from pykrx import stock as krx_stock
        today_str = date.today().strftime("%Y%m%d")
    except ImportError:
        logger.warning("[position_tracker] pykrx 없음 — 현재가 업데이트 건너뜀")
        return positions

    results = []
    for pos in positions:
        ticker = pos["ticker"]
        try:
            df = krx_stock.get_market_ohlcv_by_date(
                fromdate=today_str, todate=today_str, ticker=ticker,
            )
            if df.empty or "종가" not in df.columns:
                results.append({**pos, "current_price": None, "unrealized_pnl": None, "pnl_pct": None, "alert": None})
                continue

            current_price    = float(df["종가"].iloc[-1])
            unrealized_pnl   = (current_price - pos["entry_price"]) * pos["quantity"]
            pnl_pct          = ((current_price / pos["entry_price"]) - 1.0) * 100

            # 경고 판단
            alert = None
            if current_price <= pos["stop_loss_price"]:
                alert = "stop_loss"
                logger.warning(f"[position_tracker] ⚠️ {ticker} 손절가 도달! 현재가={current_price:,.0f} ≤ 손절가={pos['stop_loss_price']:,.0f}")
            elif current_price >= pos["target_price"]:
                alert = "target"
                logger.info(f"[position_tracker] 🎯 {ticker} 목표가 도달! 현재가={current_price:,.0f} ≥ 목표가={pos['target_price']:,.0f}")

            results.append({
                **pos,
                "current_price":  current_price,
                "unrealized_pnl": round(unrealized_pnl, 0),
                "pnl_pct":        round(pnl_pct, 2),
                "alert":          alert,
            })
        except Exception as e:
            logger.warning(f"[position_tracker] {ticker} 현재가 조회 실패: {e}")
            results.append({**pos, "current_price": None, "unrealized_pnl": None, "pnl_pct": None, "alert": None})

    return results


def get_open_positions(with_price: bool = True) -> list[dict]:
    """현재 오픈 포지션 목록 반환."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='open' ORDER BY entry_date",
        ).fetchall()
    positions = [dict(row) for row in rows]

    if with_price and positions:
        return update_current_prices()
    return positions


def get_position_summary() -> dict:
    """
    전체 포지션 요약 (리포트 섹션용).

    Returns:
        {
          'open_count': 2,
          'total_invested': 1_800_000,
          'total_unrealized_pnl': 54_000,
          'alerts': [{'ticker': '005930', 'alert': 'stop_loss'}],
          'positions': [...]
        }
    """
    positions = get_open_positions(with_price=True)
    alerts    = [p for p in positions if p.get("alert")]
    total_invested = sum(p["entry_price"] * p["quantity"] for p in positions)
    total_pnl      = sum(p.get("unrealized_pnl") or 0 for p in positions)

    return {
        "open_count":           len(positions),
        "total_invested":       total_invested,
        "total_unrealized_pnl": total_pnl,
        "alerts":               alerts,
        "positions":            positions,
    }


def get_closed_positions_recent(limit: int = 5) -> list[dict]:
    """closed 포지션 최근 N건을 close_date 내림차순으로 반환."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT ticker, ticker_name, entry_date, entry_price, close_date,
                   close_price, allocation_pct, realized_pnl_pct, close_reason
            FROM positions
            WHERE status='closed'
            ORDER BY close_date DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_portfolio_pct_summary() -> dict:
    """
    포트폴리오 전체 현황을 시드 % 기준으로 반환.
    allocation_pct가 NULL인 포지션은 % 관련 계산에서 제외 (수량 기반과 호환).

    Returns:
      {
        "invested_pct":            float,  # open 포지션 allocation_pct 합계
        "available_pct":           float,  # max(0, 100 - invested_pct)
        "cumulative_pnl_seed_pct": float,  # closed 시드 손익 합 (allocation × realized_pnl_pct / 100)
        "unrealized_pnl_seed_pct": float,  # open 시드 손익 합 (현재가 기준, 미확정)
        "priced_positions":        int,    # 현재가를 받아온 open 종목 수
        "positions":               list[dict],  # open 포지션 (현재가/수익률/alert 포함)
        "closed_recent":           list[dict],  # 최근 5건 (시드 손익 포함)
      }
    """
    open_positions = get_open_positions(with_price=True)

    positions_out: list[dict] = []
    invested_pct = 0.0
    today = date.today()

    for p in open_positions:
        alloc = p.get("allocation_pct")
        current_price = p.get("current_price")
        entry_price   = p.get("entry_price")

        if alloc is not None:
            invested_pct += float(alloc)

        # 보유일 계산
        holding_days: Optional[int] = None
        try:
            ed = p.get("entry_date")
            if ed:
                holding_days = (today - date.fromisoformat(ed)).days
        except Exception:
            holding_days = None

        # 포지션 수익률 / 시드 손익
        position_pnl_pct: Optional[float] = None
        seed_pnl_pct: Optional[float] = None
        if current_price is not None and entry_price:
            position_pnl_pct = round((current_price / entry_price - 1.0) * 100, 2)
            if alloc is not None:
                seed_pnl_pct = round(float(alloc) * position_pnl_pct / 100.0, 2)

        positions_out.append({
            "ticker":               p["ticker"],
            "ticker_name":          p.get("ticker_name") or "",
            "entry_date":           p.get("entry_date"),
            "entry_price":          entry_price,
            "current_price":        current_price,
            "allocation_pct":       alloc,
            "position_pnl_pct":     position_pnl_pct,
            "seed_pnl_pct":         seed_pnl_pct,
            "stop_loss_price":      p.get("stop_loss_price"),
            # [2026-09-15] 리포트가 "목표선까지 N%"를 계산하려면 목표가가 필요하다.
            #   take_profit_1은 제안값이고 target_price가 실제 판정 기준선이다.
            "target_price":         p.get("target_price"),
            "take_profit_1":        p.get("take_profit_1"),
            "take_profit_2":        p.get("take_profit_2"),
            "holding_days":         holding_days,
            "holding_period_weeks": p.get("holding_period_weeks"),
            "rr_ratio":             p.get("rr_ratio"),
            "alert":                p.get("alert"),
            # CR-5: 자동 청산은 하지 않고 "지났다"만 알린다. alert 필드는 건드리지 않는다
            # (auto_close_triggered_positions가 그 필드로 청산을 판정하기 때문).
            "overdue":              bool(holding_days is not None and holding_days >= HOLDING_ALERT_DAYS),
        })

    # 청산 이력 (시드 손익 포함)
    closed_rows = get_closed_positions_recent(limit=5)
    closed_recent: list[dict] = []
    cumulative_pnl_seed_pct = 0.0

    # 누적 시드 손익은 closed 전체에서 계산
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        all_closed = conn.execute(
            """
            SELECT allocation_pct, realized_pnl_pct
            FROM positions
            WHERE status='closed' AND allocation_pct IS NOT NULL
                  AND realized_pnl_pct IS NOT NULL
            """
        ).fetchall()
        for r in all_closed:
            cumulative_pnl_seed_pct += float(r["allocation_pct"]) * float(r["realized_pnl_pct"]) / 100.0

    for r in closed_rows:
        entry_dt = r.get("entry_date")
        close_dt = r.get("close_date")
        holding_days: Optional[int] = None
        try:
            if entry_dt and close_dt:
                holding_days = (date.fromisoformat(close_dt) - date.fromisoformat(entry_dt)).days
        except Exception:
            holding_days = None

        alloc = r.get("allocation_pct")
        pnl_pct = r.get("realized_pnl_pct")
        seed_pnl_pct: Optional[float] = None
        if alloc is not None and pnl_pct is not None:
            seed_pnl_pct = round(float(alloc) * float(pnl_pct) / 100.0, 2)

        closed_recent.append({
            "ticker":           r["ticker"],
            "ticker_name":      r.get("ticker_name") or "",
            "entry_price":      r.get("entry_price"),
            "close_price":      r.get("close_price"),
            "allocation_pct":   alloc,
            "position_pnl_pct": round(float(pnl_pct), 2) if pnl_pct is not None else None,
            "seed_pnl_pct":     seed_pnl_pct,
            "holding_days":     holding_days,
            "close_reason":     r.get("close_reason"),
        })

    # [2026-09-15] 미실현 손익을 따로 낸다.
    #   cumulative_pnl_seed_pct는 closed만 합산한다. 계산은 맞지만 청산 이력이 없으면
    #   보유 종목이 -3%, -5%여도 "누적 수익 +0.00%"로 보여서 손실이 없는 것처럼 읽혔다.
    #   실현과 미실현은 성격이 다르므로 합치지 않고 나란히 낸다 — 합치면 "확정된 손실"과
    #   "아직 되돌릴 수 있는 손실"이 구분되지 않는다.
    unrealized = [p["seed_pnl_pct"] for p in positions_out if p.get("seed_pnl_pct") is not None]
    unrealized_pnl_seed_pct = round(sum(unrealized), 2) if unrealized else 0.0
    return {
        "invested_pct":            round(invested_pct, 2),
        "available_pct":           round(max(0.0, 100.0 - invested_pct), 2),
        "cumulative_pnl_seed_pct": round(cumulative_pnl_seed_pct, 2),
        "unrealized_pnl_seed_pct": unrealized_pnl_seed_pct,
        "priced_positions":        len(unrealized),   # 현재가를 못 받은 종목은 빠진다
        "positions":               positions_out,
        "closed_recent":           closed_recent,
    }


def format_position_section(summary: Optional[dict] = None) -> str:
    """
    Notion 리포트에 삽입할 포지션 현황 마크다운 텍스트 생성.

    호출처: pipeline.py의 report_formatter

    반환 예시:
        ## ⑤ 현재 포지션 현황

        | 종목 | 진입가 | 수량 | 현재가 | 손익 | 손절가 | 상태 |
        |------|--------|------|--------|------|--------|------|
        | 005930 삼성전자 | 78,500 | 12 | 80,200 | +20,400 (+2.2%) | 73,005 | 보유 |
    """
    if summary is None:
        summary = get_position_summary()

    if summary["open_count"] == 0:
        return "## ⑤ 현재 포지션 현황\n\n보유 포지션 없음\n"

    lines = ["## ⑤ 현재 포지션 현황\n"]

    # 경고 먼저 출력
    for alert_pos in summary["alerts"]:
        ticker = alert_pos["ticker"]
        if alert_pos["alert"] == "stop_loss":
            lines.append(f"⚠️ **{ticker} 손절가 도달 — 청산 검토 필요**")
        elif alert_pos["alert"] == "target":
            lines.append(f"🎯 **{ticker} 목표가 도달 — 익절 검토 필요**")
    if summary["alerts"]:
        lines.append("")

    # 포지션 테이블
    lines.append("| 종목 | 진입일 | 진입가 | 수량 | 현재가 | 손익 | 손절가 | 목표가 |")
    lines.append("|------|--------|--------|------|--------|------|--------|--------|")

    for p in summary["positions"]:
        name          = p.get("ticker_name") or p["ticker"]
        current_price = p.get("current_price")
        pnl           = p.get("unrealized_pnl")
        pnl_pct       = p.get("pnl_pct")

        if current_price is not None:
            price_str = f"{current_price:,.0f}"
            pnl_str   = f"{'+' if pnl >= 0 else ''}{pnl:,.0f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)"
        else:
            price_str = "-"
            pnl_str   = "-"

        alert_icon = " ⚠️" if p.get("alert") == "stop_loss" else (" 🎯" if p.get("alert") == "target" else "")

        lines.append(
            f"| {p['ticker']} {name}{alert_icon} "
            f"| {p['entry_date']} "
            f"| {p['entry_price']:,.0f} "
            f"| {p['quantity']} "
            f"| {price_str} "
            f"| {pnl_str} "
            f"| {p['stop_loss_price']:,.0f} "
            f"| {p['target_price']:,.0f} |"
        )

    # 합계 행
    total_pnl = summary["total_unrealized_pnl"]
    lines.append(f"\n**총 미실현 손익: {'+' if total_pnl >= 0 else ''}{total_pnl:,.0f}원** "
                 f"(투자금 {summary['total_invested']:,.0f}원 기준)\n")

    return "\n".join(lines)


# ── 초기화 ────────────────────────────────────────────────────────────────────

def setup_position_tracker() -> None:
    """포지션 추적 초기화. 파이프라인 시작 시 setup_feedback_system()과 함께 호출."""
    init_position_table()
    # CR-14: 방어책(add_position 재등록 차단)이 생기기 전에 쌓인 중복을 한 번 정리한다.
    # 멱등하므로 매 실행 호출해도 안전하고, 정리 건수는 로그에 남는다.
    removed = dedupe_open_positions()
    if removed:
        print(f"[position_tracker] CR-14 중복 오픈 포지션 {len(removed)}건 정리: "
              + ", ".join(f"{r['ticker']}(id={r['id']})" for r in removed))
    logger.info("[position_tracker] 초기화 완료")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    """
    커맨드라인 인터페이스.

    사용법:
      python -m src.data.position_tracker add 005930 78500 12 [종목명] [진입근거]
      python -m src.data.position_tracker close 005930 82000 [사유]
      python -m src.data.position_tracker list
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    setup_position_tracker()

    if len(sys.argv) < 2:
        print("사용법: position_tracker [add|close|list] ...")
        return

    cmd = sys.argv[1].lower()

    if cmd == "add":
        if len(sys.argv) < 5:
            print("사용법: add <ticker> <entry_price> <quantity> [ticker_name] [rationale]")
            return
        ticker      = sys.argv[2]
        entry_price = float(sys.argv[3])
        quantity    = int(sys.argv[4])
        ticker_name = sys.argv[5] if len(sys.argv) > 5 else ""
        rationale   = sys.argv[6] if len(sys.argv) > 6 else ""
        pos_id = add_position(ticker, entry_price, quantity, ticker_name, rationale)
        print(f"✅ 포지션 추가 완료 (id={pos_id})")
        print(f"   {ticker} {quantity}주 @{entry_price:,.0f}원")
        print(f"   손절가: {entry_price*(1-STOP_LOSS_RATIO):,.0f}원 | 목표가: {entry_price*(1+DEFAULT_TARGET_RATIO):,.0f}원")

    elif cmd == "close":
        if len(sys.argv) < 4:
            print("사용법: close <ticker> <close_price> [reason]")
            return
        ticker      = sys.argv[2]
        close_price = float(sys.argv[3])
        reason      = sys.argv[4] if len(sys.argv) > 4 else "manual"
        result = close_position(ticker, close_price, reason)
        if result:
            pnl_sign = "+" if result["realized_pnl"] >= 0 else ""
            print(f"✅ 포지션 청산 완료")
            print(f"   {ticker}: {result['entry_price']:,.0f} → {result['close_price']:,.0f}원")
            print(f"   손익: {pnl_sign}{result['realized_pnl']:,.0f}원 ({pnl_sign}{result['pnl_pct']:.1f}%)")
        else:
            print(f"⚠️ {ticker} 오픈 포지션 없음")

    elif cmd == "list":
        summary = get_position_summary()
        print(format_position_section(summary))

    else:
        print(f"알 수 없는 명령: {cmd}. [add|close|list] 중 하나를 사용하세요.")


if __name__ == "__main__":
    _cli()