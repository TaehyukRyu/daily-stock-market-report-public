"""
src/scheduler.py

AI 투자 리포트 데일리 스케줄러 (APScheduler 3.x, AsyncIOScheduler)

스케줄:
  06:00 KST 평일  — mention_tracker (뉴스 언급량 크롤링)
  06:10 KST 평일  — daily_runner.run_daily() (스크리닝 + 종목별 분석 + 통합 발행)
  15:30 KST 평일  — feedback_evaluator.evaluate_predictions() (D+1 채점)
  1/4/7/10월 1일  — universe_builder.build_universe() (분기 유니버스 갱신)

실행:
  python -m src.scheduler          # 실제 실행 (블로킹)
  python -m src.scheduler --test   # dry-run: job 등록 확인 후 종료

설계 원칙:
  - 모든 job은 max_instances=1 (중복 실행 방지)
  - 평일(mon-fri)만 실행, 공휴일 제외는 추후 확장
  - 실패 시 logs/scheduler.log에 기록 (외부 알림 없음)
  - 비동기 job은 asyncio event loop 위에서 직접 실행
  - 동기 job은 AsyncIOScheduler의 ThreadPoolExecutor에서 실행
"""

import argparse
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

# PowerShell cp949 콘솔에서 한글/유니코드 출력 깨짐 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

KST = ZoneInfo("Asia/Seoul")

# ──────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "scheduler.log"

LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,   # 10 MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)

logger = logging.getLogger("scheduler")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# APScheduler 내부 로그도 파일로 보냄
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("apscheduler").addHandler(_file_handler)


# ──────────────────────────────────────────────────────────
# Job 함수
# ──────────────────────────────────────────────────────────

async def job_mention_tracker() -> None:
    """06:00 KST — 유니버스 종목 뉴스 언급량 크롤링."""
    logger.info("[job_mention_tracker] 시작")
    try:
        from src.universe.universe_builder import load_universe
        from src.data.mention_tracker import crawl_tickers

        tickers = load_universe()
        if not tickers:
            logger.warning("[job_mention_tracker] 유니버스 비어있음 — 기본 종목(005930)만 크롤링")
            tickers = ["005930"]

        logger.info(f"[job_mention_tracker] 크롤링 대상: {len(tickers)}개 종목")
        results = await crawl_tickers(tickers)

        new_count   = sum(1 for v in results.values() if v > 0)
        cache_count = sum(1 for v in results.values() if v < 0)
        empty_count = sum(1 for v in results.values() if v == 0)
        logger.info(
            f"[job_mention_tracker] 완료 — "
            f"신규 {new_count}건 | 캐시 {cache_count}건 | 데이터없음 {empty_count}건"
        )
    except Exception as e:
        logger.error(f"[job_mention_tracker] 실패: {type(e).__name__}: {e}", exc_info=True)


async def job_run_pipeline() -> None:
    """06:10 KST — 일일 실행 (스크리닝 → 종목별 분석 → 통합 발행).

    [2026-09-14 CR-12] run_pipeline("005930") 하드코딩 → daily_runner.run_daily().
      이 파일은 Docker(Dockerfile CMD)로만 뜨고 지금은 가동되지 않는다. 그런데
      누가 컨테이너를 띄우면 스크리닝을 건너뛰고 삼성전자 1종목만 분석한 리포트가
      Notion에 나갔다. 운영 경로(ci.yml → daily_runner)와 같은 흐름으로 맞춘다.
    """
    logger.info("[job_run_pipeline] 시작 (daily_runner.run_daily)")
    try:
        from src.daily_runner import run_daily
        result = await run_daily()
        health = (result or {}).get("health") or {}
        logger.info(
            f"[job_run_pipeline] 완료 — 선정 {len(result.get('selected') or [])}개, "
            f"성공 {len(result.get('succeeded') or [])}개, "
            f"건강도={'정상' if health.get('healthy', True) else '실패'}"
        )
    except Exception as e:
        logger.error(f"[job_run_pipeline] 실패: {type(e).__name__}: {e}", exc_info=True)


def job_feedback_evaluator() -> None:
    """15:30 KST — D+1 채점 및 EMA 가중치 갱신."""
    logger.info("[job_feedback_evaluator] 시작")
    try:
        from src.data.feedback_evaluator import evaluate_predictions
        result = evaluate_predictions()
        scored = result.get("scored", 0)
        logger.info(f"[job_feedback_evaluator] 완료 — 채점 {scored}건")
    except Exception as e:
        logger.error(f"[job_feedback_evaluator] 실패: {type(e).__name__}: {e}", exc_info=True)


def job_universe_builder() -> None:
    """1/4/7/10월 1일 — 분기 유니버스 갱신.

    [2026-09-14 CR-1] 같은 갱신이 daily_runner Step 0-a에도 들어갔다(운영 경로).
    universe_refresh_due()가 "이번 분기에 이미 만들었나"를 보므로 둘 다 돌아도
    중복 재구축은 일어나지 않는다. 이 잡은 Docker 운용 시의 보조 경로다.
    """
    logger.info("[job_universe_builder] 시작 (분기 유니버스 갱신)")
    try:
        from src.universe.universe_builder import build_universe
        universe = build_universe()
        logger.info(f"[job_universe_builder] 완료 — {len(universe)}개 종목")
    except Exception as e:
        logger.error(f"[job_universe_builder] 실패: {type(e).__name__}: {e}", exc_info=True)


# ──────────────────────────────────────────────────────────
# 스케줄러 조립
# ──────────────────────────────────────────────────────────

def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=KST)

    # 1. 06:00 평일 — 언급량 크롤링 (async)
    scheduler.add_job(
        job_mention_tracker,
        trigger=CronTrigger(
            hour=6, minute=0,
            day_of_week="mon-fri",
            timezone=KST,
        ),
        id="mention_tracker",
        name="뉴스 언급량 크롤링 (06:00 KST 평일)",
        max_instances=1,
        misfire_grace_time=300,   # 5분 내 지연 실행 허용
    )

    # 2. 06:10 평일 — 파이프라인 실행 (async)
    scheduler.add_job(
        job_run_pipeline,
        trigger=CronTrigger(
            hour=6, minute=10,
            day_of_week="mon-fri",
            timezone=KST,
        ),
        id="run_pipeline",
        name="분석 파이프라인 (06:10 KST 평일)",
        max_instances=1,
        misfire_grace_time=600,   # 10분 내 지연 실행 허용
    )

    # 3. 15:30 평일 — 피드백 채점 (sync → threadpool)
    scheduler.add_job(
        job_feedback_evaluator,
        trigger=CronTrigger(
            hour=15, minute=30,
            day_of_week="mon-fri",
            timezone=KST,
        ),
        id="feedback_evaluator",
        name="D+1 피드백 채점 (15:30 KST 평일)",
        max_instances=1,
        misfire_grace_time=300,
    )

    # 4. 분기마다 (1/4/7/10월 1일) — 유니버스 갱신 (sync → threadpool)
    scheduler.add_job(
        job_universe_builder,
        trigger=CronTrigger(
            month="1,4,7,10",
            day=1,
            hour=0,
            minute=0,
            timezone=KST,
        ),
        id="universe_builder",
        name="분기 유니버스 갱신 (1/4/7/10월 1일 00:00 KST)",
        max_instances=1,
        misfire_grace_time=3600,  # 1시간 내 지연 실행 허용
    )

    return scheduler


# ──────────────────────────────────────────────────────────
# dry-run: 등록된 job 목록 출력
# ──────────────────────────────────────────────────────────

def print_jobs(scheduler: AsyncIOScheduler) -> None:
    jobs = scheduler.get_jobs()
    print(f"\n등록된 job: {len(jobs)}개\n")
    for job in jobs:
        next_run = job.next_run_time
        print(f"  [{job.id}]")
        print(f"    이름       : {job.name}")
        print(f"    트리거     : {job.trigger}")
        print(f"    다음 실행  : {next_run.strftime('%Y-%m-%d %H:%M %Z') if next_run else 'N/A'}")
        print()


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

async def _main_async(test_mode: bool) -> None:
    scheduler = build_scheduler()
    scheduler.start()

    print_jobs(scheduler)

    if test_mode:
        logger.info("[scheduler] --test 모드: job 등록 확인 후 종료")
        scheduler.shutdown(wait=False)
        return

    logger.info("[scheduler] 시작 — Ctrl+C로 종료")
    try:
        # 무한 대기 (스케줄러가 event loop에서 실행 중)
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("[scheduler] 종료 신호 수신")
    finally:
        scheduler.shutdown(wait=True)
        logger.info("[scheduler] 종료 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 투자 리포트 스케줄러")
    parser.add_argument(
        "--test",
        action="store_true",
        help="dry-run: job 등록만 확인하고 종료",
    )
    args = parser.parse_args()

    asyncio.run(_main_async(test_mode=args.test))


if __name__ == "__main__":
    main()
