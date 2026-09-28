"""
src/utils/shadow_log.py

모델 A/B 섀도 로그 — 같은 입력으로 구모델·신모델을 나란히 돌려 기록한다.
리포트에는 구모델(baseline) 결과만 쓴다. (doc/2026-09-10_benchmark-analysis.md F-6)

[왜 필요한가]
  Stage 1-C sonnet→haiku 다운그레이드(68b46a8)는 1줄 변경·측정 없음이었고,
  그래서 "품질이 떨어졌는지" 판단 자체가 불가능해졌다 (S1 3-3).
  모델을 바꾸기 전에 며칠간 두 출력을 나란히 쌓아 비교하는 장치가 이것이다.

[승격 규칙 — 코드에 박는다 (prism-insight shadow_lifecycle 발상, AGPL이라 코드는 새로 씀)]
  - 실험마다 review_by(마감)와 min_samples가 명시된다.
  - 마감이 지나면 shadow_active()가 False → 신모델 호출이 자동으로 멈춘다.
    잊어버려도 비용이 계속 나가지 않는다.
  - LIVE(교체)는 사람이 한다. 이 모듈은 절대 자동으로 모델을 바꾸지 않는다.

[테이블] data/mentions.db · llm_ab_log
  run_date, node, variant(baseline|candidate), model, input_hash,
  output_json, input_tokens, output_tokens, latency_ms, cost_usd, created_at

[비교] python -m src.utils.shadow_log stage1c_model
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.llm_budget import _price_for

logger = logging.getLogger(__name__)

DB_PATH = Path("data/mentions.db")

# ── 실험 정의 ────────────────────────────────────────────────────────────
# 날짜는 일부러 하드코딩한다. "언제 끝나는지"가 코드 리뷰에서 보여야 한다.
SHADOW_POLICIES: dict[str, dict[str, Any]] = {
    "stage1c_model": {
        "label":       "Stage 1-C 뉴스 스크린 모델 haiku 4.5 → sonnet 5",
        "started":     "2026-09-11",
        # 2026-09-14 연장: 추석 휴장(9/24~25, 9/28 대체공휴일)으로 9/26까지 거래일이
        # 부족해 min_samples=10을 채울 수 없다. 시작(09-11) 이후 거래일 10일을 확보하려면
        # 10월 중순이 필요하다. 실험 조건(baseline/candidate/min_samples)은 그대로다.
        "review_by":   "2026-10-16",        # 09-26 → 연장 (추석 휴장)
        "min_samples": 10,                  # 거래일 10일 (하루 1표본)
        "baseline":    "claude-haiku-4-5-20251001",
        "candidate":   "claude-sonnet-5",
        "promotion":   "selected_tickers 겹침률·직접언급 비율·비용·지연을 비교하고 "
                       "표본 20건 수동 블라인드 리뷰 후 사람이 NEWS_SCREEN_MODEL을 바꾼다",
    },
    # 옵션 A — chief를 파이썬 집계 + sonnet-5 문장으로 교체하는 실험.
    # CHIEF_MODE=python_sonnet 로 켜기 전까지 candidate 호출은 없다 (등록만).
    "chief_python_sonnet": {
        "label":       "Chief 산술 6개를 파이썬으로, 문장만 sonnet 5",
        "started":     "2026-09-11",
        # 같은 이유로 연장 (추석 휴장). 이쪽은 CHIEF_MODE=legacy라 candidate 호출 자체가
        # 아직 0건이므로 마감이 지나면 표본 0으로 종료됐을 것이다.
        "review_by":   "2026-10-16",
        "min_samples": 10,
        "baseline":    "claude-opus-4-6",
        "candidate":   "claude-sonnet-5",
        "promotion":   "파이썬 계산값과 opus 출력값 일치율 + 문장 품질 리뷰 후 "
                       "CHIEF_MODE 기본값을 python_sonnet으로 바꾼다",
    },
}

VALID_MODES = ("shadow", "off")

# 자기 겹침률 기준선 (2026-09-14).
#
# [왜 필요한가]
#   baseline(haiku) vs candidate(sonnet-5) 겹침률이 4회 연속 0.22~0.26이었다.
#   그런데 이 값이 "모델이 다르기 때문"인지 "Stage 1-C가 원래 불안정한 것"인지
#   구분할 기준이 없었다. 같은 모델을 같은 입력으로 한 번 더 불러 자기 자신과의
#   겹침률을 재면 그 기준선이 생긴다.
#   예: self 0.9인데 candidate 0.25면 모델 차이가 크다는 뜻이고,
#       self도 0.3이면 이 단계가 원래 흔들린다는 뜻이다.
#   비용은 하루 haiku 1콜(약 $0.006). 판단은 표본이 찬 뒤에 사람이 한다.
SELF_VARIANT = "self_baseline"

# selected_tickers 겹침률이 이 값 이하로 계속 나오면 "두 모델이 사실상 다른 답을 낸다"는 뜻이다.
# 그 경우 모델 교체는 파라미터 조정이 아니라 **시스템 동작 변경**으로 다뤄야 한다.
# 2026-09-14 관측: 첫 두 실행에서 0.250, 0.200. 표본이 차기 전이라 판단은 유보한다.
LOW_AGREEMENT_THRESHOLD = 0.3


def shadow_mode() -> str:
    """SHADOW_MODE 환경변수. 기본 shadow. 잘못된 값은 off로 취급(비용 안전 쪽)."""
    raw = os.getenv("SHADOW_MODE", "shadow").strip().lower()
    return raw if raw in VALID_MODES else "off"


def shadow_active(name: str, today: Optional[date] = None) -> bool:
    """이 실험의 candidate를 오늘 호출해도 되는가.

    False 조건: 실험 미등록 / SHADOW_MODE=off / review_by 경과.
    """
    policy = SHADOW_POLICIES.get(name)
    if policy is None:
        return False
    if shadow_mode() != "shadow":
        return False
    today = today or date.today()
    if today > date.fromisoformat(policy["review_by"]):
        logger.info(f"[shadow] {name} 마감({policy['review_by']}) 경과 — candidate 호출 안 함")
        return False
    return True


# ── 저장 ─────────────────────────────────────────────────────────────────

def init_shadow_table(db_path: Path | None = None) -> None:
    with sqlite3.connect(db_path or DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS llm_ab_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date      TEXT NOT NULL,
                node          TEXT NOT NULL,
                variant       TEXT NOT NULL,      -- baseline | candidate | self_baseline
                model         TEXT NOT NULL,
                input_hash    TEXT NOT NULL,
                output_json   TEXT,
                input_tokens  INTEGER,
                output_tokens INTEGER,
                latency_ms    INTEGER,
                cost_usd      REAL,
                created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ab_node_date ON llm_ab_log (node, run_date);
        """)
        conn.commit()


def input_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def log_shadow(
    node: str,
    variant: str,
    model: str,
    in_hash: str,
    output: Any,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    run_date: Optional[str] = None,
    db_path: Path | None = None,
) -> None:
    """한 변형의 결과 1행 기록. 실패해도 예외를 던지지 않는다 — 로그가 리포트를 막으면 안 된다."""
    pin, pout = (0.0, 0.0) if model == "python-formula" else _price_for(model)
    cost = (input_tokens / 1e6) * pin + (output_tokens / 1e6) * pout
    try:
        init_shadow_table(db_path)
        with sqlite3.connect(db_path or DB_PATH) as conn:
            conn.execute(
                "INSERT INTO llm_ab_log (run_date,node,variant,model,input_hash,output_json,"
                "input_tokens,output_tokens,latency_ms,cost_usd,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_date or date.today().isoformat(), node, variant, model, in_hash,
                 json.dumps(output, ensure_ascii=False), int(input_tokens), int(output_tokens),
                 int(latency_ms), round(cost, 6), datetime.now().isoformat()),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"[shadow] 기록 실패 ({node}/{variant}): {type(e).__name__}: {e}")


# ── 비교 ─────────────────────────────────────────────────────────────────

def _param_agreement(a: Optional[dict], b: Optional[dict], rel_tol: float = 1e-3) -> Optional[float]:
    """두 파라미터 dict의 공통 키 중 값이 (상대오차 rel_tol 안에서) 같은 비율. 키가 없으면 None."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    keys = [k for k in a if k in b]
    if not keys:
        return None
    same = 0
    for k in keys:
        x, y = a[k], b[k]
        if x is None or y is None:
            same += int(x is None and y is None)
        else:
            try:
                same += int(abs(float(x) - float(y)) <= rel_tol * max(1.0, abs(float(y))))
            except (TypeError, ValueError):
                same += int(x == y)
    return same / len(keys)


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def compare(node: str, db_path: Path | None = None) -> dict:
    """node의 baseline/candidate를 run_date·input_hash로 짝지어 비교한다.

    반환: {"pairs": n, "days": [...], "agreement": 평균 자카드, "cost": {variant: 합},
           "latency_ms": {variant: 평균}, "direct_ratio": {variant: 평균}, "enough": bool}
    direct_ratio는 output_json에 "direct_ratio"가 있을 때만 (Stage 1-C가 넣는다).
    """
    path = db_path or DB_PATH
    policy = SHADOW_POLICIES.get(node, {})
    if not path.exists():
        return {"pairs": 0, "days": [], "enough": False}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM llm_ab_log WHERE node=? ORDER BY run_date, id", (node,)
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["run_date"], r["input_hash"]), {})[r["variant"]] = dict(r)

    # CR-15 (2026-09-14): 표본은 **거래일** 단위다.
    #
    # [무엇이 문제였나]
    #   min_samples=10은 주석상 "거래일 10일"인데 예전에는 (run_date, input_hash) 쌍을 셌다.
    #   같은 날 수동 dispatch를 여러 번 돌리면 헤드라인이 달라져 별개 쌍이 되고,
    #   거래일 10일을 채우지 않고도 enough=True가 됐다.
    #   실측: run 34766864896의 출력에 "날짜: 2026-09-13, 2026-09-13".
    #
    # [같은 날 여러 행 중 어느 것을 쓰나 — 마지막 실행]
    #   id가 가장 큰 쌍을 쓴다. 그날 마지막 실행이 코드가 최신이고 헤드라인도
    #   가장 늦은 시점의 것이다. 앞선 실행은 대개 검증 도중의 중간 상태다.
    per_day: dict[str, tuple[int, tuple]] = {}
    for key, pair in by_key.items():
        if "baseline" not in pair or "candidate" not in pair:
            continue
        last_id = max(pair[v]["id"] for v in ("baseline", "candidate"))
        if key[0] not in per_day or last_id > per_day[key[0]][0]:
            per_day[key[0]] = (last_id, key)
    chosen = {k for _, k in per_day.values()}

    days, jaccards, param_scores, self_jaccards = [], [], [], []
    cost   = {"baseline": 0.0, "candidate": 0.0, SELF_VARIANT: 0.0}
    lat    = {"baseline": [],  "candidate": [],  SELF_VARIANT: []}
    direct = {"baseline": [],  "candidate": [],  SELF_VARIANT: []}
    for key, pair in sorted(by_key.items()):
        if key not in chosen:
            continue
        run_date = key[0]
        days.append(run_date)
        outs = {}
        for v in ("baseline", "candidate"):
            try:
                outs[v] = json.loads(pair[v]["output_json"] or "{}")
            except json.JSONDecodeError:
                outs[v] = {}
            cost[v] += pair[v]["cost_usd"] or 0.0
            lat[v].append(pair[v]["latency_ms"] or 0)
            if isinstance(outs[v], dict) and outs[v].get("direct_ratio") is not None:
                direct[v].append(float(outs[v]["direct_ratio"]))
        j = _jaccard(set(outs["baseline"].get("selected_tickers") or []),
                     set(outs["candidate"].get("selected_tickers") or []))
        if j is not None:
            jaccards.append(j)
        pa = _param_agreement(outs["baseline"].get("params"), outs["candidate"].get("params"))
        if pa is not None:
            param_scores.append(pa)

        # 자기 겹침률 기준선: 같은 모델·같은 입력을 한 번 더 부른 결과와의 자카드.
        # candidate 겹침률이 낮을 때 "모델이 달라서"인지 "이 단계가 원래 불안정해서"인지
        # 가르는 유일한 대조군이다. 판단은 하지 않고 숫자만 쌓는다.
        if SELF_VARIANT in pair:
            try:
                self_out = json.loads(pair[SELF_VARIANT]["output_json"] or "{}")
            except json.JSONDecodeError:
                self_out = {}
            cost[SELF_VARIANT] += pair[SELF_VARIANT]["cost_usd"] or 0.0
            lat[SELF_VARIANT].append(pair[SELF_VARIANT]["latency_ms"] or 0)
            if isinstance(self_out, dict) and self_out.get("direct_ratio") is not None:
                direct[SELF_VARIANT].append(float(self_out["direct_ratio"]))
            sj = _jaccard(set(outs["baseline"].get("selected_tickers") or []),
                          set(self_out.get("selected_tickers") or []))
            if sj is not None:
                self_jaccards.append((run_date, sj))

    def avg(xs):
        return (sum(xs) / len(xs)) if xs else None
    return {
        "node":         node,
        "label":        policy.get("label", ""),
        "review_by":    policy.get("review_by"),
        "min_samples":  policy.get("min_samples"),
        "pairs":        len(days),
        "days":         days,
        "enough":       len(days) >= int(policy.get("min_samples", 0) or 0),
        "agreement":    avg(jaccards),
        # 2026-09-14: 일별 겹침률을 그대로 남긴다. 평균만으로는 "매일 비슷하게 갈리는지"와
        # "어떤 날만 크게 갈리는지"를 구분할 수 없다.
        "agreement_by_day": list(zip(days, jaccards)) if len(days) == len(jaccards) else [],
        # 자기 겹침률 — 같은 모델을 두 번 부른 결과끼리. 비교 기준선이다.
        "self_agreement":        avg([j for _, j in self_jaccards]),
        "self_agreement_by_day": self_jaccards,
        "param_agreement": avg(param_scores),
        "cost":         {k: round(v, 4) for k, v in cost.items()},
        "latency_ms":   {k: avg(v) for k, v in lat.items()},
        "direct_ratio": {k: avg(v) for k, v in direct.items()},
    }


def has_variant_on(node: str, variant: str, run_date: str, db_path: Path | None = None) -> bool:
    """그날 이 variant를 이미 기록했는가. 같은 날 재실행에서 중복 호출을 막는다."""
    path = db_path or DB_PATH
    if not path.exists():
        return False
    with sqlite3.connect(path) as conn:
        try:
            row = conn.execute(
                "SELECT 1 FROM llm_ab_log WHERE node=? AND variant=? AND run_date=? LIMIT 1",
                (node, variant, run_date),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
    return row is not None


def _agreement_flag(res: dict) -> str:
    """겹침률이 낮게 유지되면 표시만 한다. **판단은 사람이 표본을 보고 한다.**

    낮은 겹침률 자체가 candidate가 나쁘다는 뜻은 아니다. "같은 입력에 두 모델이
    다른 종목을 고른다"는 사실일 뿐이고, 어느 쪽이 맞았는지는 D+5/10/20 성적이 말한다.
    """
    by_day = res.get("agreement_by_day") or []
    if len(by_day) < 2:
        return ""
    lows = [j for _, j in by_day if j is not None and j <= LOW_AGREEMENT_THRESHOLD]
    if len(lows) == len(by_day):
        return f"  ⚠️ 모델 의존성 높음 ({len(by_day)}일 연속 {LOW_AGREEMENT_THRESHOLD} 이하)"
    return ""


def format_compare(res: dict) -> str:
    if not res.get("pairs"):
        return f"[shadow] {res.get('node')} — 짝지어진 표본 없음"
    def f(x):
        return "n/a" if x is None else f"{x:.3f}"
    lines = [
        f"[shadow] {res['node']} — {res['label']}",
        f"  표본 {res['pairs']}일 (최소 {res['min_samples']}, 마감 {res['review_by']}) "
        f"{'→ 판정 가능' if res['enough'] else '→ 표본 부족'}",
        f"  selected_tickers 자카드 겹침률 평균: {f(res['agreement'])}{_agreement_flag(res)}",
        "  일별 겹침률: " + (", ".join(f"{d} {j:.2f}" for d, j in res.get("agreement_by_day") or [])
                          or "n/a"),
        f"  자기 겹침률(같은 모델 재호출) 평균:   {f(res.get('self_agreement'))}"
        + ("  ← 비교 기준선" if res.get("self_agreement") is not None else "  (아직 기록 없음)"),
        f"  파라미터 일치율 평균(6개 공식):        {f(res.get('param_agreement'))}",
        f"  헤드라인 직접언급 비율  baseline {f(res['direct_ratio']['baseline'])} / candidate {f(res['direct_ratio']['candidate'])}",
        f"  비용 합계(USD)          baseline {res['cost']['baseline']:.4f} / candidate {res['cost']['candidate']:.4f}",
        f"  지연 평균(ms)           baseline {f(res['latency_ms']['baseline'])} / candidate {f(res['latency_ms']['candidate'])}",
        "  날짜: " + ", ".join(res["days"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(SHADOW_POLICIES)
    for n in names:
        print(format_compare(compare(n)))
        print()
