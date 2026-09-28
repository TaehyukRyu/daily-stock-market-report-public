"""
규칙 스펙 — 7필드 가설. 스펙 §1-1.

[왜 해시인가]
  결과를 보고 파라미터를 바꿔 다시 돌리는 것이 과적합의 본체다. 스펙을 해시로 고정하면
  값 하나만 바뀌어도 새 규칙이 되고 분기 m이 1 는다(§2-2). rule_id는 해시에 안 넣는다 —
  같은 가설을 다른 이름으로 다시 올리는 것도 막는다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from statistics import NormalDist

REQUIRED = ("rule_id", "universe", "signal", "filters", "candidate", "ranking", "n_picks", "horizon")


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    universe: dict
    signal: dict
    filters: list
    candidate: dict
    ranking: str
    n_picks: int
    horizon: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_spec(d: dict) -> RuleSpec:
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError(f"규칙 스펙 필수 필드 누락: {', '.join(missing)}")
    from src.screening.signals import FILTERS, SIGNALS
    name = (d.get("signal") or {}).get("name")
    if name not in SIGNALS:
        raise ValueError(f"알 수 없는 signal 이름: {name!r} (등록: {sorted(SIGNALS)})")
    for f in d.get("filters") or []:
        if f.get("name") not in FILTERS:
            raise ValueError(f"알 수 없는 filter 이름: {f.get('name')!r}")
    return RuleSpec(
        rule_id=str(d["rule_id"]), universe=dict(d["universe"]), signal=dict(d["signal"]),
        filters=list(d["filters"]), candidate=dict(d["candidate"]), ranking=str(d["ranking"]),
        n_picks=int(d["n_picks"]), horizon=int(d["horizon"]),
    )


def spec_hash(spec: RuleSpec) -> str:
    body = {k: v for k, v in asdict(spec).items() if k != "rule_id"}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def bonferroni_t(m: int, alpha: float = 0.05) -> float:
    """양측 임계값. 검정은 평균>0 단측이라 이 값은 보수적이다(스펙 §2-2)."""
    if m < 1:
        raise ValueError("m은 1 이상")
    return NormalDist().inv_cdf(1.0 - (alpha / m) / 2.0)
