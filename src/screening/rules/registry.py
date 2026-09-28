"""
규칙 레지스트리와 상태 기계. 스펙 §1-3, §3-2, §2-2.

  screen_rules        규칙 1행. spec_hash UNIQUE.
  screen_selections   규칙별 일일 선정 (3단계가 쓴다. DDL만 여기서)
  screen_rule_events  전이 기록 — "왜 죽었나"
  screen_quarters     분기 m 확정 기록
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.screening.rules.spec import RuleSpec, parse_spec, spec_hash

DB_PATH = Path(os.getenv("MENTIONS_DB_PATH", "data/mentions.db"))

STATUSES = ("draft", "gated", "shadow", "active", "observe", "retired", "rejected")
ALLOWED: dict[str, set[str]] = {
    "draft":    {"gated", "rejected"},
    "gated":    {"shadow"},
    "shadow":   {"active", "retired"},
    "active":   {"retired"},
    "observe":  {"draft"},
    "retired":  set(),
    "rejected": set(),
}
NULL_RULE_SIGNAL = "random_score"     # 대조군. m에 안 센다 (§2-2)


def quarter_of(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _next_quarter(q: str) -> str:
    y, n = int(q[:4]), int(q[-1])
    return f"{y + 1}Q1" if n == 4 else f"{y}Q{n + 1}"


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    c = sqlite3.connect(db_path or DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_rule_tables(db_path: Optional[Path] = None) -> None:
    with _conn(db_path) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS screen_rules (
                rule_id          TEXT PRIMARY KEY,
                spec_json        TEXT NOT NULL,
                spec_hash        TEXT NOT NULL UNIQUE,
                status           TEXT NOT NULL DEFAULT 'draft',
                quarter          TEXT NOT NULL,
                gate_result_json TEXT,
                shadow_since     TEXT,
                active_since     TEXT,
                retired_at       TEXT,
                retired_reason   TEXT,
                created_at       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS screen_selections (
                run_date       TEXT NOT NULL,
                rule_id        TEXT NOT NULL,
                ticker         TEXT NOT NULL,
                score          REAL,
                rank           INTEGER,
                universe_rank  INTEGER,
                status_at_run  TEXT NOT NULL,
                PRIMARY KEY (run_date, rule_id, ticker)
            );
            CREATE TABLE IF NOT EXISTS screen_rule_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id      TEXT NOT NULL,
                from_status  TEXT NOT NULL,
                to_status    TEXT NOT NULL,
                reason       TEXT NOT NULL,
                metrics_json TEXT,
                at           TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS screen_quarters (
                quarter  TEXT PRIMARY KEY,
                m        INTEGER NOT NULL,
                fixed_at TEXT NOT NULL
            );
        """)
        c.commit()


def register(spec: RuleSpec, quarter: Optional[str] = None, db_path: Optional[Path] = None) -> str:
    init_rule_tables(db_path)
    h = spec_hash(spec)
    q = quarter or quarter_of(date.today())
    with _conn(db_path) as c:
        if c.execute("SELECT 1 FROM screen_rules WHERE spec_hash=?", (h,)).fetchone():
            raise ValueError(f"같은 spec_hash가 이미 등록돼 있다: {h}")
        while not _has_room(c, q):
            q = _next_quarter(q)
        c.execute(
            "INSERT INTO screen_rules (rule_id, spec_json, spec_hash, status, quarter, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (spec.rule_id, spec.to_json(), h, "draft", q, datetime.now().isoformat()),
        )
        c.commit()
    return q


def _counted(c: sqlite3.Connection, quarter: str) -> int:
    """그 분기에 등록된 가설 수(대조군 제외, 상태 무관)."""
    rows = c.execute("SELECT spec_json FROM screen_rules WHERE quarter=?", (quarter,)).fetchall()
    return sum(1 for x in rows if json.loads(x["spec_json"])["signal"]["name"] != NULL_RULE_SIGNAL)


def _has_room(c: sqlite3.Connection, quarter: str) -> bool:
    """m이 아직 안 정해졌거나, 정해진 m보다 등록된 가설이 적으면 이 분기에 넣는다.
    게이트가 정한 m은 그때의 draft 수라 늘 꽉 차 있다 — 여유는 declare_quarter_m으로만 생긴다."""
    r = c.execute("SELECT m FROM screen_quarters WHERE quarter=?", (quarter,)).fetchone()
    return r is None or _counted(c, quarter) < int(r["m"])


def declare_quarter_m(quarter: str, m: int, db_path: Optional[Path] = None) -> None:
    """스펙이 분기 가설 수를 미리 정했는데 일부만 준비됐을 때, 첫 게이트 전에 m을 선언한다.
    통과선을 준비된 가설 수에 맞춰 낮추지 않기 위해서다. 선언 뒤에도 m개까지는 이 분기에 등록된다."""
    init_rule_tables(db_path)
    with _conn(db_path) as c:
        r = c.execute("SELECT m FROM screen_quarters WHERE quarter=?", (quarter,)).fetchone()
        if r:
            raise ValueError(f"{quarter}는 이미 m={r['m']}로 고정됐다")
        n = _counted(c, quarter)
        if m < max(1, n):
            raise ValueError(f"선언 m={m}이 이미 등록된 가설 {n}개보다 작다")
        c.execute("INSERT INTO screen_quarters (quarter, m, fixed_at) VALUES (?,?,?)",
                  (quarter, m, datetime.now().isoformat()))
        c.commit()


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["spec"] = json.loads(d.pop("spec_json"))
    raw = d.pop("gate_result_json", None)
    d["gate_result"] = json.loads(raw) if raw else None
    return d


def get_rule(rule_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with _conn(db_path) as c:
        r = c.execute("SELECT * FROM screen_rules WHERE rule_id=?", (rule_id,)).fetchone()
    return _row(r) if r else None


def list_rules(status: Optional[str] = None, db_path: Optional[Path] = None) -> list[dict]:
    sql, params = "SELECT * FROM screen_rules", ()
    if status:
        sql, params = sql + " WHERE status=?", (status,)
    with _conn(db_path) as c:
        return [_row(r) for r in c.execute(sql + " ORDER BY created_at, rule_id", params)]


def transition(rule_id: str, to_status: str, reason: str,
               metrics: Optional[dict] = None, db_path: Optional[Path] = None) -> None:
    if to_status not in STATUSES:
        raise ValueError(f"알 수 없는 상태: {to_status}")
    now = datetime.now().isoformat()
    with _conn(db_path) as c:
        r = c.execute("SELECT status FROM screen_rules WHERE rule_id=?", (rule_id,)).fetchone()
        if r is None:
            raise ValueError(f"규칙 없음: {rule_id}")
        frm = r["status"]
        if to_status not in ALLOWED[frm]:
            raise ValueError(f"허용되지 않는 전이: {frm} → {to_status}")
        sets: dict = {"status": to_status}
        if to_status == "shadow":
            sets["shadow_since"] = now
        elif to_status == "active":
            sets["active_since"] = now
        elif to_status in ("retired", "rejected"):
            sets["retired_at"] = now
            sets["retired_reason"] = reason
        if to_status == "gated" and metrics is not None:
            sets["gate_result_json"] = json.dumps(metrics, ensure_ascii=False, default=str)
        cols = ", ".join(f"{k}=?" for k in sets)
        c.execute(f"UPDATE screen_rules SET {cols} WHERE rule_id=?", (*sets.values(), rule_id))
        c.execute(
            "INSERT INTO screen_rule_events (rule_id, from_status, to_status, reason, metrics_json, at) "
            "VALUES (?,?,?,?,?,?)",
            (rule_id, frm, to_status, reason,
             json.dumps(metrics, ensure_ascii=False, default=str) if metrics is not None else None, now),
        )
        c.commit()


def fix_quarter_m(quarter: str, db_path: Optional[Path] = None) -> int:
    """그 분기 draft 수(대조군 제외)를 m으로 고정. 이미 고정이면 그 값. 멱등."""
    with _conn(db_path) as c:
        r = c.execute("SELECT m FROM screen_quarters WHERE quarter=?", (quarter,)).fetchone()
        if r:
            return int(r["m"])
        rows = c.execute("SELECT spec_json FROM screen_rules WHERE quarter=? AND status='draft'", (quarter,)).fetchall()
        m = sum(1 for x in rows if json.loads(x["spec_json"])["signal"]["name"] != NULL_RULE_SIGNAL)
        c.execute("INSERT INTO screen_quarters (quarter, m, fixed_at) VALUES (?,?,?)",
                  (quarter, m, datetime.now().isoformat()))
        c.commit()
    return m


def register_from_dir(dir_: Path, quarter: str, db_path: Optional[Path] = None) -> list[str]:
    """디렉터리의 *.json을 등록. 같은 해시는 건너뛴다. 새로 등록한 rule_id 목록."""
    done: list[str] = []
    for p in sorted(Path(dir_).glob("*.json")):
        spec = parse_spec(json.loads(p.read_text(encoding="utf-8")))
        try:
            register(spec, quarter=quarter, db_path=db_path)
            done.append(spec.rule_id)
        except ValueError as e:
            if "spec_hash" not in str(e):
                raise
    return done
