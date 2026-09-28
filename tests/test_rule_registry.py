"""레지스트리·상태 기계·분기 m. 스펙 §1-3, §3-2, §2-2."""
import sqlite3
from datetime import date

import pytest

from tests.test_rule_spec import BASE


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src.screening.rules import registry as R
    path = tmp_path / "m.db"
    monkeypatch.setattr(R, "DB_PATH", path)
    R.init_rule_tables()
    return path


def _spec(**over):
    from src.screening.rules.spec import parse_spec
    return parse_spec({**BASE, **over})


def test_register_and_get(db):
    from src.screening.rules import registry as R
    assert R.register(_spec(), quarter="2026Q4") == "2026Q4"
    r = R.get_rule("example_rule")
    assert r["status"] == "draft" and r["quarter"] == "2026Q4"
    assert r["spec"]["horizon"] == 10 and len(r["spec_hash"]) == 16 and r["gate_result"] is None


def test_duplicate_hash_rejected(db):
    from src.screening.rules import registry as R
    R.register(_spec(), quarter="2026Q4")
    with pytest.raises(ValueError, match="spec_hash"):
        R.register(_spec(rule_id="renamed"), quarter="2026Q4")


def test_quarter_of():
    from src.screening.rules.registry import quarter_of
    assert quarter_of(date(2026, 9, 23)) == "2026Q3"
    assert quarter_of(date(2026, 10, 1)) == "2026Q4"
    assert quarter_of(date(2027, 1, 15)) == "2027Q1"


def test_fix_quarter_m_once_and_late_registration_moves_quarter(db):
    from src.screening.rules import registry as R
    R.register(_spec(rule_id="a"), quarter="2026Q4")
    R.register(_spec(rule_id="b", n_picks=4), quarter="2026Q4")
    assert R.fix_quarter_m("2026Q4") == 2
    assert R.register(_spec(rule_id="c", n_picks=3), quarter="2026Q4") == "2027Q1"
    assert R.fix_quarter_m("2026Q4") == 2, "한 번 고정된 m은 안 바뀐다"


def test_null_random_is_not_counted_in_m(db):
    from src.screening.rules import registry as R
    R.register(_spec(rule_id="a"), quarter="2026Q4")
    R.register(_spec(rule_id="null_random", signal={"name": "random_score", "params": {}}), quarter="2026Q4")
    assert R.fix_quarter_m("2026Q4") == 1


@pytest.mark.parametrize("frm, to, ok", [
    ("draft", "gated", True), ("draft", "rejected", True), ("draft", "active", False),
    ("gated", "shadow", True), ("shadow", "active", True), ("shadow", "retired", True),
    ("active", "retired", True), ("active", "shadow", False),
    ("observe", "draft", True), ("observe", "active", False), ("observe", "shadow", False),
    ("retired", "draft", False), ("rejected", "draft", False),
])
def test_transition_table(db, frm, to, ok):
    from src.screening.rules import registry as R
    R.register(_spec(), quarter="2026Q4")
    with sqlite3.connect(db) as c:
        c.execute("UPDATE screen_rules SET status=? WHERE rule_id='example_rule'", (frm,))
        c.commit()
    if ok:
        R.transition("example_rule", to, reason="test")
        assert R.get_rule("example_rule")["status"] == to
        with sqlite3.connect(db) as c:
            ev = c.execute("SELECT from_status, to_status, reason FROM screen_rule_events").fetchall()
        assert ev == [(frm, to, "test")]
    else:
        with pytest.raises(ValueError):
            R.transition("example_rule", to, reason="test")


def test_gated_stores_result_and_shadow_sets_timestamp(db):
    from src.screening.rules import registry as R
    R.register(_spec(), quarter="2026Q4")
    R.transition("example_rule", "gated", reason="gate pass", metrics={"t": 2.5})
    R.transition("example_rule", "shadow", reason="auto")
    r = R.get_rule("example_rule")
    assert r["gate_result"] == {"t": 2.5} and r["shadow_since"] is not None


def test_list_rules_by_status(db):
    from src.screening.rules import registry as R
    R.register(_spec(rule_id="a"), quarter="2026Q4")
    R.register(_spec(rule_id="b", n_picks=4), quarter="2026Q4")
    R.transition("a", "rejected", reason="x")
    assert [r["rule_id"] for r in R.list_rules("draft")] == ["b"]
    assert len(R.list_rules()) == 2


def test_db_path_follows_mentions_db_path_env(monkeypatch, tmp_path):
    """다른 DB 모듈(ohlcv_cache·decision_memory)처럼 MENTIONS_DB_PATH를 따른다 —
    운영 DB 사본에 규칙을 등록할 때 로컬 data/mentions.db에 조용히 쓰지 않게."""
    import importlib
    from pathlib import Path
    from src.screening.rules import registry as R
    monkeypatch.setenv("MENTIONS_DB_PATH", str(tmp_path / "prod_copy.db"))
    try:
        assert importlib.reload(R).DB_PATH == Path(tmp_path / "prod_copy.db")
    finally:
        monkeypatch.delenv("MENTIONS_DB_PATH")
        importlib.reload(R)


def test_declared_m_is_used_and_leaves_room_for_the_rest(db):
    """스펙이 분기 가설 수를 미리 정했는데 일부만 준비됐을 때(2027Q1: m=2, buyback_mid는 DART 이력 대기)."""
    from src.screening.rules import registry as R
    R.register(_spec(rule_id="a"), quarter="2027Q1")
    R.declare_quarter_m("2027Q1", 2)
    assert R.fix_quarter_m("2027Q1") == 2, "첫 게이트가 선언값을 쓴다"
    assert R.register(_spec(rule_id="b", n_picks=4), quarter="2027Q1") == "2027Q1", "선언한 m까지는 같은 분기"
    assert R.register(_spec(rule_id="c", n_picks=3), quarter="2027Q1") == "2027Q2", "m이 차면 다음 분기"


def test_declare_rejects_fixed_quarter_or_m_below_registered(db):
    from src.screening.rules import registry as R
    R.register(_spec(rule_id="a"), quarter="2027Q1")
    R.register(_spec(rule_id="b", n_picks=4), quarter="2027Q1")
    with pytest.raises(ValueError, match="등록된"):
        R.declare_quarter_m("2027Q1", 1)
    R.declare_quarter_m("2027Q1", 2)
    with pytest.raises(ValueError, match="고정"):
        R.declare_quarter_m("2027Q1", 3)


def test_cli_declare_m(db, monkeypatch, capsys):
    import sys
    from src.screening import gate as G
    from src.screening.rules import registry as R
    R.register(_spec(rule_id="a"), quarter="2027Q1")
    monkeypatch.setattr(sys, "argv", ["gate", "--declare-m", "2", "--quarter", "2027Q1"])
    G.main()
    assert R.fix_quarter_m("2027Q1") == 2 and "2027Q1" in capsys.readouterr().out
