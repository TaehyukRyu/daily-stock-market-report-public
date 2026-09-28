"""규칙 스펙 — 7필드, 해시 불변, Bonferroni. 스펙 §1-1, §2-2."""
import pytest

BASE = {
    "rule_id": "example_rule",
    "universe": {"kospi_rank": [1, 50], "kosdaq_rank": [1, 10],
                 "min_turnover_krw": 1e9, "exclude": ["preferred"]},
    "signal": {"name": "ret_20d", "params": {}},
    "filters": [{"name": "ret_5d_max", "params": {"max": 0.20}}],
    "candidate": {"percentile_max": 0.15},
    "ranking": "signal asc",
    "n_picks": 5,
    "horizon": 10,
}


def test_parse_roundtrip_and_hash_is_16_hex():
    from src.screening.rules.spec import parse_spec, spec_hash
    s = parse_spec(BASE)
    assert s.rule_id == "example_rule" and s.horizon == 10
    h = spec_hash(s)
    assert len(h) == 16 and int(h, 16) >= 0


def test_hash_ignores_key_order_but_not_values():
    from src.screening.rules.spec import parse_spec, spec_hash
    reordered = {k: BASE[k] for k in reversed(list(BASE))}
    assert spec_hash(parse_spec(reordered)) == spec_hash(parse_spec(BASE))
    assert spec_hash(parse_spec({**BASE, "n_picks": 6})) != spec_hash(parse_spec(BASE))


def test_hash_ignores_rule_id():
    """rule_id는 이름표다. 같은 가설을 다른 이름으로 다시 올릴 수 없어야 한다."""
    from src.screening.rules.spec import parse_spec, spec_hash
    assert spec_hash(parse_spec({**BASE, "rule_id": "other"})) == spec_hash(parse_spec(BASE))


@pytest.mark.parametrize("missing", ["universe", "signal", "candidate", "n_picks", "horizon"])
def test_missing_field_raises(missing):
    from src.screening.rules.spec import parse_spec
    with pytest.raises(ValueError, match=missing):
        parse_spec({k: v for k, v in BASE.items() if k != missing})


def test_unknown_signal_name_raises_at_parse():
    from src.screening.rules.spec import parse_spec
    with pytest.raises(ValueError, match="signal"):
        parse_spec({**BASE, "signal": {"name": "made_up", "params": {}}})


def test_unknown_filter_name_raises_at_parse():
    from src.screening.rules.spec import parse_spec
    with pytest.raises(ValueError, match="filter"):
        parse_spec({**BASE, "filters": [{"name": "nope", "params": {}}]})


@pytest.mark.parametrize("m, expected", [(1, 1.96), (2, 2.24), (3, 2.39), (5, 2.58), (20, 3.02)])
def test_bonferroni_two_sided(m, expected):
    from src.screening.rules.spec import bonferroni_t
    assert bonferroni_t(m) == pytest.approx(expected, abs=0.01)


def test_bonferroni_rejects_nonpositive_m():
    from src.screening.rules.spec import bonferroni_t
    with pytest.raises(ValueError):
        bonferroni_t(0)
