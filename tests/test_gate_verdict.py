"""판정선 — 스펙 §2-2 표를 그대로 고정."""
import pytest


def _sig(mean=0.008, t=3.0, lo=0.002, n=200):
    return {"n_dates": n, "n_trades": 5 * n, "mean": mean, "sd": 0.03, "t": t, "boot_ci95": [lo, 2 * mean]}

def _oos(t=0.5, n=28):
    return {"n_dates": n, "n_trades": 5 * n, "mean": 0.003, "sd": 0.03, "t": t, "boot_ci95": [-0.01, 0.02]}

def _ctrl(pct=99.5):
    return {"n": 1000, "strategy_percentile": pct}

def _wf(pos=28, n=40):
    return {"n": n, "positive": pos, "negative": n - pos, "positive_ratio": pos / n}


def _run(**over):
    from src.screening.gate import verdict
    kw = dict(sig_index=_sig(), sig_uni=_sig(), oos_index=_oos(), oos_uni=_oos(),
              ctrl_index=_ctrl(), ctrl_uni=_ctrl(), wf_index=_wf(), wf_uni=_wf(), m=3, history_months=128)
    kw.update(over)
    return verdict(**kw)


def test_all_pass():
    v = _run()
    assert v["passed"] and v["failed"] == [] and v["t_star"] == pytest.approx(2.39, abs=0.01)


def test_too_few_dates_fails_first():
    assert _run(sig_index=_sig(n=40))["failed"][0] == "min_dates"


def test_too_few_oos_dates():
    assert "min_oos_dates" in _run(oos_index=_oos(n=6))["failed"]


def test_t_below_bonferroni():
    assert "t_index" in _run(sig_index=_sig(t=2.2))["failed"]            # t*=2.39


def test_ci_lower_must_be_positive():
    assert "ci_uni" in _run(sig_uni=_sig(lo=-0.001))["failed"]


def test_oos_only_blocks_clear_collapse():
    """OOS는 붕괴 확인만 — t가 음수여도 −1 위면 통과, −1 이하면 탈락."""
    assert _run(oos_index=_oos(t=-0.6))["passed"]
    assert "oos_index" in _run(oos_index=_oos(t=-1.4))["failed"]


def test_control_top_pct_divided_by_m():
    assert "control_index" in _run(ctrl_index=_ctrl(pct=98.0))["failed"]   # 필요 100−5/3=98.33
    assert _run(ctrl_index=_ctrl(pct=98.5))["passed"]


def test_walk_forward_ratio_and_window_count():
    assert "wf_index" in _run(wf_index=_wf(pos=22, n=40))["failed"]         # 55%
    assert "wf_uni" in _run(wf_uni=_wf(pos=4, n=4))["failed"]               # 창 4개


def test_history_minimum():
    assert "min_history" in _run(history_months=12)["failed"]


def test_null_rule_uses_m_equals_1():
    v = _run(m=1, sig_index=_sig(t=2.0), sig_uni=_sig(t=2.0), ctrl_index=_ctrl(96.0), ctrl_uni=_ctrl(96.0))
    assert v["t_star"] == pytest.approx(1.96, abs=0.01) and v["passed"]


def test_missing_keys_fail_safely():
    """거래가 0건이면 significance가 n_dates만 준다 — 예외가 아니라 탈락이어야 한다."""
    v = _run(sig_index={"n_dates": 0, "n_trades": 0}, oos_index={"n_dates": 0, "n_trades": 0})
    assert not v["passed"] and "min_dates" in v["failed"]
