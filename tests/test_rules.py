import pandas as pd

from src.models.rules import no_leak, should_promote


def test_first_model_is_promoted():
    assert should_promote(30.0, None) == (True, None)


def test_clear_improvement_is_promoted():
    ok, gain = should_promote(90.0, 100.0, min_gain=0.01)
    assert ok and abs(gain - 0.10) < 1e-9


def test_gain_below_margin_is_rejected():
    ok, _ = should_promote(99.5, 100.0, min_gain=0.01)       # 0.5% < 1%
    assert not ok


def test_gain_above_margin_is_promoted():
    ok, _ = should_promote(98.0, 100.0, min_gain=0.01)       # 2% >= 1%
    assert ok


def test_equal_model_is_rejected():
    ok, gain = should_promote(100.0, 100.0)
    assert not ok and gain == 0


def test_worse_model_is_rejected():
    ok, gain = should_promote(120.0, 100.0)
    assert not ok and gain < 0


def test_zero_production_error_does_not_crash():
    assert should_promote(1.0, 0.0) == (False, None)


def test_guard_accepts_model_trained_before_holdout():
    assert no_leak("2023-10-02", pd.Timestamp("2023-10-02"))[0]
    assert no_leak("2023-09-01", pd.Timestamp("2023-10-02"))[0]


def test_guard_refuses_model_that_saw_the_holdout():
    assert not no_leak("2023-12-04", pd.Timestamp("2023-10-02"))[0]


def test_guard_refuses_untagged_model():
    assert not no_leak(None, pd.Timestamp("2023-10-02"))[0]