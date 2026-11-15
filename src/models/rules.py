"""The promotion rule and the leakage guard, as pure functions (easy to test)."""
import pandas as pd


def should_promote(cand_werr, prod_werr, min_gain=0.01):
    """Return (promote?, relative_gain). Lower weighted error is better.
    No production model yet -> the candidate is promoted."""
    if prod_werr is None:
        return True, None
    if prod_werr <= 0:                    # cannot beat a zero error, avoid dividing by 0
        return False, None
    gain = 1 - cand_werr / prod_werr
    return gain >= min_gain, gain


def no_leak(prod_trained_until, holdout_start):
    """The production model may be scored on the holdout only if it never trained
    on it. `trained_until` = first week NOT used for training."""
    if prod_trained_until is None:
        return False, ("production model has no 'trained_until' tag, so it cannot "
                       "be shown to be free of the holdout")
    if pd.Timestamp(prod_trained_until) > pd.Timestamp(holdout_start):
        return False, (f"production trained until {prod_trained_until}, after the "
                       f"holdout start {pd.Timestamp(holdout_start).date()}: it has seen the holdout")
    return True, ""