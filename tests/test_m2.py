"""Tests for src/m2_coverage.py: no look-ahead in any rule, and the two coverage tests on hand examples."""

import numpy as np
import pandas as pd
import pytest

import data
import m2_coverage as m2


@pytest.fixture(scope="module")
def brent():
    return data.load_brent()


def test_no_look_ahead(brent):
    """Changing every price after a date must not change any margin set on or before it."""
    cut = pd.Timestamp("2015-06-30")
    before = m2.build_panel(brent, 2)
    shocked = brent.copy()
    shocked[shocked.index > cut] *= np.exp(np.random.default_rng(0).normal(0, 0.2, (shocked.index > cut).sum()))
    after = m2.build_panel(shocked, 2)
    margins = [c for c in before.columns if c.startswith("R")]
    pd.testing.assert_frame_equal(before.loc[:cut, margins], after.loc[:cut, margins])


def test_move_is_the_h_days_after_t(brent):
    """The move on row t is ln(p at t+2 / p at t), using the next two observed prices."""
    panel = m2.build_panel(brent, 2)
    t = panel.index[100]
    p = brent.dropna()
    i = p.index.get_loc(t)
    assert panel.loc[t, "move"] == pytest.approx(np.log(p.iloc[i + 2] / p.iloc[i]))


def test_kupiec_hand_example():
    """No breaches in 100 trials: LR = -2 x 100 x ln(0.99) = 2.010, p = 0.156. Exactly 1%: p = 1."""
    n, x, rate, p = m2.kupiec([False] * 100)
    assert (n, x, rate) == (100, 0, 0.0)
    assert p == pytest.approx(0.1563, abs=1e-4)
    assert m2.kupiec([True] * 10 + [False] * 990)[3] == pytest.approx(1.0)


def test_christoffersen_hand_example():
    """97 quiet days then two breaches in a row: n00=97, n01=1, n11=1, LR = 8.41, p = 0.004.
    Breaches spread out (never two in a row) do not count as clustering. No breaches: n/a."""
    assert m2.christoffersen([0] * 98 + [1, 1]) == pytest.approx(0.0037, abs=2e-4)
    spread = ([0] * 49 + [1]) * 2
    assert m2.christoffersen(spread) > 0.5
    assert np.isnan(m2.christoffersen([0] * 100))


def test_non_overlapping_rows_are_h_apart(brent):
    """At h=5 the test sample keeps every 5th margin date, so no two test moves share a day."""
    panel = m2.build_panel(brent, 5)
    panel = panel.dropna(subset=["move"])
    sample = m2.non_overlapping(panel, 5)
    assert len(sample) == len(panel[::5])
    assert sample.index[1] == panel.index[5]


def test_no_look_ahead_with_changed_settings():
    """The same scramble test for dollar changes (WTI) with lambda 0.97 and a 250-day FHS window."""
    wti = data.load_wti().dropna()
    cut = pd.Timestamp("2015-06-30")
    settings = {"lam": 0.97, "fhs_window": 250, "kind": "dollar"}
    before = m2.build_panel(wti, 2, **settings)
    shocked = wti.copy()
    shocked[shocked.index > cut] += np.random.default_rng(1).normal(0, 5, (shocked.index > cut).sum())
    after = m2.build_panel(shocked, 2, **settings)
    margins = [c for c in before.columns if c.startswith("R")]
    pd.testing.assert_frame_equal(before.loc[:cut, margins], after.loc[:cut, margins])


def test_conditional_coverage_adds_the_two_tests():
    """No breaches in 100: Kupiec LR = 2.010, independence cannot be computed (counts as 0), so the joint
    p-value is chi-square(2) at 2.010 = exp(-2.010 / 2) = 0.366."""
    assert m2.conditional_coverage([False] * 100) == pytest.approx(np.exp(-2.0101 / 2), abs=1e-3)


def test_spot_moves_are_removed_but_margins_kept(brent):
    """without_spot_moves blanks the moves that could touch the spring 2020 window and leaves every margin as it was."""
    panel = m2.build_panel(brent, 2)
    out = m2.without_spot_moves(panel)
    a, b = data.SPOT_EVENT
    inside = (panel.index >= a - pd.Timedelta(days=10)) & (panel.index <= b)
    assert out.loc[inside, "move"].isna().all()
    assert out.loc[~inside, "move"].equals(panel.loc[~inside, "move"])
    rules = [c for c in panel.columns if c.startswith("R")]
    assert out[rules].equals(panel[rules])


def test_margin_exists_the_day_after_a_long_holiday():
    """After Easter and Anzac Day 2011 no clean 2-day AUD/USD move ends on 28 Apr, but the margin set that day must
    still exist: the 10-year floor known then is the latest one computed (an earlier bug dropped the whole day)."""
    panel = m2.build_panel(data.load_audusd(), 2)
    assert pd.Timestamp("2011-04-28") in panel.index
    assert panel.loc["2011-04-28", "R4 floored long"] >= panel.loc["2011-04-28", "R2 EWMA long"]
