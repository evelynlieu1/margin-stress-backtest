"""Tests for src/m4_robustness.py: the scoring windows, the rise measure, and that each variant changes one thing."""

import numpy as np
import pandas as pd
import pytest

import m2_coverage as m2
import m4_robustness as m4

DAYS = pd.bdate_range("2019-12-02", "2021-01-29")


def test_scored_windows():
    """'no 2020' drops every 2020 row and nothing else; a date range keeps only that range."""
    panel = pd.DataFrame({"move": 0.0}, index=DAYS)
    no2020 = m4.scored(panel, "no 2020")
    assert not (no2020.index.year == 2020).any()
    assert len(no2020) == (panel.index.year != 2020).sum()
    assert m4.scored(panel, ("2020-03-02", "2020-03-06")).index.tolist() == list(pd.bdate_range("2020-03-02", "2020-03-06"))
    assert m4.scored(panel, None) is panel


def test_largest_rise_skips_the_spot_event():
    """A margin that jumps 10x only inside the spring 2020 window: the skip leaves a rise of 1x."""
    m = pd.Series(1.0, index=DAYS)
    m["2020-04-15":"2020-04-20"] = 10.0
    panel = pd.DataFrame({f"{r} {s}": m for r in m2.RULES for s in ("long", "short")})
    assert m4.largest_rise(panel, "R2 EWMA", skip_spot=False) == pytest.approx(10.0)
    assert m4.largest_rise(panel, "R2 EWMA", skip_spot=True) == pytest.approx(1.0)


def test_each_variant_changes_one_thing():
    """Every variant other than 'main' changes exactly one thing: a margin setting, the scoring window, or (WTI)
    the series with its move type. The AUD/USD sub-periods change only the window, so they test AUD/USD only."""
    for name, settings, window, scope in m4.VARIANTS[1:]:
        assert len(settings) + (window is not None) == 1, name
        if scope == "WTI":
            assert settings == {"kind": "dollar"}, name
        if scope == "AUD":
            assert window is not None and not settings, name


def test_higher_confidence_raises_every_margin():
    """At 99.5% every rule asks for at least as much margin as at 99%, on every day."""
    p = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 4000))),
                  index=pd.bdate_range("1995-01-02", periods=4000))
    base, high = m2.build_panel(p, 2), m2.build_panel(p, 2, confidence=0.995)
    rules = [c for c in base.columns if c.startswith("R")]
    common = base.index.intersection(high.index)
    assert (high.loc[common, rules] >= base.loc[common, rules] - 1e-12).all().all()
