"""Tests for src/m3_liquidity.py: the sign of each hedger, margin in dollars, and windows over holes."""

import numpy as np
import pandas as pd
import pytest

import m3_liquidity as m3

DAYS = pd.bdate_range("2020-03-02", periods=3)


def test_long_pays_when_price_falls():
    """Price 100 -> 95 -> 90 with a flat 10% (log) margin: the long's cash paid peaks on the last day (loss
    USD 10m on USD 100m, less the margin it gets back as the position shrinks). The short gains every day,
    so its highest running total is after the first day: a gain of 5m."""
    prices = pd.Series([100.0, 95.0, 90.0], index=DAYS)
    margin = pd.Series(0.10, index=DAYS)
    long = m3.hedger_cash_need(prices, margin, "long", 2).iloc[0]
    short = m3.hedger_cash_need(prices, margin, "short", 2).iloc[0]
    assert long["loss"] == pytest.approx(10e6) and long["peak day"] == DAYS[2]
    assert short["loss"] == pytest.approx(-5e6) and short["peak day"] == DAYS[1]
    assert long["margin up"] == pytest.approx(1e6 * (90 - 100) * (1 - np.exp(-0.10)))
    assert long["need"] == pytest.approx(long["loss"] + long["margin up"])


def test_margin_rise_adds_to_need():
    """Flat price, margin from 5% to 20% (log): no loss, but the hedger must post the margin rise."""
    prices = pd.Series([100.0, 100.0, 100.0], index=DAYS)
    margin = pd.Series([0.05, 0.10, 0.20], index=DAYS)
    out = m3.hedger_cash_need(prices, margin, "short", 2).iloc[0]
    assert out["loss"] == pytest.approx(0) and out["peak day"] == DAYS[2]
    assert out["need"] == pytest.approx(100e6 * ((np.exp(0.20) - 1) - (np.exp(0.05) - 1)))


def test_margin_dollars():
    """A log margin of ln(2) covers a halving for a long (50% of price) and a doubling for a short (100%)."""
    assert m3.margin_dollars_per_unit(10.0, np.log(2), "long") == pytest.approx(5.0)
    assert m3.margin_dollars_per_unit(10.0, np.log(2), "short") == pytest.approx(10.0)


def test_window_over_a_hole_is_dropped():
    """Two observed prices 30 calendar days apart are not a 1-business-day window."""
    idx = pd.DatetimeIndex(["2020-03-02", "2020-03-03", "2020-04-02"])
    prices = pd.Series([100.0, 99.0, 50.0], index=idx)
    out = m3.hedger_cash_need(prices, pd.Series(0.1, index=idx), "long", 1)
    assert list(out.index) == [pd.Timestamp("2020-03-02")]


def test_rate_up_separates_rate_from_price():
    """Price up 10% at a flat margin rate: margin rises, but none of it is a rate rise. Flat price, rate up: all of it is."""
    prices = pd.Series([100.0, 105.0, 110.0], index=DAYS)
    out = m3.hedger_cash_need(prices, pd.Series(0.1, index=DAYS), "short", 2).iloc[0]
    assert out["margin up"] > 0 and out["rate up"] == pytest.approx(0)
    flat = pd.Series(100.0, index=DAYS)
    out = m3.hedger_cash_need(flat, pd.Series([0.05, 0.10, 0.20], index=DAYS), "short", 2).iloc[0]
    assert out["rate up"] == pytest.approx(out["margin up"])


def test_spot_event_windows():
    """A window touches the spring 2020 window if it starts on or before its end and ends on or after its start."""
    a, b = m3.data.SPOT_EVENT
    need = pd.DataFrame({"end": [a - pd.Timedelta(days=1), a, b + pd.Timedelta(days=5)]},
                        index=[a - pd.Timedelta(days=30), a - pd.Timedelta(days=3), b + pd.Timedelta(days=1)])
    assert m3.touches_spot_event(need).tolist() == [False, True, False]


def test_max_span_allows_holiday_weeks():
    """A 5-day move may span 11 calendar days (two weekends and a holiday), a 2-day move 6, a 1-day move 5."""
    assert [m3.data.max_span_days(h) for h in (1, 2, 5)] == [5, 6, 11]
