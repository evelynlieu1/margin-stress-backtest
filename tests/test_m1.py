"""Tests for the loaders in src/data.py. Each one pins a rule later scripts rely on."""

import numpy as np
import pandas as pd
import pytest

import data


def test_series_cover_expected_dates():
    """Each series starts and ends where the README says, with unique sorted dates."""
    brent, wti, aud = data.load_brent(), data.load_wti(), data.load_audusd()
    assert brent.index[0] == pd.Timestamp("1987-05-20")
    assert wti.index[0] == pd.Timestamp("1986-01-02")
    assert aud.index[0] == pd.Timestamp("1983-12-12")
    for s in (brent, wti, aud):
        assert s.index.is_monotonic_increasing and not s.index.has_duplicates


def test_text_cells_become_missing_not_zero():
    """'Closed' and 'N/A' are missing values; any other text stops the loader."""
    assert np.isnan(data.to_number("Closed")) and np.isnan(data.to_number("CLOSED"))
    assert np.isnan(data.to_number("N/A")) and np.isnan(data.to_number(""))
    with pytest.raises(ValueError):
        data.to_number("see note")
    aud = data.load_audusd()
    assert aud.isna().sum() == 31 and (aud.dropna() > 0).all()


def test_audusd_join_and_sources():
    """The 2022 workbook and the 2023 CSV meet without a gap of more than a holiday."""
    aud = data.load_audusd(with_source=True)
    assert aud.loc["2022-12-30", "audusd"] == pytest.approx(0.6775)
    assert aud.loc["2023-01-03", "audusd"] == pytest.approx(0.6828)
    assert set(aud["source"]) == {"RBA", "RBA and WM/Reuters", "WM/Reuters"}


def test_moves_across_holes_are_dropped():
    """A '1-day' move spanning a 10-day hole is not kept; a normal weekend is."""
    prices = pd.Series([100.0, 101.0, 102.0, 103.0],
                       index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-16", "2020-01-17"]))
    r = data.log_returns(prices, 1)
    assert list(r.index) == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-17")]
    assert r.iloc[0] == pytest.approx(np.log(101 / 100))


def test_wti_needs_dollar_changes():
    """WTI was negative on 20 Apr 2020, so log returns refuse it and dollar changes work."""
    wti = data.load_wti()
    assert wti.loc["2020-04-20"] == pytest.approx(-36.98)
    with pytest.raises(ValueError):
        data.log_returns(wti, 1)
    assert data.dollar_changes(wti, 1).loc["2020-04-20"] == pytest.approx(-36.98 - wti.loc["2020-04-17"])
