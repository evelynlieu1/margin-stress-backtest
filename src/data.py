"""Loaders shared by every script in this project.

Three daily price series, each returned as a pandas Series indexed by date:
    load_brent()  - Europe Brent spot, USD per barrel (EIA)
    load_wti()    - Cushing WTI spot, USD per barrel (EIA)
    load_audusd() - US dollars per Australian dollar (RBA table F11)

Rules every loader follows:
    - Blank cells stay missing (NaN). A blank is never read as zero.
    - Dates are sorted and unique; a repeated date stops the loader.
    - Nothing in data/raw is ever changed.

Also here: log_returns() and dollar_changes(), the two ways later scripts turn
prices into moves over a horizon of h business days.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"

BRENT_FILE = "eia_brent_spot_daily.xls"
WTI_FILE = "eia_wti_spot_daily.xls"
F11_XLS_FILES = [  # oldest first; each workbook covers the years in its name
    "rba_f11_exchange_rates_1983-1986.xls",
    "rba_f11_exchange_rates_1987-1990.xls",
    "rba_f11_exchange_rates_1991-1994.xls",
    "rba_f11_exchange_rates_1995-1998.xls",
    "rba_f11_exchange_rates_1999-2002.xls",
    "rba_f11_exchange_rates_2003-2006.xls",
    "rba_f11_exchange_rates_2007-2009.xls",
    "rba_f11_exchange_rates_2010-2013.xls",
    "rba_f11_exchange_rates_2014-2017.xls",
    "rba_f11_exchange_rates_2018-2022.xls",
]
F11_CSV_FILE = "rba_f11_1_exchange_rates_2023_current.csv"
AUDUSD_SERIES_ID = "FXRUSD"  # the RBA's code for the A$1=USD column


def excel_date(value):
    """Turn an Excel date cell into a pandas Timestamp.

    Excel stores dates as days since 30 Dec 1899 (the 1900 date system, used by
    both the EIA and RBA workbooks). Some RBA cells hold that number as text, so
    text is converted to a number first.
    """
    return pd.Timestamp(xlrd.xldate_as_datetime(float(value), 0))


TEXT_AS_MISSING = {"closed", "n/a"}  # F11 writes 'Closed' (any case) on market holidays, 'N/A' once (2014-2017)


def to_number(cell):
    """A price cell as a float, or NaN when the cell is blank, 'Closed' or 'N/A'.

    Any other text stops the loader, so an unexpected note in the data is seen
    rather than silently dropped. M1 counts these text cells.
    """
    if cell is None:
        return np.nan
    if isinstance(cell, str):
        text = cell.strip()
        if text == "" or text.lower() in TEXT_AS_MISSING:
            return np.nan
    return float(cell)


def check_dates(series, name):
    """Sort by date and stop if any date appears twice. Returns the sorted series."""
    series = series.sort_index()
    if series.index.has_duplicates:
        dupes = series.index[series.index.duplicated()].unique()
        raise ValueError(f"{name}: repeated dates {list(dupes[:5])}")
    return series


def load_eia(file_name, name):
    """One EIA daily spot series.

    Layout of the EIA workbook: a 'Contents' sheet, then 'Data 1' with three
    header rows (a back link, the source key, then 'Date' and the series title)
    followed by one row per trading day: Excel date, price.
    """
    sheet = xlrd.open_workbook(RAW_DIR / file_name).sheet_by_name("Data 1")
    start = next(i for i in range(sheet.nrows) if sheet.cell_value(i, 0) == "Date") + 1
    dates, prices = [], []
    for i in range(start, sheet.nrows):
        date_cell, price_cell = sheet.cell_value(i, 0), sheet.cell_value(i, 1)
        if date_cell == "":
            continue  # trailing empty rows
        dates.append(excel_date(date_cell))
        prices.append(to_number(price_cell))
    return check_dates(pd.Series(prices, index=pd.DatetimeIndex(dates), name=name), name)


def load_brent():
    """Europe Brent spot price FOB, USD per barrel, daily from 20 May 1987."""
    return load_eia(BRENT_FILE, "brent")


def load_wti():
    """Cushing, Oklahoma WTI spot price FOB, USD per barrel, daily from 2 Jan 1986.

    Note: it is negative on 20 Apr 2020 (-36.98), so log returns do not work for
    WTI; later scripts use dollar_changes() for it.
    """
    return load_eia(WTI_FILE, "wti")


def f11_header(rows):
    """Find the AUD/USD column and the first data row in an F11 table.

    Input: the table as a list of rows (lists of cells), as read from a workbook
    or the CSV. The RBA puts a metadata block on top; its 'Series ID' row names
    each column, and data starts on the next row. The 'Source' row says who
    supplied the rate (RBA, or WM/Reuters from 2007 on).
    Output: (column index, first data row index, source label).
    """
    labels = {str(r[0]).strip(): r for r in rows[:20] if r}
    ids = labels["Series ID"]
    col = [str(c).strip() for c in ids].index(AUDUSD_SERIES_ID)
    first = next(i for i, r in enumerate(rows) if r and str(r[0]).strip() == "Series ID") + 1
    return col, first, str(labels["Source"][col]).strip()


def load_audusd(with_source=False):
    """AUD/USD (US dollars per Australian dollar), daily from 12 Dec 1983.

    Joins the ten F11 history workbooks (1983-2022) and the current CSV (2023 on)
    into one series. Each part must start after the previous one ends; any
    overlap stops the loader rather than silently keeping one of two values.

    with_source=True returns a DataFrame with columns 'audusd', 'source' (the
    file's Source row: 'RBA' before 2007, 'WM/Reuters' after) and 'file', which
    M1 uses to check the join and the change of data supplier.
    """
    parts = []
    for file_name in F11_XLS_FILES:
        sheet = xlrd.open_workbook(RAW_DIR / file_name).sheet_by_name("Data")
        rows = [sheet.row_values(i) for i in range(sheet.nrows)]
        col, first, source = f11_header(rows)
        data = [(excel_date(r[0]), to_number(r[col])) for r in rows[first:] if str(r[0]).strip()]
        parts.append(pd.DataFrame(data, columns=["date", "audusd"]).assign(source=source, file=file_name))

    # The CSV's first line has one field and later lines have many, so it is read
    # line by line with the csv module rather than pandas.
    with open(RAW_DIR / F11_CSV_FILE, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    col, first, source = f11_header(rows)
    data = [(pd.to_datetime(r[0], format="%d-%b-%Y"), to_number(r[col])) for r in rows[first:] if r and r[0].strip()]
    parts.append(pd.DataFrame(data, columns=["date", "audusd"]).assign(source=source, file=F11_CSV_FILE))

    for before, after in zip(parts, parts[1:]):
        if after["date"].min() <= before["date"].max():
            raise ValueError(f"F11 files overlap: {before['file'].iloc[0]} and {after['file'].iloc[0]}")
    table = pd.concat(parts).set_index("date")
    table.index = pd.DatetimeIndex(table.index)
    check_dates(table["audusd"], "audusd")
    return table if with_source else table["audusd"]


EVAL_START = pd.Timestamp("2007-01-01")  # margin models are scored from here (M1: Brent is sparse before 2007)

# Dated (spot) Brent came apart from Brent futures in spring 2020 as storage filled: -23% on 31 Mar, +35% on
# 2 Apr, $9.12 on 21 Apr, then a doubling in May. Futures moved far less. Brent results that touch these
# dates are shown separately, not as headlines.
SPOT_EVENT = (pd.Timestamp("2020-03-30"), pd.Timestamp("2020-05-29"))


def max_span_days(h):
    """Longest calendar span an h-business-day move may cover: the h business days
    stretched to calendar days (h x 7/5, whole days) plus 4 for a weekend and public
    holidays. A longer span means prices are missing in between (Brent has many such
    holes in 1993-2006), so the move is not really an h-day move. At h = 5 this allows
    11 days: a week with a holiday and two weekends, but not a two-week hole."""
    return int(h * 7 / 5) + 4


def keep_short_spans(moves, prices, h):
    """Drop moves whose calendar span is longer than max_span_days(h)."""
    dates = prices.index.to_series()
    span = (dates - dates.shift(h)).dt.days.reindex(moves.index)
    return moves[span <= max_span_days(h)]


def log_returns(prices, h):
    """Log return over h observed prices: ln(p_t / p_(t-h)), dated t.

    Missing prices are dropped first, then any return spanning a hole in the data
    (keep_short_spans) is dropped too. Use for prices that are always positive
    (Brent, AUD/USD).
    """
    p = prices.dropna()
    if (p <= 0).any():
        raise ValueError("log_returns needs positive prices; use dollar_changes")
    return keep_short_spans(np.log(p / p.shift(h)).dropna(), p, h)


def dollar_changes(prices, h):
    """Price change in dollars over h observed prices, p_t - p_(t-h), dated t. For WTI,
    whose price went negative. Moves spanning a hole are dropped, as in log_returns."""
    p = prices.dropna()
    return keep_short_spans((p - p.shift(h)).dropna(), p, h)
