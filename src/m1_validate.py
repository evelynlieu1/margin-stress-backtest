"""M1: load the three price series and check them before any margin model uses them.

Writes outputs/m1/validation_report.md. Each check prints PASS (as expected) or
FLAG (something to know before M2, not necessarily an error).

Run from the repo folder:  python src/m1_validate.py
"""

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

OUT_DIR = data.PROJECT_DIR / "outputs" / "m1"
REPORT = OUT_DIR / "validation_report.md"
MANIFEST = data.PROJECT_DIR / "data" / "MANIFEST.csv"

GAP_DAYS = 5          # a gap longer than this many calendar days between prices is listed
TOP_MOVES = 5         # how many of the largest one-day moves to list per series


def sha256(path, lf=False):
    """SHA-256 of a file. lf=True first turns Windows line endings (CRLF) into LF,
    because git on Windows can rewrite text files that way after download."""
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def check_checksums():
    """Every file in MANIFEST.csv against its recorded size and SHA-256."""
    with open(MANIFEST, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table, ok = [], True
    for r in rows:
        path = data.RAW_DIR / r["file"]
        if not path.exists():
            table.append([r["file"], "missing", ""])
            ok = False
            continue
        match = r["sha256"] in (sha256(path), sha256(path, lf=True))
        ok &= match
        table.append([r["file"], "yes" if match else "NO", f"{path.stat().st_size:,}"])
    return ok, table


def describe(series):
    """One summary row for a price series: rows, first and last date, missing, min and max with dates."""
    s = series.dropna()
    return [series.name, f"{len(series):,}", f"{series.index[0]:%d %b %Y}", f"{series.index[-1]:%d %b %Y}",
            int(series.isna().sum()), f"{s.min():.4g} ({s.idxmin():%d %b %Y})", f"{s.max():.4g} ({s.idxmax():%d %b %Y})"]


def gaps_by_year(series):
    """Per year: prices on file, and holes longer than GAP_DAYS calendar days.
    A normal year has about 250 prices and no holes."""
    gaps = series.index.to_series().diff().dt.days > GAP_DAYS
    table = pd.DataFrame({"prices": series.groupby(series.index.year).size(),
                          "holes": gaps.groupby(series.index.year).sum().astype(int)})
    return [[y, r.prices, r.holes] for y, r in table.iterrows() if r.holes > 0]


def largest_moves(series, kind):
    """The TOP_MOVES largest one-day moves, as log returns (in %) or dollar changes."""
    moves = data.log_returns(series, 1) * 100 if kind == "log" else data.dollar_changes(series, 1)
    top = moves.abs().sort_values(ascending=False).head(TOP_MOVES).index
    unit = "%" if kind == "log" else " USD"
    return [[f"{d:%d %b %Y}", f"{moves[d]:+.2f}{unit}"] for d in top]


def join_table(aud):
    """For each F11 file: its first and last date, and the jump from the previous file.

    The 'jump' is the log return from the last price of one file to the first of
    the next, compared with the typical (median absolute) daily move. A join that
    lines up correctly looks like an ordinary day.
    """
    rows, typical = [], data.log_returns(aud["audusd"], 1).abs().median() * 100
    prev = None
    for name, part in aud.groupby("file", sort=False):
        p = part["audusd"].dropna()
        jump = "" if prev is None else f"{np.log(p.iloc[0] / prev) * 100:+.2f}%"
        rows.append([name, f"{p.index[0]:%d %b %Y}", f"{p.index[-1]:%d %b %Y}", part["source"].iloc[0], jump])
        prev = p.iloc[-1]
    return rows, typical


def vol_by_year(aud, first=2003, last=2013):
    """Annualised volatility of daily AUD/USD log returns by year, around the change of supplier
    (RBA until 2006, 'RBA and WM/Reuters' 2007-2009, WM/Reuters from 2010)."""
    r = data.log_returns(aud["audusd"], 1)
    out = []
    for y in range(first, last + 1):
        part = r[str(y)]
        out.append([y, aud.loc[str(y), "source"].iloc[0], f"{part.std() * np.sqrt(252) * 100:.1f}%"])
    return out


def md(header, rows):
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    brent, wti = data.load_brent(), data.load_wti()
    aud = data.load_audusd(with_source=True)
    audusd = aud["audusd"]

    sums_ok, sums = check_checksums()
    joins, typical = join_table(aud)
    negative = wti[wti <= 0]
    gaps = {name: gaps_by_year(s) for name, s in [("brent", brent), ("wti", wti), ("audusd", audusd)]}
    missing_by_year = audusd[audusd.isna()].groupby(audusd[audusd.isna()].index.year).size()
    last5 = brent.tail(5)
    daily = data.log_returns(brent, 1) * 100
    end_big = daily.tail(3).abs().idxmax()                       # the largest move of the last three days
    end_rank = int((daily.abs() >= abs(daily[end_big])).sum())
    spikes = daily.tail(3)

    # Checks computed from the loaded data (the loaders also stop on repeated dates or overlapping files).
    dates_ok = all(s.index.is_monotonic_increasing and s.index.is_unique for s in (brent, wti, audusd))
    spans = aud.reset_index().groupby("file", sort=False)["date"].agg(["min", "max"])
    joins_ok = bool((spans["min"].iloc[1:].to_numpy() > spans["max"].iloc[:-1].to_numpy()).all())
    n_missing = int(audusd.isna().sum())
    apr_fall = brent["2020-04-20":"2020-04-21"]
    apr_log = 100 * np.log(apr_fall.iloc[1] / apr_fall.iloc[0])

    checks = [
        ("Checksums against MANIFEST.csv", "PASS" if sums_ok else "FLAG"),
        ("All three series load, dates unique and sorted", "PASS" if dates_ok else "FLAG"),
        ("AUD/USD files join without overlap", "PASS" if joins_ok else "FLAG"),
        (f"AUD/USD has {n_missing} missing values ('Closed' / 'N/A' text; 31 expected), kept as missing",
         "PASS" if n_missing == 31 else "FLAG"),
        ("WTI has non-positive prices (use dollar changes, not log returns)", "FLAG"),
        ("AUD/USD data supplier changes 2007-2010", "FLAG"),
        ("Brent is sparse before 2007: margin models scored from 2007", "FLAG"),
        (f"Brent spot falls {100 * (1 - np.exp(apr_log / 100)):.0f}% ({apr_log:.0f} log %) on 21 Apr 2020 "
         "(dated Brent, not futures)", "FLAG"),
    ]
    parts = [
        "# M1 - data validation report",
        f"Generated by `src/m1_validate.py` on {pd.Timestamp.today():%d %b %Y}.",
        "## Answer",
        f"All three series load cleanly and are ready for the margin models. Brent has {len(brent):,} daily "
        f"prices ({brent.index[0]:%b %Y} - {brent.index[-1]:%b %Y}) with none missing. AUD/USD has {len(audusd):,} "
        f"days from 11 files that join without overlap; its {int(audusd.isna().sum())} missing values are "
        "'Closed' or 'N/A' text on market holidays, kept as missing. Two things to carry into M2: WTI goes "
        f"negative on {len(negative)} day(s), so WTI uses dollar changes, and the AUD/USD supplier changes from "
        "the RBA to WM/Reuters in 2007-2010. Two decisions came out of M1: margin models are scored from 2007, "
        f"because Brent has holes before that, and the {100 * (1 - np.exp(apr_log / 100)):.0f}% fall in spot Brent on 21 Apr "
        "2020 is kept but labelled (M2 adds coverage without spring 2020; M3 leaves it out of its tables).",
        "## Summary",
        md(["check", "result"], checks),
        "## 1. Checksums",
        md(["file", "matches MANIFEST", "bytes on disk"], sums),
        "## 2. Coverage",
        md(["series", "rows", "first", "last", "missing", "min (date)", "max (date)"],
           [describe(brent), describe(wti), describe(audusd)]),
        "## 3. Missing values in AUD/USD, by year",
        "The F11 workbooks write 'Closed' (or 'CLOSED') on market holidays and 'N/A' once; `data.to_number` "
        "turns these into missing values. Any other text would stop the loader.",
        md(["year", "missing"], [[y, n] for y, n in missing_by_year.items()]),
        "## 4. How the AUD/USD files join",
        f"Typical daily move (median absolute log return): {typical:.2f}%.",
        md(["file", "first", "last", "source", "jump from previous file"], joins),
        "## 5. Data supplier change",
        "The F11 'Source' row changes from RBA to 'RBA and WM/Reuters' (2007-2009) and WM/Reuters (2010 on). "
        "If the change altered how the rate is measured, volatility could shift in those years. It is reported "
        "here and tested in M4 by splitting the sample at 2010.",
        md(["year", "source", "annualised volatility"], vol_by_year(aud)),
        "## 6. Largest one-day moves",
        "### Brent (log return)", md(["date", "move"], largest_moves(brent, "log")),
        "### WTI (dollar change)", md(["date", "move"], largest_moves(wti, "dollar")),
        "### AUD/USD (log return)", md(["date", "move"], largest_moves(audusd, "log")),
        "## 7. WTI at or below zero",
        md(["date", "price (USD)"], [[f"{d:%d %b %Y}", v] for d, v in negative.items()]),
        "## 8. Holes in the data (gaps of more than 5 calendar days), by year",
        "Brent has only about 130-190 prices a year in 1994-1997 and holes through 2006. Decision: margin models "
        f"are scored from {data.EVAL_START:%d %b %Y}; earlier years only set the fixed-percentage rule. Any move "
        "spanning a hole is dropped (`data.keep_short_spans`), so a 1-day move is never really a 10-day one.",
        "\n\n".join(f"**{k}**\n\n" + ("none" if not v else md(["year", "prices", "holes"], v)) for k, v in gaps.items()),
        "## 9. Brent in April 2020: a spot-price event",
        md(["date", "Brent (USD)"], [[f"{d:%d %b %Y}", v] for d, v in brent["2020-04-17":"2020-04-24"].items()])
        + "\n\nEIA's Brent is the physical 'dated Brent' price, which fell to $9.12 on 21 Apr 2020 as storage ran "
        "out; Brent futures, on which margin is actually set, fell far less. Decision: kept in the main run, "
        "labelled as a spot-price event. Later steps widen this to 30 Mar - 29 May 2020 (`data.SPOT_EVENT`), when "
        "dated Brent and futures came apart: M2 also scores Brent without it, M3 leaves it out of its tables, and M4 "
        "re-runs everything without 2020.",
        "## 10. The end of the Brent sample",
        md(["date", "Brent (USD)"], [[f"{d:%d %b %Y}", v] for d, v in last5.items()])
        + "\n\nLast three daily moves (log %): " + ", ".join(f"{d:%d %b} {v:+.1f}" for d, v in spikes.items())
        + f". The largest, {daily[end_big]:+.1f}% on {end_big:%d %b %Y}, ranks {end_rank} of all one-day moves "
        "since 1987. "
        "It sits at the very end of the data, so no later prices can show how margins handled what came next.",
    ]
    REPORT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    for name, result in checks:
        print(f"[{result}] {name}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
