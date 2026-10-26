"""M4: which M2 and M3 results survive when one setting changes?

Each variant changes one thing and reruns the margin rules (M2) and the hedger
cash needs (M3). Six claims from the M2 and M3 answers are then re-checked:

    C1  R2 (EWMA) under-covers oil falls: long breach rate (all days) above target, Kupiec p < 0.05 on every
        sample start day
    C2  R4 (floored) over-covers AUD/USD: breach rate (all days) below target on both sides, Kupiec p < 0.05
        on every sample start day
    C3  R1 (fixed %) breaches cluster: Christoffersen p < 0.05 on BOTH sample start days, on at least one
        side, in every series
    C4  R4 damps the largest 1-month margin rise against R2, every series (Brent outside spring 2020)
    C5  the margin rise is a bigger share of the worst 1-day cash need than of the worst 20-day one
    C6  R4 changes crisis cash needs by less than USD 5 million against R2 but ties up more collateral on average

Variants: the main settings; EWMA lambda 0.97; a 250-day FHS window; 99.5% confidence; 2020
left out of scoring; AUD/USD scored on 2007-2009 and on 2010 on (the change of data supplier);
WTI in dollar changes (its price went negative in April 2020), for C1, C3 and C4 only.

Writes outputs/m4/robustness_report.md and robustness.csv.
Run from the repo folder:  python src/m4_robustness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import m2_coverage as m2  # noqa: E402
import m3_liquidity as m3  # noqa: E402

OUT_DIR = data.PROJECT_DIR / "outputs" / "m4"
H = m2.MAIN_H
CRISIS_GAP = 5.0   # USD million: C6's bound on |R4 - R2| crisis cash need

VARIANTS = [  # (name, margin settings, scoring window or None, which claims apply)
    ("main", {}, None, "all"),
    ("lambda 0.97", {"lam": 0.97}, None, "all"),
    ("FHS window 250", {"fhs_window": 250}, None, "all"),
    ("99.5% confidence", {"confidence": 0.995}, None, "all"),
    ("without 2020", {}, "no 2020", "all"),
    ("AUD/USD 2007-2009", {}, ("2007-01-01", "2009-12-31"), "AUD"),
    ("AUD/USD 2010 on", {}, ("2010-01-01", "2100-01-01"), "AUD"),
    ("WTI (dollar changes)", {"kind": "dollar"}, None, "WTI"),
]


def scored(panel, window):
    """The rows a variant scores: all, all but 2020, or one date range. Margins still use all earlier data."""
    if window is None:
        return panel
    if window == "no 2020":
        return panel[panel.index.year != 2020]
    return panel.loc[window[0]:window[1]]


def coverage(panel, confidence, window):
    """M2's coverage table at H days for one series, on the variant's scored rows."""
    return m2.coverage_table({("x", H): scored(panel, window)}, confidence).set_index(["rule", "side"])


def largest_rise(panel, rule, skip_spot):
    """M2's 'largest 1-month rise' of the larger-side margin; skip_spot leaves out windows touching data.SPOT_EVENT."""
    m = np.maximum(panel[f"{rule} long"], panel[f"{rule} short"])
    rise = m.rolling(m2.RISE_WINDOW).max() / m.shift(m2.RISE_WINDOW - 1)
    if skip_spot:
        keep = pd.Series(m2.outside_spot_event(m.index), index=m.index).rolling(m2.RISE_WINDOW).min() == 1
        rise = rise[keep]
    return rise.max()


def m2_claims(panels, settings, window, series):
    """C1-C4 and the CC pass count for one variant. panels: {name: build_panel output}."""
    conf = settings.get("confidence", m2.CONFIDENCE)
    target = 100 * (1 - conf)
    cov = {n: coverage(p, conf, window) for n, p in panels.items()}
    out = {"CC passes (all starts)": sum(int((c["CC passes"] == f"{H}/{H}").sum()) for c in cov.values()),
           "rule-sides": sum(len(c) for c in cov.values())}
    oil = [n for n in series if n != "AUD/USD"]
    if oil:
        r = cov[oil[0]].loc[("R2 EWMA", "long")]
        out["C1"] = bool(r["rate % all days"] > target and r["Kupiec fails"] == f"{H}/{H}")
        out["C1 detail"] = f"{oil[0]} R2 long {r['rate % all days']:.2f}% (Kupiec fails on {r['Kupiec fails']} starts)"
    if "AUD/USD" in series:
        r4 = cov["AUD/USD"].loc["R4 floored"]
        out["C2"] = bool(((r4["rate % all days"] < target) & (r4["Kupiec fails"] == f"{H}/{H}")).all())
        out["C2 detail"] = "AUD/USD R4 " + " / ".join(f"{v:.2f}% (fails {k})" for v, k in
                                                      zip(r4["rate % all days"], r4["Kupiec fails"]))
    out["C3"] = all((c.loc["R1 fixed", "clustered"] == f"{H}/{H}").any() for c in cov.values())
    out["C3 detail"] = "; ".join(f"{n} R1 clustered on " + " / ".join(c.loc["R1 fixed", "clustered"])
                                 + " start days (long / short)" for n, c in cov.items())
    if window is None:   # the rise is a property of the margin path, so only full-sample variants re-measure it
        rises = {n: (largest_rise(p, "R4 floored", n != "AUD/USD"), largest_rise(p, "R2 EWMA", n != "AUD/USD"))
                 for n, p in panels.items()}
        out["C4"] = all(r4 < r2 for r4, r2 in rises.values())
        out["C4 detail"] = "; ".join(f"{n} {r4:.1f}x vs {r2:.1f}x" for n, (r4, r2) in rises.items())
    return out


def m3_claims(prices, settings, window):
    """C5 and C6 for one variant, from M3's tables; 'no 2020' drops windows that touch 2020."""
    needs, _, standing = m3.build(prices, settings)
    if window == "no 2020":
        needs = {k: t[(t.index.year != 2020) & (t["end"].dt.year != 2020)] for k, t in needs.items()}
    worst = m3.worst_table(needs)
    reactive = worst[worst["rule"].isin(m3.REACTIVE)]
    share = (reactive["margin up $m"] / reactive["need $m"]).groupby(reactive["days"]).median()
    gap = (lambda e: e["R4 floored"] - e["R2 EWMA"])(m3.episode_table(needs))
    held = np.mean([standing[(n, s, "R4 floored")] - standing[(n, s, "R2 EWMA")] for n, s in m3.HEDGERS]) / 1e6
    return {"C5": bool(share[1] > share[20]), "C5 detail": f"{100 * share[1]:.0f}% / {100 * share[5]:.0f}% / "
            f"{100 * share[20]:.0f}% (1 / 5 / 20 days)",
            "C6": bool(gap.abs().max() < CRISIS_GAP and held > 0),
            "C6 detail": f"R4 - R2 crisis need {gap.min():+.1f} to {gap.max():+.1f} $m; R4 ties up {held:+.1f} $m more"}


def run_variant(name, settings, window, scope, prices):
    """All claims that apply to one variant, as one row."""
    if scope == "WTI":
        series = {"WTI": data.load_wti().dropna()}
    elif scope == "AUD":
        series = {"AUD/USD": prices["AUD/USD"]}
    else:
        series = prices
    panels = {n: m2.build_panel(p, H, **settings) for n, p in series.items()}
    row = {"variant": name, **m2_claims(panels, settings, window, list(series))}
    if scope == "all":
        row.update(m3_claims(prices, settings, window))
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = {name: load().dropna() for name, load in m2.SERIES.items()}
    rows = pd.DataFrame([run_variant(*v, prices) for v in VARIANTS])
    rows.to_csv(OUT_DIR / "robustness.csv", index=False)

    claims = ["C1", "C2", "C3", "C4", "C5", "C6"]
    verdict = rows.set_index("variant")[claims].apply(lambda c: c.map({True: "yes", False: "NO"}).fillna("n/a"))
    held = {c: [v for v, ok in verdict[c].items() if ok == "NO"] for c in claims}
    survives = [c for c in claims if not held[c]]
    answer = (
        f"Claims that hold in every variant where they apply: {', '.join(survives) or 'none'}. "
        + " ".join(f"{c} fails under: {', '.join(v)}." for c, v in held.items() if v)
    )
    detail = rows.set_index("variant")[[f"{c} detail" for c in claims]].fillna("")
    detail.columns = claims
    passes = rows[["variant", "CC passes (all starts)", "rule-sides"]]

    text = {
        "C1": "R2 (EWMA) under-covers oil falls (Brent; WTI in its own row)",
        "C2": "R4 (floored) over-covers AUD/USD on both sides",
        "C3": "R1 (fixed %) breaches cluster in every series, on both sample start days",
        "C4": "R4 damps the largest 1-month margin rise against R2",
        "C5": "Margin rises are a bigger share of the worst 1-day cash need than of the worst 20-day one",
        "C6": f"R4 changes crisis cash needs by under USD {CRISIS_GAP:.0f}m (a bound chosen here) but ties up more "
              "collateral on average",
    }
    parts = [
        "# M4 - robustness report",
        f"Generated by `src/m4_robustness.py` on {pd.Timestamp.today():%d %b %Y}. Each variant changes one setting "
        f"and reruns M2 (coverage at {H} days, non-overlapping) and M3 (hedger cash needs). 'yes' = the claim holds, "
        "'NO' = it fails, 'n/a' = the variant cannot test it.",
        "## Answer", answer,
        "## 1. The claims",
        m2.md(pd.DataFrame({"claim": list(text), "statement": list(text.values())})),
        "## 2. What survives",
        m2.md(verdict.reset_index()),
        "## 3. The numbers behind each verdict",
        m2.md(detail.reset_index()),
        "## 4. Coverage passes (conditional coverage, every sample start day)",
        m2.md(passes),
        "## Limitations",
        "- Variants change one setting at a time; combinations are not tried.\n"
        "- 'without 2020' leaves 2020 out of scoring only; the margins still learn from 2020 data.\n"
        "- The AUD/USD sub-periods have about 380 and 2,100 test periods: C2 has little power in 2007-2009.\n"
        "- WTI's R1 is a fixed dollar amount from before 2007, when oil was much cheaper, so it under-covers by design.",
    ]
    (OUT_DIR / "robustness_report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(answer)
    print(verdict.to_string())


if __name__ == "__main__":
    main()
