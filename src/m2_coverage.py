"""M2: do margin rules cover the moves they claim to cover, and what do they cost?

A margin is a cash deposit meant to cover the loss on a position over the next h
business days with 99% confidence. Four rules set it, each using only prices up
to and including the day t it is set:

    R1 fixed     - one percentage, set once from the years before 2007, then held
    R2 EWMA      - 2.326 x EWMA volatility x sqrt(h); volatility reacts to recent moves
    R3 FHS       - filtered historical simulation: today's EWMA volatility times the
                   1% / 99% quantiles of past moves divided by their own volatility
                   (last 500 days), times sqrt(h)
    R4 floored   - the larger of R2 and a plain 10-year historical 99% margin; the
                   10-year floor is a standard clearing-house tool against margins
                   that collapse in calm times and then jump in a crisis

Everything is in log returns. Each side is tested on its own: a long position
(a buyer, loses when the price falls) breaches when the h-day move is below
-margin; a short (a seller) breaches when it is above +margin. A good 99% rule
breaches about 1% of the time on each side, and not in clusters.

Writes outputs/m2/ (coverage_report.md, CSV tables, one chart).
Run from the repo folder:  python src/m2_coverage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

OUT_DIR = data.PROJECT_DIR / "outputs" / "m2"

CONFIDENCE = 0.99
LAMBDA = 0.94                        # EWMA decay (RiskMetrics daily value)
FHS_WINDOW = 500                     # business days of standardised moves for R3
FLOOR_WINDOW = 2520                  # about 10 years of business days for R4's floor
EWMA_SEED = 100                      # first returns used to start the EWMA variance
HORIZONS = [1, 2, 5]                 # business days; 2 is the headline (futures margin period)
MAIN_H = 2
RISE_WINDOW = 21                     # about one month of business days, for the procyclicality measure
RULES = ["R1 fixed", "R2 EWMA", "R3 FHS", "R4 floored"]
SERIES = {"Brent": data.load_brent, "AUD/USD": data.load_audusd}


def ewma_vol(daily, lam=LAMBDA):
    """EWMA volatility of daily moves, known at the close of each day.

    sigma2_t = lam x sigma2_(t-1) + (1 - lam) x r_t^2, so the value dated t
    already includes day t's move (it is set after the close). Started from the
    variance of the first EWMA_SEED returns, which lie long before 2007.
    """
    r = daily.to_numpy()
    var = np.empty(len(r))
    var[0] = np.var(r[:EWMA_SEED])
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i] ** 2
    return pd.Series(np.sqrt(var), index=daily.index)


def build_panel(prices, h, lam=LAMBDA, fhs_window=FHS_WINDOW, confidence=CONFIDENCE, kind="log"):
    """One row per possible margin date t: the h-day move that follows, and each rule's margin.

    Inputs: prices (one series from data.py), h (business days); M4 changes the settings lam,
    fhs_window and confidence, and uses kind="dollar" for WTI (its price went negative).
    Columns: 'move' = ln(p_(t+h) / p_t) (or p_(t+h) - p_t), dated by its START date t, and the margin
    of each rule for a long ('<rule> long') and a short ('<rule> short'), in the same units. Moves spanning a hole in the data are dropped (data.keep_short_spans).
    No look-ahead: every margin on row t uses returns dated t or earlier.
    """
    moves_over = data.log_returns if kind == "log" else data.dollar_changes
    z_conf = stats.norm.ppf(confidence)
    p = prices.dropna()
    daily = moves_over(p, 1)
    sigma = ewma_vol(daily, lam)

    # The move that follows each start date t (keep_short_spans works on end dates, then shift back).
    end_move = moves_over(p, h)
    start_of = pd.Series(p.index, index=p.index).shift(h)
    move = pd.Series(end_move.to_numpy(), index=pd.DatetimeIndex(start_of.reindex(end_move.index).to_numpy()))
    # One row per day with a volatility; 'move' is NaN where no clean h-day move follows (the last h days,
    # or a hole). Those rows still have margins, which M3 needs; the coverage tests skip them.
    panel = sigma.rename("sigma").to_frame().join(move.rename("move"))

    # R1: fixed, from all h-day moves that end before EVAL_START (set once, never updated).
    calib = end_move[end_move.index < data.EVAL_START]
    panel["R1 fixed long"] = -np.quantile(calib, 1 - confidence)
    panel["R1 fixed short"] = np.quantile(calib, confidence)

    # R2: parametric EWMA, the same on both sides.
    panel["R2 EWMA long"] = panel["R2 EWMA short"] = z_conf * panel["sigma"] * np.sqrt(h)

    # R3: filtered historical simulation. z_s = r_s / sigma_(s-1) is each day's move in units of the
    # volatility known the day before; its tail quantiles over the last fhs_window days, scaled by
    # today's volatility and sqrt(h), give the margin.
    z = daily / sigma.shift(1)
    lo = z.rolling(fhs_window).quantile(1 - confidence)
    hi = z.rolling(fhs_window).quantile(confidence)
    panel["R3 FHS long"] = -(lo * sigma * np.sqrt(h)).reindex(panel.index)
    panel["R3 FHS short"] = (hi * sigma * np.sqrt(h)).reindex(panel.index)

    # R4: R2 with a floor = the plain historical 1% / 99% quantile of h-day moves ending in the last
    # FLOOR_WINDOW days up to t (moves ending after t are not yet known, so they are excluded).
    hist_lo = end_move.rolling(FLOOR_WINDOW, min_periods=FLOOR_WINDOW // 2).quantile(1 - confidence)
    hist_hi = end_move.rolling(FLOOR_WINDOW, min_periods=FLOOR_WINDOW // 2).quantile(confidence)
    # The floor known on day t is the latest one computed up to t: after a long holiday no clean h-day move
    # ends on t, so the value is carried forward rather than left missing (which would drop the whole day).
    panel["R4 floored long"] = np.maximum(panel["R2 EWMA long"], -hist_lo.reindex(panel.index, method="ffill"))
    panel["R4 floored short"] = np.maximum(panel["R2 EWMA short"], hist_hi.reindex(panel.index, method="ffill"))
    rules = [c for c in panel.columns if c.startswith("R")]
    return panel[panel.index >= data.EVAL_START].dropna(subset=rules)


def non_overlapping(panel, h, offset=0):
    """Every h-th row, so consecutive h-day moves do not share days (the coverage tests assume
    independent trials). offset picks which of the h possible starting rows is used; the result
    can change with it, so the coverage table reports every offset."""
    return panel.iloc[offset::h]


def kupiec_lr(breaches, confidence=CONFIDENCE):
    """Kupiec likelihood ratio: LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x]."""
    b = np.asarray(breaches, dtype=bool)
    n, x, p = len(b), int(b.sum()), 1 - confidence
    rate = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - rate) + x * np.log(rate) if 0 < x < n else 0.0  # log(1) = 0 at x = 0 or n
    return -2 * (ll0 - ll1)


def kupiec(breaches, confidence=CONFIDENCE):
    """Kupiec proportion-of-failures test: is the breach rate equal to 1 - CONFIDENCE?

    The LR is chi-square with 1 degree of freedom under the null. Returns (n, breaches, rate, p-value).
    """
    b = np.asarray(breaches, dtype=bool)
    return len(b), int(b.sum()), b.mean(), float(stats.chi2.sf(kupiec_lr(b, confidence), 1))


def christoffersen(breaches):
    """Christoffersen independence test: does a breach make the next breach more likely?

    Counts transitions n_ij (state i yesterday, j today; 1 = breach) and compares a model where
    the breach probability depends on yesterday's state with one where it does not. LR is
    chi-square with 1 degree of freedom. Returns the p-value, or NaN when there are too few
    breaches to tell (no transitions out of a breach).
    """
    lr = independence_lr(breaches)
    return np.nan if np.isnan(lr) else float(stats.chi2.sf(lr, 1))


def conditional_coverage(breaches, confidence=CONFIDENCE):
    """Christoffersen's conditional-coverage test: the right breach rate AND no clustering, jointly.

    LR = Kupiec LR + independence LR, chi-square with 2 degrees of freedom. When the independence
    part cannot be computed (no breach is followed by another observation), it counts as 0.
    """
    ind = independence_lr(breaches)
    return float(stats.chi2.sf(kupiec_lr(breaches, confidence) + (0.0 if np.isnan(ind) else ind), 2))


def independence_lr(breaches):
    """The likelihood ratio behind christoffersen(), or NaN when it cannot be computed."""
    b = np.asarray(breaches, dtype=int)
    prev, cur = b[:-1], b[1:]
    n00, n01 = int(((prev == 0) & (cur == 0)).sum()), int(((prev == 0) & (cur == 1)).sum())
    n10, n11 = int(((prev == 1) & (cur == 0)).sum()), int(((prev == 1) & (cur == 1)).sum())
    if n10 + n11 == 0 or n01 == 0:
        return np.nan

    def ll(p, a, c):  # log-likelihood of a no-breach and c breach transitions at probability p
        return (a * np.log(1 - p) if a else 0.0) + (c * np.log(p) if c and p > 0 else 0.0)

    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    return -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi01, n00, n01) - ll(pi11, n10, n11))


def coverage_table(panels, confidence=CONFIDENCE):
    """Breach rate and tests for every series, horizon, rule and side, on non-overlapping moves.

    'rate % all days' is the breach rate over every day's h-day move (overlapping): the best estimate of the
    rate, and the same whichever day a sample starts on. The tests need independent moves, so they run on
    every h-th day: n, breaches, rate % and the p-values are for the sample starting on the first row
    (offset 0); 'Kupiec fails', 'CC passes' and 'clustered' count, over all h possible start days, how many
    give Kupiec p < 0.05, conditional-coverage p >= 0.05 and Christoffersen p < 0.05.
    """
    rows = []
    for (name, h), panel in panels.items():
        moves = panel.dropna(subset=["move"])
        for rule in RULES:
            for side, sign in (("long", -1), ("short", 1)):
                def breaches(offset):  # long: move < -margin; short: move > margin
                    s = non_overlapping(moves, h, offset)
                    return (sign * s["move"]) > s[f"{rule} {side}"]
                b = breaches(0)
                n, x, rate, p_k = kupiec(b, confidence)
                all_days = 100 * ((sign * moves["move"]) > moves[f"{rule} {side}"]).mean()
                k_fails = sum(kupiec(breaches(o), confidence)[3] < 0.05 for o in range(h))
                passes = sum(conditional_coverage(breaches(o), confidence) >= 0.05 for o in range(h))
                clustered = sum(christoffersen(breaches(o)) < 0.05 for o in range(h))  # NaN counts as not clustered
                rows.append({"series": name, "h": h, "rule": rule, "side": side, "rate % all days": all_days,
                             "n": n, "breaches": x, "rate %": 100 * rate, "Kupiec fails": f"{k_fails}/{h}", "Kupiec p": p_k, "Christoffersen p": christoffersen(b),
                             "CC p": conditional_coverage(b, confidence), "CC passes": f"{passes}/{h}",
                             "clustered": f"{clustered}/{h}"})
    return pd.DataFrame(rows)


def outside_spot_event(index):
    """True for dates outside data.SPOT_EVENT."""
    a, b = data.SPOT_EVENT
    return (index < a) | (index > b)


def without_spot_moves(panel):
    """The same panel with every move that could touch data.SPOT_EVENT set to missing, so the coverage
    tests skip it: moves starting from 10 calendar days before the event (a 5-day move can reach into it)
    to its last day. The margins themselves are unchanged."""
    a, b = data.SPOT_EVENT
    out = panel.copy()
    out.loc[(out.index >= a - pd.Timedelta(days=10)) & (out.index <= b), "move"] = np.nan
    return out


def cost_table(panels):
    """What each rule asks for, on every business day (not only the test days), at MAIN_H.

    average margin (% of notional, the larger side each day), peak / average, and the largest
    rise over RISE_WINDOW business days (peak margin in the window divided by the margin at its
    start): how big a margin call a hedger can face within about a month. For R4, 'floor sets %' is the
    share of days on which the 10-year floor, not EWMA, sets the margin on at least one side. Brent is shown twice:
    all days, and with the spring 2020 spot dislocation (data.SPOT_EVENT) left out of the peak
    and of every rise window that touches it. The fixed rule never rises, so it has no rise date.
    """
    rows = []
    for (name, h), panel in panels.items():
        if h != MAIN_H:
            continue
        variants = [(name, None)] + ([(f"{name} ex spring 2020", outside_spot_event(panel.index))]
                                     if name == "Brent" else [])
        for label, keep in variants:
            for rule in RULES:
                m = np.maximum(panel[f"{rule} long"], panel[f"{rule} short"])
                rise = m.rolling(RISE_WINDOW).max() / m.shift(RISE_WINDOW - 1)
                peak = m
                if keep is not None:
                    clean_window = pd.Series(keep, index=m.index).rolling(RISE_WINDOW).min() == 1
                    rise, peak = rise[clean_window], m[keep]
                fixed = rule == "R1 fixed"
                floor_sets = ((panel["R4 floored long"] > panel["R2 EWMA long"])
                              | (panel["R4 floored short"] > panel["R2 EWMA short"])).mean()
                rows.append({"series": label, "rule": rule, "average margin %": 100 * m.mean(),
                             "peak margin %": 100 * peak.max(), "peak / average": peak.max() / m.mean(),
                             "largest 1-month rise x": rise.max(),
                             "rise ends": "n/a" if fixed else f"{rise.idxmax():%b %Y}",
                             "floor sets %": f"{100 * floor_sets:.0f}" if rule == "R4 floored" else ""})
    return pd.DataFrame(rows)


def breaches_by_year(panels):
    """Breaches per year at MAIN_H, both sides added together, per series and rule."""
    out = []
    for (name, h), panel in panels.items():
        if h != MAIN_H:
            continue
        s = non_overlapping(panel.dropna(subset=["move"]), h)
        for rule in RULES:
            b = (-s["move"] > s[f"{rule} long"]) | (s["move"] > s[f"{rule} short"])
            out.append(b.groupby(s.index.year).sum().rename(f"{name} {rule}"))
    return pd.concat(out, axis=1).fillna(0).astype(int)


def plot_margins(panels, path):
    """Margin of each rule over time at MAIN_H (larger side), Brent and AUD/USD, with the moves."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, name in zip(axes, SERIES):
        panel = panels[(name, MAIN_H)]
        ax.plot(panel.index, 100 * panel["move"].abs(), color="#cccccc", linewidth=0.6, label="actual 2-day move (size)")
        for rule, color in zip(RULES, ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]):
            m = np.maximum(panel[f"{rule} long"], panel[f"{rule} short"])
            ax.plot(panel.index, 100 * m, color=color, linewidth=1.1, label=rule)
        ax.set_title(f"{name}: 2-day margin by rule, log % of position", loc="left", fontsize=10)
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fmt_p(p):
    """p-value as text: '<0.001', 3 decimals, or 'n/a' when the test cannot run."""
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md(frame, digits=2):
    """A DataFrame as a markdown table, floats rounded."""
    f = frame.copy()
    for c in f.columns:
        if "p" == c.split()[-1]:
            f[c] = [fmt_p(v) for v in f[c]]
        elif pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].round(digits)
    head = "| " + " | ".join(map(str, f.columns)) + " |\n|" + "---|" * len(f.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in f.itertuples(index=False))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = {name: load() for name, load in SERIES.items()}
    panels = {(name, h): build_panel(p, h) for name, p in prices.items() for h in HORIZONS}

    # Coverage also for Brent with the spring 2020 spot dislocation left out (data.SPOT_EVENT).
    ex_spot = {(f"{n} ex spring 2020", h): without_spot_moves(p) for (n, h), p in panels.items() if n == "Brent"}
    cov, cost, by_year = coverage_table({**panels, **ex_spot}), cost_table(panels), breaches_by_year(panels)
    cov.to_csv(OUT_DIR / "coverage.csv", index=False)
    cost.to_csv(OUT_DIR / "cost_and_procyclicality.csv", index=False)
    by_year.to_csv(OUT_DIR / "breaches_by_year.csv")
    plot_margins(panels, OUT_DIR / "margins_over_time.png")

    main_cov = cov[(cov["h"] == MAIN_H) & (cov["series"].isin(list(SERIES)))]
    ex_cov = cov[(cov["h"] == MAIN_H) & ~cov["series"].isin(list(SERIES))]
    always = main_cov[main_cov["CC passes"] == f"{MAIN_H}/{MAIN_H}"]
    never = main_cov[main_cov["CC passes"] == f"0/{MAIN_H}"]
    wrong_rate = never["Kupiec fails"] == f"{MAIN_H}/{MAIN_H}"
    under = never[wrong_rate & (never["rate % all days"] > 100 * (1 - CONFIDENCE))]
    over = never[wrong_rate & (never["rate % all days"] < 100 * (1 - CONFIDENCE))]
    clustered = never[never["clustered"] == f"{MAIN_H}/{MAIN_H}"]
    floor_sets = cost[cost["rule"] == "R4 floored"].set_index("series")["floor sets %"]
    n_by = main_cov.groupby("series", sort=False)["n"].first()
    worst = main_cov.sort_values("rate % all days", ascending=False).iloc[0]
    clean_cost = cost[cost["series"] != "Brent"]   # Brent rises measured outside the spring 2020 dislocation
    cheapest = clean_cost.sort_values("average margin %").groupby("series").head(1)
    jumpiest = clean_cost.sort_values("largest 1-month rise x", ascending=False).iloc[0]
    rise = clean_cost.set_index(["series", "rule"])["largest 1-month rise x"]
    n_test = int(n_by.min())

    def names(frame):
        return ", ".join(f"{r.series} {r.rule} {r.side}" for r in frame.itertuples()) or "none"

    answer = (
        f"At the 2-day horizon ({', '.join(f'{n:,} {s}' for s, n in n_by.items())} non-overlapping test periods "
        f"from 2007), {len(always)} of {len(main_cov)} rule-side combinations pass the conditional-coverage test "
        f"(right breach rate and no clustering) whichever day the sample starts on; {len(never)} fail on every start "
        f"and the rest depend on the start day. Of those that always fail, too many breaches (Kupiec fails on every start day): {names(under)}; too few, "
        f"so too much margin: {names(over)}; breaches clustered on both start days: {names(clustered)}. The highest breach rate is {worst['series']} {worst['rule']} "
        f"({worst['side']}), {worst['rate % all days']:.1f}% of all 2-day moves against a 1% target. Cost side: the jumpiest margin is "
        f"{jumpiest['series']} {jumpiest['rule']}, up {jumpiest['largest 1-month rise x']:.1f}x within a month "
        f"(to {jumpiest['rise ends']}). The 10-year floor (R4) damps that rise for AUD/USD "
        f"({rise['AUD/USD', 'R4 floored']:.1f}x vs {rise['AUD/USD', 'R2 EWMA']:.1f}x for plain EWMA) and for Brent "
        f"outside spring 2020 ({rise['Brent ex spring 2020', 'R4 floored']:.1f}x vs "
        f"{rise['Brent ex spring 2020', 'R2 EWMA']:.1f}x). R4 is mostly a 10-year historical margin: the floor, "
        f"not EWMA, sets it on {floor_sets['Brent']}% of Brent days and {floor_sets['AUD/USD']}% of AUD/USD days."
    )
    parts = [
        "# M2 - margin coverage report",
        f"Generated by `src/m2_coverage.py` on {pd.Timestamp.today():%d %b %Y}. Margins and moves in log terms, "
        f"99% target per side, models scored from {data.EVAL_START:%d %b %Y}.",
        "## Answer", answer,
        f"## 1. Coverage at {MAIN_H} days (non-overlapping moves)",
        "A long breaches when the move is below -margin, a short when it is above +margin. Kupiec p tests "
        "'breach rate = 1%'; Christoffersen p tests 'breaches do not cluster' (n/a = too few breaches to test); "
        "CC p tests both together. 'rate % all days' is the breach rate over every day's 2-day move: the best "
        "estimate, the same whatever the start day. The tests need independent moves, so they use every 2nd day: "
        "n, breaches, 'rate %' and the p-values are for the sample starting on the first day; 'Kupiec fails', "
        f"'CC passes' and 'clustered' say on how many of the {MAIN_H} possible start days Kupiec p is below 0.05, "
        "CC p is 0.05 or more, and Christoffersen p is below 0.05.",
        md(main_cov.drop(columns=["h"])),
        f"### Brent without the spring 2020 spot dislocation ({data.SPOT_EVENT[0]:%d %b} - {data.SPOT_EVENT[1]:%d %b %Y})",
        "Moves starting from 10 days before the window to its end are left out; margins are unchanged.",
        md(ex_cov.drop(columns=["h"])),
        "## 2. Cost and procyclicality (2 days, every business day)",
        "Average and peak margin in log % of the position (larger side). 'Largest 1-month rise' is the biggest "
        "jump in margin within 21 business days: the margin call a hedger faces when a crisis starts. "
        f"'Brent ex spring 2020' leaves {data.SPOT_EVENT[0]:%d %b} - {data.SPOT_EVENT[1]:%d %b %Y} (the spot "
        "dislocation) out of the peak and of every rise window touching it; its average uses all days.",
        md(cost),
        "Cheapest rule on average: " + "; ".join(f"{r.series} {r.rule} ({r._3:.1f}%)" for r in cheapest.itertuples()) + ".",
        "## 3. Breaches by year (2 days, both sides)",
        md(by_year.reset_index().rename(columns={"index": "year"}), 0),
        "![Margins over time](margins_over_time.png)",
        "## 4. Other horizons (1 and 5 days)",
        md(cov[(cov["h"] != MAIN_H) & cov["series"].isin(list(SERIES))]),
        "## Limitations",
        "- Spot prices stand in for futures. Dated Brent came apart from futures from 30 Mar to May 2020 "
        "(-23% on 31 Mar, +35% on 2 Apr, $9.12 on 21 Apr); the EWMA volatility carries that shock for weeks "
        "after, so Brent margins into mid-2020 are higher than a futures margin would have been.\n"
        "- 16 rule-side tests at 2 days and 32 more at 1 and 5 days, with no correction for multiple testing: "
        "a few 'fails' at p just under 0.05 are expected by chance.\n"
        "- No margin-period buffer, concentration add-on or liquidity add-on, which real clearing houses use; "
        "and margins here are recomputed every day.\n"
        "- R3 scales a 1-day distribution by sqrt(h), which ignores any momentum or mean reversion over 2-5 days.\n"
        f"- At 2 days there are about {n_test:,} test periods per series and side, so about {n_test // 100} breaches "
        "are expected: the tests cannot reliably tell 1% from 1.3%.\n"
        "- Margins are in log terms. A long margin of m covers a price fall of 1 - e^-m (10 log % = 9.5%); "
        "above about 70 log % (April 2020 Brent) the gap to a plain % is large.",
    ]
    (OUT_DIR / "coverage_report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(answer)


if __name__ == "__main__":
    main()
