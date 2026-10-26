"""M3: how much cash does a hedger need to keep a futures hedge open through a stress?

A hedger with a futures position pays two things in cash when the market moves
against them:
    variation margin - the daily loss on the position, paid every day
    a rise in initial margin - the clearing house asks for a bigger deposit when
                               volatility jumps (M2's rules)
Cash paid from the start s of a window to a day d = loss on the position from s to
d + (initial margin at d - initial margin at s). The cash need of a window of k
business days is the HIGHEST that running total reaches on any day inside it, not
its value on the last day: EWMA margin rises and then decays within a few weeks,
so the end of the window can hide the largest call. A hedger who cannot pay is
closed out, and loses the hedge when it is needed most.

Four hedgers, each with a position worth USD 100 million on the first day of the window:
    Brent long  - an airline buying oil forward: loses when oil falls
    Brent short - a producer selling oil forward: loses when oil rises
    AUD/USD long  - an exporter selling USD for AUD forward: loses when AUD falls
    AUD/USD short - an importer buying USD with AUD forward: loses when AUD rises
Everything scales linearly, so USD 50 million needs half the cash.

Margins come from M2 at the 2-day horizon (log terms there); here they are turned
into dollars: a long margin of m on a position worth V is V x (1 - e^-m), a short
one V x (e^m - 1).

Writes outputs/m3/ (liquidity_report.md, CSV tables, one chart).
Run from the repo folder:  python src/m3_liquidity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import m2_coverage as m2  # noqa: E402

OUT_DIR = data.PROJECT_DIR / "outputs" / "m3"

NOTIONAL = 100e6                     # USD 100 million at the start of each window
WINDOWS = [1, 5, 20]                 # business days
SIDES = {"long": "loses when the price falls", "short": "loses when the price rises"}
HEDGERS = {("Brent", "long"): "airline", ("Brent", "short"): "producer",
           ("AUD/USD", "long"): "exporter", ("AUD/USD", "short"): "importer"}
EPISODES = {  # named stress periods; the worst 20-day window starting inside each is reported
    "2008 crash": ("2008-07-01", "2008-12-31"),
    "2014-16 oil glut": ("2014-06-01", "2016-02-29"),
    "2020 Covid": ("2020-02-15", "2020-04-30"),
    "2022 Ukraine": ("2022-02-15", "2022-06-30"),
    "Mar-Apr 2026 oil spike": ("2026-02-15", "2026-05-31"),   # Brent about $70 to $138 in six weeks
    "Sep-Oct 2026": ("2026-09-01", "2026-12-31"),
}
CHART = [  # (episode, series, side) drawn in the chart: the side that lost in each episode
    ("2008 crash", "Brent", "long"), ("2020 Covid", "Brent", "long"),
    ("2022 Ukraine", "Brent", "short"), ("Mar-Apr 2026 oil spike", "Brent", "short"),
]
REACTIVE = ["R2 EWMA", "R3 FHS", "R4 floored"]  # rules whose margin responds to volatility (R1 only to price)


def margin_dollars_per_unit(price, m, side):
    """Initial margin per unit of the asset, in dollars: price x (1 - e^-m) for a long, price x (e^m - 1)
    for a short. Converts M2's log margin into what the clearing house actually holds."""
    return price * (1 - np.exp(-m)) if side == "long" else price * (np.exp(m) - 1)


def hedger_cash_need(prices, margin, side, k):
    """Cash the hedger must pay over every window of k business days, per USD 100 million.

    Inputs: prices and margin (log, from M2) on the same dates; side 'long' or 'short'; k.
    The position is fixed in units at the start s: q = NOTIONAL / p_s (barrels, or Australian dollars).
    For each day d = s+1 ... s+k, cash paid so far = loss(s, d) + margin up(s, d), where
        loss      = q x (p_s - p_d) for a long, q x (p_d - p_s) for a short (variation margin paid)
        margin up = q x (margin per unit at d - margin per unit at s)
    Returns a DataFrame indexed by the start date s with columns, all in USD:
        'need'      the highest cash paid so far on any day of the window (negative: cash only came back)
        'loss', 'margin up'  the two parts on that peak day
        'rate up'   the part of 'margin up' from the clearing house raising the margin RATE (q x p_d x the
                    rise in margin per dollar); the rest is the same rate on a dearer (or cheaper) position
        'peak day'  the day it was reached; 'end' the last day of the window
    Windows whose calendar span is longer than data.max_span_days(k) (a hole in the data) are dropped.
    """
    p, im = prices.to_numpy(), margin_dollars_per_unit(prices, margin, side).to_numpy()
    rate = margin_dollars_per_unit(1.0, margin, side).to_numpy()      # margin per dollar of position
    n = len(p) - k
    q, sign = NOTIONAL / p[:n], 1 if side == "long" else -1
    # Row s, column j-1: the running totals after j days.
    loss = np.column_stack([q * sign * (p[:n] - p[j:n + j]) for j in range(1, k + 1)])
    up = np.column_stack([q * (im[j:n + j] - im[:n]) for j in range(1, k + 1)])
    rate_up = np.column_stack([q * p[j:n + j] * (rate[j:n + j] - rate[:n]) for j in range(1, k + 1)])
    best = (loss + up).argmax(axis=1)
    rows = np.arange(n)
    out = pd.DataFrame({"need": (loss + up)[rows, best], "loss": loss[rows, best], "margin up": up[rows, best], "rate up": rate_up[rows, best],
                        "peak day": prices.index[rows + best + 1], "end": prices.index[k:]},
                       index=prices.index[:n])
    span = (out["end"] - out.index).dt.days
    return out[span <= data.max_span_days(k)]


def worst_windows(need):
    """The single worst window (largest 'need') with its dates and the split into loss and margin rise."""
    s = need["need"].idxmax()
    r = need.loc[s]
    return {"need $m": r["need"] / 1e6, "loss $m": r["loss"] / 1e6, "margin up $m": r["margin up"] / 1e6,
            "of which rate up $m": r["rate up"] / 1e6,
            "start": f"{s:%d %b %Y}", "peak day": f"{r['peak day']:%d %b %Y}"}


def touches_spot_event(need):
    """True for windows that include any day of data.SPOT_EVENT (spring 2020 dated-Brent dislocation)."""
    a, b = data.SPOT_EVENT
    return (need.index <= b) & (need["end"] >= a)


def build(prices, settings=None):
    """Every cash-need table, plus what it leaves out.

    needs[(series, side, rule, k)] = hedger_cash_need(...), with Brent windows touching data.SPOT_EVENT removed;
    spot[(side, rule, k)] = the worst of those removed Brent windows;
    standing[(series, side, rule)] = average initial margin held on a USD 100 million position, every day from 2007.
    settings: optional keyword arguments for m2.build_panel (M4 changes lam, fhs_window, confidence).
    """
    needs, spot, standing = {}, {}, {}
    for name, p in prices.items():
        panel = m2.build_panel(p, m2.MAIN_H, **(settings or {}))
        px = p.reindex(panel.index)
        for side in SIDES:
            for rule in m2.RULES:
                standing[(name, side, rule)] = NOTIONAL * (margin_dollars_per_unit(px, panel[f"{rule} {side}"], side)
                                                           / px).mean()
                for k in WINDOWS:
                    t = hedger_cash_need(px, panel[f"{rule} {side}"], side, k)
                    if name == "Brent":
                        spot[(side, rule, k)] = worst_windows(t[touches_spot_event(t)])
                        t = t[~touches_spot_event(t)]
                    needs[(name, side, rule, k)] = t
    return needs, spot, standing


def worst_table(needs):
    """Worst window for every series, side, rule and window length."""
    rows = [{"series": n, "hedger": HEDGERS[(n, s)], "side": s, "days": k, "rule": r, **worst_windows(t)}
            for (n, s, r, k), t in needs.items()]
    return pd.DataFrame(rows)


def episode_table(needs):
    """Worst 20-day cash need starting inside each named episode, per series, side and rule ($ million)."""
    rows = []
    for ep, (a, b) in EPISODES.items():
        for (n, s, r, k), t in needs.items():
            part = t.loc[a:b]
            if k != 20 or part.empty:
                continue
            rows.append({"episode": ep, "series": n, "hedger": HEDGERS[(n, s)], "rule": r,
                         "need $m": part["need"].max() / 1e6})
    wide = pd.DataFrame(rows).pivot_table(index=["episode", "series", "hedger"], columns="rule",
                                          values="need $m", sort=False)
    return wide.reset_index()[["episode", "series", "hedger"] + m2.RULES]


def plot_episodes(prices, path):
    """Cumulative cash paid from the first day of each episode, Brent, the side that lost, per rule."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (ep, name, side) in zip(axes.flat, CHART):
        panel = m2.build_panel(prices[name], m2.MAIN_H)
        a, b = EPISODES[ep]
        part = panel.loc[a:b]
        if name == "Brent" and ep == "2020 Covid":
            part = part[part.index < data.SPOT_EVENT[0]]  # stop the 2020 panel before the spot dislocation
        px = prices[name].reindex(part.index)
        q, sign = NOTIONAL / px.iloc[0], 1 if side == "long" else -1
        loss = q * sign * (px.iloc[0] - px)
        ax.plot(part.index, loss / 1e6, color="#cccccc", linewidth=2, label="loss only (no margin rise)")
        for rule, color in zip(m2.RULES, ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]):
            im = q * margin_dollars_per_unit(px, part[f"{rule} {side}"], side)
            ax.plot(part.index, (loss + im - im.iloc[0]) / 1e6, color=color, linewidth=1.1, label=rule)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{ep}: Brent {HEDGERS[(name, side)]} ({side})", loc="left", fontsize=10)
        ax.set_ylabel("USD million paid")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Cash paid per USD 100 million hedged, from the first day of the episode", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = {name: load().dropna() for name, load in m2.SERIES.items()}
    needs, spot, standing = build(prices)
    worst, episodes = worst_table(needs), episode_table(needs)
    worst.to_csv(OUT_DIR / "worst_windows.csv", index=False)
    episodes.to_csv(OUT_DIR / "episodes_20d.csv", index=False)
    plot_episodes(prices, OUT_DIR / "episodes.png")

    # Headline: the worst 20 days under the reactive rules (R1's fixed % is not a realistic margin).
    reactive = worst[worst["rule"].isin(REACTIVE)]
    w20 = worst[worst["days"] == 20]
    top = reactive[reactive["days"] == 20].sort_values("need $m", ascending=False).iloc[0]
    share = (reactive["margin up $m"] / reactive["need $m"]).groupby(reactive["days"]).median()
    gap = episodes["R4 floored"] - episodes["R2 EWMA"]           # USD million, negative = R4 needs less cash
    extra = {(n, s): (standing[(n, s, "R4 floored")] - standing[(n, s, "R2 EWMA")]) / 1e6 for n, s in HEDGERS}
    extra_held = np.mean(list(extra.values()))
    extra_by = {n: np.mean([v for (m, _), v in extra.items() if m == n]) for n in m2.SERIES}
    spot_top = max(spot.items(), key=lambda kv: kv[1]["need $m"] if kv[0][2] == 20 else -np.inf)
    answer = (
        f"Under the volatility-based rules (R2-R4), the worst 20 business days for any hedger cost the "
        f"{top['series']} {top['hedger']} ({top['side']}) USD {top['need $m']:.0f} million per USD 100 million "
        f"hedged at the peak ({top['start']} - {top['peak day']}, under {top['rule']}), of which USD "
        f"{top['margin up $m']:.0f} million is higher initial margin (USD {top['of which rate up $m']:.0f} million "
        f"from a higher margin rate, the rest from the same rate on a dearer position). Across hedgers and these rules, "
        f"the margin rise is a median {100 * share[1]:.0f}% of the worst 1-day cash need, {100 * share[5]:.0f}% of "
        f"the worst 5-day one and {100 * share[20]:.0f}% of the worst 20-day one: over short windows the margin call "
        f"is a large share of the bill, over a month the loss on the position dominates. Against plain EWMA (R2), the "
        f"10-year floor (R4) changes the worst 20-day cash need in the named crises by USD {gap.min():+.1f} to "
        f"{gap.max():+.1f} million, small next to the crisis bills, but it ties up on average USD {extra_held:.1f} million more "
        f"collateral on every day of the sample (Brent hedgers {extra_by['Brent']:.1f}m, AUD/USD hedgers "
        f"{extra_by['AUD/USD']:.1f}m): cash posted, whose cost is the funding spread on it. Left out: Brent windows touching {data.SPOT_EVENT[0]:%d %b} - "
        f"{data.SPOT_EVENT[1]:%d %b %Y}, when spot Brent came apart from futures; there the Brent "
        f"{HEDGERS[('Brent', spot_top[0][0])]} would have needed USD {spot_top[1]['need $m']:.0f} million in 20 days, "
        f"a spot-price artefact."
    )

    def worst_md(k):
        t = worst[worst["days"] == k].copy()
        t["cash"] = [f"{v:.1f} ({s})" for v, s in zip(t["need $m"], t["start"])]
        wide = t.pivot_table(index=["series", "hedger"], columns="rule", values="cash", aggfunc="first", sort=False)
        return m2.md(wide.reset_index()[["series", "hedger"] + m2.RULES])

    split = w20[["series", "hedger", "rule", "need $m", "loss $m", "margin up $m", "of which rate up $m", "start",
                 "peak day"]]
    held = pd.DataFrame([{"series": n, "hedger": HEDGERS[(n, s)], **{r: standing[(n, s, r)] / 1e6 for r in m2.RULES}}
                         for n, s in HEDGERS])
    parts = [
        "# M3 - hedger liquidity report",
        f"Generated by `src/m3_liquidity.py` on {pd.Timestamp.today():%d %b %Y}. USD million of cash per USD 100 "
        f"million hedged, windows starting from {data.EVAL_START:%d %b %Y}, margins from M2 at {m2.MAIN_H} days. "
        "A window's cash need is the highest running total of cash paid on any day inside it.",
        "## Answer", answer,
        "## 1. Worst window per hedger (USD million, start date)",
        f"Brent windows touching {data.SPOT_EVENT[0]:%d %b} - {data.SPOT_EVENT[1]:%d %b %Y} (the spot dislocation, "
        "see `data.SPOT_EVENT`) are left out here and in sections 2-3.",
        *[f"### {k} business day{'s' if k > 1 else ''}\n\n{worst_md(k)}" for k in WINDOWS],
        "## 2. The worst 20 days split into loss and margin rise, on the peak day",
        "A negative 'margin up' means margin had fallen by the peak day, giving cash back. 'of which rate up' is the "
        "part due to a higher margin rate; the rest is the same rate on a position whose value moved. R1's rate never "
        "changes, so its margin moves only with the price.",
        m2.md(split, 1),
        "## 3. Named crises: worst 20-day cash need starting inside each (USD million)",
        m2.md(episodes, 1),
        "![Cash paid in four crises](episodes.png)",
        "## 4. Collateral held (USD million on a USD 100 million position, average of every day since 2007)",
        "What each rule ties up with the clearing house. It is a stock of cash, not a cost: its cost is the funding "
        "spread on it.",
        m2.md(held, 1),
        "## Limitations",
        "- Spot prices stand in for futures. The spring 2020 Brent dislocation is left out, not modelled, and the "
        "EWMA volatility still carries it into June 2020.\n"
        "- The position is fixed in barrels (or AUD) from the first day; a real hedger may cut the position.\n"
        "- Margin is the M2 rule, recomputed daily; real clearing houses change margin less often and add buffers.\n"
        "- Worst windows are picked by total cash need, so the shares by window length compare different dates; "
        "they say how big the margin part of the worst bill is, not when in a crisis it arrives.\n"
        "- The Brent data end on 6 Oct 2026, so 'Sep-Oct 2026' windows mostly lie in September; the jump of 2 Oct "
        "(+18%) and the fall of 5 Oct (-7.4%) show only in the 1-day and 5-day windows.",
    ]
    (OUT_DIR / "liquidity_report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(answer)


if __name__ == "__main__":
    main()
