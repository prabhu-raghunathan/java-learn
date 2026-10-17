"""Every figure on the Matplotlib page. Run from this folder:  python figures_matplotlib.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

import site_style
from shopstream import make_shopstream
from clean import clean_orders

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "static", "img", "data-exploration")
site_style.apply()
raw, customers = make_shopstream()
orders = clean_orders(raw, customers)
enriched = orders.merge(customers, on="customer_id", how="left", validate="m:1")
rupees_m = FuncFormatter(lambda v, pos: f"₹{v / 1e6:.1f}M")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


# ---- 1. anatomy -------------------------------------------------------------------
def anatomy():
    weekly = orders.set_index("order_ts")["revenue"].resample("W").sum().iloc[1:-1]   # drop partial weeks
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.10, right=0.70, top=0.88, bottom=0.14)                # room for callouts
    ax.plot(weekly.index, weekly.to_numpy(), label="Weekly revenue")
    ax.plot(weekly.index, weekly.rolling(4).mean().to_numpy(), linestyle="--", label="4-week average")
    ax.set_title("ShopStream weekly revenue")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(rupees_m)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.legend(loc="lower right", fontsize=8)
    fig.patches.append(plt.Rectangle((0.003, 0.006), 0.994, 0.988, transform=fig.transFigure, fill=False,
                                     edgecolor=site_style.PALETTE[1], linewidth=1.5, linestyle=":"))
    callout = dict(xycoords="axes fraction", textcoords="figure fraction", fontsize=9, color=site_style.PALETTE[1],
                   va="center", arrowprops=dict(arrowstyle="->", color=site_style.PALETTE[1], linewidth=1))
    ax.annotate("Title: ax.set_title", xy=(0.80, 1.04), xytext=(0.73, 0.90), **callout)
    ax.annotate("Axes: the plot area", xy=(0.98, 0.75), xytext=(0.73, 0.72), **callout)
    fig.text(0.73, 0.025, "Figure: the whole canvas\n(dotted border)", fontsize=9, color=site_style.PALETTE[1])
    ax.annotate("Lines: ax.plot (Line2D)", xy=(0.93, 0.45), xytext=(0.73, 0.58), **callout)
    ax.annotate("Legend: ax.legend", xy=(0.985, 0.10), xytext=(0.73, 0.44), **callout)
    ax.annotate("Spine (axis line)", xy=(1.0, 0.0), xytext=(0.73, 0.30), **callout)
    ax.annotate("Tick labels: formatter", xy=(0.975, -0.06), xytext=(0.73, 0.17), **callout)
    save(fig, "mpl-anatomy.png")
    return weekly


# ---- 2. line plot: indexed monthly revenue -----------------------------------------
def line_indexed():
    monthly = orders.pivot_table(index="order_month", columns="category", values="revenue",
                                 aggfunc="sum", observed=True)
    indexed = monthly / monthly.iloc[0] * 100                      # January = 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cat in indexed.columns:
        ax.plot(indexed.index, indexed[cat], marker="o", label=cat)
        nudge = {"Home": 5, "Books": -5}.get(cat, 0)                 # keep end labels apart
        ax.annotate(cat, xy=(indexed.index[-1], indexed[cat].iloc[-1]), xytext=(8, nudge),
                    textcoords="offset points", va="center", fontsize=9)
    ax.axhline(100, color=site_style.INK_2, linewidth=1, linestyle=":")
    ax.set_title("Monthly revenue by category, indexed to January = 100")
    ax.set_ylabel("Index (Jan = 100)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.set_xlim(indexed.index[0] - pd.Timedelta(days=8), indexed.index[-1] + pd.Timedelta(days=25))
    ax.set_xticks(indexed.index)
    save(fig, "mpl-line-indexed.png")
    return indexed


# ---- 3. bars: vertical, horizontal, stacked ----------------------------------------
def bars():
    rev = orders.groupby("category", observed=True)["revenue"].sum().sort_values()
    share = pd.crosstab(orders["category"], orders["device"], normalize="index")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    ax.bar(rev.index, rev.to_numpy(), color=site_style.PALETTE[0], width=0.6)
    ax.set_title("Vertical bar: revenue")
    ax.yaxis.set_major_formatter(rupees_m)
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    bars_ = ax.barh(rev.index, rev.to_numpy(), color=site_style.PALETTE[0], height=0.6)
    ax.bar_label(bars_, labels=[f"₹{v / 1e6:.1f}M" for v in rev], padding=3, fontsize=8)
    ax.set_title("Horizontal bar, sorted + labels")
    ax.xaxis.set_major_formatter(rupees_m)
    ax.set_xlim(0, rev.max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax = axes[2]
    left = np.zeros(len(share))
    for i, device in enumerate(share.columns):
        ax.barh(share.index, share[device], left=left, color=site_style.PALETTE[i], label=device,
                edgecolor=site_style.SURFACE, linewidth=1.5, height=0.6)
        left += share[device].to_numpy()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Stacked bar: device mix")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "mpl-bars.png")
    return rev, share


# ---- 4. histograms: raw vs log bins -----------------------------------------------
def hists():
    p = orders["unit_price"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    counts, edges, _ = axes[0].hist(p, bins=50, color=site_style.PALETTE[0], edgecolor=site_style.SURFACE)
    axes[0].set_title("50 equal-width bins: one tall spike")
    axes[0].set_xlabel("Unit price (₹)")
    axes[0].set_ylabel("Orders")
    log_bins = np.logspace(np.log10(p.min()), np.log10(p.max()), 50)
    axes[1].hist(p, bins=log_bins, color=site_style.PALETTE[0], edgecolor=site_style.SURFACE)
    axes[1].set_xscale("log")
    axes[1].axvline(p.median(), color=site_style.PALETTE[1], linestyle="--", label=f"median ₹{p.median():,.0f}")
    axes[1].axvline(p.mean(), color=site_style.PALETTE[2], linestyle="--", label=f"mean ₹{p.mean():,.0f}")
    axes[1].set_title("Log-spaced bins on a log axis")
    axes[1].set_xlabel("Unit price (₹, log scale)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "mpl-hist.png")
    return counts, edges


# ---- 5. scatter: customers ---------------------------------------------------------
def scatter():
    cust = (enriched.dropna(subset=["tier"]).groupby(["customer_id", "tier"], observed=True)
            .agg(n_orders=("order_id", "count"), spend=("revenue", "sum"), avg_rating=("rating", "mean"))
            .reset_index())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, tier in enumerate(["Bronze", "Silver", "Gold"]):
        d = cust[cust["tier"] == tier]
        ax.scatter(d["n_orders"], d["spend"], s=40, alpha=0.75, color=site_style.PALETTE[i],
                   edgecolor=site_style.SURFACE, linewidth=0.8, label=f"{tier} ({len(d)})")
    slope, intercept = np.polyfit(cust["n_orders"], cust["spend"], 1)
    xs = np.array([cust["n_orders"].min(), cust["n_orders"].max()])
    ax.plot(xs, slope * xs + intercept, color=site_style.INK_2, linewidth=1.2, linestyle="--",
            label=f"fit: ₹{slope:,.0f} per order")
    ax.set_title("Customers: orders vs total spend")
    ax.set_xlabel("Orders in 6 months")
    ax.set_ylabel("Total spend")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"₹{v / 1e3:.0f}k"))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(title="Tier")
    save(fig, "mpl-scatter.png")
    return cust, slope, intercept


# ---- 6. error bars ---------------------------------------------------------------
def errorbars():
    rated = orders[orders["rated"]]
    stats = (rated.assign(days=rated["delivery_days"].clip(upper=10))
             .groupby("days")["rating"].agg(["mean", "std", "count"]))
    stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(stats.index, stats["mean"], yerr=stats["ci95"], fmt="o-", capsize=4,
                color=site_style.PALETTE[0], ecolor=site_style.INK_2, elinewidth=1)
    ax.set_title("Average rating by delivery time (95% confidence interval)")
    ax.set_xlabel("Delivery days (10 = 10 or more)")
    ax.set_ylabel("Mean rating")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(1, 5)
    save(fig, "mpl-errorbar.png")
    return stats


# ---- 7. formatting + annotate + twinx vs two panels ------------------------------
def twin_vs_panels():
    m = orders.groupby("order_month").agg(orders=("order_id", "count"), avg_order=("revenue", "mean"))
    labels = m.index.strftime("%b")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, m["orders"], color=site_style.PALETTE[0], width=0.6, label="Orders")
    ax.set_ylabel("Orders", color=site_style.PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(labels, m["avg_order"], color=site_style.PALETTE[1], marker="o", label="Avg order value")
    ax2.set_ylabel("Avg order value (₹)", color=site_style.PALETTE[1])
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_title("twinx: two y-scales on one plot (avoid)")
    save(fig, "mpl-twinx.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    top.bar(labels, m["orders"], color=site_style.PALETTE[0], width=0.6)
    top.set_title("Orders per month")
    top.yaxis.set_major_locator(MultipleLocator(200))
    bottom.plot(labels, m["avg_order"], color=site_style.PALETTE[0], marker="o")
    bottom.set_title("Average order value")
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"₹{v:,.0f}"))
    peak = m["avg_order"].idxmax()
    i = list(m.index).index(peak)
    bottom.annotate(f"Peak: ₹{m['avg_order'].max():,.0f}", xy=(i, m["avg_order"].max()),
                    xytext=(i - 2.2, m["avg_order"].max() - 120),
                    arrowprops=dict(arrowstyle="->", color=site_style.INK_2), fontsize=9)
    fig.tight_layout()
    save(fig, "mpl-two-panels.png")
    return m


if __name__ == "__main__":
    w = anatomy(); print("weekly", len(w), round(w.max(), 2))
    idx = line_indexed(); print(idx.round(1).iloc[-1].to_dict())
    rev, share = bars(); print(rev.round(0).to_dict()); print(share.round(3).to_dict())
    counts, edges = hists(); print("first bin", int(counts[0]), round(edges[1], 2), "share", round(counts[0] / counts.sum(), 3))
    cust, slope, intercept = scatter(); print(len(cust), round(slope, 1), round(intercept, 1), round(cust[["n_orders", "spend"]].corr().iloc[0, 1], 3))
    print(errorbars().round(3).to_string())
    print(twin_vs_panels().round(1).to_string())
