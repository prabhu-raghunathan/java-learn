"""ShopStream end-to-end EDA: generate -> profile -> clean -> features -> dashboard -> findings.

Run from this folder:  python capstone_eda.py
Needs: numpy, pandas, matplotlib, seaborn (and shopstream.py next to it).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                   # render to files, no window
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns

from shopstream import make_shopstream

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "img", "data-exploration")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CAT_ORDER = ["Books", "Beauty", "Fashion", "Home", "Electronics"]
CITY_ORDER = ["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"]
SNAPSHOT = pd.Timestamp("2026-07-01")


# ---------------------------------------------------------------- 1. load + profile
def profile(df):
    """One row per column: dtype, missing share, distinct values, an example."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_pct": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
        "example": df.iloc[0],
    })


def quality_flags(raw):
    """The red flags worth fixing, as numbers."""
    return {
        "rows": len(raw),
        "exact_duplicates": int(raw.duplicated().sum()),
        "category_labels": raw["category"].nunique(),
        "day_first_dates": int(raw["order_ts"].str.contains("/").sum()),
        "bad_quantity": int((~raw["quantity"].between(1, 50)).sum()),
        "orphan_customer_rows": int((raw["customer_id"] == "C9999").sum()),
        "price_skew": round(float(raw["unit_price"].skew()), 2),
        "price_mean_vs_median": (round(float(raw["unit_price"].mean()), 2), round(float(raw["unit_price"].median()), 2)),
    }


# ---------------------------------------------------------------- 2. clean
def parse_timestamps(s):
    ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts = ts.fillna(pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce"))
    assert ts.notna().all(), "unparsed timestamps"
    return ts


def group_log_iqr_outlier(price, group, k=1.5):
    """True where log(price) is beyond the IQR fence of its own group."""
    lp = np.log(price)
    g = lp.groupby(group, observed=True)
    q1, q3 = g.transform(lambda x: x.quantile(0.25)), g.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    return (lp > q3 + k * iqr) | (lp < q1 - k * iqr)


def clean(raw, customers):
    log = [("raw rows", len(raw))]
    o = raw.drop_duplicates()
    log.append(("drop exact duplicates", len(o)))

    o = o.assign(
        order_ts=parse_timestamps(o["order_ts"]),
        category=pd.Categorical(o["category"].str.strip().str.title(), categories=CAT_ORDER),
        payment_method=o["payment_method"].fillna("Unknown").astype("category"),
        device=o["device"].astype("category"),
        discount_pct=o["discount_pct"].fillna(0.0),
    )
    assert o["category"].notna().all(), "unknown category label"

    o = o[o["quantity"].between(1, 50)]
    log.append(("drop impossible quantity", len(o)))

    extreme = group_log_iqr_outlier(o["unit_price"], o["category"], k=3.0)   # k=3: 'extreme' fence, per category
    o = o[~extreme]
    log.append(("drop extreme prices (per category)", len(o)))

    o = o.merge(customers, on="customer_id", how="left", validate="m:1", indicator=True)
    city_median = o.groupby("city")["delivery_days"].transform("median")
    o = o.assign(
        is_orphan=o["_merge"].eq("left_only"),
        delivery_missing=o["delivery_days"].isna(),
        delivery_days=o["delivery_days"].fillna(city_median).fillna(o["delivery_days"].median()),
        rated=o["rating"].notna(),
    ).drop(columns="_merge")
    return o.reset_index(drop=True), pd.DataFrame(log, columns=["step", "rows"])


# ---------------------------------------------------------------- 3. features + aggregation
def add_order_features(o):
    return o.assign(
        revenue=(o["unit_price"] * o["quantity"] * (1 - o["discount_pct"])).round(2),
        order_month=o["order_ts"].dt.to_period("M").dt.to_timestamp(),
        weekday=o["order_ts"].dt.day_name(),
        delivery_bucket=pd.cut(o["delivery_days"], bins=[0, 3, 5, 7, np.inf], labels=["1-3", "4-5", "6-7", "8+"]),
        price_vs_category=o["unit_price"] / o.groupby("category", observed=True)["unit_price"].transform("median"),
    )


def customer_table(o):
    real = o[~o["is_orphan"]]
    return (real.groupby(["customer_id", "tier", "city"], observed=True)
            .agg(n_orders=("order_id", "count"), total_spend=("revenue", "sum"),
                 avg_order=("revenue", "mean"), avg_delivery=("delivery_days", "mean"),
                 avg_rating=("rating", "mean"), last_order=("order_ts", "max"))
            .assign(recency_days=lambda d: (SNAPSHOT - d["last_order"]).dt.days)
            .reset_index())


def summaries(o):
    monthly = o.pivot_table(index="order_month", columns="category", values="revenue",
                            aggfunc="sum", observed=True)
    by_category = (o.groupby("category", observed=True)
                   .agg(orders=("order_id", "count"), revenue=("revenue", "sum"),
                        median_price=("unit_price", "median"), avg_rating=("rating", "mean"))
                   .assign(revenue_share=lambda d: d["revenue"] / d["revenue"].sum()))
    rating_by_delivery = (o[o["rated"]].assign(days=o["delivery_days"].clip(upper=10))
                          .groupby("days")["rating"].agg(["mean", "count"]))
    city_bucket = o.pivot_table(index="city", columns="delivery_bucket", values="rating",
                                aggfunc="mean", observed=True)
    return monthly, by_category, rating_by_delivery, city_bucket


# ---------------------------------------------------------------- 4. dashboard
def kpi(ax, value, label, color=INK):
    ax.axis("off")
    ax.text(0.0, 0.62, value, fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.0, 0.22, label, fontsize=10, color=INK_2, transform=ax.transAxes)


def dashboard(o, cust, monthly, by_category, path):
    sns.set_theme(style="whitegrid", palette=PALETTE,
                  rc={"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.spines.top": False,
                      "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                      "axes.titlelocation": "left", "legend.frameon": False})
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.35, 1, 1])

    # row 0: KPI tiles
    rated = o[o["rated"]]
    kpi(fig.add_subplot(gs[0, 0]), f"₹{o['revenue'].sum() / 1e6:.1f}M", f"revenue, {len(o):,} orders (Jan to Jun 2026)")
    kpi(fig.add_subplot(gs[0, 1]), f"₹{o['revenue'].median():,.0f}", "median order value (mean is higher: skew)")
    kpi(fig.add_subplot(gs[0, 2]), f"{rated['rating'].mean():.2f} / 5", f"average rating ({o['rated'].mean():.0%} of orders rated)")
    slow = (o["delivery_days"] >= 8).mean()
    kpi(fig.add_subplot(gs[0, 3]), f"{slow:.0%}", "orders delivered in 8+ days", color=PALETTE[7])

    # row 1 left: indexed monthly revenue (Matplotlib)
    ax = fig.add_subplot(gs[1, :2])
    indexed = monthly / monthly.iloc[0] * 100
    for i, cat in enumerate(CAT_ORDER):
        ax.plot(indexed.index, indexed[cat], marker="o", color=PALETTE[i], label=cat)
    ax.axhline(100, color=INK_2, linewidth=1, linestyle=":")
    ax.set_xticks(indexed.index)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_title("Revenue growth by category (January = 100)")
    ax.set_ylabel("Index")
    ax.legend(ncols=5, loc="upper left", fontsize=8)

    # row 1 right: revenue share by category (Matplotlib barh + labels)
    ax = fig.add_subplot(gs[1, 2:])
    share = by_category["revenue_share"].reindex(CAT_ORDER)
    bars = ax.barh(share.index.astype(str), share.to_numpy(), color=PALETTE[0], height=0.6)
    ax.bar_label(bars, labels=[f"{v:.0%}  (₹{r / 1e6:.1f}M)" for v, r in zip(share, by_category["revenue"].reindex(CAT_ORDER))],
                 padding=4, fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, share.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of revenue: Electronics dominates")

    # row 2 left: rating vs delivery days (Seaborn lineplot with CI)
    ax = fig.add_subplot(gs[2, :2])
    r = o[o["rated"]].assign(days=o["delivery_days"].clip(upper=10))
    sns.lineplot(data=r, x="days", y="rating", errorbar=("ci", 95), marker="o", color=PALETTE[0], ax=ax)
    ax.set_title("Every extra delivery day costs about a third of a star")
    ax.set_xlabel("Delivery days (10 = 10 or more)")
    ax.set_ylabel("Mean rating (95% CI)")
    ax.set_ylim(1, 5)

    # row 2 middle: delivery days by city (Seaborn boxplot)
    ax = fig.add_subplot(gs[2, 2])
    sns.boxplot(data=o.dropna(subset=["city"]), x="city", y="delivery_days", order=CITY_ORDER,
                fill=False, width=0.5, color=PALETTE[0], fliersize=2, ax=ax)
    ax.set_title("Delhi and Hyderabad deliver slower")
    ax.set_xlabel("")
    ax.set_ylabel("Delivery days")
    ax.tick_params(axis="x", rotation=30)

    # row 2 right: payment method vs delivery (Seaborn barplot)
    ax = fig.add_subplot(gs[2, 3])
    sns.barplot(data=o, x="payment_method", y="delivery_days", order=["UPI", "Card", "Wallet", "COD", "Unknown"],
                errorbar=("ci", 95), color=PALETTE[0], width=0.6, ax=ax)
    ax.set_title("COD orders arrive later")
    ax.set_xlabel("")
    ax.set_ylabel("Mean delivery days (95% CI)")

    fig.suptitle("ShopStream EDA dashboard, H1 2026", fontsize=16, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 5. findings
def findings(o, cust, by_category, rating_by_delivery):
    rated = o[o["rated"]]
    slope = np.polyfit(rated["delivery_days"], rated["rating"], 1)[0]
    city_days = o.groupby("city")["delivery_days"].mean()
    pay = o.groupby("payment_method", observed=True)[["delivery_days", "rating"]].mean()
    return {
        "revenue_total_m": round(float(o["revenue"].sum()) / 1e6, 2),
        "electronics_share": round(float(by_category.loc["Electronics", "revenue_share"]), 3),
        "rating_per_extra_day": round(float(slope), 3),
        "spearman_delivery_rating": round(float(rated[["delivery_days", "rating"]].corr(method="spearman").iloc[0, 1]), 3),
        "rating_1_day_vs_10_plus": (round(float(rating_by_delivery.loc[1.0, "mean"]), 2), round(float(rating_by_delivery.loc[10.0, "mean"]), 2)),
        "slow_cities_days": city_days[["Delhi", "Hyderabad"]].round(2).to_dict(),
        "fast_cities_days": city_days[["Pune", "Mumbai", "Bengaluru"]].round(2).to_dict(),
        "cod_vs_upi_days": (round(float(pay.loc["COD", "delivery_days"]), 2), round(float(pay.loc["UPI", "delivery_days"]), 2)),
        "cod_vs_upi_rating": (round(float(pay.loc["COD", "rating"]), 2), round(float(pay.loc["UPI", "rating"]), 2)),
        "rated_share": round(float(o["rated"].mean()), 3),
        "orphan_orders": int(o["is_orphan"].sum()),
        "tier_avg_order": cust.groupby("tier", observed=True)["avg_order"].mean().round(0).to_dict(),
        "corr_orders_spend": round(float(cust[["n_orders", "total_spend"]].corr().iloc[0, 1]), 3),
    }


def main():
    raw, customers = make_shopstream()

    print(profile(raw).to_string(), "\n")
    print(quality_flags(raw), "\n")

    orders, cleaning_log = clean(raw, customers)
    print(cleaning_log.to_string(index=False), "\n")

    orders = add_order_features(orders)
    cust = customer_table(orders)
    monthly, by_category, rating_by_delivery, city_bucket = summaries(orders)
    print(by_category.round(3).to_string(), "\n")
    print(city_bucket.round(2).to_string(), "\n")

    dashboard(orders, cust, monthly, by_category, os.path.join(OUT_DIR, "capstone-dashboard.png"))
    for key, value in findings(orders, cust, by_category, rating_by_delivery).items():
        print(f"{key:28s} {value}")


if __name__ == "__main__":
    main()
