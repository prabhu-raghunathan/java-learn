"""Every figure on the Seaborn page. Run from this folder:  python figures_seaborn.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import site_style
from shopstream import make_shopstream
from clean import clean_orders

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "static", "img", "data-exploration")
raw, customers = make_shopstream()
orders = clean_orders(raw, customers)
enriched = orders.merge(customers, on="customer_id", how="left", validate="m:1")
CAT_ORDER = ["Books", "Beauty", "Fashion", "Home", "Electronics"]

# Seaborn theme with the site palette: set_theme resets rcParams, then layers style + palette + overrides
sns.set_theme(context="notebook", style="whitegrid", palette=site_style.PALETTE,
              rc={"figure.facecolor": site_style.SURFACE, "axes.facecolor": site_style.SURFACE,
                  "savefig.facecolor": site_style.SURFACE, "axes.spines.top": False,
                  "axes.spines.right": False, "grid.color": site_style.GRID, "axes.edgecolor": site_style.INK_2,
                  "axes.titleweight": "bold", "axes.titlesize": 12, "savefig.dpi": 150, "savefig.bbox": "tight",
                  "legend.frameon": False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


def figure_level():
    g = sns.displot(data=orders, x="unit_price", col="device", log_scale=True, bins=30,
                    height=3, aspect=1.2, color=site_style.PALETTE[0])
    g.set_axis_labels("Unit price (₹, log scale)", "Orders")
    g.set_titles("device = {col_name}")
    g.figure.suptitle("displot: one call, one panel per device", y=1.05, fontweight="bold")
    save(g.figure, "sns-figure-level.png")
    return type(g).__name__, g.axes.shape


def distributions():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    sns.histplot(data=orders, x="unit_price", hue="category", hue_order=CAT_ORDER, log_scale=True,
                 element="step", fill=False, bins=40, ax=axes[0, 0])
    axes[0, 0].set_title("histplot: hue + log_scale + step")
    for bw, color in zip([0.2, 1, 3], site_style.PALETTE):
        sns.kdeplot(data=orders, x="delivery_days", bw_adjust=bw, ax=axes[0, 1], color=color,
                    label=f"bw_adjust={bw}", cut=0)
    axes[0, 1].legend()
    axes[0, 1].set_title("kdeplot: bandwidth changes the shape")
    sns.ecdfplot(data=orders, x="unit_price", hue="category", hue_order=CAT_ORDER, log_scale=True, ax=axes[1, 0])
    axes[1, 0].set_title("ecdfplot: share of orders at or below a price")
    sns.histplot(data=enriched.dropna(subset=["city"]), x="delivery_days", hue="city", stat="density",
                 common_norm=False, discrete=True, multiple="layer", element="step", fill=False,
                 hue_order=["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"], ax=axes[1, 1])
    axes[1, 1].set_title("histplot: stat=density, common_norm=False")
    fig.tight_layout()
    save(fig, "sns-distributions.png")


def customer_table():
    return (enriched.dropna(subset=["tier"])
            .groupby(["customer_id", "tier", "city"], observed=True)
            .agg(n_orders=("order_id", "count"), total_spend=("revenue", "sum"),
                 avg_order=("revenue", "mean"), avg_delivery=("delivery_days", "mean"),
                 avg_rating=("rating", "mean"))
            .reset_index())


def joint(cust):
    g = sns.jointplot(data=cust, x="avg_delivery", y="avg_rating", kind="reg", height=5,
                      color=site_style.PALETTE[0], marginal_kws=dict(bins=20))
    g.set_axis_labels("Average delivery days (per customer)", "Average rating (per customer)")
    g.figure.suptitle("jointplot(kind='reg'): 200 customers", y=1.02, fontweight="bold")
    save(g.figure, "sns-jointplot.png")
    return cust[["avg_delivery", "avg_rating"]].corr().iloc[0, 1]


def categorical():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=orders, x="category", y="unit_price", order=CAT_ORDER, log_scale=True,
                color=site_style.PALETTE[0], fill=False, width=0.5, ax=axes[0, 0])
    axes[0, 0].set_title("boxplot: quartiles, whiskers, outliers")
    sns.violinplot(data=enriched.dropna(subset=["city"]), x="city", y="delivery_days",
                   order=["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"], inner="quart",
                   color=site_style.PALETTE[0], fill=False, cut=0, ax=axes[0, 1])
    sns.stripplot(data=enriched.dropna(subset=["city"]).sample(600, random_state=0), x="city", y="delivery_days",
                  order=["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"], jitter=0.25, size=2.5,
                  alpha=0.5, color=site_style.PALETTE[1], ax=axes[0, 1])
    axes[0, 1].set_title("violinplot + stripplot: shape and raw points")
    sns.barplot(data=orders, x="payment_method", y="rating", order=["UPI", "Card", "Wallet", "COD", "Unknown"],
                errorbar=("ci", 95), color=site_style.PALETTE[0], width=0.6, ax=axes[1, 0])
    axes[1, 0].set_ylim(1, 5)
    axes[1, 0].set_title("barplot: mean rating with 95% CI")
    sns.countplot(data=orders, x="payment_method", hue="device", order=["UPI", "Card", "Wallet", "COD", "Unknown"],
                  hue_order=["mobile", "app", "desktop"], ax=axes[1, 1])
    axes[1, 1].set_title("countplot: orders per payment method and device")
    fig.tight_layout()
    save(fig, "sns-categorical.png")


def relational(cust):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    sns.scatterplot(data=cust, x="avg_delivery", y="total_spend", hue="tier", hue_order=["Bronze", "Silver", "Gold"],
                    size="n_orders", sizes=(15, 120), alpha=0.8, ax=axes[0])
    axes[0].set_title("scatterplot: hue + size encodings")
    axes[0].legend(fontsize=8, ncols=2, loc="upper right")
    rated = enriched[enriched["rated"]].dropna(subset=["city"])
    rated = rated.assign(days=rated["delivery_days"].clip(upper=10))
    sns.lineplot(data=rated, x="days", y="rating", hue="city",
                 hue_order=["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"],
                 errorbar=("ci", 95), marker="o", ax=axes[1])
    axes[1].set_title("lineplot: mean and 95% CI band per city")
    axes[1].set_xlabel("Delivery days (10 = 10+)")
    fig.tight_layout()
    save(fig, "sns-relational.png")


def heatmap(cust):
    cols = ["unit_price", "quantity", "discount_pct", "revenue", "delivery_days", "rating"]
    corr = orders[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=site_style.diverging_cmap().reversed(),
                vmin=-1, vmax=1, center=0, square=True, linewidths=1, linecolor=site_style.SURFACE,
                cbar_kws={"shrink": 0.75, "label": "Spearman correlation"}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation matrix, upper triangle masked")
    save(fig, "sns-heatmap.png")
    return corr


def pair(cust):
    g = sns.pairplot(cust, vars=["n_orders", "avg_order", "avg_delivery", "avg_rating"], hue="tier",
                     hue_order=["Bronze", "Silver", "Gold"], corner=True, height=2.2,
                     plot_kws=dict(s=18, alpha=0.7), diag_kind="kde", diag_kws=dict(common_norm=False))
    g.figure.suptitle("pairplot(hue='tier', corner=True)", y=1.02, fontweight="bold")
    save(g.figure, "sns-pairplot.png")


def grids(cust):
    city_order = ["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"]
    g = sns.FacetGrid(enriched.dropna(subset=["city"]), col="city", col_order=city_order, col_wrap=5,
                      height=2.6, aspect=0.9, sharey=True)
    g.map_dataframe(sns.histplot, x="delivery_days", discrete=True, color=site_style.PALETTE[0])
    g.map(plt.axvline, x=orders["delivery_days"].median(), color=site_style.PALETTE[1], linestyle="--")
    g.set_titles("{col_name}")
    g.set_axis_labels("Delivery days", "Orders")
    g.figure.suptitle("FacetGrid: same plot, one panel per city (dashed = overall median)", y=1.06, fontweight="bold")
    save(g.figure, "sns-facetgrid.png")

    pg = sns.PairGrid(cust, vars=["avg_order", "avg_delivery", "avg_rating"], height=2.4)
    pg.map_upper(sns.scatterplot, s=15, color=site_style.PALETTE[0])
    pg.map_lower(sns.kdeplot, fill=True, color=site_style.PALETTE[0], levels=5)
    pg.map_diag(sns.histplot, bins=20, color=site_style.PALETTE[0])
    pg.figure.suptitle("PairGrid: scatter above, KDE below, histogram on the diagonal", y=1.02, fontweight="bold")
    save(pg.figure, "sns-pairgrid.png")
    return type(g).__name__, type(pg).__name__


if __name__ == "__main__":
    print(sns.__version__, figure_level())
    distributions()
    cust = customer_table(); print(cust.shape)
    print("joint r", round(joint(cust), 3))
    categorical(); relational(cust)
    print(heatmap(cust).round(2).to_string())
    pair(cust); print(grids(cust))
