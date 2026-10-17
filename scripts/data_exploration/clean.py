"""Cleaning steps from the 'Cleaning, Types, Dates, Strings & Missing Data' page,
collected into one function so later pages can start from clean data.

    from shopstream import make_shopstream
    from clean import clean_orders
    raw, customers = make_shopstream()
    orders = clean_orders(raw, customers)
"""
import numpy as np
import pandas as pd


def clean_orders(raw, customers):
    o = raw.drop_duplicates()                                                     # 20 exact duplicates

    # dates: one explicit format per known pattern, never format="mixed"
    ts = pd.to_datetime(o["order_ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts = ts.fillna(pd.to_datetime(o["order_ts"], format="%d/%m/%Y %H:%M", errors="coerce"))
    assert ts.notna().all()

    o = o.assign(
        order_ts=ts,
        category=o["category"].str.strip().str.title().astype("category"),
        payment_method=o["payment_method"].fillna("Unknown").astype("category"),
        device=o["device"].astype("category"),
        discount_pct=o["discount_pct"].fillna(0.0),                               # missing = no discount
    )

    # validity: impossible quantities and x40 data-entry prices (log-scale IQR fence)
    log_p = np.log(o["unit_price"])
    q1, q3 = log_p.quantile([0.25, 0.75])
    o = o[o["quantity"].between(1, 50) & (log_p <= q3 + 1.5 * (q3 - q1))]

    # delivery_days: median per customer city, then the global median; keep a flag
    city = o["customer_id"].map(customers.set_index("customer_id")["city"])
    city_median = o.groupby(city)["delivery_days"].transform("median")
    o = o.assign(
        delivery_missing=o["delivery_days"].isna(),
        delivery_days=o["delivery_days"].fillna(city_median).fillna(o["delivery_days"].median()),
        rated=o["rating"].notna(),                                                # rating stays NaN (MNAR)
        revenue=(o["unit_price"] * o["quantity"] * (1 - o["discount_pct"])).round(2),
        order_month=ts.dt.to_period("M").dt.to_timestamp(),
    )
    return o.reset_index(drop=True)
