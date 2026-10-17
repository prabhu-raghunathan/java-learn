"""ShopStream: the synthetic, deliberately messy e-commerce dataset used on every
"Python for Data Exploration" page. Seeded, so every run produces identical data.

    from shopstream import make_shopstream
    orders, customers = make_shopstream()
"""
import numpy as np
import pandas as pd


def make_shopstream(n_orders=5_000, n_customers=200, seed=42):
    rng = np.random.default_rng(seed)

    # ---- customers: one row per customer --------------------------------------
    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_customers + 1)],
        "name": rng.choice(["Alex", "Jordan", "Morgan", "Taylor", "Riley",
                            "Casey", "Sam", "Jamie", "Avery", "Quinn"], n_customers),
        "city": rng.choice(["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"],
                           n_customers, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
        "tier": rng.choice(["Bronze", "Silver", "Gold"], n_customers, p=[0.6, 0.3, 0.1]),
        "signup_date": pd.Timestamp("2025-01-01")
                       + pd.to_timedelta(rng.integers(0, 365, n_customers), unit="D"),
    })

    # ---- orders: one row per order --------------------------------------------
    log_mean = {"Electronics": 9.0, "Fashion": 7.2, "Home": 7.6, "Books": 5.8, "Beauty": 6.5}
    category = rng.choice(list(log_mean), n_orders, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    unit_price = np.exp(rng.normal([log_mean[c] for c in category], 0.5))        # log-normal → right-skewed
    customer_id = rng.choice(customers["customer_id"].to_numpy(), n_orders)
    seconds = rng.integers(0, 181 * 24 * 3600, n_orders)                         # 1 Jan – 30 Jun 2026
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s")
    payment = rng.choice(["UPI", "Card", "COD", "Wallet"], n_orders, p=[0.45, 0.25, 0.15, 0.15])
    city = customers.set_index("customer_id").loc[customer_id, "city"].to_numpy()
    slow_city = np.isin(city, ["Delhi", "Hyderabad"])
    delivery = np.round(rng.gamma(2.0, 1.5, n_orders) + 1 + 2 * slow_city + 1.5 * (payment == "COD"))
    rating = np.clip(np.round(5.2 - 0.35 * delivery + rng.normal(0, 0.8, n_orders)), 1, 5)

    orders = pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(1, n_orders + 1)],
        "customer_id": customer_id,
        "order_ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "unit_price": unit_price.round(2),
        "quantity": rng.poisson(0.8, n_orders) + 1,
        "discount_pct": rng.choice([0.0, 0.05, 0.10, 0.20, 0.30], n_orders, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "payment_method": payment,
        "device": rng.choice(["mobile", "app", "desktop"], n_orders, p=[0.5, 0.3, 0.2]),
        "delivery_days": delivery,
        "rating": rating,
    })

    # ---- inject realistic messiness (each step touches a random subset) -------
    def pick(frac):
        return rng.choice(n_orders, int(frac * n_orders), replace=False)

    i = pick(0.10)                                                                 # inconsistent labels
    orders.loc[i, "category"] = [rng.choice([c.lower(), c.upper(), f" {c} "]) for c in orders.loc[i, "category"]]
    i = pick(0.05)                                                                 # day-first date strings
    orders.loc[i, "order_ts"] = ts[i].strftime("%d/%m/%Y %H:%M")
    orders.loc[pick(0.005), "unit_price"] *= 40                                    # data-entry outliers
    orders.loc[pick(0.004), "quantity"] = 0                                        # impossible quantities
    orders.loc[pick(0.004), "quantity"] = -1
    orders.loc[pick(0.0006), "quantity"] = 999
    orders.loc[pick(0.08), "discount_pct"] = np.nan                                # missing values
    orders.loc[pick(0.03), "payment_method"] = None
    orders.loc[pick(0.06), "delivery_days"] = np.nan
    orders.loc[pick(0.40), "rating"] = np.nan                                      # many people never rate
    orders.loc[pick(0.02), "customer_id"] = "C9999"                                # orphan rows (no customer)
    dupes = orders.sample(20, random_state=seed)                                   # exact duplicate rows
    orders = pd.concat([orders, dupes], ignore_index=True)
    return orders, customers


if __name__ == "__main__":
    o, c = make_shopstream()
    print(o.shape, c.shape)
    print(o.head())
