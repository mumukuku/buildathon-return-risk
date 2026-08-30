"""
RiskGuard business-impact extrapolation
==========================================

Turns the held-out test-set cost comparison into a number a merchant (or
a non-ML judge) can actually reason about: rupees per month, at a few
example order volumes.

METHODOLOGY (stated explicitly -- this is a projection, not a guarantee):
    1. On the 90-day test window, the rule-based baseline's total cost
       was Rs.810,063 vs. the ML model's Rs.448,090 (see
       model/baseline_comparison.csv) -- a savings of Rs.361,973 across
       12,489 total orders placed in that window (not just the returned
       ones; savings is expressed per ORDER because that's the number a
       merchant actually plans around).
    2. That gives a savings rate of ~Rs.29 per order placed, GIVEN this
       population's return rate (~20.5%), abuse rate among returns
       (~11%), and category/value mix.
    3. We scale that per-order rate linearly to a few example monthly
       order volumes. This assumes the merchant's return/fraud mix looks
       similar to our synthetic population -- a real merchant should
       recompute this rate on their own historical data, not trust ours
       blindly. We say so.

This script does NOT invent a new cost model -- it reuses the exact
FP/FN costs and comparison already computed in train_model.py, so the
projection is traceable back to the same assumptions, not a separate
set of numbers pulled from nowhere.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

TEST_PATH = "data/test.csv"
ORDERS_PATH = "data/orders.csv"
BASELINE_COMPARISON_PATH = "model/baseline_comparison.csv"

EXAMPLE_MONTHLY_VOLUMES = [50_000, 500_000, 2_000_000]  # small / mid / large merchant


def main():
    test = pd.read_csv(TEST_PATH, parse_dates=["order_date"])
    orders = pd.read_csv(ORDERS_PATH, parse_dates=["order_date"])
    baseline_cmp = pd.read_csv(BASELINE_COMPARISON_PATH)

    window_start = test["order_date"].min()
    window_end = test["order_date"].max()
    window_days = (window_end - window_start).days
    window_orders = orders[orders["order_date"] >= window_start]
    total_orders_in_window = len(window_orders)

    baseline_cost = baseline_cmp.loc[baseline_cmp["name"] == "Rule-based baseline", "total_cost"].iloc[0]
    ml_cost = baseline_cmp.loc[baseline_cmp["name"].str.contains("XGBoost"), "total_cost"].iloc[0]
    savings_in_window = baseline_cost - ml_cost

    savings_per_day = savings_in_window / window_days
    savings_per_order = savings_in_window / total_orders_in_window

    print(f"Test window: {window_start.date()} to {window_end.date()} ({window_days} days)")
    print(f"Total orders in window: {total_orders_in_window}")
    print(f"Baseline cost: Rs.{baseline_cost:,.0f} | ML cost: Rs.{ml_cost:,.0f}")
    print(f"Savings in window: Rs.{savings_in_window:,.0f}")
    print(f"Savings per day: Rs.{savings_per_day:,.2f}")
    print(f"Savings per order placed: Rs.{savings_per_order:,.4f}")

    projections = []
    for monthly_orders in EXAMPLE_MONTHLY_VOLUMES:
        monthly_savings = savings_per_order * monthly_orders
        annual_savings = monthly_savings * 12
        projections.append({
            "monthly_order_volume": monthly_orders,
            "projected_monthly_savings_rs": round(monthly_savings, 0),
            "projected_annual_savings_rs": round(annual_savings, 0),
        })

    print("\n=== Projected savings at example merchant scales ===")
    print("(ASSUMES similar return rate / abuse rate / order-value mix to our synthetic "
          "population -- recompute on real historical data before using this number "
          "in any real business decision)")
    for p in projections:
        print(f"  {p['monthly_order_volume']:>10,} orders/month  ->  "
              f"Rs.{p['projected_monthly_savings_rs']:>12,.0f} / month  |  "
              f"Rs.{p['projected_annual_savings_rs']:>14,.0f} / year")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{v:,}" for v in EXAMPLE_MONTHLY_VOLUMES]
    monthly_vals = [p["projected_monthly_savings_rs"] for p in projections]
    bars = ax.bar(labels, monthly_vals, color="#2a9d8f")
    ax.set_xlabel("Merchant monthly order volume")
    ax.set_ylabel("Projected monthly savings (Rs.)")
    ax.set_title("Projected savings vs. rule-based baseline, by merchant scale\n"
                  "(linear extrapolation from held-out test-set results -- see caveats in README)")
    for bar, val in zip(bars, monthly_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"Rs.{val:,.0f}",
                 ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("plots/business_impact_projection.png", dpi=150)
    plt.close()

    with open("model/business_impact.json", "w") as f:
        json.dump({
            "test_window_days": window_days,
            "total_orders_in_window": total_orders_in_window,
            "baseline_cost_rs": float(baseline_cost),
            "ml_cost_rs": float(ml_cost),
            "savings_in_window_rs": float(savings_in_window),
            "savings_per_order_rs": float(savings_per_order),
            "projections": projections,
            "caveat": "Linear extrapolation assuming the merchant's return rate, abuse rate, "
                      "and order-value distribution match this synthetic population. Recompute "
                      "on real historical data before relying on this for a business decision.",
        }, f, indent=2)
    print("\nSaved: model/business_impact.json, plots/business_impact_projection.png")


if __name__ == "__main__":
    main()
