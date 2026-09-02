"""
RiskGuard feature engineering + train/test split
==================================================

Builds the model-ready table for the Return-Risk Scorer and produces a
CHRONOLOGICAL train/test split (not random).

Why chronological, not random:
A random split can leak information: a customer's orders from later in
the year end up in "train" alongside earlier orders from the same
customer in "test", so the model can partially learn a customer's future
via their other rows. A chronological split -- train on the first ~75%
of the timeline, test on the last ~25% -- mimics how the model would
actually be deployed (trained on the past, scored on the future) and
gives an honest estimate of out-of-time performance.

Modeling population:
Only RETURNED orders are used. The question this model answers is
"given a return is happening, how likely is it abusive?" -- not
"will this order be returned?", which is a different (also useful, but
separate) problem.

Feature list intentionally EXCLUDES:
    - order_id, customer_id, order_date          (identifiers / leakage risk)
    - device/address/payment fingerprint          (reserved for the abuse-ring
                                                     sentinel, a separate model)
    - _behavior_type                              (audit-only ground-truth
                                                     generator, never a feature)
"""

import json
import pandas as pd

RAW_PATH = "data/orders.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
FEATURE_LIST_PATH = "model/feature_columns.json"

NUMERIC_FEATURES = [
    "order_value",
    "delivery_days",
    "order_hour",
    "is_weekend",
    "account_age_days_at_order",
    "hist_orders_before",
    "hist_return_rate_before",
    "hist_abusive_return_rate_before",
    "hist_chargebacks_before",
    "price_vs_category_avg",
    "days_to_return",
]
CATEGORICAL_FEATURES = ["category", "payment_method", "return_reason"]
TARGET = "is_abusive_return"


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["order_date"])

    # Modeling population: returns only.
    returns = df[df["is_returned"] == 1].copy()
    returns = returns.sort_values("order_date").reset_index(drop=True)

    # One-hot encode categoricals. drop_first=False so the feature list is
    # stable and self-explanatory when served later (each category gets an
    # explicit column rather than an implicit reference level).
    encoded = pd.get_dummies(returns[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    feature_df = pd.concat(
        [returns[["order_id", "customer_id", "order_date"]],
         returns[NUMERIC_FEATURES],
         encoded,
         returns[[TARGET]]],
        axis=1,
    )

    feature_cols = NUMERIC_FEATURES + list(encoded.columns)

    # Chronological split at the 75th percentile of order_date.
    cutoff = feature_df["order_date"].quantile(0.75, interpolation="nearest")
    train = feature_df[feature_df["order_date"] <= cutoff].copy()
    test = feature_df[feature_df["order_date"] > cutoff].copy()

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump({
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "encoded_feature_columns": feature_cols,
            "target": TARGET,
            "split_cutoff_date": str(cutoff.date()),
        }, f, indent=2)

    print(f"Total returned orders: {len(feature_df)}")
    print(f"Split cutoff date: {cutoff.date()}")
    print(f"Train: {len(train)} rows | abusive rate: {train[TARGET].mean():.1%}")
    print(f"Test:  {len(test)} rows | abusive rate: {test[TARGET].mean():.1%}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
