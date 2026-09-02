"""
RiskGuard cost-sensitivity grid
==================================

Precomputes the cost-optimal threshold across a GRID of (FP review cost,
max flag rate cap) assumptions, so the frontend's interactive
cost-sensitivity slider can respond instantly to a drag without a live
model round-trip per frame.

This is still REAL data -- every cell in the grid is computed the exact
same way as the single point already reported in deployment_config.json
(cost_sweep in train_model.py), just repeated across a range of
assumptions instead of the one pair we chose to deploy with. Nothing here
is interpolated or faked; the frontend interpolates only WITHIN this grid
for in-between slider positions, never invents a number outside it.
"""

import json
import joblib
import numpy as np
import pandas as pd

FP_COSTS = [50, 100, 150, 200, 300, 500, 750, 1000]
MAX_FLAG_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]


def main():
    test = pd.read_csv("data/test.csv")
    with open("model/feature_columns.json") as f:
        feat_meta = json.load(f)
    X_test = test[feat_meta["encoded_feature_columns"]]
    y_test = test[feat_meta["target"]].values
    order_values = test["order_value"].values

    model = joblib.load("model/deployed_model.pkl")
    proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    n_total = len(y_test)

    # Precompute per-threshold stats once (fp_count, fn_count, tp_count,
    # fn_cost) -- reused across every (fp_cost, flag_rate_cap) combination
    # so we don't rescan the test set 72 times.
    per_threshold = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        fp_mask = (preds == 1) & (y_test == 0)
        fn_mask = (preds == 0) & (y_test == 1)
        tp_mask = (preds == 1) & (y_test == 1)
        per_threshold.append({
            "threshold": t,
            "fp_count": int(fp_mask.sum()),
            "fn_count": int(fn_mask.sum()),
            "tp_count": int(tp_mask.sum()),
            "fn_cost_base": float(order_values[fn_mask].sum()),  # value-weighted FN cost
            "flag_rate": float((fp_mask.sum() + tp_mask.sum()) / n_total),
        })

    grid = []
    for fp_cost in FP_COSTS:
        for max_flag_rate in MAX_FLAG_RATES:
            best = None
            for row in per_threshold:
                if row["flag_rate"] > max_flag_rate:
                    continue
                total_cost = row["fp_count"] * fp_cost + row["fn_cost_base"]
                if best is None or total_cost < best["total_cost"]:
                    tp, fp, fn = row["tp_count"], row["fp_count"], row["fn_count"]
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    best = {
                        "threshold": float(row["threshold"]),
                        "flag_rate": row["flag_rate"],
                        "total_cost": float(total_cost),
                        "precision": precision,
                        "recall": recall,
                    }
            if best is None:  # no threshold satisfies the cap (shouldn't happen given 1.0 is in the list)
                continue
            grid.append({"fp_cost": fp_cost, "max_flag_rate": max_flag_rate, **best})

    output = {
        "fp_costs": FP_COSTS,
        "max_flag_rates": MAX_FLAG_RATES,
        "grid": grid,
    }
    with open("model/cost_sensitivity_grid.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Computed {len(grid)} grid cells across {len(FP_COSTS)} FP costs x {len(MAX_FLAG_RATES)} flag-rate caps")
    print("Sample cell (fp_cost=150, max_flag_rate=0.20):")
    sample = next(g for g in grid if g["fp_cost"] == 150 and g["max_flag_rate"] == 0.20)
    print(f"  threshold={sample['threshold']:.2f}, flag_rate={sample['flag_rate']:.1%}, "
          f"total_cost=Rs.{sample['total_cost']:,.0f}, precision={sample['precision']:.3f}, recall={sample['recall']:.3f}")
    print("Saved: model/cost_sensitivity_grid.json")


if __name__ == "__main__":
    main()
