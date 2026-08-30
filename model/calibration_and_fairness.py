"""
RiskGuard calibration + fairness (operational bias) check
=============================================================

Two questions this script answers, both beyond plain AUC/precision/recall:

1. CALIBRATION: if the model says "73% risk," do ~73% of such cases
   actually turn out to be abusive? A risk manager needs scores that
   mean what they claim -- a human reviewer prioritizing a queue by
   score, or a policy that auto-declines above some probability, is
   silently broken if the scores are badly calibrated even when the
   RANKING (AUC) looks great.

2. OPERATIONAL BIAS / DISPARATE IMPACT: does the model flag some
   groups far more than their true abuse rate would justify? We check
   this across payment method, product category, and account-tenure
   band -- the only group-like attributes that exist in this dataset.

   IMPORTANT SCOPE NOTE: this synthetic dataset has NO demographic
   attributes (no gender, age-of-person, location, etc.), so this is
   NOT a legally-protected-class fairness audit -- it's an operational
   bias check on business attributes. A real deployment should run the
   equivalent analysis against whatever protected classes apply in its
   jurisdiction, using real (properly governed) demographic data.

   We report an "over-flagging ratio" = flag_rate / true_abuse_rate per
   group. A ratio of 1.0 means the model flags a group exactly as often
   as that group actually turns out to be abusive (proportionate). A
   ratio well above 1.0 for one group relative to others is worth
   investigating -- it MAY be legitimate (the group really is riskier
   for reasons the features capture) or may indicate the model is
   keying off a proxy in an unwanted way. We flag it; we don't assume
   the conclusion either way.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

TEST_PATH = "data/test.csv"
ORDERS_PATH = "data/orders.csv"


def load_scored_test():
    test = pd.read_csv(TEST_PATH)
    orders = pd.read_csv(ORDERS_PATH)

    with open("model/feature_columns.json") as f:
        feat_meta = json.load(f)
    with open("model/deployment_config.json") as f:
        deploy_cfg = json.load(f)

    xgb_raw = joblib.load("model/xgb_model.pkl")           # uncalibrated, for before/after comparison
    deployed = joblib.load("model/deployed_model.pkl")     # calibrated, what's actually served
    X_test = test[feat_meta["encoded_feature_columns"]]
    proba_raw = xgb_raw.predict_proba(X_test)[:, 1]
    proba_calibrated = deployed.predict_proba(X_test)[:, 1]
    threshold = deploy_cfg["deployed_threshold"]

    df = test[["order_id", "account_age_days_at_order", feat_meta["target"]]].copy()
    df = df.rename(columns={feat_meta["target"]: "y_true"})
    df["proba"] = proba_calibrated
    df["proba_raw_uncalibrated"] = proba_raw
    df["flagged"] = (proba_calibrated >= threshold).astype(int)

    # bring back raw categorical labels (one-hot encoded away in train/test.csv)
    df = df.merge(orders[["order_id", "category", "payment_method"]], on="order_id", how="left")

    df["account_age_band"] = pd.cut(
        df["account_age_days_at_order"], bins=[-1, 90, 365, 100000],
        labels=["new (<90d)", "established (90-365d)", "veteran (>365d)"]
    )
    return df, threshold


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------
def calibration_check(df):
    prob_true_cal, prob_pred_cal = calibration_curve(df["y_true"], df["proba"], n_bins=10, strategy="quantile")
    prob_true_raw, prob_pred_raw = calibration_curve(df["y_true"], df["proba_raw_uncalibrated"], n_bins=10, strategy="quantile")

    plt.figure(figsize=(7, 6.5))
    plt.plot(prob_pred_raw, prob_true_raw, "o--", color="tab:red", alpha=0.7, label="Before calibration (raw XGBoost)")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", color="tab:blue", label="After isotonic calibration (deployed)")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    plt.xlabel("Mean predicted probability (per bin)")
    plt.ylabel("Observed abusive-return rate (per bin)")
    plt.title("Calibration / Reliability Diagram\n(deciles of predicted risk, quantile-binned)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/calibration_curve.png", dpi=150)
    plt.close()

    max_gap_cal = float(np.max(np.abs(prob_true_cal - prob_pred_cal)))
    max_gap_raw = float(np.max(np.abs(prob_true_raw - prob_pred_raw)))
    print("=== Calibration ===")
    print(f"Max gap BEFORE calibration: {max_gap_raw:.3f}")
    print(f"Max gap AFTER calibration:  {max_gap_cal:.3f}")
    print("(Smaller is better; <0.05-0.10 is generally considered well-calibrated for this kind of model)")
    return {
        "prob_true_calibrated": prob_true_cal.tolist(), "prob_pred_calibrated": prob_pred_cal.tolist(),
        "prob_true_raw": prob_true_raw.tolist(), "prob_pred_raw": prob_pred_raw.tolist(),
        "max_gap_calibrated": max_gap_cal, "max_gap_raw": max_gap_raw,
    }


# ---------------------------------------------------------------------
# Operational fairness / disparate impact
# ---------------------------------------------------------------------
def fairness_breakdown(df, group_col):
    rows = []
    for group, sub in df.groupby(group_col, observed=True):
        n = len(sub)
        true_rate = sub["y_true"].mean()
        flag_rate = sub["flagged"].mean()
        tp = ((sub["flagged"] == 1) & (sub["y_true"] == 1)).sum()
        fp = ((sub["flagged"] == 1) & (sub["y_true"] == 0)).sum()
        fn = ((sub["flagged"] == 0) & (sub["y_true"] == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        over_flag_ratio = flag_rate / true_rate if true_rate > 0 else np.nan
        rows.append({
            "group": group, "n": n, "true_abuse_rate": true_rate, "flag_rate": flag_rate,
            "precision": precision, "recall": recall, "over_flagging_ratio": over_flag_ratio,
        })
    result = pd.DataFrame(rows).sort_values("over_flagging_ratio", ascending=False)
    print(f"\n=== Operational bias check: by {group_col} ===")
    print(result.to_string(index=False))
    return result


def plot_fairness(result_df, group_col, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(result_df))
    width = 0.35
    ax.bar(x - width/2, result_df["true_abuse_rate"], width, label="True abuse rate")
    ax.bar(x + width/2, result_df["flag_rate"], width, label="Model flag rate")
    ax.set_xticks(x)
    ax.set_xticklabels(result_df["group"], rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(f"True abuse rate vs. model flag rate, by {group_col}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df, threshold = load_scored_test()
    print(f"Deployed threshold: {threshold:.2f}\n")

    calib_result = calibration_check(df)

    payment_fairness = fairness_breakdown(df, "payment_method")
    plot_fairness(payment_fairness, "payment_method", "plots/fairness_payment_method.png")

    category_fairness = fairness_breakdown(df, "category")
    plot_fairness(category_fairness, "category", "plots/fairness_category.png")

    tenure_fairness = fairness_breakdown(df, "account_age_band")
    plot_fairness(tenure_fairness, "account_age_band", "plots/fairness_account_tenure.png")

    print("\n=== Interpretation note ===")
    print("An over-flagging ratio above ~1.0 for a group means the model flags that group more "
          "often than its own true abuse rate alone would predict. This CAN be legitimate (other "
          "correlated features may genuinely explain the extra risk) but is worth a human policy "
          "review before deployment, especially for any group tied to a payment channel or "
          "customer segment a business wants to treat consistently.")

    with open("model/calibration_fairness_report.json", "w") as f:
        json.dump({
            "calibration": calib_result,
            "fairness_by_payment_method": payment_fairness.to_dict(orient="records"),
            "fairness_by_category": category_fairness.to_dict(orient="records"),
            "fairness_by_account_tenure": tenure_fairness.to_dict(orient="records"),
            "scope_note": "No demographic attributes exist in this synthetic dataset -- this is "
                          "an operational bias check (payment method, category, tenure), not a "
                          "protected-class fairness audit.",
        }, f, indent=2, default=str)
    print("\nSaved: model/calibration_fairness_report.json")
    print("Saved plots: calibration_curve.png, fairness_payment_method.png, "
          "fairness_category.png, fairness_account_tenure.png")


if __name__ == "__main__":
    main()
