"""
RiskGuard model training + evaluation
=======================================

Trains two models on the return-risk task and evaluates both HONESTLY on
the held-out (chronological) test set:

    1. Logistic Regression -- fully interpretable, coefficients read
       directly as feature contributions.
    2. XGBoost              -- captures feature interactions, typically
                                stronger on tabular fraud-style data.

We report both. We DEPLOY XGBoost (see README for the reasoning: on this
task it meaningfully outperforms logistic regression, and we recover
explainability via SHAP rather than by picking the weaker model).

This script also implements the piece the track explicitly asks for:
a COST-SENSITIVE THRESHOLD, not just a default 0.5 cutoff. A fraud
classifier's job isn't "get the highest accuracy" -- it's "minimize the
merchant's total dollar loss," and false positives and false negatives
have very different, documented costs here:

    - False Positive cost  = FP_REVIEW_COST (flat)
        A genuine return gets flagged for manual review / friction.
        Assumption: this costs the merchant a flat review/ops cost plus
        customer-goodwill risk. We use Rs. 150 as a placeholder -- this
        number should be replaced with a merchant's real support-ticket
        cost + estimated churn risk in a production setting.

    - False Negative cost  = the order's value (order_value)
        A genuinely abusive return slips through undetected. Assumption:
        the merchant's loss is proportional to the value of the item
        being fraudulently returned (not a flat number), since the
        actual financial exposure scales with order size. This is more
        realistic than a single flat FN cost.

We sweep decision thresholds and report the TOTAL EXPECTED COST at each
one, picking the threshold that minimizes it -- then compare that to the
naive default of 0.5, to show explicitly how much a cost-aware threshold
saves versus an accuracy-only mindset.

All numbers here are ASSUMPTIONS, clearly labeled. Swap in real costs if
you have them; the machinery (sweep + argmin) doesn't change.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, average_precision_score,
)
from xgboost import XGBClassifier
import shap

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
FEATURE_LIST_PATH = "model/feature_columns.json"
PLOTS_DIR = "plots"

FP_REVIEW_COST = 150.0  # Rs. -- flat cost of a false positive (see docstring)

with open(FEATURE_LIST_PATH) as f:
    feat_meta = json.load(f)
FEATURE_COLS = feat_meta["encoded_feature_columns"]
TARGET = feat_meta["target"]


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]
    return train, test, X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------
# 1. Train both models
# ---------------------------------------------------------------------
def train_models(X_train, y_train):
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)

    # class imbalance ratio for XGBoost's scale_pos_weight
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=42,
    )
    xgb.fit(X_train, y_train)

    return logreg, xgb


# ---------------------------------------------------------------------
# 2. Standard evaluation: PR curve, ROC, average precision
# ---------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, pr_thresholds = precision_recall_curve(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_test, proba)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc_auc:.3f} | Average Precision (PR-AUC): {ap:.3f}")
    preds_05 = (proba >= 0.5).astype(int)
    print("Classification report @ threshold 0.5:")
    print(classification_report(y_test, preds_05, digits=3))

    return {
        "proba": proba, "precision": precision, "recall": recall,
        "pr_thresholds": pr_thresholds, "fpr": fpr, "tpr": tpr,
        "roc_auc": roc_auc, "avg_precision": ap,
    }


def plot_pr_and_roc(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results.items():
        axes[0].plot(r["recall"], r["precision"], label=f"{name} (AP={r['avg_precision']:.3f})")
        axes[1].plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")

    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 3. Cost-sensitive threshold sweep (the key "false-positive cost" deliverable)
# ---------------------------------------------------------------------
def cost_sweep(proba, y_test, order_values, thresholds=None, max_flag_rate=0.20):
    """
    Sweeps thresholds and reports total expected cost at each one.

    IMPORTANT FINDING (kept in, not smoothed over): minimizing the flat-FP /
    value-weighted-FN cost function with NO other constraint pushes the
    threshold down to ~0.04, which flags roughly 80% of all returns for
    review. That is mathematically "cheapest" under this cost function, but
    operationally absurd -- no fraud-ops team can manually review 80% of
    returns, and a flat Rs.150 review cost does not capture the real cost of
    treating most genuine customers as suspects (trust erosion, team
    overload, brand damage aren't linear in flag count).

    So we report TWO thresholds:
      - "unconstrained_optimal": minimizes total_cost with no other limits.
      - "capacity_constrained_optimal": minimizes total_cost subject to
        flag_rate <= max_flag_rate (default 20% of all returns), a stand-in
        for a fraud-ops team's real review capacity. This is the one we
        actually deploy -- see README for the reasoning.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n_total = len(y_test)
    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        fp_mask = (preds == 1) & (y_test.values == 0)
        fn_mask = (preds == 0) & (y_test.values == 1)
        tp_mask = (preds == 1) & (y_test.values == 1)

        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        tp_count = tp_mask.sum()
        fp_cost = fp_count * FP_REVIEW_COST
        fn_cost = order_values.values[fn_mask].sum()  # value-weighted, not flat
        total_cost = fp_cost + fn_cost
        flag_rate = (fp_count + tp_count) / n_total

        rows.append({
            "threshold": t, "fp_count": int(fp_count), "fn_count": int(fn_count),
            "tp_count": int(tp_count), "fp_cost": fp_cost, "fn_cost": fn_cost,
            "total_cost": total_cost, "flag_rate": flag_rate,
        })

    cost_df = pd.DataFrame(rows)
    unconstrained_best = cost_df.loc[cost_df["total_cost"].idxmin()]

    feasible = cost_df[cost_df["flag_rate"] <= max_flag_rate]
    constrained_best = feasible.loc[feasible["total_cost"].idxmin()] if len(feasible) else unconstrained_best

    return cost_df, unconstrained_best, constrained_best


def plot_cost_curve(cost_df, unconstrained_best, constrained_best, out_path):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(cost_df["threshold"], cost_df["total_cost"], color="tab:blue", label="Total expected cost")
    ax1.axvline(unconstrained_best["threshold"], color="orange", linestyle="--",
                label=f"Unconstrained optimum = {unconstrained_best['threshold']:.2f} "
                      f"(flags {unconstrained_best['flag_rate']:.0%} of returns)")
    ax1.axvline(constrained_best["threshold"], color="green", linestyle="--",
                label=f"Capacity-constrained optimum = {constrained_best['threshold']:.2f} "
                      f"(flags {constrained_best['flag_rate']:.0%} of returns) -- DEPLOYED")
    ax1.axvline(0.5, color="red", linestyle=":", label="Naive default = 0.50")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Total expected cost (Rs.)", color="tab:blue")
    ax1.set_title("Cost-sensitive threshold selection\n(FP = flat review cost, FN = value-weighted loss, "
                   "constrained by review capacity)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cost_df["threshold"], cost_df["flag_rate"], color="gray", alpha=0.5, label="Flag rate")
    ax2.set_ylabel("Flag rate (% of returns sent to review)", color="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(y_test, proba, threshold, out_path, title_suffix=""):
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Pred: Genuine", "Pred: Abusive"])
    plt.yticks([0, 1], ["True: Genuine", "True: Abusive"])
    plt.title(f"Confusion Matrix @ threshold={threshold:.2f}{title_suffix}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 4. SHAP explainability for the deployed model (XGBoost)
# ---------------------------------------------------------------------
def plot_shap_summary(xgb_model, X_test, out_path, sample_n=500):
    sample = X_test.sample(min(sample_n, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample)
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 5. Baseline: naive rule-based heuristic (what a merchant might do without ML)
# ---------------------------------------------------------------------
def baseline_heuristic_predictions(test_df: pd.DataFrame) -> np.ndarray:
    """
    A simple, fixed-threshold rule a fraud-ops team might use without any
    ML at all -- no tuning, no probability, just business logic:

        Flag if ANY of:
          (a) returned within 3 days AND priced well above category average
              (classic "fast expensive return" red flag)
          (b) customer's historical abusive-return rate is already high
              (known repeat offender)
          (c) customer's overall historical return rate is very high
              (serial returner / bracketer, even if not yet confirmed abusive)

    This is the honest comparison point: if our ML model can't beat this,
    the ML isn't earning its complexity.
    """
    fast_expensive = (test_df["days_to_return"] <= 3) & (test_df["price_vs_category_avg"] >= 1.5)
    known_abuser = test_df["hist_abusive_return_rate_before"] >= 0.4
    serial_returner = test_df["hist_return_rate_before"] >= 0.6
    flagged = fast_expensive | known_abuser | serial_returner
    return flagged.astype(int).values


def compare_to_baseline(test_df, y_test, ml_proba, ml_threshold, order_values):
    baseline_preds = baseline_heuristic_predictions(test_df)

    def summarize(preds, name):
        tp = ((preds == 1) & (y_test.values == 1)).sum()
        fp = ((preds == 1) & (y_test.values == 0)).sum()
        fn = ((preds == 0) & (y_test.values == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        flag_rate = (preds == 1).mean()
        fp_cost = fp * FP_REVIEW_COST
        fn_cost = order_values.values[(preds == 0) & (y_test.values == 1)].sum()
        total_cost = fp_cost + fn_cost
        return {"name": name, "precision": precision, "recall": recall, "f1": f1,
                "flag_rate": flag_rate, "total_cost": total_cost, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    ml_preds = (ml_proba >= ml_threshold).astype(int)

    rows = [
        summarize(baseline_preds, "Rule-based baseline"),
        summarize(ml_preds, "XGBoost (capacity-constrained threshold)"),
    ]
    comparison_df = pd.DataFrame(rows)
    print("\n=== Baseline vs. ML comparison (held-out test set) ===")
    print(comparison_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_to_plot = ["precision", "recall", "f1"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    ax.bar(x - width/2, comparison_df.iloc[0][metrics_to_plot], width, label="Rule-based baseline")
    ax.bar(x + width/2, comparison_df.iloc[1][metrics_to_plot], width, label="XGBoost (ours)")
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_title("ML model vs. naive rule-based baseline")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(metrics_to_plot):
        for j, row_idx in enumerate([0, 1]):
            val = comparison_df.iloc[row_idx][m]
            ax.text(i + (j - 0.5) * width, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/baseline_comparison.png", dpi=150)
    plt.close()

    comparison_df.to_csv("model/baseline_comparison.csv", index=False)
    return comparison_df


def main():
    train, test, X_train, y_train, X_test, y_test = load_data()

    logreg, xgb = train_models(X_train, y_train)

    results = {
        "Logistic Regression": evaluate_model("Logistic Regression", logreg, X_test, y_test),
        "XGBoost": evaluate_model("XGBoost", xgb, X_test, y_test),
    }
    plot_pr_and_roc(results, f"{PLOTS_DIR}/pr_roc_comparison.png")

    # Cost-sensitive threshold using the DEPLOYED model (XGBoost)
    deployed_proba = results["XGBoost"]["proba"]
    order_values = test["order_value"]
    cost_df, unconstrained_best, constrained_best = cost_sweep(
        deployed_proba, y_test, order_values, max_flag_rate=0.20
    )
    cost_df.to_csv("model/cost_sweep.csv", index=False)
    plot_cost_curve(cost_df, unconstrained_best, constrained_best, f"{PLOTS_DIR}/cost_curve.png")

    optimal_threshold = float(constrained_best["threshold"])  # what we actually deploy
    naive_cost = cost_df.loc[cost_df["threshold"].sub(0.5).abs().idxmin(), "total_cost"]

    print(f"\n=== Cost-sensitive threshold (deployed model: XGBoost) ===")
    print(f"Unconstrained optimum: threshold={unconstrained_best['threshold']:.2f}, "
          f"cost=Rs.{unconstrained_best['total_cost']:,.0f}, "
          f"flag_rate={unconstrained_best['flag_rate']:.1%}  <- mathematically cheapest, "
          f"operationally unrealistic (flags most returns)")
    print(f"Capacity-constrained optimum (<=20% flag rate) -- DEPLOYED: "
          f"threshold={optimal_threshold:.2f}, cost=Rs.{constrained_best['total_cost']:,.0f}, "
          f"flag_rate={constrained_best['flag_rate']:.1%}")
    print(f"Naive default (0.5): cost=Rs.{naive_cost:,.0f}")
    print(f"Savings vs naive default: Rs. {naive_cost - constrained_best['total_cost']:,.0f}")

    plot_confusion(y_test, deployed_proba, optimal_threshold,
                    f"{PLOTS_DIR}/confusion_optimal.png", " (capacity-constrained optimal)")
    plot_confusion(y_test, deployed_proba, 0.5,
                    f"{PLOTS_DIR}/confusion_naive.png", " (naive default)")

    plot_shap_summary(xgb, X_test, f"{PLOTS_DIR}/shap_summary.png")

    # Baseline comparison
    baseline_df = compare_to_baseline(test, y_test, deployed_proba, optimal_threshold, order_values)

    # Save artifacts
    joblib.dump(logreg, "model/logreg_model.pkl")
    joblib.dump(xgb, "model/xgb_model.pkl")
    joblib.dump(xgb, "model/deployed_model.pkl")  # XGBoost is deployed

    with open("model/deployment_config.json", "w") as f:
        json.dump({
            "deployed_model": "xgboost",
            "deployed_threshold": optimal_threshold,
            "deployed_threshold_basis": "capacity_constrained_optimal (max 20% flag rate)",
            "unconstrained_optimal_threshold": float(unconstrained_best["threshold"]),
            "unconstrained_optimal_flag_rate": float(unconstrained_best["flag_rate"]),
            "fp_review_cost_rs": FP_REVIEW_COST,
            "fn_cost_basis": "order_value (value-weighted)",
            "max_flag_rate_assumption": 0.20,
            "actions": {
                "below_threshold": "approve",
                "at_or_above_threshold_below_high": "manual_review",
                "high_confidence": "auto_decline",
            },
            "roc_auc": results["XGBoost"]["roc_auc"],
            "avg_precision": results["XGBoost"]["avg_precision"],
        }, f, indent=2)

    print("\nSaved: model/logreg_model.pkl, model/xgb_model.pkl, model/deployed_model.pkl")
    print("Saved: model/deployment_config.json, model/cost_sweep.csv")
    print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
