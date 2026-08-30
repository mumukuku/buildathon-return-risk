"""
RiskGuard bootstrap confidence intervals
===========================================

Point estimates (a single precision/recall number) hide how much
uncertainty there is, especially for the abuse-ring sentinel, which has
only ~50 test clusters. This script resamples the held-out test sets
WITH REPLACEMENT many times, recomputes each metric on every resample,
and reports the 2.5th/97.5th percentiles as a 95% confidence interval.

Why this matters for an "honest metrics" bar:
"Precision 1.00" on 14 positive examples sounds airtight but could
plausibly have been "0.85" or "1.00" with a slightly different sample --
the interval makes that visible instead of hiding it behind a clean
point estimate. Wide intervals are not a flaw to hide; they're an
accurate description of how much the number can be trusted, and this
script is what makes that describable at all.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def bootstrap_metric_ci(y_true, y_pred_or_proba, threshold=None, n_boot=N_BOOTSTRAP):
    """Returns dict of metric -> (point_estimate, ci_low, ci_high)."""
    y_true = np.asarray(y_true)
    if threshold is not None:
        y_pred = (np.asarray(y_pred_or_proba) >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred_or_proba)

    n = len(y_true)
    boot_precision, boot_recall, boot_f1 = [], [], []

    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if yt.sum() == 0 or (yp.sum() == 0 and yt.sum() > 0):
            # degenerate resample (no positives, or model predicts nothing) -- skip
            continue
        boot_precision.append(precision_score(yt, yp, zero_division=0))
        boot_recall.append(recall_score(yt, yp, zero_division=0))
        boot_f1.append(f1_score(yt, yp, zero_division=0))

    def summarize(vals, point):
        if len(vals) < 10:
            return (point, None, None)
        return (point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    point_precision = precision_score(y_true, y_pred, zero_division=0)
    point_recall = recall_score(y_true, y_pred, zero_division=0)
    point_f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "n_test": n,
        "n_positive": int(y_true.sum()),
        "precision": summarize(boot_precision, point_precision),
        "recall": summarize(boot_recall, point_recall),
        "f1": summarize(boot_f1, point_f1),
        "n_valid_bootstraps": len(boot_precision),
    }


def fmt(m):
    point, lo, hi = m
    if lo is None:
        return f"{point:.3f} (CI unavailable -- too few valid resamples)"
    return f"{point:.3f}  [95% CI: {lo:.3f} - {hi:.3f}]"


def main():
    results = {}

    # --- Return-risk model (XGBoost, deployed threshold) ---
    test = pd.read_csv("data/test.csv")
    with open("model/feature_columns.json") as f:
        feat_meta = json.load(f)
    X_test = test[feat_meta["encoded_feature_columns"]]
    y_test = test[feat_meta["target"]].values

    xgb = joblib.load("model/deployed_model.pkl")  # calibrated -- must match the deployed threshold's scale
    proba = xgb.predict_proba(X_test)[:, 1]

    with open("model/deployment_config.json") as f:
        deploy_cfg = json.load(f)
    threshold = deploy_cfg["deployed_threshold"]

    return_risk_ci = bootstrap_metric_ci(y_test, proba, threshold=threshold)
    results["return_risk_scorer"] = return_risk_ci

    print("=== Return-Risk Scorer (XGBoost, threshold={:.2f}) ===".format(threshold))
    print(f"n_test={return_risk_ci['n_test']}, n_positive={return_risk_ci['n_positive']}")
    print(f"Precision: {fmt(return_risk_ci['precision'])}")
    print(f"Recall:    {fmt(return_risk_ci['recall'])}")
    print(f"F1:        {fmt(return_risk_ci['f1'])}")

    # --- Abuse-ring sentinel ---
    clusters = pd.read_csv("model/abuse_ring_clusters.csv")
    with open("model/ring_detector_config.json") as f:
        ring_cfg = json.load(f)

    from sklearn.model_selection import train_test_split
    X = clusters[ring_cfg["feature_columns"]]
    y = clusters[ring_cfg["target"]]
    _, X_test_ring, _, y_test_ring = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    ring_model = joblib.load("model/ring_detector_model.pkl")
    ring_proba = ring_model.predict_proba(X_test_ring)[:, 1]
    ring_threshold = ring_cfg["deployed_threshold"]

    ring_ci = bootstrap_metric_ci(y_test_ring.values, ring_proba, threshold=ring_threshold)
    results["abuse_ring_sentinel"] = ring_ci

    print("\n=== Abuse-Ring Sentinel (threshold={:.2f}) ===".format(ring_threshold))
    print(f"n_test={ring_ci['n_test']}, n_positive={ring_ci['n_positive']}  <-- small sample, expect wide intervals")
    print(f"Precision: {fmt(ring_ci['precision'])}")
    print(f"Recall:    {fmt(ring_ci['recall'])}")
    print(f"F1:        {fmt(ring_ci['f1'])}")
    print("NOTE: the interval above is bootstrap resampling of ONE fixed test set that happened "
          "to have zero errors -- resampling a fixed, error-free set can never introduce new "
          "errors, so it trivially returns [1.0, 1.0] and is NOT informative here. See the "
          "repeated-split analysis below instead, which is the honest version of this question.")

    # Repeated random train/test splits: the informative version of "how much can we trust
    # this number" for a small dataset. Unlike bootstrapping one fixed split, this varies
    # WHICH clusters the model has never seen, across many different splits.
    from sklearn.linear_model import LogisticRegression
    n_repeats = 200
    rep_precision, rep_recall, rep_f1 = [], [], []
    for i in range(n_repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=1000 + i, stratify=y
        )
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        preds = (p >= ring_threshold).astype(int)
        if yte.sum() == 0 or preds.sum() == 0:
            continue
        rep_precision.append(precision_score(yte, preds, zero_division=0))
        rep_recall.append(recall_score(yte, preds, zero_division=0))
        rep_f1.append(f1_score(yte, preds, zero_division=0))

    rep_summary = {
        "n_repeats_valid": len(rep_precision),
        "precision": [float(np.mean(rep_precision)), float(np.percentile(rep_precision, 2.5)), float(np.percentile(rep_precision, 97.5))],
        "recall": [float(np.mean(rep_recall)), float(np.percentile(rep_recall, 2.5)), float(np.percentile(rep_recall, 97.5))],
        "f1": [float(np.mean(rep_f1)), float(np.percentile(rep_f1, 2.5)), float(np.percentile(rep_f1, 97.5))],
    }
    results["abuse_ring_sentinel_repeated_splits"] = rep_summary

    print(f"\n=== Abuse-Ring Sentinel: repeated random splits (n={n_repeats}) -- the honest version ===")
    print(f"Precision: {rep_summary['precision'][0]:.3f}  [2.5-97.5 pct: {rep_summary['precision'][1]:.3f} - {rep_summary['precision'][2]:.3f}]")
    print(f"Recall:    {rep_summary['recall'][0]:.3f}  [2.5-97.5 pct: {rep_summary['recall'][1]:.3f} - {rep_summary['recall'][2]:.3f}]")
    print(f"F1:        {rep_summary['f1'][0]:.3f}  [2.5-97.5 pct: {rep_summary['f1'][1]:.3f} - {rep_summary['f1'][2]:.3f}]")

    with open("model/bootstrap_confidence_intervals.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: model/bootstrap_confidence_intervals.json")


if __name__ == "__main__":
    main()
