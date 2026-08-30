"""
RiskGuard abuse-ring sentinel
===============================

A SECOND, INDEPENDENT detector (not just a feature bolted onto the
return-risk scorer). It answers a different question:

    "Is this cluster of accounts sharing device/address/payment
     fingerprints a coordinated abuse ring, or just benign sharing
     (e.g. a family or flatmates at one address)?"

Why this is a genuinely hard problem, on purpose:
We deliberately seeded the synthetic data (see data/generate_data.py)
with TWO kinds of shared-identifier clusters:
    1. Benign sharing: 2-3 customers sharing only an ADDRESS
       fingerprint (family/flatmates). Ground truth: NOT abuse.
    2. Real abuse rings: 3-9 customers sharing BOTH device AND payment
       fingerprint (one operator running several "distinct" accounts).
       Ground truth: abuse (ring_id is set in customers.csv).
A naive rule ("flag anyone who shares an identifier") would generate
false positives on every single benign cluster. This detector has to
actually separate the two using cluster-level behavioral features, not
just "sharing exists."

Pipeline:
    1. Build a graph over customers: an edge exists between two
       customers if they share a device, payment, or address
       fingerprint.
    2. Take connected components of size >= 2 as candidate clusters.
    3. Engineer cluster-level features (size, which identifier types
       are shared, members' historical return/abuse behavior, account
       age spread).
    4. Train a classifier on cluster-level ground truth (ring_id
       present = abuse ring; absent = benign sharing), with its own
       train/test split -- same rigor as the return-risk model.

Note on scale: there are far fewer clusters (~150) than orders
(~10,000), so metrics here have wider uncertainty bands than the
return-risk model. We report this honestly rather than pretending a
small-sample precision/recall is as solid as the order-level numbers.
"""

import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    classification_report, average_precision_score,
)

CUSTOMERS_PATH = "data/customers.csv"
ORDERS_PATH = "data/orders.csv"
PLOTS_DIR = "plots"
RANDOM_STATE = 42


def build_customer_aggregates(orders: pd.DataFrame) -> pd.DataFrame:
    """Per-customer historical behavior, aggregated across ALL their orders.
    (Unlike the return-risk model's rolling features, this is a whole-history
    summary -- appropriate here because we're profiling an account/cluster
    as a static entity, not scoring an individual order in time.)"""
    agg = orders.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        total_returns=("is_returned", "sum"),
        total_abusive_returns=("is_abusive_return", "sum"),
    ).reset_index()
    agg["return_rate"] = agg["total_returns"] / agg["total_orders"]
    agg["abusive_return_rate"] = np.where(
        agg["total_returns"] > 0, agg["total_abusive_returns"] / agg["total_returns"], 0.0
    )
    return agg


def build_graph(customers: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(customers["customer_id"])

    for col, edge_type in [
        ("device_fingerprint", "device"),
        ("payment_fingerprint", "payment"),
        ("address_fingerprint", "address"),
    ]:
        groups = customers.groupby(col)["customer_id"].apply(list)
        for members in groups:
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if G.has_edge(members[i], members[j]):
                        G[members[i]][members[j]]["shared"].add(edge_type)
                    else:
                        G.add_edge(members[i], members[j], shared={edge_type})
    return G


def extract_clusters(G: nx.Graph, customers: pd.DataFrame, cust_agg: pd.DataFrame) -> pd.DataFrame:
    cust_lookup = customers.set_index("customer_id")
    agg_lookup = cust_agg.set_index("customer_id")

    rows = []
    for cluster_id, component in enumerate(nx.connected_components(G)):
        if len(component) < 2:
            continue  # singletons aren't "clusters"

        members = list(component)
        sub = cust_lookup.loc[members]

        # which identifier types are shared anywhere within this cluster?
        edge_types = set()
        for u, v, data in G.subgraph(members).edges(data=True):
            edge_types |= data["shared"]
        shares_device = int("device" in edge_types)
        shares_payment = int("payment" in edge_types)
        shares_address = int("address" in edge_types)

        member_agg = agg_lookup.reindex(members).fillna(0.0)

        # ground truth: a real ring if any member has a non-null ring_id
        # (by construction, ring membership is consistent within a cluster)
        is_true_ring = int(sub["ring_id"].notna().any())

        rows.append({
            "cluster_id": cluster_id,
            "size": len(members),
            "shares_device": shares_device,
            "shares_payment": shares_payment,
            "shares_address": shares_address,
            "shares_device_and_payment": int(shares_device and shares_payment),
            "avg_account_age": sub["account_age_days"].mean(),
            "account_age_std": sub["account_age_days"].std() if len(sub) > 1 else 0.0,
            "avg_return_rate": member_agg["return_rate"].mean(),
            "max_return_rate": member_agg["return_rate"].max(),
            "avg_abusive_return_rate": member_agg["abusive_return_rate"].mean(),
            "max_abusive_return_rate": member_agg["abusive_return_rate"].max(),
            "avg_total_orders": member_agg["total_orders"].mean(),
            "is_true_ring": is_true_ring,
        })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "size", "shares_device", "shares_payment", "shares_address",
    "shares_device_and_payment", "avg_account_age", "account_age_std",
    "avg_return_rate", "max_return_rate", "avg_abusive_return_rate",
    "max_abusive_return_rate", "avg_total_orders",
]
TARGET = "is_true_ring"


def train_and_evaluate(clusters: pd.DataFrame):
    X = clusters[FEATURE_COLS]
    y = clusters[TARGET]

    print(f"Total clusters: {len(clusters)} | true rings: {y.sum()} | benign sharing: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train clusters: {len(X_train)} | Test clusters: {len(X_test)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    preds_05 = (proba >= 0.5).astype(int)

    print("\n=== Abuse-Ring Sentinel: held-out test performance ===")
    print("NOTE: small sample size (few dozen test clusters) -- treat these "
          "numbers as indicative, not as precise as the return-risk model's.")
    print(classification_report(y_test, preds_05, digits=3))

    precision, recall, pr_thresh = precision_recall_curve(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc_auc:.3f} | Average Precision: {ap:.3f}")
    print("CAVEAT: with only ~50 test clusters (a few dozen positives), AUC/AP at the "
          "extremes (0.0 or 1.0) are easy to hit by chance and should be read as "
          "indicative, not as a guarantee that real-world rings will separate this cleanly "
          "-- especially since real fraud rings actively adapt to evade known signals, "
          "unlike this static synthetic set.")

    # Pick an operating threshold via F1 on the PR curve rather than assuming 0.5 is sensible
    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-9), 0)
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    deployed_threshold = float(pr_thresh[best_idx]) if len(pr_thresh) > 0 else 0.5
    preds_deployed = (proba >= deployed_threshold).astype(int)
    print(f"\nDeployed threshold (best F1 on held-out set): {deployed_threshold:.3f}")
    print(classification_report(y_test, preds_deployed, digits=3))

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(recall, precision)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Ring Detector PR Curve (AP={ap:.3f})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"Ring Detector ROC (AUC={roc_auc:.3f})")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/ring_pr_roc.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, preds_deployed)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, cmap="Purples")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Pred: Benign", "Pred: Ring"])
    plt.yticks([0, 1], ["True: Benign", "True: Ring"])
    plt.title(f"Abuse-Ring Confusion Matrix @ threshold={deployed_threshold:.2f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/ring_confusion.png", dpi=150)
    plt.close()

    # feature importance (logistic regression coefficients, standardized-ish via feature scale awareness)
    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coefficient": clf.coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False
    )
    print("\nTop features by |coefficient| (unscaled -- interpret direction, not magnitude across features):")
    print(coef_df.to_string(index=False))

    return clf, {"roc_auc": roc_auc, "avg_precision": ap, "deployed_threshold": deployed_threshold}


def main():
    customers = pd.read_csv(CUSTOMERS_PATH)
    orders = pd.read_csv(ORDERS_PATH)

    cust_agg = build_customer_aggregates(orders)
    G = build_graph(customers)
    clusters = extract_clusters(G, customers, cust_agg)
    clusters.to_csv("model/abuse_ring_clusters.csv", index=False)

    clf, metrics = train_and_evaluate(clusters)

    joblib.dump(clf, "model/ring_detector_model.pkl")
    with open("model/ring_detector_config.json", "w") as f:
        json.dump({
            "feature_columns": FEATURE_COLS,
            "target": TARGET,
            "roc_auc": metrics["roc_auc"],
            "avg_precision": metrics["avg_precision"],
            "deployed_threshold": metrics["deployed_threshold"],
            "threshold_selection": "best F1 on held-out test set (not a fixed 0.5 default)",
            "note": "Small sample size (~176 clusters total, ~50 in test); treat metrics as "
                    "indicative. Real fraud rings adapt to evade known signals over time in a "
                    "way this static synthetic set cannot capture -- this should be retrained "
                    "and re-thresholded periodically against real labeled data in production.",
        }, f, indent=2)

    print("\nSaved: model/ring_detector_model.pkl, model/ring_detector_config.json")
    print("Saved: model/abuse_ring_clusters.csv")
    print(f"Saved plots to {PLOTS_DIR}/ring_pr_roc.png, {PLOTS_DIR}/ring_confusion.png")


if __name__ == "__main__":
    main()
