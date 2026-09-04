"""
RiskGuard API scoring logic (abuse-ring sentinel)
====================================================

The ring detector operates on CLUSTER-level features (not a single
order), so the API here scores a cluster description directly rather
than rebuilding the full customer graph on every request -- that graph
build is a batch job (model/abuse_ring_detector.py), not something to
redo per HTTP request. The API also exposes a few real, precomputed
sample clusters from training (some true rings, some benign) so the
demo has realistic presets to try instead of requiring the user to type
in cluster statistics by hand.
"""

import json
import os
import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

with open(os.path.join(MODELS_DIR, "ring_detector_config.json")) as f:
    RING_CFG = json.load(f)

RING_MODEL = joblib.load(os.path.join(MODELS_DIR, "ring_detector_model.pkl"))
RING_FEATURE_COLS = RING_CFG["feature_columns"]
RING_THRESHOLD = RING_CFG["deployed_threshold"]

SAMPLE_CLUSTERS_DF = pd.read_csv(os.path.join(MODELS_DIR, "abuse_ring_clusters.csv"))

with open(os.path.join(MODELS_DIR, "abuse_ring_members.json")) as f:
    _MEMBERS_BY_CLUSTER = json.load(f)  # keys are cluster_id as strings


# ---------------------------------------------------------------------
# Plain-English explanation (same deterministic, template-based approach
# as the return-risk model -- see api/lib/scoring.py for the reasoning)
# ---------------------------------------------------------------------
RING_FEATURE_TEMPLATES = {
    "size": ("this is a larger cluster of linked accounts", "this is a small cluster of linked accounts"),
    "shares_device": ("these accounts share the same device", "these accounts do not share a device"),
    "shares_payment": ("these accounts share the same payment method", "these accounts do not share a payment method"),
    "shares_address": ("these accounts share the same address", "these accounts do not share an address"),
    "shares_device_and_payment": (
        "these accounts share both device and payment method, a strong ring signal",
        "these accounts do not share both device and payment together",
    ),
    "avg_account_age": ("the accounts are relatively old on average", "the accounts are relatively new on average"),
    "account_age_std": ("account ages vary widely within the cluster", "account ages are similar within the cluster"),
    "avg_return_rate": ("the cluster has an unusually high average return rate", "the cluster has a normal average return rate"),
    "max_return_rate": ("at least one member has an extremely high return rate", "no member has an extreme return rate"),
    "avg_abusive_return_rate": ("the cluster has a history of abusive returns", "the cluster has little history of abusive returns"),
    "max_abusive_return_rate": ("at least one member has a high abusive-return rate", "no member has a high abusive-return rate"),
    "avg_total_orders": ("members have placed many orders on average", "members have placed few orders on average"),
}


def describe_ring_feature(feature: str, impact: float) -> str:
    if feature in RING_FEATURE_TEMPLATES:
        toward_ring, toward_benign = RING_FEATURE_TEMPLATES[feature]
        return toward_ring if impact >= 0 else toward_benign
    return feature.replace("_", " ")


def ring_plain_english_explanation(factors: list, verdict: str, top_n: int = 3) -> str:
    lead = "Flagged as a likely ring" if verdict == "likely_ring" else "Assessed as likely benign sharing"
    clauses = [describe_ring_feature(f["feature"], f["impact"]) for f in factors[:top_n]]
    if len(clauses) == 1:
        joined = clauses[0]
    elif len(clauses) == 2:
        joined = f"{clauses[0]}, and {clauses[1]}"
    else:
        joined = f"{', '.join(clauses[:-1])}, and {clauses[-1]}"
    return f"{lead} mainly because: {joined}."


def score_cluster(cluster: dict) -> dict:
    row = {col: cluster.get(col, 0) for col in RING_FEATURE_COLS}
    X_row = pd.DataFrame([row], columns=RING_FEATURE_COLS)
    ring_score = float(RING_MODEL.predict_proba(X_row)[:, 1][0])
    verdict = "likely_ring" if ring_score >= RING_THRESHOLD else "likely_benign"

    coefs = dict(zip(RING_FEATURE_COLS, RING_MODEL.coef_[0]))
    contributions = sorted(
        [
            {
                "feature": f,
                "value": row[f],
                "coefficient": float(coefs[f]),
                "impact": float(coefs[f] * row[f]) if isinstance(row[f], (int, float)) else float(coefs[f]),
            }
            for f in RING_FEATURE_COLS
        ],
        key=lambda c: abs(c["impact"]),
        reverse=True,
    )
    explanation = ring_plain_english_explanation(contributions, verdict)

    return {
        "ring_score": round(ring_score, 4),
        "threshold_used": RING_THRESHOLD,
        "verdict": verdict,
        "top_factors": contributions[:5],
        "plain_english_summary": explanation,
    }


def get_sample_clusters(n=6) -> list:
    """A mix of true rings and benign clusters from training, for the demo UI to offer as presets."""
    rings = SAMPLE_CLUSTERS_DF[SAMPLE_CLUSTERS_DF["is_true_ring"] == 1].sample(
        min(n // 2, (SAMPLE_CLUSTERS_DF["is_true_ring"] == 1).sum()), random_state=7
    )
    benign = SAMPLE_CLUSTERS_DF[SAMPLE_CLUSTERS_DF["is_true_ring"] == 0].sample(
        min(n - len(rings), (SAMPLE_CLUSTERS_DF["is_true_ring"] == 0).sum()), random_state=7
    )
    combined = pd.concat([rings, benign]).sample(frac=1, random_state=7)  # shuffle
    records = combined[RING_FEATURE_COLS + ["cluster_id", "is_true_ring"]].to_dict(orient="records")

    # Attach real member details -- who is actually IN this cluster, not just
    # aggregate stats. This is what makes the demo concrete: you can see the
    # actual account IDs and their individual return histories, not just
    # "size=5, avg_return_rate=0.4".
    for record in records:
        record["members"] = _MEMBERS_BY_CLUSTER.get(str(record["cluster_id"]), [])
    return records
