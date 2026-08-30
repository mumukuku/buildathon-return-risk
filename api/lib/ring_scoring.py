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


def score_cluster(cluster: dict) -> dict:
    row = {col: cluster.get(col, 0) for col in RING_FEATURE_COLS}
    X_row = pd.DataFrame([row], columns=RING_FEATURE_COLS)
    ring_score = float(RING_MODEL.predict_proba(X_row)[:, 1][0])
    verdict = "likely_ring" if ring_score >= RING_THRESHOLD else "likely_benign"

    coefs = dict(zip(RING_FEATURE_COLS, RING_MODEL.coef_[0]))
    contributions = sorted(
        [{"feature": f, "value": row[f], "coefficient": float(coefs[f])} for f in RING_FEATURE_COLS],
        key=lambda c: abs(c["coefficient"] * (c["value"] if isinstance(c["value"], (int, float)) else 1)),
        reverse=True,
    )

    return {
        "ring_score": round(ring_score, 4),
        "threshold_used": RING_THRESHOLD,
        "verdict": verdict,
        "top_factors": contributions[:5],
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
    return records
