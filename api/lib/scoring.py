"""
RiskGuard API scoring logic (return-risk model)
==================================================

Loaded once per serverless function cold start, reused across warm
invocations. Encodes a raw order dict into the model's feature space
EXACTLY the way prepare_features.py did during training (same one-hot
columns, same order, zero-filled if a category/reason wasn't seen),
scores it with the calibrated model, and returns a decision + a
SHAP-equivalent explanation.

NOTE ON EXPLAINABILITY: this deliberately does NOT import the `shap`
package. `import shap` transitively pulls in numba (which bundles a full
LLVM via llvmlite -- commonly 150-250MB installed), plus matplotlib and
its font/backend stack, none of which are needed for scoring a single
order. That pushed our first Vercel deployment to an 889MB bundle,
blowing past the platform's 500MB Python function limit.

Instead we use XGBoost's OWN native `pred_contribs=True` prediction
mode, which computes the exact same Tree SHAP algorithm inside XGBoost's
C++ core -- mathematically identical output to shap.TreeExplainer for a
tree model, with zero extra dependencies. This is not an approximation
or a downgrade; it's the same numbers via a lighter path.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
    FEATURE_META = json.load(f)
with open(os.path.join(MODELS_DIR, "deployment_config.json")) as f:
    DEPLOY_CFG = json.load(f)

DEPLOYED_MODEL = joblib.load(os.path.join(MODELS_DIR, "deployed_model.pkl"))  # calibrated -- for the score
RAW_XGB_MODEL = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))       # uncalibrated tree -- for contributions
_BOOSTER = RAW_XGB_MODEL.get_booster()

ENCODED_COLS = FEATURE_META["encoded_feature_columns"]
NUMERIC_COLS = FEATURE_META["numeric_features"]
CATEGORICAL_COLS = FEATURE_META["categorical_features"]
THRESHOLD = DEPLOY_CFG["deployed_threshold"]


def encode_order(order: dict) -> pd.DataFrame:
    """Turn a raw order dict into a single-row DataFrame matching ENCODED_COLS exactly."""
    row = {col: 0 for col in ENCODED_COLS}
    for col in NUMERIC_COLS:
        row[col] = order.get(col, 0)
    for cat_col in CATEGORICAL_COLS:
        value = order.get(cat_col)
        one_hot_col = f"{cat_col}_{value}"
        if one_hot_col in row:
            row[one_hot_col] = 1
        # unseen category -> all-zero for that group, which is a defined (if
        # imperfect) fallback rather than an error; the model just sees no
        # signal from that categorical field.
    return pd.DataFrame([row], columns=ENCODED_COLS)


def top_shap_factors(X_row: pd.DataFrame, n=5):
    dmat = xgb.DMatrix(X_row, feature_names=list(X_row.columns))
    contribs = _BOOSTER.predict(dmat, pred_contribs=True)[0]  # last entry is the bias term
    feature_contribs = contribs[:-1]
    contributions = list(zip(X_row.columns, feature_contribs))
    contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    return [
        {"feature": feat, "value": float(X_row.iloc[0][feat]), "impact": float(impact)}
        for feat, impact in contributions[:n]
    ]


def decide_action(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "auto_decline"
    if risk_score >= THRESHOLD:
        return "manual_review"
    return "approve"


def score_order(order: dict) -> dict:
    X_row = encode_order(order)
    risk_score = float(DEPLOYED_MODEL.predict_proba(X_row)[:, 1][0])
    action = decide_action(risk_score)
    factors = top_shap_factors(X_row)

    # Audit record: in production this would be persisted to a durable,
    # queryable store (a database or a service like Vercel KV) for
    # compliance/dispute review. Here it's returned in the response itself
    # so the demo can display/export it -- the SHAPE of the record is the
    # point, not where it's stored.
    record = {
        "decision_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": order,
        "risk_score": round(risk_score, 4),
        "threshold_used": THRESHOLD,
        "action": action,
        "top_factors": factors,
        "model_version": DEPLOY_CFG.get("deployed_model", "unknown"),
    }
    return record
