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


# ---------------------------------------------------------------------
# Plain-English explanation
# ---------------------------------------------------------------------
# Deliberately TEMPLATE-BASED, not LLM-generated. A risk decision needs an
# explanation that is reproducible and auditable -- the same inputs must
# always produce the same explanation, with no chance of an LLM paraphrasing
# differently on retry or, worse, hallucinating a reason the model didn't
# actually use. Every phrase below is a direct, deterministic translation of
# a real feature contribution, never a generated summary of one.
FEATURE_TEMPLATES = {
    "hist_abusive_return_rate_before": (
        "this customer has a history of abusive returns",
        "this customer has no history of abusive returns",
    ),
    "hist_return_rate_before": (
        "this customer returns items unusually often",
        "this customer rarely returns items",
    ),
    "price_vs_category_avg": (
        "the item is priced well above similar products (a wardrobing red flag)",
        "the item is priced in line with similar products",
    ),
    "days_to_return": (
        "the return came in close to the return-window deadline",
        "the return came in quickly after purchase",
    ),
    "hist_chargebacks_before": (
        "this customer has past chargebacks on file",
        "this customer has no chargeback history",
    ),
    "hist_orders_before": (
        "this is a newer account with limited order history",
        "this customer has an established order history",
    ),
    "account_age_days_at_order": (
        "the account is relatively new",
        "the account is well-established",
    ),
    "order_value": (
        "this is a high-value order",
        "this is a lower-value order",
    ),
    "delivery_days": (
        "delivery took longer than usual",
        "delivery was quick",
    ),
    "order_hour": (
        "the order was placed at an unusual hour",
        "the order was placed at a typical hour",
    ),
    "is_weekend": (
        "the order was placed on a weekend",
        "the order was placed on a weekday",
    ),
}
ONEHOT_PREFIX_LABELS = {
    "category_": "the item category is",
    "payment_method_": "payment was made via",
    "return_reason_": "the stated return reason was",
}


def describe_feature(feature: str, impact: float) -> str:
    """Translate one feature contribution into a plain-English clause."""
    if feature in FEATURE_TEMPLATES:
        risk_up_text, risk_down_text = FEATURE_TEMPLATES[feature]
        return risk_up_text if impact >= 0 else risk_down_text
    for prefix, label in ONEHOT_PREFIX_LABELS.items():
        if feature.startswith(prefix):
            name = feature[len(prefix):].replace("_", " ")
            return f"{label} '{name}'"
    return feature.replace("_", " ")  # fallback for anything unmapped


def plain_english_explanation(factors: list, action: str, top_n: int = 3) -> str:
    lead = {
        "approve": "Approved",
        "manual_review": "Sent to manual review",
        "auto_decline": "Auto-declined",
    }[action]

    clauses = [describe_feature(f["feature"], f["impact"]) for f in factors[:top_n]]
    if len(clauses) == 1:
        joined = clauses[0]
    elif len(clauses) == 2:
        joined = f"{clauses[0]}, and {clauses[1]}"
    else:
        joined = f"{', '.join(clauses[:-1])}, and {clauses[-1]}"

    return f"{lead} mainly because: {joined}."


def score_order(order: dict) -> dict:
    X_row = encode_order(order)
    risk_score = float(DEPLOYED_MODEL.predict_proba(X_row)[:, 1][0])
    action = decide_action(risk_score)
    factors = top_shap_factors(X_row)
    explanation = plain_english_explanation(factors, action)

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
        "plain_english_summary": explanation,
        "model_version": DEPLOY_CFG.get("deployed_model", "unknown"),
    }
    return record
