"""
RiskGuard API
==============
FastAPI app deployed as a Vercel Python serverless function. Routes:

    GET  /api/health              - liveness check
    POST /api/score               - score a single order (return-risk model)
    POST /api/abuse-check         - score a cluster description (ring sentinel)
    GET  /api/sample-clusters     - a few real example clusters for the demo UI
    GET  /api/metrics             - precomputed evaluation metrics for the dashboard tab

Note on the /api/score and /api/abuse-check "audit records" in the
response: in a real deployment these would be persisted to a durable,
queryable store for compliance/dispute review, not just returned once.
This demo returns the full record so the frontend can display/export
it -- see api/lib/scoring.py for the reasoning.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from lib.scoring import score_order
from lib.ring_scoring import score_cluster, get_sample_clusters

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

app = FastAPI(title="RiskGuard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrderInput(BaseModel):
    order_value: float = Field(..., description="Order value in Rs.")
    category: str = Field(..., description="apparel|footwear|electronics|mobile|beauty|home|accessories")
    payment_method: str = Field(..., description="UPI|card|COD|wallet")
    return_reason: str = Field(..., description="size_issue|changed_mind|damaged|not_as_described|wrong_item|no_longer_needed")
    delivery_days: int = 3
    order_hour: int = 12
    is_weekend: int = 0
    account_age_days_at_order: int = 100
    hist_orders_before: int = 0
    hist_return_rate_before: float = 0.0
    hist_abusive_return_rate_before: float = 0.0
    hist_chargebacks_before: int = 0
    price_vs_category_avg: float = 1.0
    days_to_return: int = 5


class ClusterInput(BaseModel):
    size: int
    shares_device: int
    shares_payment: int
    shares_address: int
    shares_device_and_payment: int
    avg_account_age: float
    account_age_std: float
    avg_return_rate: float
    max_return_rate: float
    avg_abusive_return_rate: float
    max_abusive_return_rate: float
    avg_total_orders: float


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/score")
def score(order: OrderInput):
    try:
        return score_order(order.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/abuse-check")
def abuse_check(cluster: ClusterInput):
    try:
        return score_cluster(cluster.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sample-clusters")
def sample_clusters():
    return {"clusters": get_sample_clusters()}


import math

def _sanitize(obj):
    """Recursively replace NaN/inf with None so standard JSON encoding doesn't choke."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@app.get("/api/metrics")
def metrics():
    def load(name):
        path = os.path.join(MODELS_DIR, name)
        if name.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        else:
            import pandas as pd
            return pd.read_csv(path).to_dict(orient="records")

    return _sanitize({
        "deployment_config": load("deployment_config.json"),
        "ring_detector_config": load("ring_detector_config.json"),
        "baseline_comparison": load("baseline_comparison.csv"),
        "calibration_fairness": load("calibration_fairness_report.json"),
        "business_impact": load("business_impact.json"),
    })
