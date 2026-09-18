"""
DeliveryGuard AI — Week 7: FastAPI Backend
==============================================
Serves the trained model + decision engine + simulator as a real API.

Endpoints:
  GET  /zones                     — list available zones (for dashboard dropdown)
  GET  /snapshots?zone_id=Z01     — list available test-set zone-hours for a zone (demo data)
  GET  /analyze?zone_id=Z01&idx=0 — full pipeline: predict -> explain -> recommend -> simulate

Run: uvicorn api.main:app --reload --port 8000
Docs auto-generated at: http://localhost:8000/docs

NOTE ON DATA SOURCE: for this portfolio project, /analyze reads a specific
row from the held-out TEST set (never used in training) rather than
accepting arbitrary live input — this keeps the demo grounded in real,
already-evaluated data rather than letting a user submit values the model
was never validated against. A production version would take live sensor/
ops inputs instead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "decision"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "simulation"))

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from decision_engine import identify_dominant_cause, decide_action
from simulator import simulate_intervention

app = FastAPI(title="DeliveryGuard AI API", version="0.1")

# Allow the dashboard (served separately) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model + data once at startup ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

model = xgb.XGBClassifier()
model.load_model(os.path.join(DATA_DIR, "xgboost_model.json"))

test = pd.read_csv(os.path.join(DATA_DIR, "test_predictions.csv"), parse_dates=["timestamp"])
tier_map = {"short": 0, "medium": 1, "long": 2}
test["distance_tier_ordinal"] = test["distance_tier"].map(tier_map)
zone_dummies = pd.get_dummies(test["zone_id"], prefix="zone")

BASE_FEATURES = [
    "current_rainfall", "forecast_rainfall_1h", "temperature", "humidity", "wind_speed",
    "rain_probability", "congestion_pct", "avg_speed", "travel_time_index",
    "current_orders", "recent_orders_2h", "expected_next_hour_orders",
    "historical_avg_demand", "demand_surge_rate", "available_riders", "busy_riders",
    "riders_in_transit", "rider_utilization", "required_riders_est", "capacity_gap",
    "inventory_level", "coverage_hours", "stockout_risk_flag", "incoming_replenishment",
    "recent_delivery_time_avg", "historical_sla_breach_rate", "previous_disruption_flag",
    "orders_per_rider", "rain_x_traffic", "demand_x_rider_shortage",
    "hour_of_day", "day_of_week", "is_weekend", "is_peak_hour", "distance_tier_ordinal",
]
X_test = pd.concat([test[BASE_FEATURES], zone_dummies], axis=1)
FEATURE_COLS = X_test.columns.tolist()

explainer = shap.TreeExplainer(model)

print(f"DeliveryGuard API ready. {len(test):,} test zone-hours loaded, "
      f"{test['zone_id'].nunique()} zones.")


@app.get("/")
def root():
    return {"status": "ok", "service": "DeliveryGuard AI API", "docs": "/docs"}


@app.get("/zones")
def list_zones():
    zones = test[["zone_id", "zone_name", "distance_tier"]].drop_duplicates().sort_values("zone_id")
    return zones.to_dict(orient="records")


@app.get("/snapshots")
def list_snapshots(zone_id: str = Query(...), limit: int = 20):
    """Returns available demo zone-hours for a zone, sorted by predicted risk (highest first)."""
    subset = test[test["zone_id"] == zone_id].sort_values("predicted_prob", ascending=False).head(limit)
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No data for zone {zone_id}")
    return subset[["timestamp", "predicted_prob", "disruption_label"]].reset_index().rename(
        columns={"index": "row_idx"}
    ).to_dict(orient="records")


@app.get("/analyze")
def analyze(row_idx: int = Query(..., description="Row index from /snapshots")):
    """Full pipeline: predict risk, explain via SHAP, recommend action, simulate outcome."""
    if row_idx not in test.index:
        raise HTTPException(status_code=404, detail="row_idx not found")

    row_data = test.loc[row_idx]
    row_pos = test.index.get_loc(row_idx)
    feature_row = X_test.loc[row_idx]

    # 1. PREDICT
    predicted_prob = float(row_data["predicted_prob"])

    # 2. EXPLAIN
    shap_row_vals = explainer.shap_values(feature_row.to_frame().T.astype(float))[0]
    shap_row = pd.Series(shap_row_vals, index=X_test.columns)
    top_factors = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(5)
    explanation = [{"feature": f, "shap_value": round(float(v), 4)} for f, v in top_factors.items()]

    # 3. DECIDE
    dominant_cause = identify_dominant_cause(row_data, shap_row)
    decision = decide_action(predicted_prob, dominant_cause)

    # 4. SIMULATE
    sim = simulate_intervention(model, feature_row, decision["action"], FEATURE_COLS)

    return {
        "zone_id": row_data["zone_id"],
        "zone_name": row_data["zone_name"],
        "timestamp": row_data["timestamp"].isoformat(),
        "actual_disruption": bool(row_data["disruption_label"]),
        "prediction": {
            "risk_probability": round(predicted_prob, 4),
            "risk_level": "high" if predicted_prob >= 0.7 else "moderate" if predicted_prob >= 0.3 else "low",
        },
        "explanation": explanation,
        "dominant_cause": dominant_cause,
        "recommendation": decision,
        "simulation": {
            "risk_before": round(sim["risk_before"], 4),
            "risk_after": round(sim["risk_after"], 4),
            "risk_reduction": round(sim["risk_reduction"], 4),
            "note": "Simulated/illustrative — based on documented hand-tuned effect-size assumptions.",
        },
        "current_state": {
            "current_orders": int(row_data["current_orders"]),
            "available_riders": int(row_data["available_riders"]),
            "capacity_gap": round(float(row_data["capacity_gap"]), 1),
            "congestion_pct": round(float(row_data["congestion_pct"]), 1),
            "current_rainfall": round(float(row_data["current_rainfall"]), 1),
            "inventory_level": round(float(row_data["inventory_level"]), 1),
        },
    }
