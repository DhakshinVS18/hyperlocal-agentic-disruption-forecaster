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

# --- Static per-zone profile (used for live simulation, since there's no
# real live weather/traffic/order feed to connect to for this project) ---
import datetime as dt

ZONE_PROFILE = {
    "Z01": {"tier": "medium", "base_demand": 55, "base_riders": 32},
    "Z02": {"tier": "short", "base_demand": 70, "base_riders": 41},
    "Z03": {"tier": "short", "base_demand": 85, "base_riders": 49},
    "Z04": {"tier": "medium", "base_demand": 60, "base_riders": 35},
    "Z05": {"tier": "long", "base_demand": 40, "base_riders": 23},
    "Z06": {"tier": "medium", "base_demand": 58, "base_riders": 34},
    "Z07": {"tier": "long", "base_demand": 35, "base_riders": 20},
    "Z08": {"tier": "short", "base_demand": 65, "base_riders": 38},
    "Z09": {"tier": "short", "base_demand": 62, "base_riders": 36},
    "Z10": {"tier": "long", "base_demand": 50, "base_riders": 29},
}
TIER_MINUTES = {"short": 12, "medium": 18, "long": 28}


def hour_demand_multiplier(hour):
    if 12 <= hour <= 14: return 1.6
    if 19 <= hour <= 22: return 1.9
    if 7 <= hour <= 9: return 1.2
    if 0 <= hour <= 5: return 0.15
    return 0.8


def hour_traffic_base(hour):
    if 8 <= hour <= 10 or 17 <= hour <= 20: return 65
    if 0 <= hour <= 5: return 10
    return 35


def generate_live_snapshot(zone_id: str, rainfall_mm: float = 0.0):
    """
    Generates a CURRENT zone-hour using real wall-clock time and the same
    causal relationships as the training data generator, plus a
    user-supplied rainfall estimate (since we have no live weather feed).
    This is a live SIMULATION, not a live SENSOR feed — labeled as such
    everywhere it's shown.
    """
    now = dt.datetime.now()
    hour = now.hour
    dow = now.weekday()
    is_weekend = int(dow >= 5)
    profile = ZONE_PROFILE[zone_id]
    tier = profile["tier"]

    congestion = min(98, max(5, hour_traffic_base(hour) + rainfall_mm * 2.2 + np.random.normal(0, 5)))
    avg_speed = np.clip(45 - congestion * 0.35, 5, 45)
    travel_time_index = 1 + congestion / 60

    weekend_mult = 1.15 if is_weekend else 1.0
    rain_bump = 1 + (rainfall_mm / 40) if rainfall_mm > 2 else 1.0
    expected_orders = profile["base_demand"] * hour_demand_multiplier(hour) * weekend_mult * rain_bump
    current_orders = max(0, int(np.random.poisson(max(expected_orders, 0.1))))

    typical_riders = profile["base_riders"] * (0.55 + 0.45 * hour_demand_multiplier(hour) / 1.9)
    available_riders = max(1, int(typical_riders - np.random.binomial(int(max(typical_riders, 1)), 0.08)))
    busy_riders = int(available_riders * np.random.uniform(0.3, 0.7))
    required_riders_est = current_orders / 2.2
    capacity_gap = required_riders_est - available_riders

    inventory_level = float(np.clip(np.random.normal(75, 15), 5, 100))
    coverage_hours = inventory_level / max(current_orders * 0.5, 1)

    row = {
        "zone_id": zone_id, "zone_name": zone_id, "distance_tier": tier,
        "timestamp": now,
        "current_rainfall": rainfall_mm, "forecast_rainfall_1h": rainfall_mm,
        "temperature": 28.0, "humidity": 65.0, "wind_speed": 10.0,
        "rain_probability": min(100, rainfall_mm * 8),
        "congestion_pct": congestion, "avg_speed": avg_speed, "travel_time_index": travel_time_index,
        "current_orders": current_orders, "recent_orders_2h": current_orders * 2,
        "expected_next_hour_orders": expected_orders,
        "historical_avg_demand": profile["base_demand"] * hour_demand_multiplier(hour),
        "demand_surge_rate": current_orders / max(profile["base_demand"] * hour_demand_multiplier(hour), 1),
        "available_riders": available_riders, "busy_riders": busy_riders, "riders_in_transit": 1,
        "rider_utilization": busy_riders / max(busy_riders + available_riders, 1),
        "required_riders_est": required_riders_est, "capacity_gap": capacity_gap,
        "inventory_level": inventory_level, "coverage_hours": coverage_hours,
        "stockout_risk_flag": int(coverage_hours < 1), "incoming_replenishment": 0,
        "recent_delivery_time_avg": TIER_MINUTES[tier] + 10,
        "historical_sla_breach_rate": 0.15, "previous_disruption_flag": 0,
        "orders_per_rider": current_orders / max(available_riders, 1),
        "rain_x_traffic": rainfall_mm * congestion,
        "demand_x_rider_shortage": (current_orders / max(profile["base_demand"], 1)) * max(capacity_gap, 0),
        "hour_of_day": hour, "day_of_week": dow, "is_weekend": is_weekend,
        "is_peak_hour": int((12 <= hour <= 14) or (19 <= hour <= 22)),
        "distance_tier_ordinal": tier_map[tier],
    }
    return row


@app.get("/live")
def live_analyze(zone_id: str = Query(...), rainfall_mm: float = Query(0.0, description="Current rainfall estimate in mm/hr")):
    """
    Live mode: generates a snapshot for the ACTUAL current hour (your PC's
    real clock) using the same causal model as training data, then runs
    the full predict->explain->decide->simulate pipeline on it.
    NOT a real sensor feed — there is no live weather/traffic/order API
    connected. rainfall_mm lets you manually set current conditions.
    """
    if zone_id not in ZONE_PROFILE:
        raise HTTPException(status_code=404, detail=f"Unknown zone {zone_id}")

    row_dict = generate_live_snapshot(zone_id, rainfall_mm)
    row_df = pd.DataFrame([row_dict])
    zone_dummy_row = pd.get_dummies(row_df["zone_id"], prefix="zone").reindex(columns=zone_dummies.columns, fill_value=0)
    feature_row_df = pd.concat([row_df[BASE_FEATURES], zone_dummy_row], axis=1).astype(float)
    feature_row = feature_row_df.iloc[0]

    predicted_prob = float(model.predict_proba(feature_row_df)[:, 1][0])

    shap_row_vals = explainer.shap_values(feature_row_df)[0]
    shap_row = pd.Series(shap_row_vals, index=feature_row_df.columns)
    top_factors = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(5)
    explanation = [{"feature": f, "shap_value": round(float(v), 4)} for f, v in top_factors.items()]

    dominant_cause = identify_dominant_cause(pd.Series(row_dict), shap_row)
    decision = decide_action(predicted_prob, dominant_cause)
    sim = simulate_intervention(model, feature_row, decision["action"], FEATURE_COLS)

    zone_names = {"Z01": "Ambattur", "Z02": "Anna Nagar", "Z03": "T Nagar", "Z04": "Velachery",
                  "Z05": "Porur", "Z06": "Adyar", "Z07": "Tambaram", "Z08": "Guindy",
                  "Z09": "Mylapore", "Z10": "Sholinganallur"}

    return {
        "mode": "live_simulated",
        "disclaimer": "Generated for the current real-world hour using the same causal model as training data. No live weather/traffic/order feed is connected in this portfolio version.",
        "zone_id": zone_id,
        "zone_name": zone_names.get(zone_id, zone_id),
        "timestamp": row_dict["timestamp"].isoformat(),
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
            "current_orders": int(row_dict["current_orders"]),
            "available_riders": int(row_dict["available_riders"]),
            "capacity_gap": round(float(row_dict["capacity_gap"]), 1),
            "congestion_pct": round(float(row_dict["congestion_pct"]), 1),
            "current_rainfall": round(float(row_dict["current_rainfall"]), 1),
            "inventory_level": round(float(row_dict["inventory_level"]), 1),
        },
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "DeliveryGuard AI API", "docs": "/docs"}


@app.get("/zones")
def list_zones():
    zones = test[["zone_id", "zone_name", "distance_tier"]].drop_duplicates().sort_values("zone_id")
    return zones.to_dict(orient="records")


@app.get("/snapshots")
def list_snapshots(zone_id: str = Query(...), limit: int = 20):
    """
    Returns a MIX of low/moderate/high risk demo zone-hours for a zone —
    not just the worst cases — so the dashboard demonstrates the model
    discriminating between risk levels, not just detecting disasters.
    """
    subset = test[test["zone_id"] == zone_id].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No data for zone {zone_id}")

    per_bucket = max(1, limit // 3)
    high = subset[subset["predicted_prob"] >= 0.7].sample(min(per_bucket, len(subset[subset["predicted_prob"] >= 0.7])), random_state=1)
    moderate = subset[(subset["predicted_prob"] >= 0.3) & (subset["predicted_prob"] < 0.7)]
    moderate = moderate.sample(min(per_bucket, len(moderate)), random_state=1)
    low = subset[subset["predicted_prob"] < 0.3]
    low = low.sample(min(per_bucket, len(low)), random_state=1)

    mixed = pd.concat([high, moderate, low]).sort_values("timestamp")
    return mixed[["timestamp", "predicted_prob", "disruption_label"]].reset_index().rename(
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
