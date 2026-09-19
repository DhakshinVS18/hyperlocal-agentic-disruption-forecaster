"""
DeliveryGuard AI — Stockout Scenario Injector (Post-Week-8 addition)
========================================================================
The original synthetic dataset has rare stockouts (a few % of rows), so the
decision engine's `reposition_inventory` action was never exercised in the
Week 6 evaluation — a limitation the final report already states honestly.

This script does NOT retrain the model or touch the original dataset/metrics
(which stay exactly as reported). Instead, it generates a separate,
targeted stress-test set with FORCED severe stockouts, then runs the
ALREADY-TRAINED model + decision engine + simulator against it, to see
whether the inventory branch behaves sensibly when it's actually exercised.

This is an honest validation exercise: if the model underweights stockouts
(because it saw so few during training), that is itself a reportable,
useful finding — not something to hide.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import sys
sys.path.insert(0, "src/decision")
sys.path.insert(0, "src/simulation")
from decision_engine import identify_dominant_cause, decide_action
from simulator import simulate_intervention

np.random.seed(7)

# --- Load the ALREADY-TRAINED model (unchanged) ---
model = xgb.XGBClassifier()
model.load_model("data/processed/xgboost_model.json")

# --- Load real test rows as a base, then override inventory fields ---
test = pd.read_csv("data/processed/test_predictions.csv", parse_dates=["timestamp"])

N_SCENARIOS = 300
base_rows = test.sample(N_SCENARIOS, random_state=7).copy().reset_index(drop=True)

# Force a severe stockout: very low inventory, near-zero coverage, flag set,
# and NOT confounded by rider/traffic stress — isolate the inventory signal
base_rows["inventory_level"] = np.random.uniform(2, 12, N_SCENARIOS)
base_rows["coverage_hours"] = base_rows["inventory_level"] / np.maximum(base_rows["current_orders"] * 0.5, 1)
base_rows["stockout_risk_flag"] = (base_rows["coverage_hours"] < 1).astype(int)
base_rows["incoming_replenishment"] = 0
# Keep rider/traffic/weather at mild/normal levels so inventory is the dominant story
base_rows["capacity_gap"] = np.random.uniform(-3, 2, N_SCENARIOS)
base_rows["congestion_pct"] = np.random.uniform(20, 40, N_SCENARIOS)
base_rows["current_rainfall"] = 0.0

tier_map = {"short": 0, "medium": 1, "long": 2}
base_rows["distance_tier_ordinal"] = base_rows["distance_tier"].map(tier_map)
zone_dummies = pd.get_dummies(base_rows["zone_id"], prefix="zone")

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
X_stress = pd.concat([base_rows[BASE_FEATURES], zone_dummies], axis=1)
FEATURE_COLS = X_stress.columns.tolist()

predicted_probs = model.predict_proba(X_stress)[:, 1]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_stress)

results = []
for i in range(N_SCENARIOS):
    row = X_stress.iloc[i]
    shap_row = pd.Series(shap_values[i], index=X_stress.columns)
    dominant_cause = identify_dominant_cause(base_rows.iloc[i], shap_row)
    decision = decide_action(predicted_probs[i], dominant_cause)
    sim = simulate_intervention(model, row, decision["action"], FEATURE_COLS)
    results.append({
        "zone_id": base_rows.iloc[i]["zone_id"],
        "inventory_level": round(base_rows.iloc[i]["inventory_level"], 1),
        "coverage_hours": round(base_rows.iloc[i]["coverage_hours"], 2),
        "predicted_risk": round(predicted_probs[i], 3),
        "dominant_cause": dominant_cause,
        "action": decision["action"],
        "risk_before": round(sim["risk_before"], 3),
        "risk_after": round(sim["risk_after"], 3),
    })

results_df = pd.DataFrame(results)

print("=" * 60)
print(f"STOCKOUT STRESS TEST — {N_SCENARIOS} forced severe-stockout scenarios")
print("=" * 60)
print(f"Mean predicted risk: {results_df['predicted_risk'].mean():.1%}")
print(f"\nDominant cause distribution:")
print(results_df["dominant_cause"].value_counts())
print(f"\nAction distribution:")
print(results_df["action"].value_counts())

inventory_triggered = (results_df["action"] == "reposition_inventory").sum()
print(f"\n'reposition_inventory' action triggered: {inventory_triggered} / {N_SCENARIOS} "
      f"({inventory_triggered/N_SCENARIOS:.1%}) of forced stockout scenarios")

if inventory_triggered > 0:
    inv_rows = results_df[results_df["action"] == "reposition_inventory"]
    print(f"Mean risk reduction when triggered: {(inv_rows['risk_before'] - inv_rows['risk_after']).mean():.1%} points")
else:
    print("\nFINDING: the model, trained on rare stockouts (~a few % of original data),")
    print("does not treat inventory signals as dominant even under severe forced stockouts.")
    print("This is an honest limitation: the current model under-weights inventory risk")
    print("relative to rider/traffic signals, because it saw too few real examples during")
    print("training. Recommended fix: retrain with denser stockout injection in Week 2's")
    print("generator (not done here, to avoid invalidating already-reported Week 4-6 metrics).")

results_df.to_csv("data/processed/stockout_stress_test_results.csv", index=False)
print("\nSaved to data/processed/stockout_stress_test_results.csv")
