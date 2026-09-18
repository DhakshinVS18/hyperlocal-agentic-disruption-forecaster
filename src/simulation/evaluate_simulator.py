"""
DeliveryGuard AI — Week 6c: Evaluation
==========================================
Runs the decision engine + simulator across every high-risk zone-hour in
the test set, and compares three scenarios:
1. No intervention        — baseline, what actually would happen
2. Reactive intervention  — action applied only AFTER disruption starts
                             (approximated here as: no benefit, since by
                             the time it's reactive the hour is already lost)
3. Proactive intervention — action applied BEFORE, using this system's
                             prediction, at full simulated effect

This produces the headline comparison for the project report.
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

model = xgb.XGBClassifier()
model.load_model("data/processed/xgboost_model.json")

test = pd.read_csv("data/processed/test_predictions.csv", parse_dates=["timestamp"])

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
shap_values = explainer.shap_values(X_test)

# --- Focus on zone-hours where the model predicts meaningful risk ---
HIGH_RISK_THRESHOLD = 0.30
candidates = test[test["predicted_prob"] >= HIGH_RISK_THRESHOLD].copy()
print(f"Evaluating {len(candidates):,} zone-hours with predicted risk >= {HIGH_RISK_THRESHOLD:.0%} "
      f"({len(candidates)/len(test):.1%} of test set)")

results = []
for idx in candidates.index:
    row_pos = test.index.get_loc(idx)
    row = X_test.loc[idx]
    shap_row = pd.Series(shap_values[row_pos], index=X_test.columns)

    dominant_cause = identify_dominant_cause(test.loc[idx], shap_row)
    decision = decide_action(test.loc[idx, "predicted_prob"], dominant_cause)
    sim = simulate_intervention(model, row, decision["action"], FEATURE_COLS)

    results.append({
        "zone_id": test.loc[idx, "zone_id"],
        "timestamp": test.loc[idx, "timestamp"],
        "actual_disruption": test.loc[idx, "disruption_label"],
        "dominant_cause": dominant_cause,
        "action": decision["action"],
        "risk_no_action": sim["risk_before"],
        "risk_proactive": sim["risk_after"],
        "risk_reduction": sim["risk_reduction"],
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ACTION DISTRIBUTION")
print("=" * 60)
print(results_df["action"].value_counts())

print("\n" + "=" * 60)
print("SCENARIO COMPARISON: No Action vs Reactive vs Proactive")
print("=" * 60)

no_action_mean_risk = results_df["risk_no_action"].mean()
proactive_mean_risk = results_df["risk_proactive"].mean()
# Reactive: by definition, action happens only after the disruption window has
# already occurred, so it cannot reduce THIS hour's risk — modeled as zero benefit
# for the hour itself (its value is in preventing the NEXT hour, which is out of
# scope for a 1-hour-horizon single-step simulation).
reactive_mean_risk = no_action_mean_risk

print(f"No intervention   — mean predicted disruption risk: {no_action_mean_risk:.1%}")
print(f"Reactive          — mean predicted disruption risk: {reactive_mean_risk:.1%} (no same-hour benefit, by definition)")
print(f"Proactive         — mean predicted disruption risk: {proactive_mean_risk:.1%}")
print(f"\nAbsolute risk reduction from proactive intervention: {no_action_mean_risk - proactive_mean_risk:.1%} points")
print(f"Relative risk reduction: {(no_action_mean_risk - proactive_mean_risk) / no_action_mean_risk:.1%}")

# How many high-risk zone-hours drop BELOW the 30% disruption threshold after action?
below_threshold_before = (results_df["risk_no_action"] < 0.30).sum()
below_threshold_after = (results_df["risk_proactive"] < 0.30).sum()
print(f"\nZone-hours pushed below 30% risk threshold by intervention: "
      f"{below_threshold_after - below_threshold_before} additional "
      f"({(below_threshold_after - below_threshold_before) / len(results_df):.1%} of evaluated cases)")

# Breakdown by action type — which actions are most effective?
print("\n" + "=" * 60)
print("RISK REDUCTION BY ACTION TYPE")
print("=" * 60)
action_summary = results_df.groupby("action").agg(
    count=("action", "size"),
    avg_risk_before=("risk_no_action", "mean"),
    avg_risk_after=("risk_proactive", "mean"),
    avg_reduction=("risk_reduction", "mean"),
).round(3)
print(action_summary)

results_df.to_csv("data/processed/intervention_simulation_results.csv", index=False)
print("\nSaved to data/processed/intervention_simulation_results.csv")

print("\n" + "!" * 60)
print("ALL RESULTS ABOVE ARE SIMULATED/ILLUSTRATIVE, based on documented")
print("hand-tuned effect-size assumptions, not real operational outcomes.")
print("!" * 60)
