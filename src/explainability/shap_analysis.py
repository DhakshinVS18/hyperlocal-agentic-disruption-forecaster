"""
DeliveryGuard AI — Week 5: Explainability + Sensitivity + Robustness
=======================================================================
Three separate questions, kept separate on purpose:
1. SHAP: WHY does the model assign high/low risk to a given zone-hour?
2. Sensitivity: how much does the 0.30 SLA-breach threshold actually matter?
3. Robustness: does the model still work when multiple stressors hit at once,
   or does it fall apart outside the "normal" range it was trained on?
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# --- Load model and test predictions from Week 4 ---
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

# ============================================================
# 1. SHAP — GLOBAL EXPLANATIONS
# ============================================================
print("=" * 60)
print("SHAP GLOBAL FEATURE IMPORTANCE")
print("=" * 60)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
print(mean_abs_shap.head(15).to_string(index=False))
mean_abs_shap.to_csv("data/processed/shap_global_importance.csv", index=False)

# ============================================================
# 2. SHAP — LOCAL EXPLANATIONS (example zone-hours)
# ============================================================
print("\n" + "=" * 60)
print("SHAP LOCAL EXPLANATIONS — 3 example high-risk predictions")
print("=" * 60)
top_risk_idx = test.sort_values("predicted_prob", ascending=False).head(3).index
for idx in top_risk_idx:
    row_pos = test.index.get_loc(idx)
    zone = test.loc[idx, "zone_id"]
    ts = test.loc[idx, "timestamp"]
    prob = test.loc[idx, "predicted_prob"]
    print(f"\n{zone} at {ts} — predicted risk: {prob:.1%}")
    row_shap = pd.Series(shap_values[row_pos], index=X_test.columns).sort_values(key=abs, ascending=False)
    print("Top contributing factors:")
    print(row_shap.head(5).to_string())

# ============================================================
# 3. SENSITIVITY ANALYSIS — does the 30% threshold matter?
# ============================================================
print("\n" + "=" * 60)
print("SENSITIVITY: relabeling at different SLA-breach thresholds")
print("=" * 60)
print("(Model was trained on 30% threshold. This checks how many zone-hours")
print(" would flip class if the business used a different threshold — and how")
print(" the model's fixed 0.5 probability cutoff would perform against each.)")

for thresh in [0.20, 0.25, 0.30, 0.35, 0.40]:
    relabeled = (test["sla_breach_rate_actual"] > thresh).astype(int)
    pct_positive = relabeled.mean()
    # Using the SAME model probabilities, see how well they'd classify under this threshold
    pred_at_default_cutoff = (test["predicted_prob"] >= 0.5).astype(int)
    f1 = f1_score(relabeled, pred_at_default_cutoff)
    print(f"Threshold={thresh:.2f} -> {pct_positive:.1%} of zone-hours positive | "
          f"F1 vs current model @0.5 cutoff: {f1:.3f}")

# ============================================================
# 4. ROBUSTNESS — combined-shock scenarios
# ============================================================
print("\n" + "=" * 60)
print("ROBUSTNESS: performance under simultaneous shocks")
print("=" * 60)

overall_recall = recall_score(test["disruption_label"], test["predicted_label"])
overall_precision = precision_score(test["disruption_label"], test["predicted_label"])
print(f"Overall test set: precision={overall_precision:.3f}, recall={overall_recall:.3f}")

# Define a "combined shock" as: rain > 10mm AND congestion > 60% AND capacity_gap > 0
shock_mask = (test["current_rainfall"] > 10) & (test["congestion_pct"] > 60) & (test["capacity_gap"] > 0)
shock_subset = test[shock_mask]
print(f"\nCombined-shock scenarios (heavy rain + high congestion + rider shortage): "
      f"{len(shock_subset)} zone-hours ({len(shock_subset)/len(test):.1%} of test set)")
if len(shock_subset) > 0:
    shock_recall = recall_score(shock_subset["disruption_label"], shock_subset["predicted_label"])
    shock_precision = precision_score(shock_subset["disruption_label"], shock_subset["predicted_label"], zero_division=0)
    print(f"Under combined shock: precision={shock_precision:.3f}, recall={shock_recall:.3f}")
    print(f"Actual disruption rate in shock scenarios: {shock_subset['disruption_label'].mean():.1%}")
    if shock_recall < overall_recall:
        print("NOTE: recall drops under combined shock vs overall — model may under-flag"
              " the most extreme scenarios. Worth mentioning honestly in the report.")
    else:
        print("Model holds up (recall does not degrade) under combined-shock scenarios.")

print("\nSaved: data/processed/shap_global_importance.csv")
