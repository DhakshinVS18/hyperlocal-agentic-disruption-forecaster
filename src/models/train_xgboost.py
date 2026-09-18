"""
DeliveryGuard AI — Week 4: Feature Engineering + XGBoost
===========================================================
Trains the primary disruption-risk classifier. Compares directly against
the Week 3 baselines using the SAME time-aware split, so the comparison
is fair.

FEATURE ENGINEERING ADDED HERE (beyond what generation already produced):
- zone_id: one-hot encoded (10 zones — cheap, no ordinality assumption)
- distance_tier: ordinal encoded (short=0, medium=1, long=2 — genuinely
  ordinal, so ordinal encoding is defensible here unlike zone_id)

Everything else reuses the Week 3 FEATURES list — deliberately, so any
performance gain over logistic regression is attributable to XGBoost's
capacity to model nonlinear interactions, not to new information.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

df = pd.read_csv("data/synthetic/zone_hourly_data.csv", parse_dates=["timestamp"])

# --- Same time-aware split as Week 3 baselines ---
split_date = df["timestamp"].quantile(0.78)
train = df[df["timestamp"] < split_date].copy()
test = df[df["timestamp"] >= split_date].copy()

# --- Feature engineering: encode categoricals ---
tier_map = {"short": 0, "medium": 1, "long": 2}
for d in (train, test):
    d["distance_tier_ordinal"] = d["distance_tier"].map(tier_map)

zone_dummies_train = pd.get_dummies(train["zone_id"], prefix="zone")
zone_dummies_test = pd.get_dummies(test["zone_id"], prefix="zone")
# align columns in case a zone is missing from one split (shouldn't happen here, but safe)
zone_dummies_test = zone_dummies_test.reindex(columns=zone_dummies_train.columns, fill_value=0)

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

X_train = pd.concat([train[BASE_FEATURES], zone_dummies_train], axis=1)
X_test = pd.concat([test[BASE_FEATURES], zone_dummies_test], axis=1)
y_train, y_test = train["disruption_label"], test["disruption_label"]

FEATURE_COLS = X_train.columns.tolist()
print(f"Total features: {len(FEATURE_COLS)}")

# --- Train/val split WITHIN train (still time-ordered) for early stopping ---
val_split = train["timestamp"].quantile(0.85 / 0.78 * 0.78)  # last ~15% of train period
val_cut = train["timestamp"].quantile(0.85)
fit_mask = train["timestamp"] < val_cut
X_fit, y_fit = X_train[fit_mask], y_train[fit_mask]
X_val, y_val = X_train[~fit_mask], y_train[~fit_mask]

print(f"Fit: {len(X_fit):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# --- XGBoost with early stopping (light manual tuning, not exhaustive grid search) ---
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),  # handle class imbalance
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")

# --- Evaluate on held-out TEST set (never touched during training/tuning) ---
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

print("\n--- XGBoost Test Set Performance ---")
print(f"Precision: {precision_score(y_test, test_pred):.3f}")
print(f"Recall:    {recall_score(y_test, test_pred):.3f}")
print(f"F1:        {f1_score(y_test, test_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_prob):.3f}")
print(f"PR-AUC:    {average_precision_score(y_test, test_prob):.3f}")
print(f"Brier score (lower=better calibrated): {brier_score_loss(y_test, test_prob):.4f}")
cm = confusion_matrix(y_test, test_pred)
print(f"Confusion matrix [[TN FP][FN TP]]:\n{cm}")

# --- Calibration check ---
frac_pos, mean_pred = calibration_curve(y_test, test_prob, n_bins=10)
print("\n--- Calibration (predicted prob bucket vs actual observed rate) ---")
calib_df = pd.DataFrame({"predicted_prob": mean_pred.round(3), "observed_rate": frac_pos.round(3)})
print(calib_df.to_string(index=False))

# --- Feature importance ---
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n--- Top 15 features by XGBoost importance ---")
print(importance.head(15).to_string(index=False))

# --- Compare against Week 3 baselines ---
try:
    baseline_results = pd.read_csv("data/processed/baseline_results.csv", index_col=0)
    xgb_row = pd.DataFrame({
        "precision": [precision_score(y_test, test_pred)],
        "recall": [recall_score(y_test, test_pred)],
        "f1": [f1_score(y_test, test_pred)],
        "roc_auc": [roc_auc_score(y_test, test_prob)],
        "pr_auc": [average_precision_score(y_test, test_prob)],
    }, index=["xgboost"])
    comparison = pd.concat([baseline_results, xgb_row])
    print("\n" + "=" * 60)
    print("FULL MODEL COMPARISON (rule-based vs logreg vs xgboost)")
    print("=" * 60)
    print(comparison.round(3))
    comparison.to_csv("data/processed/model_comparison.csv")
    print("\nSaved to data/processed/model_comparison.csv")
except FileNotFoundError:
    print("\n(baseline_results.csv not found — run baselines.py first for full comparison)")

# Save the trained model for use in Week 5 (SHAP) and beyond
model.save_model("data/processed/xgboost_model.json")
print("Model saved to data/processed/xgboost_model.json")

# Save test set predictions for later SHAP/error analysis
test_with_preds = test.copy()
test_with_preds["predicted_prob"] = test_prob
test_with_preds["predicted_label"] = test_pred
test_with_preds.to_csv("data/processed/test_predictions.csv", index=False)
