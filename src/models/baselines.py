"""
DeliveryGuard AI — Week 3 Baselines
======================================
Two baselines to beat before XGBoost gets to claim it's "better":
1. Rule-based baseline — hand-written thresholds, no learning at all.
2. Logistic regression — simplest real model, linear decision boundary.

If XGBoost (Week 4) can't meaningfully beat these, that's a real finding,
not a failure — it would mean the relationships here are mostly linear/
rule-shaped, which is itself worth reporting honestly.

TIME-AWARE SPLIT: never shuffle-split time series data. We use the first
~7 months as train, the remaining ~2 months as test — this respects the
leakage rule (no future data used to predict earlier data) and estimates
how the model would perform on genuinely unseen future weeks.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

df = pd.read_csv("data/synthetic/zone_hourly_data.csv", parse_dates=["timestamp"])

# --- TIME-AWARE SPLIT ---
split_date = df["timestamp"].quantile(0.78)  # ~7 of 9 months train
train = df[df["timestamp"] < split_date].copy()
test = df[df["timestamp"] >= split_date].copy()
print(f"Train: {len(train):,} rows ({train['timestamp'].min().date()} to {train['timestamp'].max().date()})")
print(f"Test:  {len(test):,} rows ({test['timestamp'].min().date()} to {test['timestamp'].max().date()})")
print(f"Train disruption rate: {train['disruption_label'].mean():.1%} | "
      f"Test disruption rate: {test['disruption_label'].mean():.1%}")


def evaluate(y_true, y_pred, y_prob=None, name=""):
    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred):.3f}")
    if y_prob is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.3f}")
        print(f"PR-AUC:    {average_precision_score(y_true, y_prob):.3f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix [[TN FP][FN TP]]:\n{cm}")
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        "pr_auc": average_precision_score(y_true, y_prob) if y_prob is not None else None,
    }


results = {}

# ============================================================
# BASELINE 1: RULE-BASED
# ============================================================
# Hand-written thresholds from domain intuition + EDA correlations.
# No fitting to data — this is meant to represent "what an experienced
# ops manager would flag by eye," not a tuned model.
def rule_based_predict(d):
    return (
        ((d["capacity_gap"] > 5) & (d["congestion_pct"] > 50)) |
        (d["current_rainfall"] > 10) |
        (d["stockout_risk_flag"] == 1) |
        ((d["is_peak_hour"] == 1) & (d["capacity_gap"] > 10))
    ).astype(int)

test_pred_rule = rule_based_predict(test)
results["rule_based"] = evaluate(test["disruption_label"], test_pred_rule, name="Rule-Based Baseline")

# ============================================================
# BASELINE 2: LOGISTIC REGRESSION
# ============================================================
# Only features that are genuinely known at prediction time T.
# Explicitly excludes: sla_breach_rate_actual, disruption_label (label leakage),
# and zone_name/timestamp (identifiers, not signal).
FEATURES = [
    "current_rainfall", "forecast_rainfall_1h", "temperature", "humidity", "wind_speed",
    "rain_probability", "congestion_pct", "avg_speed", "travel_time_index",
    "current_orders", "recent_orders_2h", "expected_next_hour_orders",
    "historical_avg_demand", "demand_surge_rate", "available_riders", "busy_riders",
    "riders_in_transit", "rider_utilization", "required_riders_est", "capacity_gap",
    "inventory_level", "coverage_hours", "stockout_risk_flag", "incoming_replenishment",
    "recent_delivery_time_avg", "historical_sla_breach_rate", "previous_disruption_flag",
    "orders_per_rider", "rain_x_traffic", "demand_x_rider_shortage",
    "hour_of_day", "day_of_week", "is_weekend", "is_peak_hour",
]

X_train, y_train = train[FEATURES], train["disruption_label"]
X_test, y_test = test[FEATURES], test["disruption_label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)

test_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
test_pred_logreg = (test_prob_logreg >= 0.5).astype(int)
results["logistic_regression"] = evaluate(y_test, test_pred_logreg, test_prob_logreg, name="Logistic Regression Baseline")

# Top coefficients (which features the linear model leans on most)
coef_df = pd.DataFrame({"feature": FEATURES, "coefficient": logreg.coef_[0]})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
print("\n--- Logistic Regression: Top 10 features by |coefficient| ---")
print(coef_df.sort_values("abs_coef", ascending=False).head(10)[["feature", "coefficient"]].to_string(index=False))

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("BASELINE COMPARISON SUMMARY")
print("=" * 60)
summary = pd.DataFrame(results).T
print(summary.round(3))
summary.to_csv("data/processed/baseline_results.csv")
print("\nSaved to data/processed/baseline_results.csv")
