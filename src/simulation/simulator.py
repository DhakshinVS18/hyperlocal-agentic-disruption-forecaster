"""
DeliveryGuard AI — Week 6b: Intervention Simulator
======================================================
Given a zone-hour and a chosen action, projects what the operational
state would look like AFTER the action, then re-runs the trained model
on that projected state to estimate the new risk. This is the "before
vs after" that gets shown to the operator before they approve anything.

IMPORTANT — what this simulator is and isn't:
- It is a SCENARIO PROJECTION using simple, documented, hand-tuned effect
  sizes (e.g. "moving 8 riders reduces capacity_gap by 8"). It is NOT a
  full discrete-event simulation of individual orders/riders.
- It reuses the TRAINED MODEL to score the projected state, so the
  "after" risk estimate carries the same assumptions/limitations as the
  model itself. This is intentional: it tests "does the model's own
  learned relationship agree that this action should help," which is a
  meaningful sanity check, not a claim of ground-truth physical accuracy.
- Every effect size below is a documented assumption. Label results as
  SIMULATED / ILLUSTRATIVE — never as guaranteed real-world outcomes.
"""
import pandas as pd
import numpy as np
import xgboost as xgb

# --- Effect size assumptions (documented, hand-tuned — not fit to data) ---
RIDERS_MOVED_PER_REBALANCE = 8          # riders shifted in from a nearby zone
INVENTORY_REPOSITION_BOOST = 25          # percentage points added to inventory_level
TRAFFIC_REROUTE_CONGESTION_REDUCTION = 0.20  # 20% relative reduction in effective congestion


def apply_action(row: pd.Series, action: str) -> pd.Series:
    """Returns a COPY of the row with projected state changes from the action."""
    new_row = row.copy()

    if action == "rebalance_riders":
        new_row["available_riders"] = row["available_riders"] + RIDERS_MOVED_PER_REBALANCE
        new_row["capacity_gap"] = max(0, row["required_riders_est"] - new_row["available_riders"])
        new_row["orders_per_rider"] = row["current_orders"] / max(new_row["available_riders"], 1)
        new_row["rider_utilization"] = row["busy_riders"] / max(row["busy_riders"] + new_row["available_riders"], 1)
        new_row["demand_x_rider_shortage"] = row["demand_surge_rate"] * new_row["capacity_gap"]

    elif action == "reposition_inventory":
        new_row["inventory_level"] = min(100, row["inventory_level"] + INVENTORY_REPOSITION_BOOST)
        new_row["coverage_hours"] = new_row["inventory_level"] / max(row["current_orders"] * 0.5, 1)
        new_row["stockout_risk_flag"] = int(new_row["coverage_hours"] < 1)

    elif action == "reroute_traffic":
        new_row["congestion_pct"] = row["congestion_pct"] * (1 - TRAFFIC_REROUTE_CONGESTION_REDUCTION)
        new_row["avg_speed"] = min(45, row["avg_speed"] * (1 + TRAFFIC_REROUTE_CONGESTION_REDUCTION * 0.5))
        new_row["travel_time_index"] = 1 + new_row["congestion_pct"] / 60
        new_row["rain_x_traffic"] = row["current_rainfall"] * new_row["congestion_pct"]

    # alert_ops and no_action: no state change — nothing to simulate, they're not
    # physical interventions, just operational postures.

    return new_row


def score_row(model, row: pd.Series, feature_cols: list) -> float:
    """Re-runs the trained model on a single row's feature vector."""
    X = row[feature_cols].to_frame().T.astype(float)
    return float(model.predict_proba(X)[:, 1][0])


def simulate_intervention(model, row: pd.Series, action: str, feature_cols: list) -> dict:
    """Returns before/after risk plus the projected state for the given action."""
    before_prob = score_row(model, row, feature_cols)

    if action in ("alert_ops", "no_action"):
        return {
            "action": action,
            "risk_before": before_prob,
            "risk_after": before_prob,  # no state change, so no projected change
            "risk_reduction": 0.0,
        }

    after_row = apply_action(row, action)
    after_prob = score_row(model, after_row, feature_cols)

    return {
        "action": action,
        "risk_before": before_prob,
        "risk_after": after_prob,
        "risk_reduction": before_prob - after_prob,
    }
