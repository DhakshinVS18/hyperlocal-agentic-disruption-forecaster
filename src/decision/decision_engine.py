"""
DeliveryGuard AI — Week 6a: Decision Engine
==============================================
Takes a zone-hour's predicted risk + SHAP-identified dominant cause and
maps it to ONE action from a fixed, closed set. This is deliberately NOT
an LLM choosing freely — it's an auditable if/else policy over causes,
exactly as the project brief requires ("constrained and auditable").

Action set (from the project brief):
- rebalance_riders        : rider capacity shortage / demand surge
- reposition_inventory    : inventory shortage / stockout risk
- reroute_traffic         : traffic is the dominant constraint
- alert_ops               : moderate risk or low model confidence
- no_action               : low risk, intervention not justified
"""
import pandas as pd
import numpy as np

# Confidence band: predictions in this probability range are treated as
# "uncertain" and get alert_ops instead of a committed action, even if
# risk is nominally above the disruption threshold — we don't want the
# agent confidently acting on a coin-flip prediction.
CONFIDENT_HIGH = 0.70
CONFIDENT_LOW = 0.30


def identify_dominant_cause(row, shap_row):
    """
    Groups the top SHAP-contributing features into one of four causal
    buckets. shap_row: a pandas Series of SHAP values indexed by feature name.
    Only considers POSITIVE contributors (pushing risk up), since those are
    what an intervention should address.
    """
    positive_contribs = shap_row[shap_row > 0].sort_values(ascending=False)
    if len(positive_contribs) == 0:
        return "none"

    RIDER_FEATURES = {"orders_per_rider", "capacity_gap", "demand_x_rider_shortage",
                       "available_riders", "required_riders_est", "demand_surge_rate",
                       "current_orders", "recent_orders_2h"}
    TRAFFIC_FEATURES = {"congestion_pct", "travel_time_index", "avg_speed", "rain_x_traffic"}
    INVENTORY_FEATURES = {"stockout_risk_flag", "coverage_hours", "inventory_level",
                           "incoming_replenishment"}

    # Score each bucket by summed SHAP contribution of its features present in top contributors
    scores = {"rider_shortage": 0.0, "traffic": 0.0, "inventory": 0.0}
    for feat, val in positive_contribs.items():
        if feat in RIDER_FEATURES:
            scores["rider_shortage"] += val
        elif feat in TRAFFIC_FEATURES:
            scores["traffic"] += val
        elif feat in INVENTORY_FEATURES:
            scores["inventory"] += val

    dominant = max(scores, key=scores.get)
    if scores[dominant] == 0:
        return "none"
    return dominant


def decide_action(predicted_prob: float, dominant_cause: str) -> dict:
    """
    Returns the chosen action plus a human-readable rationale — this
    rationale is what gets shown to the ops user for approval, so it
    must be traceable to the actual inputs, not generated freely.
    """
    if predicted_prob < CONFIDENT_LOW:
        return {"action": "no_action", "rationale": f"Low predicted risk ({predicted_prob:.1%}) — no intervention justified."}

    if CONFIDENT_LOW <= predicted_prob < CONFIDENT_HIGH:
        return {"action": "alert_ops", "rationale": f"Moderate/uncertain risk ({predicted_prob:.1%}) — alert operations, prepare capacity, no committed action yet."}

    # predicted_prob >= CONFIDENT_HIGH: confident high risk, act on dominant cause
    if dominant_cause == "rider_shortage":
        return {"action": "rebalance_riders", "rationale": f"High risk ({predicted_prob:.1%}), dominant cause is rider/demand capacity gap — rebalance riders from nearby lower-risk zones."}
    elif dominant_cause == "traffic":
        return {"action": "reroute_traffic", "rationale": f"High risk ({predicted_prob:.1%}), dominant cause is traffic congestion — recommend alternate routing policy."}
    elif dominant_cause == "inventory":
        return {"action": "reposition_inventory", "rationale": f"High risk ({predicted_prob:.1%}), dominant cause is inventory/stockout risk — recommend inventory repositioning."}
    else:
        return {"action": "alert_ops", "rationale": f"High risk ({predicted_prob:.1%}) but no single dominant cause identified — alert operations for manual review."}
