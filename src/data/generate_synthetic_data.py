"""
DeliveryGuard AI — Synthetic Data Generator
=============================================
Generates hourly zone-level operational data for Chennai quick-commerce zones.

DESIGN PHILOSOPHY (read this before changing anything):
- Every relationship below is a DOCUMENTED ASSUMPTION, not a random number.
- Noise is added deliberately so the resulting classification task is NOT
  trivially separable — a model that hits >98% accuracy on this data is
  almost certainly leaking or the noise is too small.
- The label (SLA_BreachRate > 0.30) is computed from individual simulated
  order delivery times, NOT hand-picked — so it reflects the same
  causal chain a leakage-free model would have to learn.
- All "known at T" features are computed using only current/past state.
  `expected_next_hour_orders` intentionally uses a NOISY internal forecast,
  not the true future order count, to avoid leakage.

Run: python generate_synthetic_data.py
Output: data/synthetic/zone_hourly_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

START_DATE = datetime(2025, 1, 1)
NUM_DAYS = 270  # ~9 months of hourly data
SLA_BREACH_THRESHOLD = 0.30

# Zone definitions: (zone_id, name, distance_tier, base_demand_per_hour_peak, base_rider_pool)
# distance_tier -> promised_time = 10 + {short:12, medium:18, long:28}
ZONES = [
    ("Z01", "Ambattur",       "medium", 55, 32),
    ("Z02", "Anna Nagar",     "short",  70, 41),
    ("Z03", "T Nagar",        "short",  85, 49),
    ("Z04", "Velachery",      "medium", 60, 35),
    ("Z05", "Porur",          "long",   40, 23),
    ("Z06", "Adyar",          "medium", 58, 34),
    ("Z07", "Tambaram",       "long",   35, 20),
    ("Z08", "Guindy",         "short",  65, 38),
    ("Z09", "Mylapore",       "short",  62, 36),
    ("Z10", "Sholinganallur", "long",   50, 29),
]

DISTANCE_TIER_MINUTES = {"short": 12, "medium": 18, "long": 28}
BASE_PROMISED_TIME = 10

AVG_DELIVERIES_PER_RIDER_PER_HOUR = 2.2  # used to compute required_riders_est


def promised_time(distance_tier: str) -> float:
    return BASE_PROMISED_TIME + DISTANCE_TIER_MINUTES[distance_tier]


# ---------------------------------------------------------------------------
# HOURLY PATTERN HELPERS
# ---------------------------------------------------------------------------
def hour_demand_multiplier(hour: int) -> float:
    """Lunch (12-14) and dinner (19-22) peaks; low overnight."""
    if 12 <= hour <= 14:
        return 1.6
    if 19 <= hour <= 22:
        return 1.9
    if 7 <= hour <= 9:
        return 1.2
    if 0 <= hour <= 5:
        return 0.15
    return 0.8


def hour_traffic_base(hour: int) -> float:
    """Morning (8-10) and evening (17-20) rush; base congestion %."""
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        return 65
    if 0 <= hour <= 5:
        return 10
    return 35


def is_peak_hour(hour: int) -> int:
    return int((12 <= hour <= 14) or (19 <= hour <= 22))


# ---------------------------------------------------------------------------
# MAIN GENERATION LOOP
# ---------------------------------------------------------------------------
def generate():
    rows = []
    timestamps = [START_DATE + timedelta(hours=h) for h in range(NUM_DAYS * 24)]

    # Per-day rain events (not every day rains; rain clusters realistically)
    rain_days = {}
    for day in range(NUM_DAYS):
        # ~25% of days have rain, monsoon-heavier in some months (simple seasonal bump)
        month = (START_DATE + timedelta(days=day)).month
        seasonal_boost = 0.25 if month in (10, 11, 12) else 0.0  # NE monsoon months
        rain_days[day] = np.random.random() < (0.20 + seasonal_boost)

    # Rolling history buffers per zone for lag features
    history = {z[0]: {"orders": [], "delivery_times": [], "disruptions": []} for z in ZONES}

    for ts in timestamps:
        hour = ts.hour
        day_idx = (ts - START_DATE).days
        dow = ts.weekday()  # 0=Mon
        is_weekend = int(dow >= 5)
        raining_today = rain_days[day_idx]

        # Weather (shared across zones same hour, with small per-zone jitter)
        if raining_today and 6 <= hour <= 23:
            base_rain = np.random.gamma(shape=2.0, scale=6.0)  # mm/hr, skewed
        else:
            base_rain = max(0, np.random.normal(0, 0.3))

        temperature = 28 + 4 * np.sin((hour - 6) / 24 * 2 * np.pi) + np.random.normal(0, 1)
        humidity = np.clip(60 + 20 * (base_rain > 1) + np.random.normal(0, 5), 30, 95)
        wind_speed = max(0, np.random.normal(12, 4))
        rain_probability = np.clip(base_rain * 8 + np.random.normal(0, 5), 0, 100)

        for zone_id, name, tier, base_demand_peak, base_riders in ZONES:
            zone_rain = max(0, base_rain + np.random.normal(0, 1.0))
            forecast_rainfall_1h = max(0, zone_rain + np.random.normal(0, 1.5))  # imperfect forecast

            # --- TRAFFIC: base + rain effect + noise ---
            congestion = hour_traffic_base(hour) + zone_rain * 2.2 + np.random.normal(0, 5)
            congestion = np.clip(congestion, 5, 98)
            avg_speed = np.clip(45 - congestion * 0.35 + np.random.normal(0, 2), 5, 45)
            travel_time_index = np.clip(1 + congestion / 60, 1.0, 3.0)

            # --- DEMAND: base * hour mult * weekend * rain bump + noise ---
            weekend_mult = 1.15 if is_weekend else 1.0
            rain_bump = 1 + (zone_rain / 40) if zone_rain > 2 else 1.0
            expected_orders = base_demand_peak * hour_demand_multiplier(hour) * weekend_mult * rain_bump
            current_orders = max(0, np.random.poisson(max(expected_orders, 0.1)))

            # --- RIDERS: scheduled pool sized to TYPICAL demand, not this hour's actual ---
            # Schedule is based on the average hour multiplier (~0.9), not the live spike —
            # this is what creates genuine (not constant) shortage when demand surges.
            typical_riders_needed = base_riders * (0.55 + 0.45 * hour_demand_multiplier(hour) / 1.9)
            attrition = np.random.binomial(int(max(typical_riders_needed, 1)), 0.08)  # ~8% absenteeism
            available_riders = max(1, int(typical_riders_needed - attrition + np.random.normal(0, 1)))
            busy_riders = int(available_riders * np.random.uniform(0.3, 0.7))
            riders_in_transit = int(np.random.poisson(1.5))

            required_riders_est = current_orders / AVG_DELIVERIES_PER_RIDER_PER_HOUR
            capacity_gap = required_riders_est - available_riders
            rider_utilization = busy_riders / max(busy_riders + available_riders, 1)

            # --- INVENTORY: mostly stable, periodic dips ---
            inventory_level = np.clip(np.random.normal(75, 15), 5, 100)
            coverage_hours = inventory_level / max(current_orders * 0.5, 1)
            stockout_risk_flag = int(coverage_hours < 1)
            incoming_replenishment = int(np.random.random() < 0.1)

            # --- HISTORY (lag features from buffer, past-only) ---
            hist = history[zone_id]
            recent_orders_2h = sum(hist["orders"][-2:]) if hist["orders"] else current_orders
            recent_delivery_time_avg = (
                np.mean(hist["delivery_times"][-2:]) if hist["delivery_times"] else promised_time(tier)
            )
            historical_avg_demand = np.mean(hist["orders"][-168:]) if hist["orders"] else expected_orders  # ~1wk
            historical_sla_breach_rate = (
                np.mean(hist["disruptions"][-168:]) if hist["disruptions"] else 0.15
            )
            previous_disruption_flag = hist["disruptions"][-1] if hist["disruptions"] else 0
            demand_surge_rate = current_orders / max(historical_avg_demand, 1)

            # --- EXPECTED NEXT HOUR ORDERS (noisy internal forecast, NOT ground truth) ---
            expected_next_hour_orders = max(0, expected_orders + np.random.normal(0, expected_orders * 0.2))

            # ===================================================================
            # SIMULATE INDIVIDUAL ORDER DELIVERY TIMES -> compute REAL label
            # ===================================================================
            base_time = promised_time(tier)
            n_orders = max(current_orders, 1)

            # --- BREACH-RATE MODEL (this is the causal core) ---
            # Rather than simulating each order's delivery time against a fixed cutoff
            # (which mathematically forces >50% "late" the instant any stress factor is
            # nonzero — a lognormal's median IS its expected value), we model the
            # zone-hour's breach RATE directly as a function of stress, on a logit scale.
            # A negative intercept keeps baseline breach rate low; each stressor pushes it
            # up. Weights are hand-tuned so single stressors alone rarely cross the 30%
            # disruption threshold, but combinations of 2-3 do — this is what makes SHAP's
            # later explanations meaningful (no single feature should trivially predict
            # the label on its own).
            rider_shortage_term = max(0, capacity_gap) / max(available_riders, 1)  # relative shortage
            congestion_term = max(0, congestion - 35) / 65  # normalized excess congestion, 0-1
            rain_term = min(zone_rain, 40) / 40  # normalized rain, 0-1

            breach_logit = (
                -3.2  # baseline: ~4% breach rate under normal conditions
                + 2.8 * rider_shortage_term
                + 2.2 * congestion_term
                + 1.6 * rain_term
                + np.random.normal(0, 0.5)  # unmodeled noise
            )
            breach_rate_mean = 1 / (1 + np.exp(-breach_logit))

            # Sample actual late-order count from this zone-hour's breach probability
            late_orders = np.random.binomial(n_orders, breach_rate_mean)
            sla_breach_rate_actual = late_orders / n_orders
            disruption_label = int(sla_breach_rate_actual > SLA_BREACH_THRESHOLD)
            expected_delivery_time = base_time  # kept for recent_delivery_time_avg lag feature

            # --- INTERACTIONS ---
            orders_per_rider = current_orders / max(available_riders, 1)
            rain_x_traffic = zone_rain * congestion
            demand_x_rider_shortage = demand_surge_rate * max(capacity_gap, 0)

            rows.append({
                "timestamp": ts, "zone_id": zone_id, "zone_name": name, "distance_tier": tier,
                "hour_of_day": hour, "day_of_week": dow, "is_weekend": is_weekend,
                "is_holiday": 0,  # placeholder — join a real holiday calendar later
                "is_peak_hour": is_peak_hour(hour),
                "current_rainfall": round(zone_rain, 2),
                "forecast_rainfall_1h": round(forecast_rainfall_1h, 2),
                "temperature": round(temperature, 1), "humidity": round(humidity, 1),
                "wind_speed": round(wind_speed, 1), "rain_probability": round(rain_probability, 1),
                "congestion_pct": round(congestion, 1), "avg_speed": round(avg_speed, 1),
                "travel_time_index": round(travel_time_index, 2),
                "current_orders": current_orders, "recent_orders_2h": recent_orders_2h,
                "expected_next_hour_orders": round(expected_next_hour_orders, 1),
                "historical_avg_demand": round(historical_avg_demand, 1),
                "demand_surge_rate": round(demand_surge_rate, 2),
                "available_riders": available_riders, "busy_riders": busy_riders,
                "riders_in_transit": riders_in_transit,
                "rider_utilization": round(rider_utilization, 2),
                "required_riders_est": round(required_riders_est, 1),
                "capacity_gap": round(capacity_gap, 1),
                "inventory_level": round(inventory_level, 1),
                "coverage_hours": round(coverage_hours, 2),
                "stockout_risk_flag": stockout_risk_flag,
                "incoming_replenishment": incoming_replenishment,
                "recent_delivery_time_avg": round(recent_delivery_time_avg, 1),
                "historical_sla_breach_rate": round(historical_sla_breach_rate, 3),
                "previous_disruption_flag": previous_disruption_flag,
                "orders_per_rider": round(orders_per_rider, 2),
                "rain_x_traffic": round(rain_x_traffic, 1),
                "demand_x_rider_shortage": round(demand_x_rider_shortage, 2),
                # --- LABEL / OUTCOME (never use as feature) ---
                "sla_breach_rate_actual": round(sla_breach_rate_actual, 3),
                "disruption_label": disruption_label,
            })

            # update history buffers
            hist["orders"].append(current_orders)
            hist["delivery_times"].append(expected_delivery_time)
            hist["disruptions"].append(disruption_label)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("Generating synthetic zone-hour dataset...")
    df = generate()
    print(f"Generated {len(df):,} rows across {df['zone_id'].nunique()} zones, "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Disruption rate: {df['disruption_label'].mean():.1%}")
    print(df['disruption_label'].value_counts(normalize=True))

    out_path = "data/synthetic/zone_hourly_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
