"""
DeliveryGuard AI — Week 3 EDA
==============================
Exploratory analysis of the synthetic zone-hour dataset. Run this before
touching any model — the point is to confirm the data behaves the way
Week 2's causal assumptions intended, and to catch anything surprising
before it becomes a modeling bug.
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 120)

df = pd.read_csv("data/synthetic/zone_hourly_data.csv", parse_dates=["timestamp"])

print("=" * 60)
print("SHAPE & OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df):,} | Zones: {df['zone_id'].nunique()} | "
      f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"Overall disruption rate: {df['disruption_label'].mean():.1%}")
print(f"Missing values: {df.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("CLASS BALANCE BY ZONE")
print("=" * 60)
print(df.groupby("zone_id")["disruption_label"].mean().round(3))

print("\n" + "=" * 60)
print("CLASS BALANCE BY HOUR OF DAY")
print("=" * 60)
print(df.groupby("hour_of_day")["disruption_label"].mean().round(3))

print("\n" + "=" * 60)
print("CORRELATION OF KEY FEATURES WITH LABEL")
print("=" * 60)
known_at_t_numeric = [
    "current_rainfall", "forecast_rainfall_1h", "congestion_pct", "avg_speed",
    "current_orders", "demand_surge_rate", "available_riders", "capacity_gap",
    "rider_utilization", "inventory_level", "stockout_risk_flag",
    "historical_sla_breach_rate", "previous_disruption_flag",
    "orders_per_rider", "rain_x_traffic", "demand_x_rider_shortage", "is_peak_hour",
]
corrs = df[known_at_t_numeric + ["disruption_label"]].corr()["disruption_label"].drop("disruption_label")
print(corrs.sort_values(ascending=False).round(3))

print("\n" + "=" * 60)
print("WEEKEND VS WEEKDAY")
print("=" * 60)
print(df.groupby("is_weekend")["disruption_label"].mean().round(3))

print("\n" + "=" * 60)
print("STOCKOUT RISK EFFECT")
print("=" * 60)
print(df.groupby("stockout_risk_flag")["disruption_label"].mean().round(3))
