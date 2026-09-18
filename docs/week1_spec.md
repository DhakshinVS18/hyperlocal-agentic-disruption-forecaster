# DeliveryGuard AI — Week 1 Spec
### Problem Definition v1, Zone Map, and Data Dictionary

---

## 1. Problem Definition v1 (Locked)

| Item | Definition |
|---|---|
| Prediction unit | Zone × hour |
| Prediction timestamp | T (any hour boundary) |
| Prediction horizon | T+1 hour (predict outcome for window T → T+1h) |
| SLA breach (order-level) | `actual_delivery_time > promised_time(zone)` |
| Promised time formula | `10 min (base) + distance_tier_minutes` — short=+12, medium=+18, long=+28 |
| SLA_BreachRate (zone-hour) | `late_orders / total_orders` for that zone-hour |
| Disruption label | `1` if `SLA_BreachRate(zone, T+1h) > 0.30`, else `0` |
| Threshold status | Working assumption — must be sensitivity-tested (Week 5) at 0.20 / 0.30 / 0.40 |
| Leakage boundary | Only information available **at or before T** may be used as a feature. Nothing about the T→T+1h window itself (including SLA_BreachRate, actual delivery times, or realized order outcomes in that window) may leak into features. |

### Data-Leakage Checklist
- [ ] No feature computed from info only known after T
- [ ] No feature that's an algebraic function of the label (e.g. no "late order count" as input)
- [ ] Rolling/lag features use only past windows, never centered or future
- [ ] Train/val/test split is time-based, not random
- [ ] Synthetic generator doesn't bake label logic directly into a feature

---

## 2. Zone Map — Chennai (10 zones)

| Zone ID | Area (proxy) | Distance Tier | Promised SLA | Notes |
|---|---|---|---|---|
| Z01 | Ambattur | Medium | 30 min | Mixed residential/industrial |
| Z02 | Anna Nagar | Short | 22 min | Dense, well-connected |
| Z03 | T Nagar | Short | 20 min | High density, commercial core |
| Z04 | Velachery | Medium | 28 min | IT-corridor adjacent |
| Z05 | Porur | Long | 38 min | Sprawling, farther from hubs |
| Z06 | Adyar | Medium | 26 min | Coastal residential |
| Z07 | Tambaram | Long | 40 min | Outer zone |
| Z08 | Guindy | Short | 24 min | Central, mixed use |
| Z09 | Mylapore | Short | 22 min | Dense old city core |
| Z10 | Sholinganallur | Long | 36 min | IT corridor, traffic-prone |

Each zone also carries a static `base_demand` and `base_rider_pool` (to be assigned in Week 2 during data generation — zones with higher density get higher base demand).

---

## 3. Data Dictionary

Legend — **Known@T**: is this available at prediction time? **Type**: raw (direct signal) or engineered (derived).

### Weather
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `current_rainfall` | Rainfall at time T | mm/hr | Public weather API | Yes | Raw |
| `forecast_rainfall_1h` | Forecast rainfall for T+1h | mm/hr | Public weather API (forecast) | Yes | Raw |
| `temperature` | Ambient temp at T | °C | Public weather API | Yes | Raw |
| `humidity` | Relative humidity at T | % | Public weather API | Yes | Raw |
| `wind_speed` | Wind speed at T | km/h | Public weather API | Yes | Raw |
| `rain_probability` | Forecast probability of rain in next hour | % | Public weather API | Yes | Raw |

### Traffic
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `congestion_pct` | Current congestion level | % | Public traffic API / synthetic | Yes | Raw |
| `avg_speed` | Average vehicle speed in zone | km/h | Public traffic API / synthetic | Yes | Raw |
| `travel_time_index` | Ratio of current to free-flow travel time | ratio | Derived from traffic API | Yes | Engineered |
| `recent_congestion_change` | Δ congestion over last 2h | % | Derived | Yes | Engineered |

### Orders / Demand
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `current_orders` | Orders in the current hour so far | count | Synthetic (order system) | Yes | Raw |
| `recent_orders_2h` | Orders in trailing 2h | count | Synthetic | Yes | Engineered (lag) |
| `expected_next_hour_orders` | Forecasted orders for T+1h (from a simple demand model, NOT the true future value) | count | Internal demand forecast | Yes* | Engineered |
| `historical_avg_demand` | Avg orders for this zone/hour-of-week historically | count | Synthetic historical | Yes | Engineered |
| `demand_surge_rate` | `current_orders / historical_avg_demand` | ratio | Derived | Yes | Engineered |

*`expected_next_hour_orders` must come from a **separate forecasting step available at T**, not the ground-truth realized value — otherwise this is leakage. Document this clearly in code comments.

### Riders
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `available_riders` | Idle/available riders at T | count | Synthetic (rider system) | Yes | Raw |
| `busy_riders` | Riders currently on a delivery | count | Synthetic | Yes | Raw |
| `riders_in_transit` | Riders en route to zone (inbound) | count | Synthetic | Yes | Raw |
| `rider_utilization` | `busy_riders / (busy_riders + available_riders)` | ratio | Derived | Yes | Engineered |
| `required_riders_est` | Estimated riders needed given expected demand | count | Derived (simple heuristic: orders / avg_deliveries_per_rider_per_hour) | Yes | Engineered |
| `capacity_gap` | `required_riders_est - available_riders` | count | Derived | Yes | Engineered |

### Inventory
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `inventory_level` | Current stock level (normalized) | % of full | Synthetic (dark store system) | Yes | Raw |
| `coverage_hours` | Estimated hours until stockout at current demand | hours | Derived | Yes | Engineered |
| `stockout_risk_flag` | Binary flag if coverage_hours < 1 | 0/1 | Derived | Yes | Engineered |
| `incoming_replenishment` | Replenishment scheduled within next 2h | 0/1 | Synthetic (logistics schedule) | Yes | Raw |

### Time / Context
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `hour_of_day` | Hour (0–23) | int | Timestamp | Yes | Raw |
| `day_of_week` | Day (0–6) | int | Timestamp | Yes | Raw |
| `is_weekend` | Weekend flag | 0/1 | Derived from timestamp | Yes | Engineered |
| `is_holiday` | Holiday flag | 0/1 | Public holiday calendar | Yes | Raw |
| `is_peak_hour` | Lunch/dinner peak flag | 0/1 | Derived (rule: 12-14h, 19-22h) | Yes | Engineered |

### History
| Feature | Meaning | Unit | Source | Known@T | Type |
|---|---|---|---|---|---|
| `recent_delivery_time_avg` | Avg delivery time in zone, trailing 2h | min | Synthetic, computed from past completed orders only | Yes | Engineered (lag) |
| `historical_sla_breach_rate` | Zone's long-run avg SLA breach rate for this hour-of-week | ratio | Synthetic historical | Yes | Engineered |
| `previous_disruption_flag` | Was previous hour disrupted? | 0/1 | Derived from label history (past only) | Yes | Engineered (lag) |

### Interaction Features (engineered, computed after base features exist)
| Feature | Meaning | Known@T | Type |
|---|---|---|---|
| `orders_per_rider` | `current_orders / max(available_riders,1)` | Yes | Engineered |
| `rain_x_traffic` | `current_rainfall * congestion_pct` | Yes | Engineered |
| `demand_x_rider_shortage` | `demand_surge_rate * capacity_gap` (where capacity_gap>0) | Yes | Engineered |

### Label (never a feature)
| Field | Meaning | Known@T |
|---|---|---|
| `sla_breach_rate_actual` | Realized breach rate for T→T+1h | **No — outcome only, used for labeling and evaluation, never as a model input** |
| `disruption_label` | Binary label derived from above | No — target variable |

---

## 4. Open Items Before Week 2 (Data Generation)
- [ ] Assign `base_demand` and `base_rider_pool` per zone (density-based, to be set when writing the generator)
- [ ] Write explicit causal equations: rain→traffic, traffic→delivery time, demand surge rules
- [ ] Decide noise distribution for each equation (Gaussian, params TBD in Week 2)
- [ ] Decide historical data span (recommend 6–12 months hourly) to get enough disruption examples per zone
