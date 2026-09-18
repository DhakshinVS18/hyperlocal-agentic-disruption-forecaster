# DeliveryGuard AI — Project Report

**An explainable AI decision-support system for hyperlocal quick-commerce delivery disruption prediction and mitigation.**

Author: Dhakshin Kumar | Repo: [hyperlocal-agentic-disruption-forecaster](https://github.com/DhakshinVS18/hyperlocal-agentic-disruption-forecaster)

---

## 1. Problem

Quick-commerce delivery operations experience short-lived, hyperlocal disruptions when multiple signals interact — rain, traffic, order spikes, rider shortages — often invisible in city-wide averages until SLA breaches have already happened. This project builds a system that predicts, at zone-hour granularity, whether a delivery zone is likely to breach its SLA in the next hour, explains why, recommends a mitigation from a constrained action set, and simulates the mitigation's expected effect — before a human operator commits to it.

**Core loop:** Predict → Explain → Decide → Simulate → Approve → Measure

## 2. Problem Definition

- **Unit:** zone × hour, across 10 Chennai zones with distance-tiered SLAs (20–40 min, based on short/medium/long distance profile)
- **Label:** `disruption = 1` if `SLA_BreachRate > 0.30` in the next hour, where an order is "late" if its delivery time exceeds its zone's promised SLA
- **Leakage rule:** only information available at prediction time T is used as a feature; the realized breach rate itself is never a feature
- Full data dictionary and leakage checklist: [`docs/week1_spec.md`](../docs/week1_spec.md)

## 3. Data

Public weather/traffic patterns were not available for direct integration in this timeframe, so a **synthetic hybrid dataset** was built instead — 64,800 zone-hours (10 zones × ~9 months hourly) generated from documented, hand-tuned causal equations (rain → traffic → delay, demand rhythm by hour, rider scheduling that lags real-time demand), with noise added deliberately so the task isn't trivially separable.

**A real bug surfaced during generation, worth reporting honestly:** the first version of the label-generation logic produced 99% disruption rate — the underlying math meant that any order-time distribution with only positive-valued delay penalties automatically pushed a majority of orders "late," regardless of actual stress conditions (a lognormal distribution's median equals its mean, so once any stressor pushed the mean above the SLA cutoff, over half of all orders "failed" by construction). This was diagnosed and fixed by remodeling the breach mechanism as a direct logistic function of relative stress rather than simulating individual order times against a fixed cutoff. Final dataset: 24.5% disruption rate, with clean, interpretable relationships (rain: 15% baseline → 97% under heavy rain; no single feature trivially predicts the label).

## 4. Modeling

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Rule-based baseline | 0.53 | 0.93 | 0.68 | — | — |
| Logistic Regression | 0.72 | 0.88 | 0.79 | 0.960 | 0.902 |
| **XGBoost** | 0.72 | 0.89 | **0.80** | **0.965** | **0.907** |

Time-aware split (train: first ~7 months, test: last ~2 months) — never randomly shuffled, to respect the leakage rule and simulate genuine future prediction.

**Honest finding:** XGBoost's improvement over logistic regression is small (+0.005 ROC-AUC). This is not a shortcoming to hide — it's evidence that most of the causal relationships in this data are close to additive/monotonic (by construction of the data generator), so a linear model already captures the bulk of the signal. XGBoost's value is in the smaller residual of nonlinear feature interactions.

**Calibration** was checked explicitly (not just discrimination): predicted probability buckets track observed outcome rates closely (e.g. predicted 3% → observed 0.8%; predicted 97% → observed 94.5%), meaning the model's probability outputs are trustworthy for downstream decisions, not just its rank-ordering.

## 5. Explainability (SHAP)

Global feature importance is dominated by `orders_per_rider`, `travel_time_index`, and `congestion_pct` — consistent with the causal structure built into the data generator, confirming the pipeline is coherent end to end rather than fitting to noise. Local (per-prediction) explanations were also validated: the top 3 highest-risk zone-hours in the test set were all driven by the same combination (rider overload + rain-traffic interaction), demonstrating consistent, legible reasoning rather than arbitrary per-case attributions.

## 6. Sensitivity & Robustness

- **Threshold sensitivity:** the 30% SLA-breach threshold (a working assumption per the project's own rules) was tested against 20%/25%/30%/35%/40%. F1 actually peaks at 25% (0.816) rather than the trained 30% (0.796), and degrades further by 40% (0.682) — evidence the threshold has real influence on downstream performance and was correctly treated as a hypothesis to test, not an assumed constant.
- **Robustness under combined shocks:** in the 4.3% of test-set zone-hours with simultaneous heavy rain + high congestion + rider shortage, the model's recall *improves* (99.8% vs 88.8% overall) — these extreme cases are unambiguous rather than a blind spot.

## 7. Decision Engine + Intervention Simulator (the project's differentiator)

A constrained, auditable decision policy (never a free-form LLM) maps the SHAP-identified dominant cause to one of five fixed actions: rebalance riders, reposition inventory, reroute traffic, alert operations, or no action. A simulator then projects the operational state after that action (using documented, hand-tuned effect sizes — e.g. "+8 riders reduces capacity gap by 8") and re-scores the projected state with the trained model.

**Headline result:** across 4,493 high-risk test-set zone-hours, proactive intervention reduces mean predicted disruption risk from **77.3% → 55.6%** (a 28.1% relative reduction), pushing 795 additional zone-hours (17.7%) below the disruption threshold entirely. Effectiveness varies meaningfully by action — rider rebalancing (the most common trigger) cuts risk by 39.3 points on average, traffic rerouting by 14.6 points, reflecting that rider capacity is the dominant lever in this system, which matches the SHAP findings.

**Limitation, stated plainly:** this measures whether the model's own learned relationships agree an action should help — a real and useful sanity check — not a guarantee of real-world physical outcomes, since effect sizes are documented assumptions rather than fit to real intervention data.

## 8. System (API + Dashboard)

A FastAPI backend (`/zones`, `/snapshots`, `/analyze`, `/live`) serves the trained model, SHAP explainer, decision engine, and simulator behind a single `/analyze` call. A dashboard (dark ops-console design, not a generic template) renders this live, including a "Live" mode that generates a fresh prediction for the actual current hour using the same causal model (clearly labeled as simulated, since no real live weather/traffic feed is connected in this portfolio version).

## 9. Limitations (stated honestly, not hidden)

- All data is synthetic; real deployment would need real weather/traffic/order/rider APIs
- The intervention simulator's effect sizes are documented assumptions, not fit to real outcomes
- The 1-hour prediction horizon and single-step simulation don't capture cascading, multi-hour effects
- No inventory-shortage interventions were triggered in the test evaluation (stockouts are rare in this dataset, worth expanding in future work)
- Novelty is scoped intentionally: this doesn't claim disruption prediction, weather-aware logistics, or rider rebalancing are new — the contribution is fusing them into one auditable predict→explain→decide→simulate loop with real, honestly-reported evaluation at every stage

## 10. What's Next

Real data partnerships, multi-hour cascading simulation, an actual A/B-testable intervention framework, and inventory-scenario coverage.

---

*All results are from synthetic/public data. No real company data, proprietary benchmarks, or novelty claims beyond what's stated above.*
