# 🚦 DeliveryGuard AI

### Predicting delivery chaos before it happens.

> A hyperlocal quick-commerce zone might look fine on the city dashboard — until rain, traffic, a demand spike, and a rider shortage collide in the same neighborhood at the same hour. By the time SLA breaches show up in the numbers, it's already too late to fix it.

**DeliveryGuard AI** predicts *where* and *when* that collision is about to happen, explains *why* in plain language, recommends a fix, and simulates whether that fix would actually work — before anyone commits to it.

```
Predict → Explain → Decide → Simulate → Approve → Measure
```

---

## 🧠 What it does

| Stage | What happens |
|---|---|
| **Predict** | XGBoost estimates disruption risk per zone, 1 hour ahead |
| **Explain** | SHAP breaks the prediction down into its real drivers — rain? traffic? rider shortage? — narrated in plain English |
| **Decide** | A constrained decision engine picks a cause-appropriate fix — no LLM freelancing actions |
| **Simulate** | Before anything happens, the simulator projects the outcome of that fix vs. doing nothing |
| **Approve** | A human operator sees the full picture and signs off |
| **Measure** | Real outcomes get logged and compared against what the model predicted |

This isn't another "train XGBoost, report accuracy" project. The interesting part is what happens *after* the prediction.

## 🌦️ Live Mode

The dashboard includes a **Live** mode covering 22 real Chennai zones (10 in the model's original trained set, 12 extended), pulling **real current weather (OpenWeatherMap)** and **real current traffic (TomTom)** when API keys are configured. Order/rider/inventory data remains synthetic in all cases — no public API exposes real quick-commerce operational data for any platform.

## ⚙️ Stack

`Python` `FastAPI` `XGBoost` `SHAP` `PostgreSQL` `React + Tailwind` `Plotly` `Docker`

## 📍 Status — 8-week build complete

- [x] **Week 1** — Problem definition, Chennai zone map, full data dictionary → [`docs/week1_spec.md`](docs/week1_spec.md)
- [x] **Week 2** — Synthetic data generator (causal rain→traffic→delay relationships)
- [x] **Week 3** — EDA + baseline models
- [x] **Week 4** — Feature engineering + XGBoost (0.965 ROC-AUC)
- [x] **Week 5** — SHAP explainability + sensitivity/robustness analysis
- [x] **Week 6** — Decision engine + intervention simulator (28% relative risk reduction)
- [x] **Week 7** — FastAPI backend + React-style operations dashboard
- [x] **Week 8** — Docker deployment, real weather/traffic integration, plain-language narration → [`docs/final_report.md`](docs/final_report.md)

## 🗂️ Structure

```
data/              raw, processed, synthetic datasets
notebooks/         EDA and experiments
src/
  data/            ingestion & synthetic data generation
  features/        feature engineering
  models/          training & evaluation
  explainability/  SHAP
  decision/        decision engine
  simulation/      intervention simulator
api/               FastAPI service (predict/explain/decide/simulate + live mode)
dashboard/         Operations console dashboard
docs/              specs, data dictionary, final report, interview prep
```

## 🚀 Running it

```bash
docker compose up --build
```
API: `localhost:8000` · Dashboard: `localhost:3000`

For real weather/traffic in Live mode, create a `.env` file (see `docs/final_report.md` for setup) with:
```
OPENWEATHER_API_KEY=your_key
TOMTOM_API_KEY=your_key
```

## ⚠️ A note on honesty

Hyperlocal disruption prediction, weather-aware logistics, and rider rebalancing aren't new ideas — big quick-commerce players already do versions of this internally. What this project is actually testing: **can one system fuse those signals, name the dominant cause, pick a cause-appropriate fix, and prove the fix would help — before acting?** No claim beyond that. Full honest limitations in [`docs/final_report.md`](docs/final_report.md).

---
Built by [Dhakshin Kumar](https://github.com/DhakshinVS18) — Chennai
