\# 🚦 DeliveryGuard AI



\### Predicting delivery chaos before it happens.



> A hyperlocal quick-commerce zone might look fine on the city dashboard — until rain, traffic, a demand spike, and a rider shortage collide in the same neighborhood at the same hour. By the time SLA breaches show up in the numbers, it's already too late to fix it.



\*\*DeliveryGuard AI\*\* predicts \*where\* and \*when\* that collision is about to happen, explains \*why\*, recommends a fix, and simulates whether that fix would actually work — before anyone commits to it.



Predict → Explain → Decide → Simulate → Approve → Measure



\---



\## 🧠 What it does



| Stage | What happens |

|---|---|

| \*\*Predict\*\* | XGBoost estimates disruption risk per zone, 1 hour ahead |

| \*\*Explain\*\* | SHAP breaks the prediction down into its real drivers — rain? traffic? rider shortage? |

| \*\*Decide\*\* | A constrained decision engine picks a cause-appropriate fix — no LLM freelancing actions |

| \*\*Simulate\*\* | Before anything happens, the simulator projects the outcome of that fix vs. doing nothing |

| \*\*Approve\*\* | A human operator sees the full picture and signs off |

| \*\*Measure\*\* | Real outcomes get logged and compared against what the model predicted |



This isn't another "train XGBoost, report accuracy" project. The interesting part is what happens \*after\* the prediction.



\## ⚙️ Stack



`Python` `FastAPI` `XGBoost` `SHAP` `PostgreSQL` `React + Tailwind` `Plotly` `Docker`



\## 📍 Status



Currently in \*\*Week 1 of 8\*\* — problem definition and data architecture locked, moving into synthetic data generation next.



\- \[x] \*\*Week 1\*\* — Problem definition, Chennai zone map, full data dictionary → \[`docs/week1\_spec.md`](docs/week1\_spec.md)

\- \[ ] \*\*Week 2\*\* — Synthetic data generator (causal rain→traffic→delay relationships)

\- \[ ] \*\*Week 3\*\* — EDA + baseline models

\- \[ ] \*\*Week 4\*\* — Feature engineering + XGBoost

\- \[ ] \*\*Week 5\*\* — SHAP explainability + sensitivity analysis

\- \[ ] \*\*Week 6\*\* — Decision engine + intervention simulator

\- \[ ] \*\*Week 7\*\* — FastAPI backend + React dashboard

\- \[ ] \*\*Week 8\*\* — Deployment + full write-up



\## 🗂️ Structure



data/ raw, processed, synthetic datasets

notebooks/ EDA and experiments

src/

data/ ingestion \& validation

features/ feature engineering

models/ training \& evaluation

explainability/ SHAP

decision/ decision engine

simulation/ intervention simulator

api/ FastAPI service

dashboard/ React frontend

docs/ specs, data dictionary, reports



\## ⚠️ A note on honesty



Hyperlocal disruption prediction, weather-aware logistics, and rider rebalancing aren't new ideas — big quick-commerce players already do versions of this internally. What this project is actually testing: \*\*can one system fuse those signals, name the dominant cause, pick a cause-appropriate fix, and prove the fix would help — before acting?\*\* No claim beyond that. All results here are from synthetic/public data unless stated otherwise.



\---

Built by \[Dhakshin Kumar](https://github.com/DhakshinVS18) — Chennai

