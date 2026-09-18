# DeliveryGuard AI — Resume Bullets & Interview Prep

## Resume bullets (pick 2-3, tailor to role)

- Built an end-to-end ML decision-support system (predict → explain → decide → simulate) for hyperlocal delivery disruption forecasting; XGBoost classifier achieved 0.965 ROC-AUC with verified probability calibration, outperforming rule-based and logistic regression baselines
- Designed and debugged a synthetic data generator with documented causal assumptions; diagnosed and fixed a statistical bug (lognormal median-mean equivalence) that was producing a 99% disruption rate, recalibrating to a realistic 24.5% base rate
- Built a constrained, auditable decision engine (5-action policy, no free-form LLM decisions) paired with an intervention simulator; demonstrated 28% relative reduction in mean predicted disruption risk from proactive vs. no intervention across 4,493 test scenarios
- Validated model reliability beyond accuracy: SHAP-based local/global explainability, threshold sensitivity analysis (tested 5 SLA-breach thresholds), and robustness testing under combined-shock scenarios (rain + traffic + rider shortage)
- Shipped a full-stack demo: FastAPI backend (4 endpoints) + custom ops-console dashboard with live and historical modes, containerized with Docker

## Likely interview questions + how to answer them honestly

**"Walk me through the project."**
Use the one-paragraph description you already have memorized from this conversation — problem, loop, headline number (28% risk reduction). Keep it under 90 seconds unless asked to go deeper.

**"Why XGBoost over other models?"**
Be honest: you tested a linear baseline first (logistic regression, 0.96 ROC-AUC) and XGBoost only marginally improved on it (0.965). Explain *why*: the causal structure you built is largely additive, so most signal is linear. XGBoost's real value is the small residual of nonlinear interactions (e.g., rain only matters combined with rider shortage). This answer is *stronger* than pretending XGBoost dramatically won — it shows you understand what tree ensembles are actually for.

**"How do you know your model isn't overfitting / your data isn't leaking?"**
Point to: (1) the explicit leakage checklist from Week 1, (2) the time-aware split (never random shuffle), (3) SHAP feature importance matching the intended causal structure rather than spurious correlations, (4) calibration curve showing genuinely trustworthy probabilities, not just rank ordering.

**"What was the hardest bug you hit?"**
Tell the 99%-disruption-rate story. This is your best technical story in the whole project — it shows real debugging (not just "I fixed a typo"), statistical reasoning (recognizing a lognormal median-mean property), and intellectual honesty (you didn't just cap the output, you fixed the causal model).

**"Is this actually novel?"**
No — and say so directly. Cite your own project brief's honesty: hyperlocal prediction, weather-aware logistics, and rider rebalancing all exist in industry already (Swiggy Instamart, Zepto, Blinkit have internal versions). Your contribution is the *engineering integration question*: fusing prediction, explanation, constrained decision-making, and pre-action simulation into one auditable loop, evaluated honestly at every stage. This is a mature, credible answer — claiming false novelty is a red flag to any technical interviewer.

**"How would this be different with real data?"**
You'd need: real weather/traffic/order APIs, real rider GPS/scheduling data, and — critically — real intervention outcomes to validate (or replace) the simulator's hand-tuned effect sizes with fitted ones. You'd also need to re-derive the SLA-breach threshold from actual business cost data (false-positive alert cost vs. missed-disruption cost) rather than an assumed 30%.

**"What would you do with more time?"**
Multi-hour cascading simulation (currently single-step), a real A/B-testable intervention framework, expanding beyond rider/traffic interventions to inventory scenarios (which barely triggered in this dataset), and partnering for real operational data.

**"Why synthetic data instead of a public dataset?"**
No public dataset exists at this zone-hour, multi-signal granularity for Indian quick-commerce — this is deliberately a data-scarce problem space. The alternative (using an unrelated public dataset) would have made the problem definition dishonest. Synthetic data with documented causal assumptions was the more defensible choice, and you validated it wasn't trivially separable before building on top of it.

## Demo script (for a 2-3 min video or live walkthrough)

1. **(10s)** One-line pitch: what DeliveryGuard predicts and why it matters
2. **(30s)** Show the dashboard: pick a high-risk zone-hour, point out risk %, SHAP explanation, recommended action
3. **(30s)** Show the before/after simulation number
4. **(20s)** Switch to Live mode, adjust rainfall slider, show risk react in real time
5. **(30s)** One slide/screen: the model comparison table + the 28% headline simulator result
6. **(20s)** Close: "built end-to-end, from problem definition through a working demo, over 8 weeks" — link to repo
