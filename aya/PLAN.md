# Within-Branch Demand Forecasting & Anomaly Detection — Implementation Plan

Owner: Aya · Workspace: `aya/`

## What the data tells us up front

Measured on `cleaned_stories_transactions.csv` (51,045 lines, 2026-01-01 → 2026-06-30):

| Fact | Value | Consequence |
|---|---|---|
| Series (branch × product) | 3 × 45 = 135 | Small enough for one global model + per-series thresholds |
| Median daily units per series | ~2 (10th pct ≈ 1, 90th ≈ 3.8) | Low-count, intermittent demand → count/Tweedie models, not plain regression |
| Days with zero sales | 21% of branch-product-days | Must fill missing days with 0 before modelling |
| Branch volume | Harbour 3.2 · Garden 2.2 · Uptown 1.2 units/product/day | Evaluate and threshold per branch, never pooled |
| Weekday effect | Tue 1.52×, Sat 0.70× the average | Weekday is a primary feature |
| Trend | Monthly units 7.2k (Jan) → 11.0k (Jun), +52% | Need trend-aware features; a static average will under-forecast |
| Discounts | Mean 0.7%, max 15% | Minor driver, include as a feature |

**Split issue:** the existing `aya/data/` splits were cut by transaction count from the raw
file, so boundary days (May 16, Jun 9) fall in two splits. Daily forecasting needs
whole-day splits. We will leave the shared files untouched and build a day-level split:

- Train: 2026-01-01 → 2026-05-15
- Validation: 2026-05-16 → 2026-06-08
- Test: 2026-06-09 → 2026-06-30 (touched once, at the end)

## Folder layout

```
aya/
  PLAN.md
  notebooks/01_eda.ipynb          exploration + figures
  src/panel.py                    transactions → daily branch×product panel
  src/features.py                 lag/rolling/calendar features (leak-safe)
  src/forecast.py                 baselines + LightGBM models
  src/inventory.py                prep-quantity recommendation
  src/anomaly.py                  scoring, flags, financial impact
  outputs/                        forecasts.csv, prep_plan.csv, alerts.csv, report
  tests/                          unit tests for panel, features, anomaly logic
```

## Phase 1 — Forecast-oriented EDA

1. Compare raw vs cleaned files (132 rows removed) and document what cleaning did.
2. Build the daily panel; plot per-branch total units, weekday profile, monthly trend.
3. Product lifecycle: products launched or dropped mid-period per branch (these need
   special handling — no history ≠ zero demand).
4. Price/cost changes over time per product; discount frequency.
5. Distribution of daily counts per series (mean vs variance → Poisson vs overdispersed).
6. Look for obvious existing outlier days (data is synthetic — injected events are
   likely and useful later as sanity checks for the anomaly detector).
7. Hour-of-day profile per branch (useful later for intraday prep, optional).

Deliverable: `01_eda.ipynb` + a short findings section appended to this plan.

## Phase 2 — Data preparation

- `panel.py`: aggregate to `Branch, Product, Date → units, revenue, gross_profit, avg_price,
  avg_discount, avg_unit_cost`; reindex to the full date grid, fill units with 0.
- `features.py` (all computed from the past only, valid for the forecast horizon):
  - lags: t−1, t−7, t−14, t−28 (for horizon *h*, only lags ≥ *h*)
  - rolling means/std over 7, 14, 28 days; same-weekday mean of last 4 weeks
  - calendar: weekday, day of month, week of year, days since start (trend)
  - price, discount, unit margin
  - identifiers: branch, product, category (categorical)
  - series-level: branch-product share of branch volume (fit on train only)
- Fit any scaling/encoding on train only.

## Phase 3 — Demand forecasting

Horizons: **next day** (primary — "what to prepare tomorrow") and **next 7 days**.

Models, in order (each must beat the previous to stay):

1. Seasonal naive: same weekday last week.
2. Weekday-adjusted moving average: last-4-same-weekday mean × recent trend ratio.
3. **LightGBM global model, Tweedie/Poisson objective**, one model across all 135 series
   with branch/product features — shares weekday and trend structure across series
   while still learning each product's level within its branch.
4. Quantile LightGBM (P10, P50, P90) — gives the uncertainty band used by both
   inventory and anomaly detection.

Evaluation (per branch, then per product):
- MAE, WAPE, bias (systematic over/under-forecast)
- Interval coverage: share of actuals inside P10–P90 (target ≈ 80%)
- Rolling-origin backtest across the validation window (re-forecast each day)

## Phase 4 — Inventory preparation recommendation

Turn the forecast distribution into "how many to prepare" using the newsvendor rule:
prepare the quantile at critical ratio `underage / (underage + overage)`, where
underage (missed sale) = unit gross margin = price − cost. Overage depends on
whether the product is wasted if unsold, so each product gets a shelf-life class
(kept in `src/product_classes.csv`, editable by the team):

| Class | Products | Overage cost | Resulting quantile |
|---|---|---|---|
| **Fresh, prepared ahead** | Bakery, Breakfast, Sandwiches, Meals, Desserts, Fresh Orange Juice | Full unit cost (binned at close) | `margin / price` — typically P50–P65 |
| **Made to order** | Hot Coffee, Cold Coffee, Matcha & Tea, Lemonade, Peach Iced Tea | Small — only ingredients are stocked, and they keep | High, e.g. P90 → read as ingredient/staffing need, not finished units |
| **Packaged / shelf-stable** | Still Water, Sparkling Water | Holding cost only (≈ 5% of cost) | ≈ P95 → read as a reorder level, not daily prep |

High-margin fresh items get prepared more generously; low-margin ones more tightly.
Output `prep_plan.csv`: branch, product, class, date, P50 forecast, recommended quantity.

**Business evaluation:** simulate the validation period — profit, waste units and
missed sales under (a) the naive baseline and (b) our recommendation. This is the
headline number for the pitch.

## Phase 5 — Within-branch anomaly detection

For each branch-product-day, compare actual units with that series' own forecast:

1. **Score:** probability of seeing a value this extreme under the forecast
   distribution (negative binomial with dispersion fitted per series on validation
   residuals; fall back to quantile-band position if the fit is poor).
2. **Flag types:**
   - Demand drop / demand spike — single day outside the calibrated band
   - Possible stockout — zero or near-zero sales after a run of normal days
     (especially if sales stop at a specific time of day)
   - Sustained shift — CUSUM on residuals over several days (catches slow decline
     that single-day checks miss)
3. **Noise control:** with ~2 units/day, single-unit swings are normal. Require a
   minimum absolute gap (e.g. ≥ 3 units) as well as a low probability, and also run
   detection at category and branch-total level where counts are larger.
4. **Financial impact:** `(expected − actual) × unit gross margin` for drops (lost
   profit); for spikes, the margin gained and stockout risk for the next day.
5. **Ranking:** severity = probability score × financial impact → top alerts per
   branch per day, with a one-line explanation
   (e.g. "Uptown · Iced Latte sold 0 vs expected 6 (P10 = 3) · est. lost profit $14").

**Validating without labels:** inject synthetic anomalies (drops, spikes, zeroed
days) into a copy of the validation period and measure precision/recall; tune
thresholds per branch on that.

## Phase 6 — Outputs and presentation

- `forecasts.csv`, `prep_plan.csv`, `alerts.csv`
- A one-page HTML report or small dashboard per branch:
  tomorrow's prep list, products trending up/down, today's ranked alerts, and the
  simulated profit/waste improvement.

## Phase 7 — Final evaluation

Run the frozen pipeline once on the test window (Jun 9–30). Report forecast
accuracy, interval coverage, simulated profit gain and anomaly results. No tuning
after this step.

## Known limitations

- **Sales ≠ demand.** There is no stock data, so stockout days look like low demand
  and will bias forecasts down. Mitigation: exclude flagged stockout days from training.
- Six months of history — no yearly seasonality or holidays can be learned.
- Shelf-life classes are assumptions from product names; desserts and cookies may
  last more than one day.

## Time-boxed order of work (a few hours)

Must-have core, in order — stop at any step and still have something to show:

| # | Step | Time |
|---|---|---|
| 1 | Environment (`numpy, scikit-learn, lightgbm, matplotlib`), daily panel + day-level split | 20 min |
| 2 | Lean EDA: weekday/trend/sparsity plots, outlier days (skip lifecycle & price deep-dives) | 25 min |
| 3 | Seasonal-naive baseline + LightGBM Tweedie + P10/P50/P90 quantile models, next-day horizon only | 50 min |
| 4 | Prep quantities by shelf-life class + profit/waste simulation vs baseline | 30 min |
| 5 | Anomaly flags (drop/spike/stockout), financial impact, ranked alerts | 40 min |
| 6 | One-page HTML report per branch + final test-set run | 30 min |

Cut if short on time: 7-day horizon, CUSUM sustained-shift detection, synthetic
anomaly injection, unit tests beyond the panel/split logic.
