# Within-Branch Sales Forecasting & Anomaly Detection — Plan

Owner: Aya · Workspace: `aya/` · Status: **core system built and evaluated**

## Scope

1. **Expected units sold.** For every product at every branch, the model gives the
   expected number of units sold for each of the next 14 days. It does not recommend prep quantities.
   Deciding how much to make is left to the branch.
2. **Within-branch anomaly alerts.** These compare actual sales with that expectation
   and flag unusual drops, spikes, possible stockouts and sustained shifts, with an
   estimate of the profit impact.

Each branch is judged only against its own history. Branches are never compared
with each other.

## What the data tells us

Measured on `cleaned_stories_transactions.csv` (51,045 lines, 2026-01-01 → 2026-06-30;
the 132 rows removed from the raw file were exact duplicates):

| Fact | Value | Consequence |
|---|---|---|
| Series (branch × product) | 3 × 45 = 135 | One global model across all series |
| Mean daily units per series | ~2 (10th pct ≈ 1, 90th ≈ 3.8) | Low counts → Poisson objective; product-level error is dominated by chance |
| Days with zero sales | 21% of branch-product-days | Missing days filled with 0 |
| Variance vs mean | ≈ 1.26 × mean | Mildly overdispersed → negative binomial spread for alerts |
| Weekday effect | Tue ≈ 1.65×, Sat ≈ 0.70× an average day, steady every month and in every branch | Weekday pattern is a core input |
| Trend | Monthly units +52% Jan → Jun (≈ 8% per month) | Model must extend growth beyond the training range |
| Product lifecycle | All 135 series sell throughout the period | No launch/discontinuation handling needed |
| Prices | ±5% month-to-month changes on every product | Price ratio used as an input |

**Split.** The shared `aya/data/` files were cut by transaction count, so boundary
days fall in two splits. We use a whole-day split instead and leave the shared
files untouched:

- Train: 2026-01-01 → 2026-05-15
- Validation: 2026-05-16 → 2026-06-08 (all model choices made here)
- Test: 2026-06-09 → 2026-06-30 (final evaluation)

## Folder layout

```
aya/
  PLAN.md, README.md
  run_pipeline.py          full run: validate → test → forecast tomorrow → outputs + report
  eda.py                   exploration printout + figures
  evaluate_forecast.py     validation: first model, NB check, anomaly injection tests
  compare_models.py        validation: raw vs linear-tree vs baseline-offset LightGBM
  compare_horizons.py      validation: 1-14 day forecasts, multi-day model vs baseline
  src/panel.py             transactions → daily branch × product panel, day-level split
  src/features.py          leak-safe features + baseline (level × weekday × growth)
  src/forecast.py          baselines, LightGBM model, negative binomial helpers, accuracy
  src/horizon.py           1-14 day forecast rows (inputs as of the forecast day) + multi-day model
  src/anomaly.py           alert scoring, flags, profit impact, synthetic injection
  src/report.py            HTML report
  src/product_classes.csv  product type (fresh / made to order / packaged), used as a model input
  tests/                   panel, no-leak features (next-day and 1-14 day)
  outputs/                 forecast + category forecast CSVs, test results, alerts, report
```

## Work done

| # | Step | Status |
|---|---|---|
| 1 | Daily panel + whole-day split, unit tests | ✅ |
| 2 | Forecast-oriented EDA (weekday, trend, sparsity, dispersion, lifecycle, prices, outlier days) | ✅ |
| 3 | Baselines (same weekday last week, weekday average) + LightGBM | ✅ |
| 4 | Fix busy-day under-forecast: baseline-offset model, chosen on validation | ✅ |
| 5 | Anomaly alerts at product-day, category-day and product 7-day levels, with profit impact | ✅ |
| 6 | Alert validation with planted synthetic anomalies | ✅ |
| 7 | HTML report per branch + test-period evaluation | ✅ |
| 8 | Independent review for data leaks and bugs; fixes applied | ✅ |
| 9 | 14-day forecast (one model across horizons), leak tests, second independent review | ✅ |

## How the model works

1. **Baseline:** recent 28-day level × the branch's weekday pattern (fitted on
   training days) × recent branch growth. This carries the trend and the weekday
   effect.
2. **LightGBM** (Poisson objective, `log(baseline)` as offset), one model for all
   135 series, learns product-specific adjustments on top of the baseline. Inputs
   are recent sales (lags 1/7/14/28, rolling means), branch traffic, calendar,
   price ratio and product type, all using data up to the previous day only.
3. **Output:** expected units sold per product, branch and day for the next 14
   days, plus totals per category and per branch. For a day *h* days ahead, the
   sales inputs are taken as of the forecast day; the calendar belongs to the
   target day.

## Results (test period, Jun 9 – 30)

| | Same weekday last week | First model (raw LightGBM) | **Final model** |
|---|---|---|---|
| Product-level error (WAPE) | 66% | 49% | **49%** |
| Bias | −0.6% | −3.3% | **−0.3%** |
| Tuesday bias | — | −8.8% | **−1.1%** |

- Error on each branch's daily total: Harbour 7%, Garden 11%, Uptown 16%.
- 14 days ahead is as accurate as 1 day ahead: 49.2% vs 49.1% product error,
  about 10% branch daily total error. A review confirmed this is not a leak.
- Product-level error is near the floor. Even a perfect expected value would leave
  about 47–51% WAPE, because a product selling ~2.5 units/day swings by ±1–2 units by
  chance.
- Alerts: 57 over 22 days (≈ 0.9 per branch per day).
- Detection tests on validation found:
  - single-day spikes: 100% caught
  - sustained 7-day drops: 95% caught
  - single-day product drops: not reliably detectable at these volumes, which is why
    the category and 7-day checks exist

## Decisions

| Decision | Why |
|---|---|
| Negative binomial spread instead of quantile LightGBM | Fits the data better than Poisson, gives a probability for any count, and avoids crossing quantiles. Used for alerts. |
| Baseline-offset LightGBM instead of raw | Raw tree models can't predict above training values, so they under-forecast a growing business on its busiest days. Offset removed the bias on validation. |
| Prep recommendations removed | Scope narrowed to expected units sold. Prep quantities, two-batch prep and the waste/profit simulation were built, then removed (still in git history: commit `4f9aba8`). |
| Alerts need both a low probability and a ≥ 3-unit gap | At ~2 units/day, one-unit swings are normal. This keeps false alerts below 1% of normal days. |
| One model for all 14 days | Same accuracy as the next-day model at 1 day ahead on validation (51.3% vs 51.4%), and consistent numbers across the 14-day table. |
| Previous day's price as input | The day's own average price comes from that day's transactions; using it would leak same-day information. |
| No tuning on test | All choices were made on validation. The test period has been scored after each model change; no run tuned anything. |

## Known limitations

- **Sales ≠ demand.** There is no stock data, so a day a product ran out looks like
  low demand. The model predicts sales, not true demand.
- Six months of history, so there is no yearly seasonality or holiday effect.
- Wednesdays are over-forecast by 12% on test (3 days only, and validation showed
  no such bias). Watch this if more data arrives.
- The EDA figures above were computed on the full file, test period included. They
  were used to understand the data, not to tune the model.

## Possible next steps

- Save the trained model and retrain weekly instead of on every run.
- Retrain daily through the test period (rolling-origin evaluation).
- Alerts still use the next-day model; they could be moved to the multi-day model
  for a single model throughout.
- Product-specific weekday patterns (small expected gain, ~0.5–1 point).
- External data (weather, holidays, local events) for the biggest remaining gains.
- Stock / waste records, to forecast true demand and to evaluate stockout alerts.
